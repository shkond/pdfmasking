"""GPT-based Japanese PII masking recognizer.

This recognizer integrates the public Hugging Face model:
- cameltech/japanese-gpt-1b-PII-masking

Important behavioral note:
The model *re-generates* the sentence while replacing PII spans with fixed tags
(e.g., <name>, <address>) and may introduce punctuation/format changes.
Therefore, this recognizer uses a tolerant alignment strategy to recover
start/end offsets in the original text.

Design goals:
- Plug into Presidio Analyzer as an EntityRecognizer
- Lazy-load model/tokenizer
- Best-effort span recovery with strict safety rails
  (discard uncertain spans; log discards)

Supported model tags (fixed 8):
<name>, <birthday>, <phone-number>, <mail-address>, <customer-id>,
<address>, <post-code>, <company>

Internal entity types (recommended mapping):
JP_PERSON, DATE_OF_BIRTH_JP, PHONE_NUMBER_JP, EMAIL_ADDRESS,
CUSTOMER_ID_JP, JP_ADDRESS, JP_ZIP_CODE, JP_ORGANIZATION
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from presidio_analyzer import EntityRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import NlpArtifacts

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


_MASKING_LOGGER_NAME = "masking"


_TAGS = (
    "<name>",
    "<birthday>",
    "<phone-number>",
    "<mail-address>",
    "<customer-id>",
    "<address>",
    "<post-code>",
    "<company>",
)

TAG_TO_ENTITY_DEFAULT: dict[str, str] = {
    "<name>": "JP_PERSON",
    "<birthday>": "DATE_OF_BIRTH_JP",
    "<phone-number>": "PHONE_NUMBER_JP",
    "<mail-address>": "EMAIL_ADDRESS",
    "<customer-id>": "CUSTOMER_ID_JP",
    "<address>": "JP_ADDRESS",
    "<post-code>": "JP_ZIP_CODE",
    "<company>": "JP_ORGANIZATION",
}

_TAG_SPLIT_RE = re.compile("(" + "|".join(re.escape(t) for t in _TAGS) + ")")


def _preprocess(text: str) -> str:
    return text.replace("\n", "<LB>")


def _postprocess(text: str) -> str:
    return text.replace("<LB>", "\n")


_TRANSLATION_TABLE = str.maketrans(
    {
        "：": ":",
        "；": ";",
        "？": "?",
        "！": "!",
        "（": "(",
        "）": ")",
        "［": "[",
        "］": "]",
        "｛": "{",
        "｝": "}",
        "　": " ",  # fullwidth space
    }
)


def _normalize_for_match(s: str) -> str:
    """Normalization which mostly preserves length (1-char to 1-char).

    We avoid collapsing whitespace or removing punctuation because we need
    positions on the original text.
    """
    return s.translate(_TRANSLATION_TABLE)


@dataclass(frozen=True)
class _SpanRecoveryConfig:
    min_anchor_len: int = 6
    short_anchor_len: int = 9
    search_window: int = 700
    # Fuzzy match ratio thresholds
    fuzzy_threshold_long: float = 0.80
    fuzzy_threshold_mid: float = 0.86
    fuzzy_threshold_short: float = 0.93
    # Span limits (safety rails)
    max_span_default: int = 120
    max_span_by_entity: dict[str, int] | None = None

    def max_span_for(self, entity_type: str) -> int:
        if self.max_span_by_entity and entity_type in self.max_span_by_entity:
            return self.max_span_by_entity[entity_type]
        return self.max_span_default

    def threshold_for_anchor_len(self, anchor_len: int) -> float:
        if anchor_len <= self.short_anchor_len:
            return self.fuzzy_threshold_short
        if anchor_len <= 20:
            return self.fuzzy_threshold_mid
        return self.fuzzy_threshold_long


class GPTPIIMaskerRecognizer(EntityRecognizer):
    """GPT-based PII masking recognizer.

    The model outputs a *masked* sentence using fixed tags.
    This recognizer aligns the model output back to the original text to
    produce Presidio RecognizerResult spans.

    Notes:
    - Best-effort. If span recovery is uncertain, the tag detection is discarded
      (but logged).
    - Production is assumed to run on GPU. If require_gpu=True and CUDA is not
      available, an error is raised.
    """

    def __init__(
        self,
        model_name: str = "cameltech/japanese-gpt-1b-PII-masking",
        supported_language: str = "ja",
        supported_entities: list[str] | None = None,
        device: str = "cuda",
        require_gpu: bool = True,
        preload_model: bool = False,
        generation_config: dict[str, Any] | None = None,
        tag_to_entity: dict[str, str] | None = None,
        span_recovery: _SpanRecoveryConfig | None = None,
        base_score: float = 0.85,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "torch and transformers are required for GPTPIIMaskerRecognizer"
            )

        self.model_name = model_name
        self.device = device
        self.require_gpu = require_gpu
        self.preload_model = preload_model
        self.generation_config = generation_config or {
            "max_new_tokens": 256,
            "num_beams": 3,
            "num_return_sequences": 1,
            "early_stopping": True,
            "repetition_penalty": 3.0,
        }
        self.tag_to_entity = tag_to_entity or dict(TAG_TO_ENTITY_DEFAULT)
        self.span_recovery = span_recovery or _SpanRecoveryConfig(
            max_span_default=120,
            max_span_by_entity={
                "JP_ADDRESS": 200,  # prefecture + city/ward (+ some context)
                "JP_PERSON": 40,
                "JP_ORGANIZATION": 80,
                "CUSTOMER_ID_JP": 40,
                "EMAIL_ADDRESS": 80,
                "PHONE_NUMBER_JP": 40,
                "JP_ZIP_CODE": 20,
                "DATE_OF_BIRTH_JP": 30,
            },
        )
        self.base_score = float(base_score)

        self._model = None
        self._tokenizer = None

        if supported_entities is None:
            supported_entities = sorted(set(self.tag_to_entity.values()))

        super().__init__(
            supported_entities=supported_entities,
            supported_language=supported_language,
        )

    @property
    def _logger(self) -> logging.Logger:
        # Use the same logger name as MaskingLogger to ensure file logging.
        return logging.getLogger(_MASKING_LOGGER_NAME)

    def load(self) -> None:
        """Presidio lifecycle hook.

        Presidio calls `load()` during recognizer construction.
        We intentionally keep this lightweight to avoid downloading large
        models during unit tests or registry initialization.

        If you want eager loading (e.g., warm-up at startup), pass
        `preload_model=True`.
        """
        if self.preload_model:
            self._ensure_model_loaded()

    def _ensure_model_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        if self.require_gpu and (not torch.cuda.is_available()):
            raise RuntimeError(
                "GPTPIIMaskerRecognizer requires CUDA GPU in production, "
                "but torch.cuda.is_available() is False"
            )

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name)

        # Device placement
        if self.device and self.device != "cpu" and torch.cuda.is_available():
            self._model = self._model.to(self.device)
        self._model.eval()

    def analyze(
        self,
        text: str,
        entities: list[str],
        nlp_artifacts: NlpArtifacts | None = None,
    ) -> list[RecognizerResult]:
        requested = set(entities) & set(self.supported_entities)
        if not requested:
            return []

        # Run model (or overridden in tests)
        masked_output = self._generate_masked_text(text)
        if not masked_output:
            return []

        # Recover spans
        return self._recover_spans(original_text=text, masked_text=masked_output, requested=requested)

    # ---- Model call (override-friendly) ----

    def _generate_masked_text(self, original_text: str) -> str:
        """Generate masked text (tags) from model.

        This method is intentionally split for unit tests.
        """
        self._ensure_model_loaded()

        instruction = "# タスク\n入力文中の個人情報をマスキングせよ\n\n# 入力文\n"
        input_text = instruction + original_text + "<SEP>"
        input_text = _preprocess(input_text)

        with torch.no_grad():
            token_ids = self._tokenizer.encode(
                input_text, add_special_tokens=False, return_tensors="pt"
            )
            token_ids = token_ids.to(self._model.device)

            generation_kwargs = dict(self.generation_config)
            generation_kwargs.setdefault("eos_token_id", self._tokenizer.eos_token_id)
            generation_kwargs.setdefault("pad_token_id", self._tokenizer.pad_token_id)

            output_ids = self._model.generate(token_ids, **generation_kwargs)

        # Decode only the generated continuation after the prompt tokens
        out = self._tokenizer.decode(
            output_ids.tolist()[0][token_ids.size(1) :],
            skip_special_tokens=True,
        )
        out = _postprocess(out)
        return out

    # ---- Span recovery ----

    def _recover_spans(
        self,
        *,
        original_text: str,
        masked_text: str,
        requested: set[str],
    ) -> list[RecognizerResult]:
        pieces = [p for p in _TAG_SPLIT_RE.split(masked_text) if p != ""]

        match_text = _normalize_for_match(original_text)
        cursor = 0

        last_anchor_end = 0
        pending_tag: str | None = None
        results: list[RecognizerResult] = []

        for piece in pieces:
            if piece in self.tag_to_entity:
                pending_tag = piece
                continue

            # text anchor
            anchor_raw = piece
            anchor = _normalize_for_match(anchor_raw).strip()
            if not anchor:
                continue

            # If anchor too short, treat as unreliable, but still advance if exact found.
            anchor_len = len(anchor)
            found = self._find_anchor(match_text, anchor, start=cursor)
            # Try a fuzzy match only if anchor is not too short
            if found is None and anchor_len >= self.span_recovery.min_anchor_len:
                found = self._fuzzy_find_anchor(match_text, anchor, start=cursor)

            if found is None:
                # Can't place this anchor. For safety, do not use it.
                continue

            anchor_start, anchor_end, anchor_score = found

            # If we have a pending tag, map it to the span between last_anchor_end and anchor_start.
            if pending_tag is not None:
                rr = self._build_result_from_pending_tag(
                    original_text=original_text,
                    pending_tag=pending_tag,
                    requested=requested,
                    span_start=last_anchor_end,
                    span_end=anchor_start,
                    anchor_score=float(anchor_score),
                    approx_pos=cursor,
                )
                if rr is not None:
                    results.append(rr)
                pending_tag = None

            # Advance cursor and last_anchor_end
            cursor = anchor_end
            last_anchor_end = anchor_end

        # If there was a trailing tag with no reliable right anchor: discard (but log)
        if pending_tag is not None:
            self._log_discard(original_text, pending_tag, "no_right_anchor", approx_pos=last_anchor_end)

        return results

    def _log_discard(
        self,
        original_text: str,
        tag: str,
        reason: str,
        *,
        approx_pos: int | None = None,
    ) -> None:
        ctx_pos = approx_pos or 0
        ctx = original_text[max(0, ctx_pos - 40) : min(len(original_text), ctx_pos + 40)]
        self._logger.info(f"[GPTPIIMasker] Discarded {tag}: {reason}. Context='{ctx}'")

    def _build_result_from_pending_tag(
        self,
        *,
        original_text: str,
        pending_tag: str,
        requested: set[str],
        span_start: int,
        span_end: int,
        anchor_score: float,
        approx_pos: int,
    ) -> RecognizerResult | None:
        entity_type = self.tag_to_entity.get(pending_tag)
        if not entity_type or entity_type not in requested:
            return None

        if span_end < span_start:
            self._log_discard(original_text, pending_tag, "negative_span", approx_pos=approx_pos)
            return None

        span_len = span_end - span_start
        if span_len <= 0:
            self._log_discard(original_text, pending_tag, "empty_span", approx_pos=span_start)
            return None

        max_allowed = self.span_recovery.max_span_for(entity_type)
        if span_len > max_allowed:
            self._log_discard(
                original_text,
                pending_tag,
                f"span_too_long(len={span_len}, max={max_allowed})",
                approx_pos=span_start,
            )
            return None

        score = max(0.0, min(1.0, self.base_score * float(anchor_score)))
        return RecognizerResult(entity_type=entity_type, start=span_start, end=span_end, score=score)

    def _find_anchor(
        self,
        haystack: str,
        needle: str,
        *,
        start: int,
    ) -> tuple[int, int, float] | None:
        idx = haystack.find(needle, start)
        if idx < 0:
            return None
        return idx, idx + len(needle), 1.0

    def _fuzzy_find_anchor(
        self,
        haystack: str,
        needle: str,
        *,
        start: int,
    ) -> tuple[int, int, float] | None:
        """Find the best approximate substring match for `needle` near `start`."""
        n = len(needle)
        if n < self.span_recovery.min_anchor_len:
            return None

        window = self.span_recovery.search_window
        end_limit = min(len(haystack), start + window)
        if end_limit - start < n:
            return None

        best_ratio = 0.0
        best_pos = None

        # Sliding window (bounded); safe enough for typical document sizes.
        for pos in range(start, end_limit - n + 1):
            chunk = haystack[pos : pos + n]
            ratio = SequenceMatcher(None, needle, chunk).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_pos = pos

        if best_pos is None:
            return None

        threshold = self.span_recovery.threshold_for_anchor_len(n)
        if best_ratio < threshold:
            return None

        return best_pos, best_pos + n, float(best_ratio)

