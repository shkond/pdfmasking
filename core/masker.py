"""PII Masker - Main masking functionality.

This module provides:
- Masker class: Core masking with dependency injection
- PIIMasker class: Backward-compatible wrapper (deprecated)
- mask_pii_in_text function: Backward-compatible function (deprecated)

Layer: Domain
Dependencies: Protocols from core.protocols
"""

import re

from datetime import datetime
from typing import Any

from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from config import (
    get_detection_strategy,
    get_entities_to_mask,
    get_entity_categories,
    get_transformer_config,
    load_config,
)
from core.allow_list import get_allow_list
from core.analyzer import create_analyzer, create_multilingual_analyzer
from core.masking_result import MaskingResult
from core.processors.hybrid_detection import hybrid_detection_analyze
from core.processors.result import deduplicate_results, merge_results
from core.processors.text import preprocess_text
from core.protocols import AnonymizerProtocol, LoggerProtocol, NullLogger


_PERSON_TEXT_RE = re.compile(r"[A-Za-z\u3040-\u30FF\u4E00-\u9FFF]")


def _is_meaningful_entity(entity_text: str, entity_type: str) -> bool:
    """Best-effort filter to drop obvious garbage entities.

    Some PDF extractions include stray punctuation tokens (e.g., "~") that
    can be falsely tagged as a person by NER. Those inflate counts and cause
    downstream masking/logging noise.
    """
    if not entity_text:
        return False
    stripped = entity_text.strip()
    if not stripped:
        return False

    if entity_type in {"JP_PERSON", "PERSON"}:
        # Require at least one plausible name character.
        if not _PERSON_TEXT_RE.search(stripped):
            return False

    return True


def build_operators(config: dict[str, Any]) -> dict:
    """
    Build anonymizer operators from config.
    
    Reads masking.entity_masks from config.yaml to create entity-specific
    mask patterns (e.g., phone: ***-****-****, zip: ***-****).
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dict of entity_type -> OperatorConfig for AnonymizerEngine
    """
    masking_cfg = config.get("masking", {})
    default_mask = masking_cfg.get("default_mask", "****")
    entity_masks = masking_cfg.get("entity_masks", {})
    
    operators = {
        "DEFAULT": OperatorConfig("replace", {"new_value": default_mask})
    }
    for entity_type, mask in entity_masks.items():
        operators[entity_type] = OperatorConfig("replace", {"new_value": mask})
    return operators


class Masker:
    """Core PII Masker with dependency injection.
    
    Provides clean separation of concerns:
    - Accepts dependencies via constructor (DI)
    - Uses Protocol-based interfaces for testability
    - Returns structured MaskingResult
    
    Targets 8 PII types:
    - 名前 (PERSON, JP_PERSON)
    - メールアドレス (EMAIL_ADDRESS)
    - 郵便番号 (JP_ZIP_CODE)
    - 電話番号 (PHONE_NUMBER_JP)
    - 生年月日 (DATE_OF_BIRTH_JP)
    - 住所 (JP_ADDRESS, LOCATION)
    - 性別 (JP_GENDER)
    - 年齢 (JP_AGE)
    
    Usage:
        masker = Masker(anonymizer=AnonymizerEngine())
        result = masker.mask(text, language="ja")
        print(result.masked_text)
    """
    
    def __init__(
        self,
        anonymizer: AnonymizerProtocol | None = None,
        logger: LoggerProtocol | None = None,
        config: dict[str, Any] | None = None
    ):
        """Initialize masker with dependencies.
        
        Args:
            anonymizer: Anonymizer implementation (default: AnonymizerEngine)
            logger: Logger implementation (default: NullLogger)
            config: Configuration dict (default: load from config.yaml)
        """
        self.anonymizer = anonymizer or AnonymizerEngine()
        self.logger = logger or NullLogger()
        self.config = config or load_config()
        
        # Cache config values
        self._transformer_cfg = get_transformer_config(self.config)
        self._detection_strategy = get_detection_strategy(self.config)
        self._operators = build_operators(self.config)
        self._allow_list = get_allow_list(self.config)
    
    def analyze(
        self,
        text: str,
        language: str = "auto",
        do_preprocess: bool = False
    ) -> list:
        """Analyze text and return detection results.
        
        Args:
            text: Text to analyze
            language: Language code ("en", "ja", or "auto")
            do_preprocess: If True, normalize text before analysis
            
        Returns:
            List of RecognizerResult
        """
        if do_preprocess:
            text = preprocess_text(text)
        
        use_transformer = self._transformer_cfg.get("enabled", False)
        transformer_entities = self._detection_strategy.get("transformer_entities", [])
        pattern_entities = self._detection_strategy.get("pattern_entities", [])
        
        if use_transformer:
            results = hybrid_detection_analyze(
                text=text,
                transformer_entities=transformer_entities,
                pattern_entities=pattern_entities,
                language=language,
                app_config=self.config,
                allow_list=self._allow_list
            )
        else:
            # When ML detection is disabled, fall back to rule-based / spaCy / GiNZA
            # for *all* entities to avoid dropping PERSON/ADDRESS/etc.
            all_entities = list(dict.fromkeys([*pattern_entities, *transformer_entities]))
            
            if language == "auto":
                analyzer = create_multilingual_analyzer(use_ginza=True, use_transformer=False)
                results_en = analyzer.analyze(
                    text=text, language="en", entities=all_entities, allow_list=self._allow_list
                )
                results_ja = analyzer.analyze(
                    text=text, language="ja", entities=all_entities, allow_list=self._allow_list
                )
                results = merge_results(results_en, results_ja)
            else:
                analyzer = create_analyzer(language=language, use_transformer=False)
                results = analyzer.analyze(
                    text=text, language=language, entities=all_entities, allow_list=self._allow_list
                )
        
        results = [
            r for r in results
            if _is_meaningful_entity(text[r.start:r.end], getattr(r, "entity_type", ""))
        ]
        return deduplicate_results(results, text)
    
    def mask(
        self,
        text: str,
        language: str = "auto",
        do_preprocess: bool = False,
        log_results: bool = True
    ) -> MaskingResult:
        """Mask PII in text.
        
        Args:
            text: Text to analyze and mask
            language: Language code ("en", "ja", or "auto")
            do_preprocess: If True, normalize text before analysis
            log_results: If True, log detected entities
            
        Returns:
            MaskingResult with masked text and entity info
        """
        # Preprocess if requested
        if do_preprocess:
            text = preprocess_text(text)
            # Don't preprocess again in analyze
            results = self.analyze(text, language, do_preprocess=False)
        else:
            results = self.analyze(text, language, do_preprocess=False)
        
        # Log results
        if log_results and results:
            self._log_results(text, results)
        
        # Anonymize
        anonymized = self.anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators=self._operators,
        )
        
        return MaskingResult.from_anonymizer_result(
            anonymized_text=anonymized.text,
            original_text=text,
            analyzer_results=results
        )
    
    def _log_results(self, text: str, results: list) -> None:
        """Log detected entities.
        
        Args:
            text: Original text
            results: List of RecognizerResult
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logger.log(f"\n{'='*60}")
        self.logger.log(f"Masking Log - {timestamp}")
        self.logger.log(f"{'='*60}")
        for result in results:
            entity_text = text[result.start:result.end]
            self.logger.log(
                f"[{result.entity_type}] \"{entity_text}\" "
                f"(score: {result.score:.2f}, pos: {result.start}-{result.end})"
            )
        self.logger.log(f"Total: {len(results)} entities masked")

