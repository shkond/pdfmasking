"""Unit tests for GPTPIIMaskerRecognizer.

These tests do NOT download or run the real model.
We monkeypatch the generation method to return a controlled output.
"""

import pytest


def test_gpt_pii_masker_span_recovery_basic(monkeypatch):
    gpt_module = pytest.importorskip("recognizers.gpt_pii_masker")
    GPTPIIMaskerRecognizer = gpt_module.GPTPIIMaskerRecognizer

    # Use explicit escapes for fullwidth punctuation to avoid lint warnings while preserving test intent.
    original = "オペレーター\uFF1Aお名前は山田太郎ですか\uFF1F会員IDはABC12345です。東京都練馬区在住です。"
    mocked_output = "オペレーター:お名前は<name>ですか?会員IDは<customer-id>です。<address>在住です。"

    rec = GPTPIIMaskerRecognizer(require_gpu=False, device="cpu")

    monkeypatch.setattr(rec, "_generate_masked_text", lambda _text: mocked_output)

    results = rec.analyze(
        original,
        entities=["JP_PERSON", "CUSTOMER_ID_JP", "JP_ADDRESS"],
    )

    # Convert to dict for easy assertions
    found = {(r.entity_type, original[r.start : r.end]) for r in results}

    assert ("JP_PERSON", "山田太郎") in found
    assert ("CUSTOMER_ID_JP", "ABC12345") in found
    assert ("JP_ADDRESS", "東京都練馬区") in found


def test_gpt_pii_masker_discards_when_no_right_anchor(monkeypatch):
    gpt_module = pytest.importorskip("recognizers.gpt_pii_masker")
    GPTPIIMaskerRecognizer = gpt_module.GPTPIIMaskerRecognizer

    original = "会員IDはABC12345です。"
    mocked_output = "会員IDは<customer-id>"  # no right anchor

    rec = GPTPIIMaskerRecognizer(require_gpu=False, device="cpu")
    monkeypatch.setattr(rec, "_generate_masked_text", lambda _text: mocked_output)

    results = rec.analyze(original, entities=["CUSTOMER_ID_JP"])
    assert results == []
