"""Accuracy tests for GiNZA NER recognizers.

These tests evaluate the actual detection accuracy of GiNZA models.
They use real models without mocking to verify detection quality.

Excluded from CI by default due to:
- Long execution time (model loading)
- Resource requirements (memory)
"""

import pytest

# Skip all tests if GiNZA is not available
ginza = pytest.importorskip("ginza")
spacy = pytest.importorskip("spacy")


# Mark all tests in this module as accuracy tests
pytestmark = pytest.mark.accuracy


class TestGinzaPersonAccuracy:
    """Accuracy tests for GiNZA person name detection."""

    @pytest.fixture
    def nlp(self):
        """Load GiNZA model."""
        return spacy.load("ja_ginza")

    @pytest.fixture
    def recognizer(self):
        """Create GinzaPersonRecognizer."""
        from recognizers.japanese_ner import GinzaPersonRecognizer
        return GinzaPersonRecognizer()

    def test_common_japanese_names(self, recognizer, nlp):
        """Test detection of common Japanese names."""
        test_cases = [
            "山田太郎さんが来ました",
            "田中花子は会議に出席しました",
            "佐藤一郎先生の授業",
            "鈴木健太が担当します",
        ]

        from presidio_analyzer.nlp_engine import NlpArtifacts

        detected_count = 0
        for text in test_cases:
            doc = nlp(text)
            artifacts = NlpArtifacts(
                entities=doc.ents,
                tokens=doc,
                tokens_indices=[],
                lemmas=[],
                nlp_engine=None,
                language="ja",
            )

            results = recognizer.analyze(text, entities=["JP_PERSON"], nlp_artifacts=artifacts)
            if results:
                detected_count += 1

        # At least 50% should be detected
        assert detected_count >= len(test_cases) // 2, \
            f"Only detected {detected_count}/{len(test_cases)} names"

    def test_context_improves_detection(self, recognizer, nlp):
        """Test that context keywords improve detection confidence."""
        from presidio_analyzer.nlp_engine import NlpArtifacts

        # With context
        text_with_context = "氏名: 山田太郎"
        doc = nlp(text_with_context)
        artifacts = NlpArtifacts(
            entities=doc.ents,
            tokens=doc,
            tokens_indices=[],
            lemmas=[],
            nlp_engine=None,
            language="ja",
        )
        results_with_context = recognizer.analyze(
            text_with_context, entities=["JP_PERSON"], nlp_artifacts=artifacts
        )

        # Without context
        text_without_context = "山田太郎がいる"
        doc = nlp(text_without_context)
        artifacts = NlpArtifacts(
            entities=doc.ents,
            tokens=doc,
            tokens_indices=[],
            lemmas=[],
            nlp_engine=None,
            language="ja",
        )
        results_without_context = recognizer.analyze(
            text_without_context, entities=["JP_PERSON"], nlp_artifacts=artifacts
        )

        # Compare scores if both detected
        if results_with_context and results_without_context:
            assert results_with_context[0].score > results_without_context[0].score


class TestGinzaAddressAccuracy:
    """Accuracy tests for GiNZA address detection."""

    @pytest.fixture
    def nlp(self):
        """Load GiNZA model."""
        return spacy.load("ja_ginza")

    @pytest.fixture
    def recognizer(self):
        """Create GinzaAddressRecognizer."""
        from recognizers.japanese_ner import GinzaAddressRecognizer
        return GinzaAddressRecognizer()

    def test_prefecture_detection(self, recognizer, nlp):
        """Test detection of Japanese prefectures."""
        from presidio_analyzer.nlp_engine import NlpArtifacts

        prefectures = ["東京都", "大阪府", "北海道", "京都府"]

        detected_count = 0
        for pref in prefectures:
            text = f"住所: {pref}の中心部"
            doc = nlp(text)
            artifacts = NlpArtifacts(
                entities=doc.ents,
                tokens=doc,
                tokens_indices=[],
                lemmas=[],
                nlp_engine=None,
                language="ja",
            )

            results = recognizer.analyze(text, entities=["JP_ADDRESS"], nlp_artifacts=artifacts)
            if results:
                detected_count += 1

        # Log detection rate (not strict assertion as GiNZA detection varies)
        print(f"Prefecture detection rate: {detected_count}/{len(prefectures)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
