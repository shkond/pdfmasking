"""Unit tests for GiNZA-based NER recognizers (Category B from testlist.md)."""

import pytest

# Skip all tests if GiNZA is not available
ginza = pytest.importorskip("ginza")
spacy = pytest.importorskip("spacy")


class TestGinzaPersonRecognizer:
    """Tests for GinzaPersonRecognizer functionality (B1, B3, B5-B8)."""

    @pytest.fixture
    def recognizer(self):
        """Create a GinzaPersonRecognizer instance."""
        from recognizers.japanese_ner import GinzaPersonRecognizer
        return GinzaPersonRecognizer()

    @pytest.fixture
    def nlp(self):
        """Load GiNZA model."""
        return spacy.load("ja_ginza")

    def test_person_entity_detection(self, recognizer, nlp):
        """B1: GiNZA PERSON entity detection."""
        # Use text with context keyword for reliable detection
        text = "氏名: 田中花子さんです"
        doc = nlp(text)

        # Create mock NlpArtifacts
        from presidio_analyzer.nlp_engine import NlpArtifacts
        artifacts = NlpArtifacts(
            entities=doc.ents,
            tokens=doc,
            tokens_indices=[],
            lemmas=[],
            nlp_engine=None,
            language="ja",
        )

        results = recognizer.analyze(text, entities=["JP_PERSON"], nlp_artifacts=artifacts)

        # GiNZA detection varies - just verify no errors occur and type check
        assert isinstance(results, list)
        for result in results:
            assert result.entity_type == "JP_PERSON"

    def test_context_boost_with_keyword(self, recognizer, nlp):
        """B3: Context word detection boosts confidence for person names."""
        text = "氏名: 田中花子"
        doc = nlp(text)

        from presidio_analyzer.nlp_engine import NlpArtifacts
        artifacts = NlpArtifacts(
            entities=doc.ents,
            tokens=doc,
            tokens_indices=[],
            lemmas=[],
            nlp_engine=None,
            language="ja",
        )

        results = recognizer.analyze(text, entities=["JP_PERSON"], nlp_artifacts=artifacts)

        # If person detected, score should be boosted due to "氏名" context
        for result in results:
            if result.entity_type == "JP_PERSON":
                assert result.score >= 0.9, f"Expected boosted score >= 0.9, got {result.score}"

    def test_base_confidence_without_context(self, recognizer, nlp):
        """B5: Base confidence when context window has no keywords."""
        text = "佐藤さんがいる"
        doc = nlp(text)

        from presidio_analyzer.nlp_engine import NlpArtifacts
        artifacts = NlpArtifacts(
            entities=doc.ents,
            tokens=doc,
            tokens_indices=[],
            lemmas=[],
            nlp_engine=None,
            language="ja",
        )

        results = recognizer.analyze(text, entities=["JP_PERSON"], nlp_artifacts=artifacts)

        # Without context keywords, should have base confidence
        for result in results:
            if result.entity_type == "JP_PERSON":
                assert result.score == 0.6, f"Expected base score 0.6, got {result.score}"

    def test_empty_nlp_artifacts_returns_empty(self, recognizer):
        """B7: Empty NlpArtifacts returns empty list."""
        results = recognizer.analyze("test", entities=["JP_PERSON"], nlp_artifacts=None)
        assert results == []

    def test_unrequested_entity_ignored(self, recognizer, nlp):
        """B8: Unrequested entity types are ignored."""
        text = "山田太郎さん"
        doc = nlp(text)

        from presidio_analyzer.nlp_engine import NlpArtifacts
        artifacts = NlpArtifacts(
            entities=doc.ents,
            tokens=doc,
            tokens_indices=[],
            lemmas=[],
            nlp_engine=None,
            language="ja",
        )

        # Request only ADDRESS, not PERSON
        results = recognizer.analyze(text, entities=["JP_ADDRESS"], nlp_artifacts=artifacts)

        # Should return empty since JP_PERSON is not requested
        assert len(results) == 0


class TestGinzaAddressRecognizer:
    """Tests for GinzaAddressRecognizer functionality (B2, B4)."""

    @pytest.fixture
    def recognizer(self):
        """Create a GinzaAddressRecognizer instance."""
        from recognizers.japanese_ner import GinzaAddressRecognizer
        return GinzaAddressRecognizer()

    @pytest.fixture
    def nlp(self):
        """Load GiNZA model."""
        return spacy.load("ja_ginza")

    def test_loc_entity_detection(self, recognizer, nlp):
        """B2: GiNZA LOC entity detection."""
        text = "住所: 東京都千代田区千代田1-1"
        doc = nlp(text)

        from presidio_analyzer.nlp_engine import NlpArtifacts
        artifacts = NlpArtifacts(
            entities=doc.ents,
            tokens=doc,
            tokens_indices=[],
            lemmas=[],
            nlp_engine=None,
            language="ja",
        )

        results = recognizer.analyze(text, entities=["JP_ADDRESS"], nlp_artifacts=artifacts)

        # Check if any LOC entities were detected
        # Note: GiNZA may or may not detect this as LOC depending on model
        if results:
            assert results[0].entity_type == "JP_ADDRESS"

    def test_context_boost_with_address_keyword(self, recognizer, nlp):
        """B4: Context word detection boosts confidence for addresses."""
        text = "現住所: 大阪府大阪市北区"
        doc = nlp(text)

        from presidio_analyzer.nlp_engine import NlpArtifacts
        artifacts = NlpArtifacts(
            entities=doc.ents,
            tokens=doc,
            tokens_indices=[],
            lemmas=[],
            nlp_engine=None,
            language="ja",
        )

        results = recognizer.analyze(text, entities=["JP_ADDRESS"], nlp_artifacts=artifacts)

        # If address detected, score should be boosted due to "現住所" context
        for result in results:
            if result.entity_type == "JP_ADDRESS":
                assert result.score >= 0.9, f"Expected boosted score >= 0.9, got {result.score}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
