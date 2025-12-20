"""Tests for RecognizerRegistry functionality."""

import pytest

from recognizers.registry import (
    create_default_registry,
)


class TestRecognizerRegistry:
    """Tests for RecognizerRegistry class."""

    def test_create_default_registry(self):
        """Test that create_default_registry creates a registry with all recognizers."""
        registry = create_default_registry(use_ginza=True)

        # Should have at least 7 pattern-based recognizers
        pattern_recognizers = registry.get_by_type("pattern")
        assert len(pattern_recognizers) >= 7

        # All should be Japanese language
        ja_recognizers = registry.get_by_language("ja")
        assert len(ja_recognizers) >= 7

    def test_get_by_type_pattern(self):
        """Test filtering by pattern type."""
        registry = create_default_registry(use_ginza=False)

        patterns = registry.get_by_type("pattern")
        assert len(patterns) == 7

        for config in patterns:
            assert config.type == "pattern"

    def test_get_by_language(self):
        """Test filtering by language."""
        registry = create_default_registry(use_ginza=False)

        ja = registry.get_by_language("ja")
        assert len(ja) == 7

        # No English recognizers in default registry (Japanese-focused)
        en = registry.get_by_language("en")
        assert len(en) == 0

    def test_summary_output(self):
        """Test that summary generates readable output."""
        registry = create_default_registry(use_ginza=False)

        summary = registry.summary()

        assert "Recognizer Registry Summary:" in summary
        assert "PATTERN:" in summary
        assert "PHONE_NUMBER_JP" in summary
        assert "JP_ZIP_CODE" in summary
        assert "Total: 7 recognizers" in summary

    def test_apply_to_analyzer(self):
        """Test applying recognizers to analyzer."""
        from presidio_analyzer import AnalyzerEngine

        registry = create_default_registry(use_ginza=False)
        analyzer = AnalyzerEngine(supported_languages=["ja"])

        # Apply all recognizers
        registry.apply_to_analyzer(analyzer)

        # Check that recognizers were added (use all_fields=True)
        recognizers = analyzer.registry.get_recognizers(language="ja", all_fields=True)
        assert len(recognizers) >= 7

    def test_apply_to_analyzer_with_language_filter(self):
        """Test applying recognizers with language filter."""
        from presidio_analyzer import AnalyzerEngine

        registry = create_default_registry(use_ginza=False)
        analyzer = AnalyzerEngine(supported_languages=["ja", "en"])

        # Apply only Japanese recognizers
        registry.apply_to_analyzer(analyzer, language="ja")

        # Check that recognizers were added for Japanese (use all_fields=True)
        ja_recognizers = analyzer.registry.get_recognizers(language="ja", all_fields=True)
        assert len(ja_recognizers) >= 7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
