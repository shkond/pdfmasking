"""Error handling and edge case tests (Category D from testlist.md)."""

import pytest

from core.masker import Masker


class TestEdgeCases:
    """Tests for edge cases and error handling (D1-D7)."""

    def test_empty_text_input(self):
        """D1: Empty text input handling."""
        masker = Masker()
        result = masker.mask("", language="ja")

        assert result.masked_text == ""
        assert len(result.entities) == 0

    def test_whitespace_only_text(self):
        """D1 variant: Whitespace-only text."""
        masker = Masker()
        result = masker.mask("   \n\t  ", language="ja")

        assert result.masked_text.strip() == ""

    def test_special_characters_only(self):
        """D2: Special characters only text."""
        text = "!@#$%^&*()[]{}|;:',.<>?/"

        masker = Masker()
        result = masker.mask(text, language="ja")

        # Should return text unchanged (no PII) or with special char handling
        assert isinstance(result.masked_text, str)
        # entities can be empty when no PII found

    def test_unicode_special_characters(self):
        """D2 variant: Unicode special characters."""
        text = "〒☆★◎●○■□▲△▼▽"

        masker = Masker()
        result = masker.mask(text, language="ja")

        # Should handle unicode without errors
        assert len(result.masked_text) > 0

    def test_very_long_text(self):
        """Test handling of very long text."""
        # Create long text with PII
        base_text = "電話: 090-1234-5678\n"
        long_text = base_text * 100

        masker = Masker()
        result = masker.mask(long_text, language="ja")

        # All phone numbers should be masked
        assert "090-1234-5678" not in result.masked_text

    def test_mixed_encoding_text(self):
        """Test handling of mixed Japanese/English text."""
        text = "Name: John Smith 氏名: 山田太郎 Phone: 090-1234-5678"

        masker = Masker()
        result = masker.mask(text, language="ja")

        # Phone should be masked
        assert "090-1234-5678" not in result.masked_text

    def test_no_pii_text(self):
        """Test text with no PII."""
        text = "これはテスト文章です。個人情報は含まれていません。"

        masker = Masker()
        result = masker.mask(text, language="ja")

        # Text should be unchanged
        assert result.masked_text == text

    def test_partial_phone_number(self):
        """Test partial phone number patterns."""
        text = "電話: 090-1234"  # Incomplete phone number

        masker = Masker()
        result = masker.mask(text, language="ja")

        # Incomplete pattern may or may not be detected depending on regex
        # Just verify no error occurs
        assert isinstance(result.masked_text, str)


class TestTransformerAvailability:
    """Tests for transformer availability checks (D3)."""

    def test_transformer_import_check(self):
        """D3: Check transformer availability flag."""
        from recognizers import TRANSFORMER_AVAILABLE

        # Should be a boolean
        assert isinstance(TRANSFORMER_AVAILABLE, bool)


class TestConfigurationHandling:
    """Tests for configuration handling (D7)."""

    def test_load_config(self):
        """D7: Configuration file loading."""
        from config import load_config

        config = load_config()

        # Should load without error
        assert isinstance(config, dict)
        assert "transformer" in config or "models" in config

    def test_default_language_handling(self):
        """Test default language handling."""
        text = "Email: test@example.com"

        masker = Masker()
        result = masker.mask(text, language="en")
        
        assert "test@example.com" not in result.masked_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

