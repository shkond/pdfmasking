"""Error handling and edge case tests (Category D from testlist.md)."""

import pytest

from core.masker import mask_pii_in_text


class TestEdgeCases:
    """Tests for edge cases and error handling (D1-D7)."""

    def test_empty_text_input(self):
        """D1: Empty text input handling."""
        masked_text, entities = mask_pii_in_text("", language="ja", verbose=False)

        assert masked_text == ""
        assert entities is None or entities == []

    def test_whitespace_only_text(self):
        """D1 variant: Whitespace-only text."""
        masked_text, entities = mask_pii_in_text("   \n\t  ", language="ja", verbose=False)

        assert masked_text.strip() == ""

    def test_special_characters_only(self):
        """D2: Special characters only text."""
        text = "!@#$%^&*()[]{}|;:',.<>?/"

        masked_text, entities = mask_pii_in_text(text, language="ja", verbose=False)

        # Should return text unchanged (no PII) or with special char handling
        assert isinstance(masked_text, str)
        # entities can be empty list or None when no PII found

    def test_unicode_special_characters(self):
        """D2 variant: Unicode special characters."""
        text = "〒☆★◎●○■□▲△▼▽"

        masked_text, entities = mask_pii_in_text(text, language="ja", verbose=False)

        # Should handle unicode without errors
        assert len(masked_text) > 0

    def test_very_long_text(self):
        """Test handling of very long text."""
        # Create long text with PII
        base_text = "電話: 090-1234-5678\n"
        long_text = base_text * 100

        masked_text, entities = mask_pii_in_text(long_text, language="ja", verbose=False)

        # All phone numbers should be masked
        assert "090-1234-5678" not in masked_text

    def test_mixed_encoding_text(self):
        """Test handling of mixed Japanese/English text."""
        text = "Name: John Smith 氏名: 山田太郎 Phone: 090-1234-5678"

        masked_text, entities = mask_pii_in_text(text, language="ja", verbose=False)

        # Phone should be masked
        assert "090-1234-5678" not in masked_text

    def test_no_pii_text(self):
        """Test text with no PII."""
        text = "これはテスト文章です。個人情報は含まれていません。"

        masked_text, entities = mask_pii_in_text(text, language="ja", verbose=False)

        # Text should be unchanged
        assert masked_text == text

    def test_partial_phone_number(self):
        """Test partial phone number patterns."""
        text = "電話: 090-1234"  # Incomplete phone number

        masked_text, entities = mask_pii_in_text(text, language="ja", verbose=False)

        # Incomplete pattern may or may not be detected depending on regex
        # Just verify no error occurs
        assert isinstance(masked_text, str)


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
        from analyzer_factory import load_config

        config = load_config()

        # Should load without error
        assert isinstance(config, dict)
        assert "transformer" in config or "models" in config

    def test_default_language_handling(self):
        """Test default language handling."""
        text = "Email: test@example.com"

        # Should work with explicit language
        masked_text, entities = mask_pii_in_text(text, language="en", verbose=False)
        assert "test@example.com" not in masked_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
