"""Integration tests for the PII masking pipeline (Category C from testlist.md)."""

import pytest

from core.masker import Masker


class TestMaskingPipeline:
    """Tests for the complete PII masking pipeline (C1-C8)."""

    def test_standard_mode_masking(self):
        """C1: Standard Mode - pattern-based masking only."""
        text = "電話: 090-1234-5678 メール: test@example.com"

        masker = Masker()
        result = masker.mask(text, language="ja")

        # Phone and email should be masked
        assert "090-1234-5678" not in result.masked_text
        assert "test@example.com" not in result.masked_text

    def test_japanese_pii_all_types(self):
        """C4: Japanese 8 PII types masking success."""
        text = """氏名: 山田太郎
ふりがな: やまだたろう
生年月日: 1995年4月15日（29歳）
性別: 男性
〒100-0001
東京都千代田区千代田1-1
電話: 090-1234-5678
Email: taro.yamada@example.com
"""
        masker = Masker()
        result = masker.mask(text, language="ja")

        # All PII should be masked
        assert "090-1234-5678" not in result.masked_text
        assert "100-0001" not in result.masked_text
        assert "taro.yamada@example.com" not in result.masked_text
        assert "1995年4月15日" not in result.masked_text

        # Verify entities were detected (count varies based on detection mode)
        assert len(result.entities) >= 3, f"Expected at least 3 entities, got {result.entities}"

    def test_english_email_masking(self):
        """C5: English PII (EMAIL) masking success."""
        text = "Contact: john.doe@company.com"

        masker = Masker()
        result = masker.mask(text, language="en")

        assert "john.doe@company.com" not in result.masked_text

    def test_zip_code_masking(self):
        """Test zip code pattern detection."""
        text = "〒100-0001 東京都"

        masker = Masker()
        result = masker.mask(text, language="ja")

        assert "100-0001" not in result.masked_text

    def test_phone_number_masking(self):
        """Test phone number pattern detection."""
        text = "電話番号: 03-1234-5678"

        masker = Masker()
        result = masker.mask(text, language="ja")

        assert "03-1234-5678" not in result.masked_text

    def test_birth_date_masking(self):
        """Test birth date pattern detection."""
        text = "生年月日: 1990年1月1日"

        masker = Masker()
        result = masker.mask(text, language="ja")

        assert "1990年1月1日" not in result.masked_text

    def test_age_masking(self):
        """Test age pattern detection."""
        text = "年齢: 25歳"

        masker = Masker()
        result = masker.mask(text, language="ja")

        assert "25歳" not in result.masked_text

    def test_gender_masking(self):
        """Test gender pattern detection."""
        text = "性別: 男性"

        masker = Masker()
        result = masker.mask(text, language="ja")

        assert "男性" not in result.masked_text

    def test_duplicate_entity_handling(self):
        """C6: Duplicate entity deduplication."""
        text = "電話: 090-1234-5678 電話2: 090-1234-5678"

        masker = Masker()
        result = masker.mask(text, language="ja")

        # Both occurrences should be masked
        assert result.masked_text.count("090-1234-5678") == 0

    def test_logging_with_entities(self):
        """C8: Verify entities are returned correctly."""
        text = "Email: test@example.com"

        masker = Masker()
        result = masker.mask(text, language="ja")

        assert "test@example.com" not in result.masked_text
        assert len(result.entities) >= 1


class TestPreprocessingMode:
    """Tests for preprocessing mode."""

    def test_preprocess_flag(self):
        """Test that preprocess flag works."""
        text = "電話: 090-1234-5678"

        masker = Masker()
        result = masker.mask(text, language="ja", do_preprocess=True)

        assert "090-1234-5678" not in result.masked_text

    def test_preprocess_false(self):
        """Test that preprocess=False works."""
        text = "電話: 090-1234-5678"

        masker = Masker()
        result = masker.mask(text, language="ja", do_preprocess=False)

        assert "090-1234-5678" not in result.masked_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

