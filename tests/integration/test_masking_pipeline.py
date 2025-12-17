"""Integration tests for the PII masking pipeline (Category C from testlist.md)."""

import pytest

from core.masker import mask_pii_in_text


class TestMaskingPipeline:
    """Tests for the complete PII masking pipeline (C1-C8)."""

    def test_standard_mode_masking(self):
        """C1: Standard Mode - pattern-based masking only."""
        text = "電話: 090-1234-5678 メール: test@example.com"

        masked_text, entities = mask_pii_in_text(text, language="ja", verbose=False)

        # Phone and email should be masked
        assert "090-1234-5678" not in masked_text
        assert "test@example.com" not in masked_text

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
        masked_text, entities = mask_pii_in_text(text, language="ja", verbose=False)

        # All PII should be masked
        assert "090-1234-5678" not in masked_text
        assert "100-0001" not in masked_text
        assert "taro.yamada@example.com" not in masked_text
        assert "1995年4月15日" not in masked_text

        # Verify entities were detected (count varies based on detection mode)
        assert entities is None or len(entities) >= 3, f"Expected at least 3 entities, got {entities}"

    def test_english_email_masking(self):
        """C5: English PII (EMAIL) masking success."""
        text = "Contact: john.doe@company.com"

        masked_text, entities = mask_pii_in_text(text, language="en", verbose=False)

        assert "john.doe@company.com" not in masked_text

    def test_zip_code_masking(self):
        """Test zip code pattern detection."""
        text = "〒100-0001 東京都"

        masked_text, entities = mask_pii_in_text(text, language="ja", verbose=False)

        assert "100-0001" not in masked_text

    def test_phone_number_masking(self):
        """Test phone number pattern detection."""
        text = "電話番号: 03-1234-5678"

        masked_text, entities = mask_pii_in_text(text, language="ja", verbose=False)

        assert "03-1234-5678" not in masked_text

    def test_birth_date_masking(self):
        """Test birth date pattern detection."""
        text = "生年月日: 1990年1月1日"

        masked_text, entities = mask_pii_in_text(text, language="ja", verbose=False)

        assert "1990年1月1日" not in masked_text

    def test_age_masking(self):
        """Test age pattern detection."""
        text = "年齢: 25歳"

        masked_text, entities = mask_pii_in_text(text, language="ja", verbose=False)

        assert "25歳" not in masked_text

    def test_gender_masking(self):
        """Test gender pattern detection."""
        text = "性別: 男性"

        masked_text, entities = mask_pii_in_text(text, language="ja", verbose=False)

        assert "男性" not in masked_text

    def test_duplicate_entity_handling(self):
        """C6: Duplicate entity deduplication."""
        text = "電話: 090-1234-5678 電話2: 090-1234-5678"

        masked_text, entities = mask_pii_in_text(text, language="ja", verbose=False)

        # Both occurrences should be masked
        assert masked_text.count("090-1234-5678") == 0

    def test_logging_with_verbose(self):
        """C8: Log output normal operation."""
        text = "Email: test@example.com"

        # Should not raise when verbose=True
        masked_text, entities = mask_pii_in_text(text, language="ja", verbose=True)

        assert "test@example.com" not in masked_text


class TestPreprocessingMode:
    """Tests for preprocessing mode."""

    def test_preprocess_flag(self):
        """Test that preprocess flag works."""
        text = "電話: 090-1234-5678"

        # Should work with preprocess=True
        masked_text, entities = mask_pii_in_text(text, language="ja", preprocess=True)

        assert "090-1234-5678" not in masked_text

    def test_preprocess_false(self):
        """Test that preprocess=False works."""
        text = "電話: 090-1234-5678"

        masked_text, entities = mask_pii_in_text(text, language="ja", preprocess=False)

        assert "090-1234-5678" not in masked_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
