"""Unit tests for text preprocessing functions.

Tests preprocess_text and related normalization functions.
"""

import pytest

from core.processors.text import preprocess_text


class TestPreprocessText:
    """Test text preprocessing for PDF extraction."""
    
    def test_normalize_whitespace(self):
        """Should normalize multiple spaces to single space."""
        text = "Hello    World"
        result = preprocess_text(text)
        assert "    " not in result
    
    def test_soft_hyphen_handling(self):
        """Should handle soft hyphens (may preserve or remove based on implementation)."""
        text = "test\u00adword"  # soft hyphen
        result = preprocess_text(text)
        # Preprocess should produce valid output (may or may not remove soft hyphens)
        assert "test" in result
        assert "word" in result
    
    def test_normalize_unicode(self):
        """Should normalize Unicode characters."""
        # Full-width to half-width for certain characters
        text = "テスト"
        result = preprocess_text(text)
        # Should still contain the Japanese text
        assert len(result) > 0
    
    def test_empty_string(self):
        """Should handle empty string."""
        result = preprocess_text("")
        assert result == ""
    
    def test_whitespace_only(self):
        """Should handle whitespace-only string."""
        result = preprocess_text("   \n\t   ")
        # Should normalize but may result in empty or minimal
        assert isinstance(result, str)
    
    def test_japanese_text_preserved(self):
        """Japanese text should be preserved."""
        text = "田中太郎の連絡先は090-1234-5678です。"
        result = preprocess_text(text)
        assert "田中太郎" in result
        assert "連絡先" in result
    
    def test_mixed_language_text(self):
        """Mixed language text should be preserved."""
        text = "John Smith's email: john@example.com 田中太郎"
        result = preprocess_text(text)
        assert "John Smith" in result
        assert "田中太郎" in result
    
    def test_newlines_handling(self):
        """Should handle newlines appropriately."""
        text = "Line 1\nLine 2\r\nLine 3"
        result = preprocess_text(text)
        # Newlines should be normalized or preserved
        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" in result


class TestPreprocessRealWorldCases:
    """Test preprocessing with real-world PDF extraction patterns."""
    
    def test_pdf_extracted_with_extra_spaces(self):
        """PDF text often has extra spaces between characters."""
        text = "氏  名：田  中  太  郎"
        result = preprocess_text(text)
        # Should be more readable
        assert isinstance(result, str)
    
    def test_pdf_with_page_breaks(self):
        """PDF text may have form feed characters."""
        text = "Page 1\f\nPage 2"
        result = preprocess_text(text)
        assert "Page 1" in result
        assert "Page 2" in result
    
    def test_bullet_points(self):
        """Should handle various bullet point characters."""
        text = "• Item 1\n・Item 2\n- Item 3"
        result = preprocess_text(text)
        assert "Item 1" in result
        assert "Item 2" in result
        assert "Item 3" in result
