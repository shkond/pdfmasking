"""Unit tests for TextExtractor class.

Tests the new TextExtractor class and MockTextExtractor.
"""

import os
import tempfile
from pathlib import Path

import pytest

from file_io.extractors import TextExtractor, MockTextExtractor, extract_text


class TestTextExtractor:
    """Test TextExtractor class."""
    
    def test_extract_nonexistent_file(self):
        """Should raise FileNotFoundError for missing file."""
        extractor = TextExtractor()
        
        with pytest.raises(FileNotFoundError):
            extractor.extract("/nonexistent/file.pdf")
    
    def test_extract_unsupported_format(self, tmp_path):
        """Should raise ValueError for unsupported formats."""
        # Create a temp file with unsupported extension
        test_file = tmp_path / "test.xyz"
        test_file.write_text("test content")
        
        extractor = TextExtractor()
        
        with pytest.raises(ValueError) as exc_info:
            extractor.extract(str(test_file))
        
        assert "Unsupported file format" in str(exc_info.value)


class TestMockTextExtractor:
    """Test MockTextExtractor for testing purposes."""
    
    def test_returns_configured_text(self):
        """Should return the configured text."""
        extractor = MockTextExtractor("This is test content")
        
        result = extractor.extract("any/path.pdf")
        
        assert result == "This is test content"
    
    def test_tracks_extract_calls(self):
        """Should track all extract calls."""
        extractor = MockTextExtractor("test")
        
        extractor.extract("file1.pdf")
        extractor.extract("file2.pdf")
        
        assert len(extractor.extract_called_with) == 2
        assert "file1.pdf" in extractor.extract_called_with
        assert "file2.pdf" in extractor.extract_called_with
    
    def test_empty_return_text(self):
        """Should handle empty return text."""
        extractor = MockTextExtractor("")
        
        result = extractor.extract("any.pdf")
        
        assert result == ""


class TestExtractTextFunction:
    """Test the backward-compatible extract_text function."""
    
    def test_file_not_found(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            extract_text("/nonexistent/file.pdf")
    
    def test_unsupported_format(self, tmp_path):
        """Should raise ValueError for unsupported formats."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")
        
        with pytest.raises(ValueError):
            extract_text(str(test_file))
