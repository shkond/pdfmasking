"""Unit tests for structure_restorer module."""

import pytest
from core.processors.structure_restorer import StructureRestorer, TextSegment


class TestStructureRestorer:
    """Tests for StructureRestorer class."""
    
    @pytest.fixture
    def restorer(self):
        """Create a StructureRestorer instance with default config."""
        return StructureRestorer()
    
    def test_normalize_text_fullwidth_spaces(self, restorer):
        """Test normalization of full-width spaces."""
        text = "テスト　テキスト"
        result = restorer._normalize_text(text)
        assert "　" not in result
        assert " " in result
    
    def test_normalize_text_multiple_spaces(self, restorer):
        """Test collapsing multiple spaces."""
        text = "test    text"
        result = restorer._normalize_text(text)
        assert "    " not in result
    
    def test_split_into_lines(self, restorer):
        """Test line splitting."""
        text = "line1\nline2\n\nline3"
        lines = restorer._split_into_lines(text)
        assert len(lines) == 3
        assert lines == ["line1", "line2", "line3"]
    
    def test_detect_japanese_section_headings(self, restorer):
        """Test detection of Japanese section headings."""
        text = "名前: 田中太郎\n学歴\n東京大学\n職歴\nGoogleで勤務"
        lines = restorer._split_into_lines(text)
        
        headings = restorer._detect_sections(lines, text)
        
        heading_types = [h.section_type for h in headings]
        assert "education" in heading_types
        assert "experience" in heading_types
    
    def test_detect_english_section_headings(self, restorer):
        """Test detection of English section headings."""
        text = "Name: John Smith\nEDUCATION\nStanford University\nWORK EXPERIENCE\nWorked at Google"
        lines = restorer._split_into_lines(text)
        
        headings = restorer._detect_sections(lines, text)
        
        heading_types = [h.section_type for h in headings]
        assert "education" in heading_types
        assert "experience" in heading_types
    
    def test_restore_creates_segments(self, restorer):
        """Test that restore creates proper segments."""
        text = "Jane Smith\nEmail: jane@test.com\nEDUCATION\nStanford University"
        
        segments = restorer.restore(text)
        
        assert len(segments) > 0
        assert all(isinstance(s, TextSegment) for s in segments)
    
    def test_segments_have_section_context(self, restorer):
        """Test that segments have proper section types."""
        text = "名前: 田中太郎\n学歴\n東京大学 2015年卒業"
        
        segments = restorer.restore(text)
        
        section_types = set(s.section_type for s in segments)
        assert "education" in section_types or "header" in section_types
    
    def test_segments_have_char_positions(self, restorer):
        """Test that segments have character positions."""
        text = "Line 1\nLine 2"
        
        segments = restorer.restore(text)
        
        for segment in segments:
            assert segment.char_start >= 0
            assert segment.char_end > segment.char_start
    
    def test_get_section_text(self, restorer):
        """Test extracting text for a specific section."""
        text = "Name: John\nEDUCATION\nStanford\nMIT\nWORK EXPERIENCE\nGoogle"
        segments = restorer.restore(text)
        
        edu_text = restorer.get_section_text(segments, "education")
        
        # Should contain education-related content
        assert "Stanford" in edu_text or "MIT" in edu_text or "EDUCATION" in edu_text
    
    def test_get_sections_summary(self, restorer):
        """Test getting section summary."""
        text = "Name: John\nEDUCATION\nStanford\nWORK EXPERIENCE\nGoogle"
        segments = restorer.restore(text)
        
        summary = restorer.get_sections_summary(segments)
        
        assert isinstance(summary, dict)
        assert sum(summary.values()) == len(segments)
