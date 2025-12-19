"""Unit tests for candidate_extractor module."""

import pytest
from core.processors.structure_restorer import TextSegment
from core.processors.candidate_extractor import CandidateExtractor, Candidate


class TestCandidateExtractor:
    """Tests for CandidateExtractor class."""
    
    @pytest.fixture
    def extractor(self):
        """Create a CandidateExtractor instance."""
        return CandidateExtractor()
    
    @pytest.fixture
    def sample_segment(self):
        """Create a sample segment for testing."""
        return TextSegment(
            section_id="section_0",
            section_type="contact",
            line_text="Email: test@example.com Phone: 090-1234-5678",
            char_start=0,
            char_end=45,
            line_number=0
        )
    
    def test_extract_email(self, extractor):
        """Test email extraction."""
        segment = TextSegment(
            section_id="section_0",
            section_type="contact",
            line_text="Contact: jane_smith@outlook.com",
            char_start=0,
            char_end=31,
            line_number=0
        )
        
        candidates = extractor.extract([segment])
        
        emails = [c for c in candidates if c.entity_type == "EMAIL_ADDRESS"]
        assert len(emails) == 1
        assert emails[0].text == "jane_smith@outlook.com"
    
    def test_extract_japanese_email(self, extractor):
        """Test Japanese email with kanji in local part."""
        segment = TextSegment(
            section_id="section_0",
            section_type="contact",
            line_text="メール: 田中太郎@gmail.com",
            char_start=0,
            char_end=20,
            line_number=0
        )
        
        candidates = extractor.extract([segment])
        
        emails = [c for c in candidates if c.entity_type == "EMAIL_ADDRESS"]
        assert len(emails) == 1
    
    def test_extract_japanese_phone(self, extractor):
        """Test Japanese phone number extraction."""
        segment = TextSegment(
            section_id="section_0",
            section_type="contact",
            line_text="電話: 090-1234-5678",
            char_start=0,
            char_end=18,
            line_number=0
        )
        
        candidates = extractor.extract([segment])
        
        phones = [c for c in candidates if c.entity_type == "PHONE_NUMBER_JP"]
        assert len(phones) == 1
        assert "090-1234-5678" in phones[0].text
    
    def test_extract_us_phone(self, extractor):
        """Test US phone number extraction."""
        segment = TextSegment(
            section_id="section_0",
            section_type="contact",
            line_text="Phone: +1-555-123-4567",
            char_start=0,
            char_end=22,
            line_number=0
        )
        
        candidates = extractor.extract([segment])
        
        phones = [c for c in candidates if "PHONE" in c.entity_type]
        assert len(phones) >= 1
    
    def test_extract_japanese_postal_code(self, extractor):
        """Test Japanese postal code extraction."""
        segment = TextSegment(
            section_id="section_0",
            section_type="contact",
            line_text="〒100-0001 東京都千代田区",
            char_start=0,
            char_end=20,
            line_number=0
        )
        
        candidates = extractor.extract([segment])
        
        zips = [c for c in candidates if c.entity_type == "JP_ZIP_CODE"]
        assert len(zips) == 1
        assert "100-0001" in zips[0].text
    
    def test_extract_birth_date(self, extractor):
        """Test birth date extraction with context."""
        segment = TextSegment(
            section_id="section_0",
            section_type="contact",
            line_text="生年月日: 1990-05-15",
            char_start=0,
            char_end=18,
            line_number=0
        )
        
        candidates = extractor.extract([segment])
        
        dates = [c for c in candidates if "DATE" in c.entity_type]
        assert len(dates) >= 1
        # Should have high score due to birth context
        birth_dates = [c for c in dates if c.entity_type == "DATE_OF_BIRTH_JP"]
        assert len(birth_dates) == 1
    
    def test_extract_age(self, extractor):
        """Test age extraction."""
        segment = TextSegment(
            section_id="section_0",
            section_type="contact",
            line_text="年齢: 35歳",
            char_start=0,
            char_end=8,
            line_number=0
        )
        
        candidates = extractor.extract([segment])
        
        ages = [c for c in candidates if c.entity_type == "JP_AGE"]
        assert len(ages) == 1
        assert "35歳" in ages[0].text
    
    def test_extract_gender(self, extractor):
        """Test gender extraction."""
        segment = TextSegment(
            section_id="section_0",
            section_type="contact",
            line_text="性別: 男性",
            char_start=0,
            char_end=7,
            line_number=0
        )
        
        candidates = extractor.extract([segment])
        
        genders = [c for c in candidates if c.entity_type == "JP_GENDER"]
        assert len(genders) == 1
    
    def test_extract_japanese_address(self, extractor):
        """Test Japanese address extraction."""
        segment = TextSegment(
            section_id="section_0",
            section_type="contact",
            line_text="住所: 東京都渋谷区1-2-3",
            char_start=0,
            char_end=18,
            line_number=0
        )
        
        candidates = extractor.extract([segment])
        
        addresses = [c for c in candidates if c.entity_type == "JP_ADDRESS"]
        assert len(addresses) == 1
        assert "東京都" in addresses[0].text
    
    def test_candidates_have_section_info(self, extractor, sample_segment):
        """Test that candidates include section information."""
        candidates = extractor.extract([sample_segment])
        
        for candidate in candidates:
            assert candidate.section_id == "section_0"
            assert candidate.section_type == "contact"
    
    def test_merge_overlapping_candidates(self, extractor):
        """Test that overlapping candidates are merged."""
        segment = TextSegment(
            section_id="section_0",
            section_type="contact",
            line_text="test@example.com",
            char_start=0,
            char_end=16,
            line_number=0
        )
        
        candidates = extractor.extract([segment])
        
        # Should only have one email (not duplicates)
        emails = [c for c in candidates if c.entity_type == "EMAIL_ADDRESS"]
        assert len(emails) == 1
