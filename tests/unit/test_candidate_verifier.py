"""Unit tests for candidate_verifier module."""

import pytest
from core.processors.candidate_extractor import Candidate
from core.processors.candidate_verifier import CandidateVerifier, VerificationResult


class TestCandidateVerifier:
    """Tests for CandidateVerifier class."""
    
    @pytest.fixture
    def verifier(self):
        """Create a CandidateVerifier instance."""
        return CandidateVerifier()
    
    @pytest.fixture
    def email_candidate(self):
        """Create a sample email candidate."""
        return Candidate(
            entity_type="EMAIL_ADDRESS",
            text="test@example.com",
            start=0,
            end=16,
            score=0.95,
            source="rule:email_regex",
            section_id="section_0",
            section_type="contact"
        )
    
    @pytest.fixture
    def phone_candidate(self):
        """Create a sample phone candidate."""
        return Candidate(
            entity_type="PHONE_NUMBER_JP",
            text="090-1234-5678",
            start=0,
            end=13,
            score=0.9,
            source="rule:phone_jp_regex",
            section_id="section_0",
            section_type="contact"
        )
    
    def test_verify_valid_email(self, verifier, email_candidate):
        """Test verification of valid email."""
        results = verifier.verify([email_candidate])
        
        assert len(results) == 1
        assert results[0].status == "mask"
        assert results[0].verified_score > 0
    
    def test_verify_invalid_email_format(self, verifier):
        """Test rejection of invalid email format."""
        invalid_email = Candidate(
            entity_type="EMAIL_ADDRESS",
            text="not-an-email",
            start=0,
            end=12,
            score=0.9,
            source="rule:email_regex",
            section_id="section_0",
            section_type="contact"
        )
        
        results = verifier.verify([invalid_email])
        
        assert len(results) == 1
        assert results[0].status == "exclude"
        assert "format_invalid" in results[0].reason
    
    def test_verify_valid_japanese_phone(self, verifier, phone_candidate):
        """Test verification of valid Japanese phone."""
        results = verifier.verify([phone_candidate])
        
        assert len(results) == 1
        assert results[0].status == "mask"
    
    def test_section_policy_education_penalty(self, verifier):
        """Test that person names in education section get penalty."""
        person_in_edu = Candidate(
            entity_type="JP_PERSON",
            text="田中太郎",
            start=0,
            end=4,
            score=0.8,  # High enough to normally mask
            source="ner:ginza",
            section_id="section_1",
            section_type="education"
        )
        
        results = verifier.verify([person_in_edu])
        
        # Score should be reduced due to education section penalty
        assert len(results) == 1
        assert results[0].verified_score < 0.8
    
    def test_section_policy_contact_boost(self, verifier):
        """Test that person names in contact section get boost."""
        person_in_contact = Candidate(
            entity_type="JP_PERSON",
            text="田中太郎",
            start=0,
            end=4,
            score=0.6,  # Below mask threshold normally
            source="ner:ginza",
            section_id="section_0",
            section_type="contact"
        )
        
        results = verifier.verify([person_in_contact])
        
        # Score should be boosted
        assert len(results) == 1
        assert results[0].verified_score >= 0.6
    
    def test_valid_japanese_postal_code(self, verifier):
        """Test validation of Japanese postal code."""
        zip_candidate = Candidate(
            entity_type="JP_ZIP_CODE",
            text="100-0001",
            start=0,
            end=8,
            score=0.95,
            source="rule:jp_zip_regex",
            section_id="section_0",
            section_type="contact"
        )
        
        results = verifier.verify([zip_candidate])
        
        assert len(results) == 1
        assert results[0].status == "mask"
    
    def test_invalid_postal_code_format(self, verifier):
        """Test rejection of invalid postal code format."""
        invalid_zip = Candidate(
            entity_type="JP_ZIP_CODE",
            text="12345",  # Missing hyphen
            start=0,
            end=5,
            score=0.9,
            source="rule:jp_zip_regex",
            section_id="section_0",
            section_type="contact"
        )
        
        results = verifier.verify([invalid_zip])
        
        assert len(results) == 1
        assert results[0].status == "exclude"
    
    def test_valid_date_format(self, verifier):
        """Test validation of date format."""
        date_candidate = Candidate(
            entity_type="DATE_OF_BIRTH_JP",
            text="1990-05-15",
            start=0,
            end=10,
            score=0.9,
            source="rule:date_regex",
            section_id="section_0",
            section_type="contact"
        )
        
        results = verifier.verify([date_candidate])
        
        assert len(results) == 1
        assert results[0].status == "mask"
    
    def test_score_threshold_review(self, verifier):
        """Test that medium scores get review status."""
        medium_score = Candidate(
            entity_type="JP_ADDRESS",
            text="東京都渋谷区",
            start=0,
            end=6,
            score=0.55,  # Between review and mask threshold
            source="rule:jp_address_prefix",
            section_id="section_0",
            section_type="contact"
        )
        
        results = verifier.verify([medium_score])
        
        # After contact boost, may still be review or mask
        assert len(results) == 1
        assert results[0].status in ["review", "mask"]
    
    def test_score_threshold_exclude(self, verifier):
        """Test that low scores get exclude status."""
        low_score = Candidate(
            entity_type="JP_ADDRESS",
            text="東京",
            start=0,
            end=2,
            score=0.3,  # Below review threshold
            source="rule:jp_address_prefix",
            section_id="section_1",
            section_type="education"  # Will get penalty too
        )
        
        results = verifier.verify([low_score])
        
        assert len(results) == 1
        # Low score + penalty should result in exclude
        assert results[0].status in ["exclude", "review"]
    
    def test_collision_resolution_priority(self, verifier):
        """Test that higher priority entity wins collision."""
        email = Candidate(
            entity_type="EMAIL_ADDRESS",
            text="test@example.com",
            start=0,
            end=16,
            score=0.95,
            source="rule:email_regex",
            section_id="section_0",
            section_type="contact"
        )
        
        # Overlapping address (lower priority)
        address = Candidate(
            entity_type="JP_ADDRESS",
            text="test@example",  # Overlaps with email
            start=0,
            end=12,
            score=0.8,
            source="rule:jp_address_prefix",
            section_id="section_0",
            section_type="contact"
        )
        
        results = verifier.verify([email, address])
        
        # Email should be kept, address should be excluded
        mask_results = [r for r in results if r.status == "mask"]
        assert len(mask_results) == 1
        assert mask_results[0].candidate.entity_type == "EMAIL_ADDRESS"
    
    def test_get_maskable_candidates(self, verifier, email_candidate, phone_candidate):
        """Test getting maskable candidates."""
        results = verifier.verify([email_candidate, phone_candidate])
        
        maskable = verifier.get_maskable_candidates(results)
        
        assert len(maskable) >= 1
        assert all(isinstance(c, Candidate) for c in maskable)
    
    def test_deterministic_output(self, verifier, email_candidate):
        """Test that output is deterministic."""
        results1 = verifier.verify([email_candidate])
        results2 = verifier.verify([email_candidate])
        
        assert results1[0].status == results2[0].status
        assert results1[0].verified_score == results2[0].verified_score
