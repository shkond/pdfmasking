"""Integration test for end-to-end Japanese PII masking."""

import pytest

from analyzer_factory import create_analyzer
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from tests.sample_data import get_sample_resume, get_expected_entities


class TestJapanesePIIMasking:
    """Integration tests for Japanese PII masking."""
    
    @pytest.fixture
    def analyzer(self):
        """Create Japanese analyzer."""
        return create_analyzer("ja")
    
    @pytest.fixture
    def anonymizer(self):
        """Create anonymizer."""
        return AnonymizerEngine()
    
    def test_detect_all_entities(self, analyzer):
        """Test that all expected entities are detected."""
        text = get_sample_resume()
        results = analyzer.analyze(text, language="ja")
        
        # Should detect multiple entities
        assert len(results) > 0
        
        # Check for specific entity types
        entity_types = {result.entity_type for result in results}
        assert "JP_PERSON" in entity_types or "PERSON" in entity_types
        assert "PHONE_NUMBER_JP" in entity_types
        assert "JP_ZIP_CODE" in entity_types
    
    def test_mask_pii(self, analyzer, anonymizer):
        """Test end-to-end PII masking."""
        text = get_sample_resume()
        
        # Analyze
        results = analyzer.analyze(text, language="ja")
        
        # Anonymize
        operators = {
            "DEFAULT": OperatorConfig("replace", {"new_value": "****"})
        }
        anonymized = anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators=operators,
        )
        
        # Verify masking
        masked_text = anonymized.text
        
        # Original PII should not appear in masked text
        assert "山田太郎" not in masked_text
        assert "03-1234-5678" not in masked_text
        assert "090-9876-5432" not in masked_text
        assert "150-0001" not in masked_text
        assert "yamada.taro@example.com" not in masked_text
        
        # Mask placeholder should appear
        assert "****" in masked_text
    
    def test_phone_number_detection(self, analyzer):
        """Test phone number detection."""
        text = "TEL: 03-1234-5678 携帯: 090-9876-5432"
        results = analyzer.analyze(text, language="ja", entities=["PHONE_NUMBER_JP"])
        
        assert len(results) == 2
        for result in results:
            assert result.entity_type == "PHONE_NUMBER_JP"
    
    def test_zip_code_detection(self, analyzer):
        """Test zip code detection."""
        text = "〒150-0001"
        results = analyzer.analyze(text, language="ja", entities=["JP_ZIP_CODE"])
        
        assert len(results) > 0
        assert results[0].entity_type == "JP_ZIP_CODE"
    
    def test_date_detection(self, analyzer):
        """Test birth date detection."""
        text = "生年月日: 1990年1月1日"
        results = analyzer.analyze(text, language="ja", entities=["DATE_OF_BIRTH_JP"])
        
        assert len(results) > 0
        assert results[0].entity_type == "DATE_OF_BIRTH_JP"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
