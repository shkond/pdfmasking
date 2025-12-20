"""Unit tests for Masker class with dependency injection.

Tests the new Masker class to verify:
- Dependency injection works correctly
- Masking produces expected results
- MaskingResult structure is correct
"""

import pytest

from core.masker import Masker
from core.masking_result import MaskingResult, EntityInfo
from core.protocols import NullLogger


class MockAnonymizer:
    """Mock anonymizer for testing."""
    
    def __init__(self):
        self.anonymize_calls = []
    
    def anonymize(self, *, text: str, analyzer_results: list, operators: dict):
        """Return text with entities replaced by [MASKED]."""
        self.anonymize_calls.append({
            "text": text,
            "results": analyzer_results,
            "operators": operators
        })
        
        # Simple mock: replace entities with [MASKED]
        masked = text
        # Sort by start position descending to avoid index issues
        for result in sorted(analyzer_results, key=lambda r: r.start, reverse=True):
            masked = masked[:result.start] + "[MASKED]" + masked[result.end:]
        
        # Return object with .text attribute
        class MockResult:
            pass
        result = MockResult()
        result.text = masked
        return result


@pytest.fixture
def sample_config():
    """Minimal config for testing."""
    return {
        "transformer": {"enabled": False},
        "detection_strategy": {
            "transformer_entities": [],
            "pattern_entities": ["EMAIL_ADDRESS", "PHONE_NUMBER_JP"],
        },
        "masking": {
            "default_mask": "****",
            "entity_masks": {}
        },
        "allow_list": {
            "enabled": False,
            "dictionary_path": None
        }
    }


class TestMaskerDependencyInjection:
    """Test Masker with injected dependencies."""
    
    def test_masker_with_null_logger(self, sample_config):
        """Masker should work with NullLogger (no logging)."""
        masker = Masker(
            anonymizer=MockAnonymizer(),
            logger=NullLogger(),
            config=sample_config
        )
        assert masker.logger is not None
        # Should not raise when logging
        masker.logger.log("test message")
    
    def test_masker_with_mock_anonymizer(self, sample_config):
        """Masker should use injected anonymizer."""
        mock_anon = MockAnonymizer()
        masker = Masker(
            anonymizer=mock_anon,
            logger=NullLogger(),
            config=sample_config
        )
        
        # Mask text
        result = masker.mask("test@example.com", language="en", log_results=False)
        
        # Verify mock was called
        assert len(mock_anon.anonymize_calls) == 1
        assert mock_anon.anonymize_calls[0]["text"] == "test@example.com"


class TestMaskingResult:
    """Test MaskingResult data class."""
    
    def test_from_anonymizer_result(self):
        """Test creation from analyzer results."""
        # Create mock RecognizerResult
        class MockRecognizerResult:
            def __init__(self, entity_type, start, end, score):
                self.entity_type = entity_type
                self.start = start
                self.end = end
                self.score = score
        
        results = [
            MockRecognizerResult("EMAIL_ADDRESS", 0, 16, 0.95),
        ]
        
        result = MaskingResult.from_anonymizer_result(
            anonymized_text="****",
            original_text="test@example.com",
            analyzer_results=results
        )
        
        assert result.masked_text == "****"
        assert len(result.entities) == 1
        assert result.entities[0].entity_type == "EMAIL_ADDRESS"
        assert result.entities[0].text == "test@example.com"
        assert result.stats.total_entities == 1
    
    def test_to_entities_info(self):
        """Test conversion to legacy dict format."""
        entity = EntityInfo(
            entity_type="EMAIL_ADDRESS",
            text="test@example.com",
            score=0.95,
            start=0,
            end=16
        )
        result = MaskingResult(
            masked_text="****",
            entities=(entity,)
        )
        
        info = result.to_entities_info()
        
        assert info is not None
        assert len(info) == 1
        assert info[0]["type"] == "EMAIL_ADDRESS"
        assert info[0]["text"] == "test@example.com"
    
    def test_empty_result(self):
        """Test empty MaskingResult."""
        result = MaskingResult(masked_text="no pii here")
        
        assert result.masked_text == "no pii here"
        assert len(result.entities) == 0
        assert result.to_entities_info() is None


class TestNullLogger:
    """Test NullLogger implementation."""
    
    def test_null_logger_log_does_nothing(self):
        """NullLogger.log should not raise."""
        logger = NullLogger()
        # Should not raise
        logger.log("test message")
        logger.log("")
        logger.log("日本語メッセージ")
    
    def test_null_logger_setup_does_nothing(self, tmp_path):
        """NullLogger.setup_file_handler should not raise."""
        logger = NullLogger()
        # Should not raise
        logger.setup_file_handler(tmp_path / "test.log")
