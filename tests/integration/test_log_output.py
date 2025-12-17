"""Test that masked words are logged correctly in output log files."""

import re
import shutil
import tempfile
from pathlib import Path

import pytest

from core.masker import mask_pii_in_text
from masking_logging import MaskingLogger

# Create module-level logger instance for tests
masking_logger = MaskingLogger().logger

def setup_logger(log_path):
    """Setup logger handler for tests."""
    logger_instance = MaskingLogger()
    logger_instance.setup_file_handler(log_path)

# Expected 9 PII entities that should be masked from the sample text (no spaces)
# These are the core PII values we want to ensure are detected
EXPECTED_PII_ENTITIES = [
    "山田太郎",       # Full name
    "やまだたろう",   # Name in hiragana
    "1995年4月15日",  # Birth date
    "29歳",           # Age
    "男性",           # Gender
    "100-0001",       # Zip code
    "東京都千代田区千代田1-1",  # Address
    "090-1234-5678",  # Phone number
    "taro.yamada@example.com",  # Email
]

# Expected PII entities from document.pdf (with spaces in names)
EXPECTED_PDF_PII_ENTITIES = [
    "山田 太郎",      # Full name (with space)
    "やまだ たろう",  # Name in hiragana (with space)
    "1995年4月15日",  # Birth date
    "29歳",           # Age
    "男性",           # Gender
    "100-0001",       # Zip code
    "東京都千代田区千代田1-1",  # Address
    "090-1234-5678",  # Phone number
    "taro.yamada@example.com",  # Email
]

# Maximum number of entities that should be masked
MAX_MASKED_ENTITIES = 9

# Sample text containing the PII to be masked
SAMPLE_TEXT = """氏名: 山田太郎
ふりがな: やまだたろう
生年月日: 1995年4月15日（29歳）
性別: 男性
〒100-0001
東京都千代田区千代田1-1
電話: 090-1234-5678
Email: taro.yamada@example.com
"""


class TestLogOutput:
    """Test that log files contain expected masked words."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)

        # Close all logger handlers before cleanup
        for handler in masking_logger.handlers[:]:
            handler.close()
            masking_logger.removeHandler(handler)

        # Clean up temp directory, ignore errors on Windows
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_masked_entities_count_and_content(self, temp_output_dir):
        """
        Test that:
        1. All 9 expected PII entities are found in the log
        2. The number of masked entities does not exceed 9
        """
        log_path = temp_output_dir / "test_log.txt"

        # Setup logger for this test
        setup_logger(log_path)

        # Run masking
        masked_text, _ = mask_pii_in_text(SAMPLE_TEXT, language="ja", verbose=True)

        # Flush and close handlers to ensure all log content is written
        for handler in masking_logger.handlers:
            handler.flush()

        # Read log file content
        log_content = log_path.read_text(encoding="utf-8")

        # Extract all quoted strings from log (format: "word")
        logged_entities = re.findall(r'"([^"]+)"', log_content)

        # Test 1: Check that each expected PII entity appears in the log
        missing_pii = []
        for expected_pii in EXPECTED_PII_ENTITIES:
            found = False
            for entity in logged_entities:
                # Normalize spaces and check for match
                normalized_entity = entity.strip()
                if expected_pii in normalized_entity or normalized_entity in expected_pii:
                    found = True
                    break
            if not found:
                missing_pii.append(expected_pii)

        assert not missing_pii, f"Expected PII not found in log: {missing_pii}\nLogged entities: {logged_entities}"

        # Test 2: Check that the number of masked entities does not exceed MAX_MASKED_ENTITIES
        assert len(logged_entities) <= MAX_MASKED_ENTITIES, (
            f"Too many entities masked: {len(logged_entities)} (max: {MAX_MASKED_ENTITIES}). "
            f"Entities: {logged_entities}"
        )

    def test_phone_number_in_log(self, temp_output_dir):
        """Test that phone numbers are detected and logged."""
        log_path = temp_output_dir / "phone_log.txt"
        setup_logger(log_path)

        text = "電話: 090-1234-5678"
        mask_pii_in_text(text, language="ja", verbose=True)

        for handler in masking_logger.handlers:
            handler.flush()

        log_content = log_path.read_text(encoding="utf-8")
        assert "090-1234-5678" in log_content
        assert "PHONE_NUMBER" in log_content

    def test_zip_code_in_log(self, temp_output_dir):
        """Test that zip codes are detected and logged."""
        log_path = temp_output_dir / "zip_log.txt"
        setup_logger(log_path)

        text = "〒100-0001"
        mask_pii_in_text(text, language="ja", verbose=True)

        for handler in masking_logger.handlers:
            handler.flush()

        log_content = log_path.read_text(encoding="utf-8")
        assert "100-0001" in log_content
        assert "JP_ZIP_CODE" in log_content

    def test_date_in_log(self, temp_output_dir):
        """Test that dates are detected and logged."""
        log_path = temp_output_dir / "date_log.txt"
        setup_logger(log_path)

        text = "生年月日: 1995年4月15日"
        mask_pii_in_text(text, language="ja", verbose=True)

        for handler in masking_logger.handlers:
            handler.flush()

        log_content = log_path.read_text(encoding="utf-8")
        assert "1995年4月15日" in log_content
        assert "DATE" in log_content

    def test_email_in_log(self, temp_output_dir):
        """Test that email addresses are detected and logged."""
        log_path = temp_output_dir / "email_log.txt"
        setup_logger(log_path)

        text = "Email: test@example.com"
        mask_pii_in_text(text, language="ja", verbose=True)

        for handler in masking_logger.handlers:
            handler.flush()

        log_content = log_path.read_text(encoding="utf-8")
        assert "test@example.com" in log_content
        assert "EMAIL" in log_content

    def test_person_name_in_log(self, temp_output_dir):
        """Test that person names are detected and logged."""
        log_path = temp_output_dir / "person_log.txt"
        setup_logger(log_path)

        text = "氏名: 山田太郎"
        mask_pii_in_text(text, language="ja", verbose=True)

        for handler in masking_logger.handlers:
            handler.flush()

        log_content = log_path.read_text(encoding="utf-8")
        assert "山田太郎" in log_content
        assert "PERSON" in log_content


class TestDocumentPdfOutput:
    """Test that document.pdf processing creates correct output log with exactly 9 entities."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)

        # Close all logger handlers before cleanup
        for handler in masking_logger.handlers[:]:
            handler.close()
            masking_logger.removeHandler(handler)

        # Clean up temp directory, ignore errors on Windows
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_document_pdf_entities_count(self, temp_output_dir):
        """
        Test that document.pdf processing with Japanese language:
        1. Detects all 9 expected PII entities
        2. Does not exceed 9 entities (no duplicates)
        """
        from document_extractors import extract_text

        # Path to document.pdf
        pdf_path = Path(__file__).parent.parent.parent / "document.pdf"

        if not pdf_path.exists():
            pytest.skip("document.pdf not found")

        log_path = temp_output_dir / "document_log.txt"

        # Setup logger for this test
        setup_logger(log_path)

        # Extract text from PDF
        text = extract_text(str(pdf_path))

        # Run masking with Japanese language and preprocessing for PDF text
        masked_text, _ = mask_pii_in_text(text, language="ja", verbose=True, preprocess=True)

        # Flush and close handlers to ensure all log content is written
        for handler in masking_logger.handlers:
            handler.flush()

        # Read log file content
        log_content = log_path.read_text(encoding="utf-8")

        # Extract all quoted strings from log (format: "word")
        logged_entities = re.findall(r'"([^"]+)"', log_content)

        # Test 1: Check that each expected PII entity appears in the log
        missing_pii = []
        for expected_pii in EXPECTED_PDF_PII_ENTITIES:
            found = False
            for entity in logged_entities:
                # Normalize spaces and check for match
                normalized_entity = entity.strip()
                if expected_pii in normalized_entity or normalized_entity in expected_pii:
                    found = True
                    break
            if not found:
                missing_pii.append(expected_pii)

        assert not missing_pii, f"Expected PII not found in log: {missing_pii}\nLogged entities: {logged_entities}"

        # Test 2: Check that the number of masked entities equals MAX_MASKED_ENTITIES (exactly 9)
        assert len(logged_entities) <= MAX_MASKED_ENTITIES, (
            f"Too many entities masked: {len(logged_entities)} (max: {MAX_MASKED_ENTITIES}). "
            f"Entities: {logged_entities}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


