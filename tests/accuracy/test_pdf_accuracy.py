"""Accuracy tests for PDF processing.

These tests evaluate the accuracy of PII detection in PDF documents.
Uses document.pdf as the test file.

Excluded from CI by default due to:
- Long execution time (PDF processing + model inference)
- File dependencies (requires document.pdf)
"""

import logging
import re
from pathlib import Path

import pytest

from core.masker import Masker
from masking_logging import MaskingLogger


# Mark all tests in this module as accuracy tests
pytestmark = pytest.mark.accuracy


# Expected PII entities from document.pdf
EXPECTED_PDF_PII = {
    "name": ["山田 太郎", "やまだ たろう"],
    "date": ["1995年4月15日"],
    "age": ["29歳"],
    "gender": ["男性"],
    "zip": ["100-0001"],
    "address": ["東京都千代田区千代田1-1"],
    "phone": ["090-1234-5678"],
    "email": ["taro.yamada@example.com"],
}


class TestPdfAccuracy:
    """Accuracy tests for PDF processing."""

    @pytest.fixture
    def pdf_path(self):
        """Get path to document.pdf."""
        path = Path(__file__).parent.parent.parent / "document.pdf"
        if not path.exists():
            pytest.skip("document.pdf not found")
        return path

    @pytest.fixture
    def pdf_text(self, pdf_path):
        """Extract text from PDF."""
        from document_extractors import extract_text
        return extract_text(str(pdf_path))

    def test_pdf_text_extraction(self, pdf_text):
        """Verify PDF text extraction works."""
        assert len(pdf_text) > 0
        # Should contain some expected content
        assert any(word in pdf_text for word in ["氏名", "住所", "電話"])

    def test_pdf_pii_detection_recall(self, pdf_text, temp_output_dir):
        """Test that all expected PII is detected."""
        log_path = temp_output_dir / "pdf_accuracy_log.txt"
        logger = MaskingLogger()
        logger.setup_file_handler(log_path)
        masker = Masker(logger=logger)

        result = masker.mask(pdf_text, language="ja", do_preprocess=True)

        # Flush handlers
        logger.logger.handlers[0].flush() if logger.logger.handlers else None

        # Read log content
        log_content = log_path.read_text(encoding="utf-8")
        logged_entities = re.findall(r'"([^"]+)"', log_content)

        # Check each expected PII category
        missing_pii = []
        for category, expected_values in EXPECTED_PDF_PII.items():
            for expected in expected_values:
                found = False
                for entity in logged_entities:
                    # Normalize and check
                    if expected in entity or entity in expected:
                        found = True
                        break
                if not found:
                    missing_pii.append(f"{category}: {expected}")

        # Report missing (not strict assertion for accuracy tests)
        if missing_pii:
            print(f"Missing PII: {missing_pii}")
        
        # At least 70% should be detected
        total_expected = sum(len(v) for v in EXPECTED_PDF_PII.values())
        detected = total_expected - len(missing_pii)
        detection_rate = detected / total_expected

        assert detection_rate >= 0.7, \
            f"Detection rate {detection_rate:.1%} is below 70%. Missing: {missing_pii}"

    def test_pdf_pii_detection_precision(self, pdf_text, temp_output_dir):
        """Test that not too many false positives are detected."""
        log_path = temp_output_dir / "pdf_precision_log.txt"
        logger = MaskingLogger()
        logger.setup_file_handler(log_path)
        masker = Masker(logger=logger)

        result = masker.mask(pdf_text, language="ja", do_preprocess=True)

        # Flush handlers
        logger.logger.handlers[0].flush() if logger.logger.handlers else None

        # Read log content
        log_content = log_path.read_text(encoding="utf-8")
        logged_entities = re.findall(r'"([^"]+)"', log_content)

        # Maximum expected entities is 9 (as per original test)
        MAX_EXPECTED = 9

        assert len(logged_entities) <= MAX_EXPECTED, \
            f"Too many entities detected: {len(logged_entities)} (max: {MAX_EXPECTED}). " \
            f"Entities: {logged_entities}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

