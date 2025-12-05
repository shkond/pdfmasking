"""Unit tests for Japanese pattern-based recognizers."""

import pytest
from presidio_analyzer import RecognizerResult

from recognizers.japanese_patterns import (
    JapanesePhoneRecognizer,
    JapaneseZipCodeRecognizer,
    JapaneseBirthDateRecognizer,
)


class TestJapanesePhoneRecognizer:
    """Tests for Japanese phone number recognizer."""
    
    def test_landline_phone(self):
        recognizer = JapanesePhoneRecognizer()
        text = "電話: 03-1234-5678"
        results = recognizer.analyze(text, entities=["PHONE_NUMBER_JP"])
        
        assert len(results) > 0
        assert results[0].entity_type == "PHONE_NUMBER_JP"
        assert text[results[0].start:results[0].end] == "03-1234-5678"
    
    def test_mobile_phone(self):
        recognizer = JapanesePhoneRecognizer()
        text = "携帯: 090-1234-5678"
        results = recognizer.analyze(text, entities=["PHONE_NUMBER_JP"])
        
        assert len(results) > 0
        assert results[0].entity_type == "PHONE_NUMBER_JP"
        assert text[results[0].start:results[0].end] == "090-1234-5678"
    
    def test_multiple_phones(self):
        recognizer = JapanesePhoneRecognizer()
        text = "自宅: 03-1234-5678 携帯: 080-9876-5432"
        results = recognizer.analyze(text, entities=["PHONE_NUMBER_JP"])
        
        assert len(results) == 2


class TestJapaneseZipCodeRecognizer:
    """Tests for Japanese zip code recognizer."""
    
    def test_zip_code(self):
        recognizer = JapaneseZipCodeRecognizer()
        text = "〒150-0001"
        results = recognizer.analyze(text, entities=["JP_ZIP_CODE"])
        
        assert len(results) > 0
        assert results[0].entity_type == "JP_ZIP_CODE"
        assert text[results[0].start:results[0].end] == "150-0001"
    
    def test_zip_code_with_context(self):
        recognizer = JapaneseZipCodeRecognizer()
        text = "郵便番号: 100-0001"
        results = recognizer.analyze(text, entities=["JP_ZIP_CODE"])
        
        assert len(results) > 0


class TestJapaneseBirthDateRecognizer:
    """Tests for Japanese birth date recognizer."""
    
    def test_western_date_slash(self):
        recognizer = JapaneseBirthDateRecognizer()
        text = "生年月日: 1990/01/01"
        results = recognizer.analyze(text, entities=["DATE_OF_BIRTH_JP"])
        
        assert len(results) > 0
        assert results[0].entity_type == "DATE_OF_BIRTH_JP"
        assert text[results[0].start:results[0].end] == "1990/01/01"
    
    def test_japanese_date_kanji(self):
        recognizer = JapaneseBirthDateRecognizer()
        text = "生年月日: 1990年1月1日"
        results = recognizer.analyze(text, entities=["DATE_OF_BIRTH_JP"])
        
        assert len(results) > 0
        assert text[results[0].start:results[0].end] == "1990年1月1日"
    
    def test_reiwa_era(self):
        recognizer = JapaneseBirthDateRecognizer()
        text = "生年月日: 令和5年12月1日"
        results = recognizer.analyze(text, entities=["DATE_OF_BIRTH_JP"])
        
        assert len(results) > 0
        assert text[results[0].start:results[0].end] == "令和5年12月1日"
    
    def test_heisei_era(self):
        recognizer = JapaneseBirthDateRecognizer()
        text = "生年月日: 平成2年1月1日"
        results = recognizer.analyze(text, entities=["DATE_OF_BIRTH_JP"])
        
        assert len(results) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
