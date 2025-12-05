"""Pattern-based recognizers for Japanese PII."""

from typing import List, Optional
import re

from presidio_analyzer import Pattern, PatternRecognizer


class JapanesePhoneRecognizer(PatternRecognizer):
    """
    Recognizer for Japanese phone numbers.
    
    Supports formats:
    - Landline: 0X-XXXX-XXXX, 0XX-XXX-XXXX, 0XXX-XX-XXXX, etc.
    - Mobile: 090-XXXX-XXXX, 080-XXXX-XXXX, 070-XXXX-XXXX
    """
    
    PATTERNS = [
        Pattern(
            name="japanese_phone_general",
            regex=r"0\d{1,4}-\d{1,4}-\d{4}",
            score=0.7,
        ),
        Pattern(
            name="japanese_mobile",
            regex=r"0[789]0-\d{4}-\d{4}",
            score=0.8,
        ),
    ]
    
    CONTEXT = ["TEL", "電話", "携帯", "自宅", "tel", "Tel"]
    
    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "ja",
        supported_entity: str = "PHONE_NUMBER_JP",
    ):
        patterns = patterns if patterns else self.PATTERNS
        context = context if context else self.CONTEXT
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns,
            context=context,
            supported_language=supported_language,
        )


class JapaneseZipCodeRecognizer(PatternRecognizer):
    """
    Recognizer for Japanese postal codes.
    
    Format: XXX-XXXX (e.g., 100-0001)
    """
    
    PATTERNS = [
        Pattern(
            name="japanese_zipcode",
            regex=r"\d{3}-\d{4}",
            score=0.6,
        ),
    ]
    
    CONTEXT = ["〒", "郵便番号", "郵便", "zip", "ZIP"]
    
    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "ja",
        supported_entity: str = "JP_ZIP_CODE",
    ):
        patterns = patterns if patterns else self.PATTERNS
        context = context if context else self.CONTEXT
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns,
            context=context,
            supported_language=supported_language,
        )


class JapaneseBirthDateRecognizer(PatternRecognizer):
    """
    Recognizer for Japanese date formats, especially birth dates.
    
    Supports:
    - Western format: YYYY/MM/DD, YYYY-MM-DD
    - Japanese format: YYYY年MM月DD日
    - Japanese era: 令和X年X月X日, 平成XX年X月X日, 昭和XX年X月X日
    """
    
    PATTERNS = [
        Pattern(
            name="western_date_slash",
            regex=r"\d{4}/\d{1,2}/\d{1,2}",
            score=0.6,
        ),
        Pattern(
            name="western_date_hyphen",
            regex=r"\d{4}-\d{1,2}-\d{1,2}",
            score=0.6,
        ),
        Pattern(
            name="japanese_date_kanji",
            regex=r"\d{4}年\d{1,2}月\d{1,2}日",
            score=0.7,
        ),
        Pattern(
            name="reiwa_era",
            regex=r"令和\d{1,2}年\d{1,2}月\d{1,2}日",
            score=0.8,
        ),
        Pattern(
            name="heisei_era",
            regex=r"平成\d{1,2}年\d{1,2}月\d{1,2}日",
            score=0.8,
        ),
        Pattern(
            name="showa_era",
            regex=r"昭和\d{1,2}年\d{1,2}月\d{1,2}日",
            score=0.8,
        ),
    ]
    
    CONTEXT = ["生年月日", "年齢", "生まれ", "誕生日", "生年", "年月日"]
    
    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "ja",
        supported_entity: str = "DATE_OF_BIRTH_JP",
    ):
        patterns = patterns if patterns else self.PATTERNS
        context = context if context else self.CONTEXT
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns,
            context=context,
            supported_language=supported_language,
        )
