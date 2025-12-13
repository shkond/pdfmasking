"""Pattern-based recognizers for Japanese PII."""

from typing import List, Optional
import re

from presidio_analyzer import Pattern, PatternRecognizer, EntityRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import NlpArtifacts


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
    
    CONTEXT = ["TEL", "電話", "携帯", "自宅", "tel", "Tel", "電話番号"]
    
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
    
    CONTEXT = ["〒", "郵便番号", "郵便", "zip", "ZIP", "住所"]
    
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


class JapaneseNameRecognizer(EntityRecognizer):
    """
    Context-based recognizer for Japanese person names.
    
    Detects Japanese names (kanji, hiragana, katakana) when they appear
    near context keywords like "氏名", "ふりがな", etc.
    """
    
    CONTEXT_KEYWORDS = ["氏名", "ふりがな", "フリガナ", "名前", "お名前", "姓名", "氏", "名"]
    CONTEXT_WINDOW = 100  # characters before/after to search for context
    
    # Pattern for Japanese names (kanji, hiragana, katakana with spaces)
    NAME_PATTERN = r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\u3000 ]{2,20}'
    
    def __init__(
        self,
        supported_language: str = "ja",
        supported_entity: str = "JP_PERSON",
        context_keywords: Optional[List[str]] = None,
    ):
        self.context_keywords = context_keywords if context_keywords else self.CONTEXT_KEYWORDS
        super().__init__(
            supported_entities=[supported_entity],
            supported_language=supported_language,
        )
    
    def load(self) -> None:
        """Load is not needed for this recognizer."""
        pass
    
    def analyze(
        self, text: str, entities: List[str], nlp_artifacts: Optional[NlpArtifacts] = None
    ) -> List[RecognizerResult]:
        """
        Analyze text for Japanese names using context-based pattern matching.
        
        Args:
            text: The text to analyze
            entities: List of entities to look for
            nlp_artifacts: Not used for this recognizer
            
        Returns:
            List of RecognizerResult objects
        """
        results = []
        
        # Check if this entity type is requested
        if self.supported_entities[0] not in entities:
            return results
        
        # Find all potential name matches
        for match in re.finditer(self.NAME_PATTERN, text):
            name_text = match.group().strip()
            
            # Skip if too short or contains only spaces
            if len(name_text) < 2 or name_text.replace(' ', '').replace('　', '') == '':
                continue
            
            # Check if context keyword is nearby
            start_pos = match.start()
            end_pos = match.end()
            
            # Extract context window
            context_start = max(0, start_pos - self.CONTEXT_WINDOW)
            context_end = min(len(text), end_pos + self.CONTEXT_WINDOW)
            context = text[context_start:context_end]
            
            # Check for context keywords
            has_context = any(keyword in context for keyword in self.context_keywords)
            
            if has_context:
                # Higher score when context is found
                score = 0.85
                results.append(
                    RecognizerResult(
                        entity_type=self.supported_entities[0],
                        start=start_pos,
                        end=end_pos,
                        score=score,
                    )
                )
        
        return results

