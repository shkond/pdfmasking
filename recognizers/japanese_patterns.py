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
            regex=r"0[1-9]0-\d{4}-\d{4}",
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


class JapaneseBirthDateRecognizer(EntityRecognizer):
    """
    Recognizer for Japanese birth dates only.
    
    Only detects dates when they appear near birthdate context keywords.
    Excludes dates in education/work history sections.
    
    Supports:
    - Western format: YYYY/MM/DD, YYYY-MM-DD
    - Japanese format: YYYY年MM月DD日
    - Japanese era: 令和X年X月X日, 平成XX年X月X日, 昭和XX年X月X日
    """
    
    # Date patterns
    DATE_PATTERNS = [
        r"\d{4}/\d{1,2}/\d{1,2}",  # 1995/4/15
        r"\d{4}-\d{1,2}-\d{1,2}",  # 1995-04-15
        r"\d{4}年\d{1,2}月\d{1,2}日",  # 1995年4月15日
        r"令和\d{1,2}年\d{1,2}月\d{1,2}日",
        r"平成\d{1,2}年\d{1,2}月\d{1,2}日",
        r"昭和\d{1,2}年\d{1,2}月\d{1,2}日",
    ]
    
    # Context keywords that indicate this is a birthdate
    BIRTHDATE_CONTEXT = ["生年月日", "年齢", "生まれ", "誕生日", "生年"]
    CONTEXT_WINDOW = 50  # Window to search for birthdate context
    
    # Context keywords that indicate this is NOT a birthdate (education/work history)
    EXCLUDE_CONTEXT = ["学歴", "職歴", "職務経歴", "入学", "卒業", "入社", "退社", "現在"]
    
    def __init__(
        self,
        supported_language: str = "ja",
        supported_entity: str = "DATE_OF_BIRTH_JP",
        context: Optional[List[str]] = None,
    ):
        self.birthdate_context = context if context else self.BIRTHDATE_CONTEXT
        super().__init__(
            supported_entities=[supported_entity],
            supported_language=supported_language,
        )
    
    def load(self) -> None:
        """Load is not needed for this recognizer."""
        pass
    
    def _has_birthdate_context(self, text: str, start: int, end: int) -> bool:
        """Check if birthdate context keywords are nearby."""
        context_start = max(0, start - self.CONTEXT_WINDOW)
        context_end = min(len(text), end + self.CONTEXT_WINDOW)
        context_text = text[context_start:context_end]
        return any(keyword in context_text for keyword in self.birthdate_context)
    
    def _has_exclude_context(self, text: str, start: int, end: int) -> bool:
        """Check if exclusion context (education/work history) is nearby.
        
        Only returns True if exclusion context exists AND birthdate context is NOT immediately before.
        """
        # First check if birthdate context is immediately before the date (within 20 chars)
        immediate_context = text[max(0, start - 20):start]
        if any(keyword in immediate_context for keyword in self.birthdate_context):
            # Birthdate context is immediate, don't exclude
            return False
        
        # Check for exclusion keywords in a wider window
        context_start = max(0, start - 30)
        context_end = min(len(text), end + 30)
        context_text = text[context_start:context_end]
        return any(keyword in context_text for keyword in self.EXCLUDE_CONTEXT)
    
    def analyze(
        self, text: str, entities: List[str], nlp_artifacts: Optional[NlpArtifacts] = None
    ) -> List[RecognizerResult]:
        """
        Analyze text for Japanese birth dates.
        
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
        
        # Find all potential date matches
        for pattern in self.DATE_PATTERNS:
            for match in re.finditer(pattern, text):
                start_pos = match.start()
                end_pos = match.end()
                
                # Only match if birthdate context is found
                if not self._has_birthdate_context(text, start_pos, end_pos):
                    continue
                
                # Skip if exclusion context (education/work) is found
                if self._has_exclude_context(text, start_pos, end_pos):
                    continue
                
                results.append(
                    RecognizerResult(
                        entity_type=self.supported_entities[0],
                        start=start_pos,
                        end=end_pos,
                        score=1.0,
                    )
                )
        
        return results


class JapaneseAgeRecognizer(PatternRecognizer):
    """
    Recognizer for Japanese age patterns.
    
    Supports formats like: 29歳, 30才
    """
    
    PATTERNS = [
        Pattern(
            name="japanese_age",
            regex=r"\d{1,3}歳",
            score=0.8,
        ),
        Pattern(
            name="japanese_age_alt",
            regex=r"\d{1,3}才",
            score=0.8,
        ),
    ]
    
    CONTEXT = ["年齢", "生年月日", "（", "）", "歳"]
    
    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "ja",
        supported_entity: str = "JP_AGE",
    ):
        patterns = patterns if patterns else self.PATTERNS
        context = context if context else self.CONTEXT
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns,
            context=context,
            supported_language=supported_language,
        )


class JapaneseGenderRecognizer(PatternRecognizer):
    """
    Recognizer for Japanese gender patterns.
    
    Detects: 男性, 女性
    """
    
    PATTERNS = [
        Pattern(
            name="japanese_gender",
            regex=r"(?:男性|女性)",
            score=0.7,
        ),
    ]
    
    CONTEXT = ["性別"]
    
    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "ja",
        supported_entity: str = "JP_GENDER",
    ):
        patterns = patterns if patterns else self.PATTERNS
        context = context if context else self.CONTEXT
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns,
            context=context,
            supported_language=supported_language,
        )


class JapaneseAddressRecognizer(EntityRecognizer):
    """
    Recognizer for Japanese addresses.
    
    Detects addresses like: 東京都千代田区千代田1-1
    Excludes school names, organization names, and company names.
    """
    
    # Pattern for Japanese addresses (prefecture + district/city + details)
    ADDRESS_PATTERN = r"(?:東京都|北海道|(?:京都|大阪)府|[^\s]{2,3}県)[^\s\n]{3,30}"
    
    CONTEXT = ["住所", "〒", "現住所", "所在地"]
    CONTEXT_WINDOW = 50
    
    # Words that indicate this is NOT an address (schools, organizations, companies)
    EXCLUDE_SUFFIXES = [
        "高等学校", "高校", "中学校", "中学", "小学校", "小学",
        "大学", "大学院", "専門学校", "学園", "学院",
        "株式会社", "会社", "有限会社", "合同会社", "法人",
        "銀行", "病院", "クリニック", "事務所", "研究所",
        "センター", "協会", "組合", "財団", "社団",
    ]
    
    def __init__(
        self,
        supported_language: str = "ja",
        supported_entity: str = "JP_ADDRESS",
        context: Optional[List[str]] = None,
    ):
        self.context = context if context else self.CONTEXT
        super().__init__(
            supported_entities=[supported_entity],
            supported_language=supported_language,
        )
    
    def load(self) -> None:
        """Load is not needed for this recognizer."""
        pass
    
    def _is_excluded(self, text: str) -> bool:
        """Check if text ends with excluded suffix (school, org, company)."""
        for suffix in self.EXCLUDE_SUFFIXES:
            if text.endswith(suffix):
                return True
        return False
    
    def _has_context(self, text: str, start: int, end: int) -> bool:
        """Check if address context keywords are nearby."""
        context_start = max(0, start - self.CONTEXT_WINDOW)
        context_end = min(len(text), end + self.CONTEXT_WINDOW)
        context = text[context_start:context_end]
        return any(keyword in context for keyword in self.context)
    
    def analyze(
        self, text: str, entities: List[str], nlp_artifacts: Optional[NlpArtifacts] = None
    ) -> List[RecognizerResult]:
        """
        Analyze text for Japanese addresses.
        
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
        
        # Find all potential address matches
        for match in re.finditer(self.ADDRESS_PATTERN, text):
            address_text = match.group().strip()
            
            # Skip if this looks like a school, organization, or company
            if self._is_excluded(address_text):
                continue
            
            start_pos = match.start()
            end_pos = match.end()
            
            # Higher score if context is found
            if self._has_context(text, start_pos, end_pos):
                score = 1.0
            else:
                score = 0.7
            
            results.append(
                RecognizerResult(
                    entity_type=self.supported_entities[0],
                    start=start_pos,
                    end=end_pos,
                    score=score,
                )
            )
        
        return results


class JapaneseNameRecognizer(EntityRecognizer):
    """
    Context-based recognizer for Japanese person names.
    
    Detects Japanese names that appear directly after context keywords like "氏名", "ふりがな".
    Uses pattern: keyword + space + name (kanji or hiragana/katakana).
    """
    
    # Patterns to match name after context keyword
    # Format: keyword + optional colon/space + name (kanji or hiragana/katakana)
    NAME_AFTER_KEYWORD_PATTERNS = [
        # 氏名 followed by name (kanji with optional space between surname and given name)
        (r'氏名[:\s　]*([一-龯ぁ-んァ-ン]{1,5}[\s　]?[一-龯ぁ-んァ-ン]{1,5})', "full_name"),
        # ふりがな followed by name (hiragana with optional space)
        (r'ふりがな[:\s　]*([ぁ-ん]{1,10}[\s　]?[ぁ-ん]{1,10})', "furigana"),
        # フリガナ followed by name (katakana with optional space)
        (r'フリガナ[:\s　]*([ァ-ン]{1,10}[\s　]?[ァ-ン]{1,10})', "furigana_katakana"),
        # 名前 followed by name
        (r'名前[:\s　]*([一-龯ぁ-んァ-ン]{1,5}[\s　]?[一-龯ぁ-んァ-ン]{1,5})', "name"),
    ]
    
    def __init__(
        self,
        supported_language: str = "ja",
        supported_entity: str = "JP_PERSON",
    ):
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
        Analyze text for Japanese names that follow context keywords.
        
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
        
        # Find names after context keywords
        for pattern, pattern_name in self.NAME_AFTER_KEYWORD_PATTERNS:
            for match in re.finditer(pattern, text):
                # Get the captured name group (group 1)
                name_text = match.group(1).strip()
                
                # Skip very short matches
                if len(name_text.replace(' ', '').replace('　', '')) < 2:
                    continue
                
                # Calculate the position of the name in the full text
                start_pos = match.start(1)
                end_pos = match.end(1)
                
                results.append(
                    RecognizerResult(
                        entity_type=self.supported_entities[0],
                        start=start_pos,
                        end=end_pos,
                        score=0.85,
                    )
                )
        
        return results


