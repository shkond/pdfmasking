"""Candidate extraction for PII detection.

Extracts PII candidates using regex patterns and NER models.
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

from core.processors.structure_restorer import TextSegment
from config import load_config


@dataclass
class Candidate:
    """Represents a PII candidate detection."""
    entity_type: str          # EMAIL_ADDRESS, PHONE_NUMBER_JP, JP_PERSON, etc.
    text: str                 # Detected text
    start: int                # Start position in original text
    end: int                  # End position in original text
    score: float              # Confidence score (0.0 - 1.0)
    source: str               # Detection source (rule:email_regex, ner:ginza, etc.)
    section_id: str           # Section ID where detected
    section_type: str         # Section type (contact, education, etc.)


class CandidateExtractor:
    """Extracts PII candidates using regex patterns and NER."""
    
    # Email regex pattern
    EMAIL_PATTERN = re.compile(
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}|'
        r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\w._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    )
    
    # Phone patterns (Japanese)
    PHONE_JP_PATTERNS = [
        re.compile(r'0\d{1,4}-\d{1,4}-\d{4}'),  # 03-1234-5678, 090-1234-5678
        re.compile(r'0\d{9,10}'),                # 0312345678
        re.compile(r'\+81-?\d{1,4}-?\d{1,4}-?\d{4}'),  # +81-90-1234-5678
    ]
    
    # Phone patterns (US/International)
    PHONE_EN_PATTERNS = [
        re.compile(r'\+1-?\d{3}-?\d{3}-?\d{4}'),  # +1-123-456-7890
        re.compile(r'\(\d{3}\)\s?\d{3}-\d{4}'),    # (123) 456-7890
    ]
    
    # Japanese postal code pattern
    JP_ZIP_PATTERN = re.compile(r'〒?\s*\d{3}-\d{4}')
    
    # US postal code pattern
    US_ZIP_PATTERN = re.compile(r'\b\d{5}(?:-\d{4})?\b')
    
    # Date patterns for birth date
    DATE_PATTERNS = [
        # ISO format
        re.compile(r'\b\d{4}-\d{2}-\d{2}\b'),
        # Japanese format with kanji
        re.compile(r'\d{4}年\d{1,2}月\d{1,2}日'),
        # Slash format
        re.compile(r'\b\d{4}/\d{1,2}/\d{1,2}\b'),
    ]
    
    # Age pattern (Japanese)
    AGE_PATTERN = re.compile(r'(\d{1,3})\s*歳')
    
    # Gender patterns (Japanese)
    GENDER_PATTERNS = [
        re.compile(r'[（(]?\s*(男性?|女性?|男|女)\s*[）)]?'),
    ]
    
    # Japanese address prefecture patterns
    JP_ADDRESS_PREFIXES = [
        '北海道', '青森県', '岩手県', '宮城県', '秋田県', '山形県', '福島県',
        '茨城県', '栃木県', '群馬県', '埼玉県', '千葉県', '東京都', '神奈川県',
        '新潟県', '富山県', '石川県', '福井県', '山梨県', '長野県', '岐阜県',
        '静岡県', '愛知県', '三重県', '滋賀県', '京都府', '大阪府', '兵庫県',
        '奈良県', '和歌山県', '鳥取県', '島根県', '岡山県', '広島県', '山口県',
        '徳島県', '香川県', '愛媛県', '高知県', '福岡県', '佐賀県', '長崎県',
        '熊本県', '大分県', '宮崎県', '鹿児島県', '沖縄県'
    ]
    
    def __init__(self, config: Optional[Dict] = None, use_ner: bool = False):
        """Initialize candidate extractor.
        
        Args:
            config: Configuration dict. If None, loads from config.yaml.
            use_ner: If True, enable NER engines (GiNZA/Transformer) for extraction.
        """
        if config is None:
            config = load_config()
        
        self.config = config
        self.use_ner = use_ner
        self._load_extraction_config()
        
        # NER engines (lazy initialization)
        self._nlp_ja = None
        self._transformer_recognizers = None
        self._ner_initialized = False

    
    def _load_extraction_config(self) -> None:
        """Load extraction configuration."""
        extraction_config = self.config.get("candidate_extraction", {})
        self.entity_priority = extraction_config.get("entity_priority", [
            "EMAIL_ADDRESS",
            "PHONE_NUMBER_JP",
            "JP_ZIP_CODE",
            "JP_ADDRESS",
            "JP_PERSON"
        ])
    
    def _init_ner_engines(self) -> None:
        """Initialize NER engines (GiNZA and Transformer).
        
        Lazy initialization - only called when use_ner=True and first extraction.
        """
        if self._ner_initialized:
            return
        
        # Initialize GiNZA
        try:
            import spacy
            self._nlp_ja = spacy.load("ja_ginza")
        except ImportError:
            import warnings
            warnings.warn("spacy/ginza not available - GiNZA NER disabled")
            self._nlp_ja = None
        except OSError:
            import warnings
            warnings.warn("ja_ginza model not found - GiNZA NER disabled")
            self._nlp_ja = None
        
        # Initialize Transformer recognizers from config
        self._transformer_recognizers = []
        try:
            from model_registry import ModelRegistry
            registry = ModelRegistry(self.config)
            models = registry.list_models()
            
            for model_id in models:
                try:
                    recognizer = registry.get_recognizer(model_id)
                    if recognizer:
                        self._transformer_recognizers.append({
                            "id": model_id,
                            "recognizer": recognizer,
                            "language": models[model_id].get("language", "ja")
                        })
                except Exception as e:
                    import warnings
                    warnings.warn(f"Failed to load transformer model {model_id}: {e}")
        except ImportError:
            import warnings
            warnings.warn("ModelRegistry not available - Transformer NER disabled")
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to initialize Transformer NER: {e}")
        
        self._ner_initialized = True
    
    def _extract_ner_candidates(
        self,
        text: str,
        segments: List[TextSegment]
    ) -> List[Candidate]:
        """Extract PII candidates using NER engines.
        
        Args:
            text: Full text to analyze
            segments: Text segments with section context
            
        Returns:
            List of Candidate objects from NER
        """
        candidates = []
        
        # Build segment lookup for position-to-section mapping
        def get_section_for_position(pos: int) -> Tuple[str, str]:
            for segment in segments:
                if segment.char_start <= pos < segment.char_end:
                    return segment.section_id, segment.section_type
            return "unknown", "unknown"
        
        # GiNZA NER
        if self._nlp_ja:
            try:
                doc = self._nlp_ja(text)
                for ent in doc.ents:
                    # Map spaCy labels to our entity types
                    entity_type = None
                    if ent.label_ == "PERSON":
                        entity_type = "JP_PERSON"
                    elif ent.label_ in ["LOC", "GPE"]:
                        entity_type = "JP_ADDRESS"
                    elif ent.label_ == "ORG":
                        entity_type = "JP_ORGANIZATION"
                    
                    if entity_type:
                        section_id, section_type = get_section_for_position(ent.start_char)
                        candidates.append(Candidate(
                            entity_type=entity_type,
                            text=ent.text,
                            start=ent.start_char,
                            end=ent.end_char,
                            score=0.7,  # Base NER confidence
                            source="ner:ginza",
                            section_id=section_id,
                            section_type=section_type
                        ))
            except Exception as e:
                import warnings
                warnings.warn(f"GiNZA NER failed: {e}")
        
        # Transformer NER
        for transformer_info in self._transformer_recognizers:
            recognizer = transformer_info["recognizer"]
            try:
                results = recognizer.analyze(
                    text=text,
                    entities=["JP_PERSON", "JP_ADDRESS", "PERSON", "LOCATION"]
                )
                for result in results:
                    section_id, section_type = get_section_for_position(result.start)
                    entity_text = text[result.start:result.end]
                    candidates.append(Candidate(
                        entity_type=result.entity_type,
                        text=entity_text,
                        start=result.start,
                        end=result.end,
                        score=result.score,
                        source=f"ner:transformer:{transformer_info['id']}",
                        section_id=section_id,
                        section_type=section_type
                    ))
            except Exception as e:
                import warnings
                warnings.warn(f"Transformer NER failed ({transformer_info['id']}): {e}")
        
        return candidates

    
    def extract(self, segments: List[TextSegment]) -> List[Candidate]:
        """Extract PII candidates from structured segments.
        
        Args:
            segments: List of TextSegment from structure restorer
            
        Returns:
            List of Candidate objects with detected PII
        """
        candidates = []
        
        # Reconstruct full text for position tracking
        full_text = self._reconstruct_text(segments)
        
        # Regex-based extraction from each segment
        for segment in segments:
            segment_candidates = self._extract_from_segment(segment, full_text)
            candidates.extend(segment_candidates)
        
        # NER-based extraction (if enabled)
        if self.use_ner:
            self._init_ner_engines()
            ner_candidates = self._extract_ner_candidates(full_text, segments)
            candidates.extend(ner_candidates)
        
        # Merge overlapping candidates
        candidates = self._merge_candidates(candidates)
        
        # Sort by start position
        candidates.sort(key=lambda c: c.start)
        
        return candidates
    
    def _reconstruct_text(self, segments: List[TextSegment]) -> str:
        """Reconstruct full text from segments."""
        if not segments:
            return ""
        
        # Find the maximum end position to determine text length
        max_end = max(s.char_end for s in segments)
        
        # Create a text buffer
        text_parts = []
        last_end = 0
        
        for segment in sorted(segments, key=lambda s: s.char_start):
            # Add spacing if there's a gap
            if segment.char_start > last_end:
                text_parts.append(' ' * (segment.char_start - last_end))
            text_parts.append(segment.line_text)
            last_end = segment.char_end
        
        return ''.join(text_parts)
    
    def _extract_from_segment(
        self, 
        segment: TextSegment,
        full_text: str
    ) -> List[Candidate]:
        """Extract candidates from a single segment.
        
        Args:
            segment: TextSegment to process
            full_text: Full reconstructed text
            
        Returns:
            List of detected candidates
        """
        candidates = []
        text = segment.line_text
        offset = segment.char_start
        
        # Extract email
        candidates.extend(self._extract_emails(text, offset, segment))
        
        # Extract phone numbers
        candidates.extend(self._extract_phones(text, offset, segment))
        
        # Extract postal codes
        candidates.extend(self._extract_postal_codes(text, offset, segment))
        
        # Extract dates (potential birth dates)
        candidates.extend(self._extract_dates(text, offset, segment))
        
        # Extract age
        candidates.extend(self._extract_age(text, offset, segment))
        
        # Extract gender
        candidates.extend(self._extract_gender(text, offset, segment))
        
        # Extract Japanese addresses
        candidates.extend(self._extract_jp_addresses(text, offset, segment))
        
        return candidates
    
    def _extract_emails(
        self, 
        text: str, 
        offset: int, 
        segment: TextSegment
    ) -> List[Candidate]:
        """Extract email addresses."""
        candidates = []
        
        for match in self.EMAIL_PATTERN.finditer(text):
            candidates.append(Candidate(
                entity_type="EMAIL_ADDRESS",
                text=match.group(),
                start=offset + match.start(),
                end=offset + match.end(),
                score=0.95,  # High confidence for regex match
                source="rule:email_regex",
                section_id=segment.section_id,
                section_type=segment.section_type
            ))
        
        return candidates
    
    def _extract_phones(
        self, 
        text: str, 
        offset: int, 
        segment: TextSegment
    ) -> List[Candidate]:
        """Extract phone numbers."""
        candidates = []
        
        # Japanese phone patterns
        for pattern in self.PHONE_JP_PATTERNS:
            for match in pattern.finditer(text):
                candidates.append(Candidate(
                    entity_type="PHONE_NUMBER_JP",
                    text=match.group(),
                    start=offset + match.start(),
                    end=offset + match.end(),
                    score=0.9,
                    source="rule:phone_jp_regex",
                    section_id=segment.section_id,
                    section_type=segment.section_type
                ))
        
        # English phone patterns
        for pattern in self.PHONE_EN_PATTERNS:
            for match in pattern.finditer(text):
                candidates.append(Candidate(
                    entity_type="PHONE_NUMBER",
                    text=match.group(),
                    start=offset + match.start(),
                    end=offset + match.end(),
                    score=0.9,
                    source="rule:phone_en_regex",
                    section_id=segment.section_id,
                    section_type=segment.section_type
                ))
        
        return candidates
    
    def _extract_postal_codes(
        self, 
        text: str, 
        offset: int, 
        segment: TextSegment
    ) -> List[Candidate]:
        """Extract postal codes."""
        candidates = []
        
        # Japanese postal code
        for match in self.JP_ZIP_PATTERN.finditer(text):
            matched_text = match.group()
            
            # Skip year ranges like "2016-2024" that could match as "016-2024"
            # Check if preceded by a digit (indicating year range)
            if match.start() > 0 and text[match.start() - 1].isdigit():
                continue
            
            # Check for year range context (YYYY-YYYY pattern)
            if match.start() >= 2:
                before = text[match.start() - 2:match.start()]
                if before.isdigit() or (before[0].isdigit() and before[1] == '-'):
                    continue
            
            candidates.append(Candidate(
                entity_type="JP_ZIP_CODE",
                text=matched_text,
                start=offset + match.start(),
                end=offset + match.end(),
                score=0.95,
                source="rule:jp_zip_regex",
                section_id=segment.section_id,
                section_type=segment.section_type
            ))
        
        # US postal code (only in contact section to reduce false positives)
        if segment.section_type in ["contact", "header"]:
            for match in self.US_ZIP_PATTERN.finditer(text):
                # Skip if looks like a year (1900-2100)
                val = match.group()[:4] if len(match.group()) >= 4 else match.group()
                if val.isdigit() and 1900 <= int(val) <= 2100:
                    continue
                    
                candidates.append(Candidate(
                    entity_type="US_ZIP_CODE",
                    text=match.group(),
                    start=offset + match.start(),
                    end=offset + match.end(),
                    score=0.7,  # Lower score due to false positive risk
                    source="rule:us_zip_regex",
                    section_id=segment.section_id,
                    section_type=segment.section_type
                ))
        
        return candidates
    
    def _extract_dates(
        self, 
        text: str, 
        offset: int, 
        segment: TextSegment
    ) -> List[Candidate]:
        """Extract dates (potential birth dates)."""
        candidates = []
        
        for pattern in self.DATE_PATTERNS:
            for match in pattern.finditer(text):
                # Determine if likely birth date based on context
                is_birth_context = any(
                    kw in text.lower() 
                    for kw in ['birth', '生年月日', '誕生', '生まれ', 'date of birth']
                )
                
                score = 0.9 if is_birth_context else 0.5
                entity_type = "DATE_OF_BIRTH_JP" if is_birth_context else "DATE"
                
                candidates.append(Candidate(
                    entity_type=entity_type,
                    text=match.group(),
                    start=offset + match.start(),
                    end=offset + match.end(),
                    score=score,
                    source="rule:date_regex",
                    section_id=segment.section_id,
                    section_type=segment.section_type
                ))
        
        return candidates
    
    def _extract_age(
        self, 
        text: str, 
        offset: int, 
        segment: TextSegment
    ) -> List[Candidate]:
        """Extract age mentions."""
        candidates = []
        
        for match in self.AGE_PATTERN.finditer(text):
            age_val = int(match.group(1))
            # Reasonable age range
            if 0 <= age_val <= 120:
                candidates.append(Candidate(
                    entity_type="JP_AGE",
                    text=match.group(),
                    start=offset + match.start(),
                    end=offset + match.end(),
                    score=0.85,
                    source="rule:age_regex",
                    section_id=segment.section_id,
                    section_type=segment.section_type
                ))
        
        return candidates
    
    def _extract_gender(
        self, 
        text: str, 
        offset: int, 
        segment: TextSegment
    ) -> List[Candidate]:
        """Extract gender mentions."""
        candidates = []
        
        for pattern in self.GENDER_PATTERNS:
            for match in pattern.finditer(text):
                candidates.append(Candidate(
                    entity_type="JP_GENDER",
                    text=match.group(),
                    start=offset + match.start(),
                    end=offset + match.end(),
                    score=0.8,
                    source="rule:gender_regex",
                    section_id=segment.section_id,
                    section_type=segment.section_type
                ))
        
        return candidates
    
    def _extract_jp_addresses(
        self, 
        text: str, 
        offset: int, 
        segment: TextSegment
    ) -> List[Candidate]:
        """Extract Japanese addresses starting with prefecture."""
        candidates = []
        
        for prefecture in self.JP_ADDRESS_PREFIXES:
            if prefecture in text:
                # Find the prefecture and extract following address
                start_idx = text.index(prefecture)
                
                # Extract until end of line or common delimiter
                end_idx = len(text)
                for delimiter in ['\n', '　', '  ', 'Email', 'email', 'Phone', 'phone', 'TEL', 'tel']:
                    if delimiter in text[start_idx:]:
                        potential_end = start_idx + text[start_idx:].index(delimiter)
                        if potential_end < end_idx:
                            end_idx = potential_end
                
                address_text = text[start_idx:end_idx].strip()
                
                # Only accept if it looks like a real address (has numbers or more content)
                if len(address_text) > len(prefecture) + 2:
                    candidates.append(Candidate(
                        entity_type="JP_ADDRESS",
                        text=address_text,
                        start=offset + start_idx,
                        end=offset + start_idx + len(address_text),
                        score=0.75,  # Medium confidence, needs NER validation
                        source="rule:jp_address_prefix",
                        section_id=segment.section_id,
                        section_type=segment.section_type
                    ))
        
        return candidates
    
    def _merge_candidates(self, candidates: List[Candidate]) -> List[Candidate]:
        """Merge overlapping candidates, keeping higher priority/score.
        
        Args:
            candidates: List of candidates to merge
            
        Returns:
            Merged list without overlaps
        """
        if not candidates:
            return []
        
        # Sort by start position, then by priority (lower index = higher priority)
        def priority_key(c: Candidate) -> int:
            try:
                return self.entity_priority.index(c.entity_type)
            except ValueError:
                return len(self.entity_priority)  # Unknown types at end
        
        sorted_candidates = sorted(
            candidates, 
            key=lambda c: (c.start, priority_key(c), -c.score)
        )
        
        merged = []
        for candidate in sorted_candidates:
            # Check if overlaps with any existing
            overlaps = False
            for existing in merged:
                if self._overlaps(candidate, existing):
                    overlaps = True
                    # Keep higher priority or higher score
                    if (priority_key(candidate) < priority_key(existing) or 
                        (priority_key(candidate) == priority_key(existing) and 
                         candidate.score > existing.score)):
                        merged.remove(existing)
                        merged.append(candidate)
                    break
            
            if not overlaps:
                merged.append(candidate)
        
        return merged
    
    def _overlaps(self, c1: Candidate, c2: Candidate) -> bool:
        """Check if two candidates overlap."""
        return not (c1.end <= c2.start or c2.end <= c1.start)
