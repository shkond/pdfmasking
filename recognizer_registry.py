"""Centralized registry for managing recognizers with clear categorization."""

from dataclasses import dataclass
from typing import List, Literal, Optional
import warnings

from presidio_analyzer import EntityRecognizer, AnalyzerEngine

from recognizers import (
    JapanesePhoneRecognizer,
    JapaneseZipCodeRecognizer,
    JapaneseBirthDateRecognizer,
    JapaneseNameRecognizer,
    JapaneseAgeRecognizer,
    JapaneseGenderRecognizer,
    JapaneseAddressRecognizer,
)

# Try to import GiNZA recognizers (optional)
GINZA_AVAILABLE = False
try:
    import spacy
    from recognizers import GinzaPersonRecognizer, GinzaAddressRecognizer
    GINZA_AVAILABLE = True
except ImportError:
    pass

RecognizerType = Literal["pattern", "ner_ginza", "ner_presidio"]


@dataclass
class RecognizerConfig:
    """
    Recognizer configuration with metadata for visibility.
    
    Attributes:
        recognizer: The EntityRecognizer instance
        type: Classification of recognizer ("pattern", "ner_ginza", "ner_presidio")
        language: Supported language code
        entity_type: Entity type string this recognizer detects
        description: Human-readable description
        requires_nlp: Whether this recognizer requires NLP artifacts
    """
    recognizer: EntityRecognizer
    type: RecognizerType
    language: str
    entity_type: str
    description: str
    requires_nlp: bool = False
    
    def __repr__(self):
        return f"<{self.type.upper()}:{self.language}:{self.entity_type}>"


class RecognizerRegistry:
    """
    Centralized registry for managing recognizers with clear categorization.
    
    Provides methods to:
    - Register recognizers with metadata
    - Filter by type or language
    - Apply to AnalyzerEngine
    - Generate summary for debugging
    """
    
    def __init__(self):
        self.configs: List[RecognizerConfig] = []
    
    def register(self, config: RecognizerConfig):
        """Register a recognizer with metadata."""
        self.configs.append(config)
    
    def get_by_type(self, recognizer_type: RecognizerType) -> List[RecognizerConfig]:
        """Get all recognizers of a specific type."""
        return [c for c in self.configs if c.type == recognizer_type]
    
    def get_by_language(self, language: str) -> List[RecognizerConfig]:
        """Get all recognizers for a specific language."""
        return [c for c in self.configs if c.language == language]
    
    def apply_to_analyzer(
        self, 
        analyzer: AnalyzerEngine, 
        language: Optional[str] = None, 
        types: Optional[List[RecognizerType]] = None
    ):
        """
        Apply registered recognizers to an analyzer with filtering.
        
        Args:
            analyzer: AnalyzerEngine instance
            language: Filter by language (None = all)
            types: Filter by recognizer types (None = all)
        """
        configs = self.configs
        
        if language:
            configs = [c for c in configs if c.language == language]
        if types:
            configs = [c for c in configs if c.type in types]
        
        for config in configs:
            analyzer.registry.add_recognizer(config.recognizer)
    
    def summary(self) -> str:
        """Generate a human-readable summary of registered recognizers."""
        lines = ["Recognizer Registry Summary:"]
        for rtype in ["pattern", "ner_ginza", "ner_presidio"]:
            configs = self.get_by_type(rtype)
            if configs:
                lines.append(f"\n{rtype.upper()}:")
                for cfg in configs:
                    lines.append(f"  - [{cfg.language}] {cfg.entity_type}: {cfg.description}")
        lines.append(f"\nTotal: {len(self.configs)} recognizers")
        return "\n".join(lines)


def create_default_registry(use_ginza: bool = True) -> RecognizerRegistry:
    """
    Create a registry with all available recognizers.
    
    Args:
        use_ginza: Whether to include GiNZA-based recognizers
        
    Returns:
        RecognizerRegistry with all recognizers registered
    """
    registry = RecognizerRegistry()
    
    # === Pattern-based recognizers (Japanese) ===
    registry.register(RecognizerConfig(
        recognizer=JapanesePhoneRecognizer(),
        type="pattern",
        language="ja",
        entity_type="PHONE_NUMBER_JP",
        description="Japanese phone numbers (正規表現)",
        requires_nlp=False
    ))
    
    registry.register(RecognizerConfig(
        recognizer=JapaneseZipCodeRecognizer(),
        type="pattern",
        language="ja",
        entity_type="JP_ZIP_CODE",
        description="Japanese postal codes (〒XXX-XXXX)",
        requires_nlp=False
    ))
    
    registry.register(RecognizerConfig(
        recognizer=JapaneseBirthDateRecognizer(),
        type="pattern",
        language="ja",
        entity_type="DATE_OF_BIRTH_JP",
        description="Japanese birth dates (生年月日パターン)",
        requires_nlp=False
    ))
    
    registry.register(RecognizerConfig(
        recognizer=JapaneseNameRecognizer(),
        type="pattern",
        language="ja",
        entity_type="JP_PERSON",
        description="Japanese names (コンテキストベース)",
        requires_nlp=False
    ))
    
    registry.register(RecognizerConfig(
        recognizer=JapaneseAgeRecognizer(),
        type="pattern",
        language="ja",
        entity_type="JP_AGE",
        description="Age mentions (XX歳)",
        requires_nlp=False
    ))
    
    registry.register(RecognizerConfig(
        recognizer=JapaneseGenderRecognizer(),
        type="pattern",
        language="ja",
        entity_type="JP_GENDER",
        description="Gender (性別: 男/女)",
        requires_nlp=False
    ))
    
    registry.register(RecognizerConfig(
        recognizer=JapaneseAddressRecognizer(),
        type="pattern",
        language="ja",
        entity_type="JP_ADDRESS",
        description="Japanese addresses (都道府県パターン)",
        requires_nlp=False
    ))
    
    # === GiNZA-based recognizers (if available) ===
    if use_ginza and GINZA_AVAILABLE:
        registry.register(RecognizerConfig(
            recognizer=GinzaPersonRecognizer(),
            type="ner_ginza",
            language="ja",
            entity_type="JP_PERSON",
            description="Person names via GiNZA NER",
            requires_nlp=True
        ))
        
        registry.register(RecognizerConfig(
            recognizer=GinzaAddressRecognizer(),
            type="ner_ginza",
            language="ja",
            entity_type="JP_ADDRESS",
            description="Addresses via GiNZA NER",
            requires_nlp=True
        ))
    
    return registry
