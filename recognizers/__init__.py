"""Custom recognizers for Japanese PII detection.

This package contains:
- japanese_patterns: Pattern-based recognizers for phone, zip, date, etc.
- japanese_ner: GiNZA-based NER recognizers
- transformer_ner: Transformer-based NER recognizers
- registry: Centralized recognizer registry
"""

from .japanese_patterns import (
    JapanesePhoneRecognizer,
    JapaneseZipCodeRecognizer,
    JapaneseBirthDateRecognizer,
    JapaneseNameRecognizer,
    JapaneseAgeRecognizer,
    JapaneseGenderRecognizer,
    JapaneseAddressRecognizer,
)
from .japanese_ner import (
    GinzaPersonRecognizer,
    GinzaAddressRecognizer,
)
from .registry import (
    RecognizerRegistry,
    RecognizerConfig,
    create_default_registry,
    GINZA_AVAILABLE,
)

# Conditional import for Transformer recognizers (requires torch and transformers)
TRANSFORMER_AVAILABLE = False
try:
    from .transformer_ner import (
        TransformerNERRecognizer,
        create_transformer_recognizer,
    )
    TRANSFORMER_AVAILABLE = True
except ImportError:
    pass

__all__ = [
    # Pattern recognizers
    "JapanesePhoneRecognizer",
    "JapaneseZipCodeRecognizer",
    "JapaneseBirthDateRecognizer",
    "JapaneseNameRecognizer",
    "JapaneseAgeRecognizer",
    "JapaneseGenderRecognizer",
    "JapaneseAddressRecognizer",
    # GiNZA recognizers
    "GinzaPersonRecognizer",
    "GinzaAddressRecognizer",
    # Registry
    "RecognizerRegistry",
    "RecognizerConfig",
    "create_default_registry",
    "GINZA_AVAILABLE",
    "TRANSFORMER_AVAILABLE",
]

if TRANSFORMER_AVAILABLE:
    __all__.extend([
        "TransformerNERRecognizer",
        "create_transformer_recognizer",
    ])
