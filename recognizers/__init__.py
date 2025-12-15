"""Custom recognizers for Japanese PII detection."""

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

# Conditional import for Transformer recognizers (requires torch and transformers)
TRANSFORMER_AVAILABLE = False
try:
    from .transformer_ner import (
        TransformerNERRecognizer,
        create_english_transformer_recognizer,
        create_japanese_transformer_recognizer,
    )
    TRANSFORMER_AVAILABLE = True
except ImportError:
    pass

__all__ = [
    "JapanesePhoneRecognizer",
    "JapaneseZipCodeRecognizer",
    "JapaneseBirthDateRecognizer",
    "JapaneseNameRecognizer",
    "JapaneseAgeRecognizer",
    "JapaneseGenderRecognizer",
    "JapaneseAddressRecognizer",
    "GinzaPersonRecognizer",
    "GinzaAddressRecognizer",
    "TRANSFORMER_AVAILABLE",
]

if TRANSFORMER_AVAILABLE:
    __all__.extend([
        "TransformerNERRecognizer",
        "create_english_transformer_recognizer",
        "create_japanese_transformer_recognizer",
    ])

