"""Core PII masking functionality.

This package contains the main masking logic:
- masker: Masker class for text masking (with DI support)
- analyzer: AnalyzerEngine creation functions
- processors: Text preprocessing and result processing
- protocols: Protocol definitions for dependency abstraction
- masking_result: Structured result classes
"""

from .analyzer import (
    create_analyzer,
    create_japanese_analyzer,
    create_multilingual_analyzer,
)
from .masker import Masker, PIIMasker, mask_pii_in_text
from .masking_result import EntityInfo, MaskingResult, MaskingStats
from .protocols import AnonymizerProtocol, LoggerProtocol, NullLogger, TextExtractorProtocol

__all__ = [
    # Domain
    "Masker",
    "MaskingResult",
    "EntityInfo",
    "MaskingStats",
    # Analyzer
    "create_analyzer",
    "create_japanese_analyzer",
    "create_multilingual_analyzer",
    # Protocols
    "LoggerProtocol",
    "AnonymizerProtocol",
    "TextExtractorProtocol",
    "NullLogger",
    # Backward compatibility (deprecated)
    "PIIMasker",
    "mask_pii_in_text",
]

