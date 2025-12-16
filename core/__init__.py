"""Core PII masking functionality.

This package contains the main masking logic:
- masker: PIIMasker class for text masking
- analyzer: AnalyzerEngine creation functions
- processors: Text preprocessing and result processing
"""

from .analyzer import (
    create_analyzer,
    create_japanese_analyzer,
    create_multilingual_analyzer,
)
from .masker import PIIMasker, mask_pii_in_text

__all__ = [
    "PIIMasker",
    "create_analyzer",
    "create_japanese_analyzer",
    "create_multilingual_analyzer",
    "mask_pii_in_text",
]
