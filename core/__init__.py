"""Core PII masking functionality.

This package contains the main masking logic:
- masker: PIIMasker class for text masking
- analyzer: AnalyzerEngine creation functions
- processors: Text preprocessing and result processing
"""

from .masker import PIIMasker, mask_pii_in_text
from .analyzer import (
    create_analyzer,
    create_multilingual_analyzer,
    create_japanese_analyzer,
)

__all__ = [
    "PIIMasker",
    "mask_pii_in_text",
    "create_analyzer",
    "create_multilingual_analyzer",
    "create_japanese_analyzer",
]
