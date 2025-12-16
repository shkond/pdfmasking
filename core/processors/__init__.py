"""Text and result processing utilities."""

from .dual_detection import dual_detection_analyze, normalize_entity_type
from .result import deduplicate_results, merge_results
from .text import preprocess_text

__all__ = [
    "deduplicate_results",
    "dual_detection_analyze",
    "merge_results",
    "normalize_entity_type",
    "preprocess_text",
]
