"""Text and result processing utilities."""

from .hybrid_detection import hybrid_detection_analyze
from .result import deduplicate_results, merge_results
from .text import preprocess_text, TextPreprocessor
from .structure_restorer import StructureRestorer, TextSegment
from .candidate_extractor import CandidateExtractor, Candidate
from .candidate_verifier import CandidateVerifier, VerificationResult

__all__ = [
    "deduplicate_results",
    "hybrid_detection_analyze",
    "merge_results",
    "preprocess_text",
    "TextPreprocessor",
    "StructureRestorer",
    "TextSegment",
    "CandidateExtractor",
    "Candidate",
    "CandidateVerifier",
    "VerificationResult",
]

