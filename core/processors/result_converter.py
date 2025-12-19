"""Result converter: TextPreprocessor output to Presidio format.

Converts VerificationResult/Candidate objects to Presidio RecognizerResult
for seamless integration with AnonymizerEngine.
"""

from typing import List

from presidio_analyzer import RecognizerResult

from core.processors.candidate_extractor import Candidate
from core.processors.candidate_verifier import VerificationResult


def candidate_to_recognizer_result(candidate: Candidate) -> RecognizerResult:
    """Convert a Candidate to Presidio RecognizerResult.
    
    Args:
        candidate: Candidate object from CandidateExtractor
        
    Returns:
        RecognizerResult compatible with AnonymizerEngine
    """
    return RecognizerResult(
        entity_type=candidate.entity_type,
        start=candidate.start,
        end=candidate.end,
        score=candidate.score
    )


def verification_result_to_recognizer_result(
    vr: VerificationResult
) -> RecognizerResult:
    """Convert a VerificationResult to Presidio RecognizerResult.
    
    Uses the verified_score from verification process.
    
    Args:
        vr: VerificationResult object from CandidateVerifier
        
    Returns:
        RecognizerResult compatible with AnonymizerEngine
    """
    return RecognizerResult(
        entity_type=vr.candidate.entity_type,
        start=vr.candidate.start,
        end=vr.candidate.end,
        score=vr.verified_score
    )


def convert_verification_results(
    results: List[VerificationResult],
    include_statuses: List[str] = None
) -> List[RecognizerResult]:
    """Convert a list of VerificationResults to RecognizerResults.
    
    Args:
        results: List of VerificationResult from CandidateVerifier
        include_statuses: List of statuses to include (default: ["mask"])
                         Use ["mask", "review"] to include review items.
        
    Returns:
        List of RecognizerResult for AnonymizerEngine
    """
    if include_statuses is None:
        include_statuses = ["mask"]
    
    return [
        verification_result_to_recognizer_result(vr)
        for vr in results
        if vr.status in include_statuses
    ]


def convert_candidates(candidates: List[Candidate]) -> List[RecognizerResult]:
    """Convert a list of Candidates to RecognizerResults.
    
    Args:
        candidates: List of Candidate objects
        
    Returns:
        List of RecognizerResult for AnonymizerEngine
    """
    return [candidate_to_recognizer_result(c) for c in candidates]
