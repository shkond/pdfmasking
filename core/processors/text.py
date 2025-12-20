"""Text preprocessing pipeline for PDF/document text.

Combines structure restoration, candidate extraction, and verification
into a unified preprocessing pipeline.
"""

from typing import List, Dict, Optional, Tuple

from presidio_analyzer import RecognizerResult

from core.processors.structure_restorer import StructureRestorer, TextSegment
from core.processors.candidate_extractor import CandidateExtractor, Candidate
from core.processors.candidate_verifier import CandidateVerifier, VerificationResult
from core.processors.result_converter import convert_verification_results
from config import load_config


class TextPreprocessor:
    """Unified text preprocessing pipeline.
    
    Processes PDF-extracted text through:
    1. Structure restoration (line/block/section)
    2. Candidate extraction (regex + NER)
    3. Verification and suppression
    """
    
    def __init__(self, config: Optional[Dict] = None, use_ner: bool = False):
        """Initialize the preprocessing pipeline.
        
        Args:
            config: Configuration dict. If None, loads from config.yaml.
            use_ner: If True, enable NER engines (GiNZA/Transformer) for extraction.
        """
        if config is None:
            config = load_config()
        
        self.config = config
        self.use_ner = use_ner
        self.structure_restorer = StructureRestorer(config)
        self.candidate_extractor = CandidateExtractor(config, use_ner=use_ner)
        self.candidate_verifier = CandidateVerifier(config)
    
    def process(
        self, 
        raw_text: str
    ) -> Tuple[List[TextSegment], List[VerificationResult]]:
        """Process raw text through the full pipeline.
        
        Args:
            raw_text: Raw text extracted from PDF/DOCX
            
        Returns:
            Tuple of (segments, verification_results)
        """
        # Step 1: Structure restoration
        segments = self.structure_restorer.restore(raw_text)
        
        # Step 2: Candidate extraction
        candidates = self.candidate_extractor.extract(segments)
        
        # Step 3: Verification and suppression
        results = self.candidate_verifier.verify(candidates)
        
        return segments, results
    
    def get_maskable_candidates(
        self, 
        results: List[VerificationResult]
    ) -> List[Candidate]:
        """Get candidates that should be masked.
        
        Args:
            results: Verification results from process()
            
        Returns:
            List of candidates to mask, sorted by position
        """
        return self.candidate_verifier.get_maskable_candidates(results)
    
    def get_recognizer_results(
        self,
        results: List[VerificationResult],
        include_review: bool = False
    ) -> List[RecognizerResult]:
        """Convert verification results to Presidio RecognizerResult format.
        
        Args:
            results: Verification results from process()
            include_review: If True, include "review" status items
            
        Returns:
            List of RecognizerResult compatible with AnonymizerEngine
        """
        statuses = ["mask"]
        if include_review:
            statuses.append("review")
        return convert_verification_results(results, include_statuses=statuses)
    
    def preprocess_and_detect(
        self, 
        raw_text: str
    ) -> List[Dict]:
        """Process text and return detection results.
        
        Convenience method that returns results in a flat dict format
        suitable for logging.
        
        Args:
            raw_text: Raw text extracted from PDF/DOCX
            
        Returns:
            List of detection dicts with entity info
        """
        segments, results = self.process(raw_text)
        
        detections = []
        for result in results:
            if result.status in ["mask", "review"]:
                candidate = result.candidate
                detections.append({
                    "entity_type": candidate.entity_type,
                    "text": candidate.text,
                    "start": candidate.start,
                    "end": candidate.end,
                    "score": result.verified_score,
                    "source": candidate.source,
                    "section_id": candidate.section_id,
                    "section_type": candidate.section_type,
                    "status": result.status,
                    "reason": result.reason
                })
        
        return detections


def preprocess_text(text: str) -> str:
    """Legacy function: Preprocess text extracted from PDF.
    
    This function is maintained for backward compatibility.
    For full pipeline processing, use TextPreprocessor class.
    
    Args:
        text: Raw text extracted from PDF
        
    Returns:
        Normalized text with consistent formatting
    """
    import re
    
    # Normalize full-width spaces to regular spaces
    text = text.replace('\u3000', ' ')
    
    # Collapse multiple spaces into single space (but preserve newlines)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    
    # Normalize multiple newlines to double newline (paragraph break)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()
