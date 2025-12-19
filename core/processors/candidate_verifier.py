"""Candidate verification and suppression for PII detection.

Verifies candidates using section-based policies, format validation,
and allow-lists.
"""

import re
from dataclasses import dataclass, replace
from typing import List, Dict, Optional, Set
from pathlib import Path

from core.processors.candidate_extractor import Candidate
from config import load_config


@dataclass
class VerificationResult:
    """Result of candidate verification."""
    candidate: Candidate          # Original candidate
    verified_score: float         # Adjusted score after verification
    status: str                   # "mask", "review", "exclude"
    reason: str                   # Reason for status


class CandidateVerifier:
    """Verifies and filters PII candidates."""
    
    # Email validation pattern (RFC 5322 simplified)
    EMAIL_VALID_PATTERN = re.compile(
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$|'
        r'^[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\w._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    )
    
    # Phone validation patterns
    PHONE_JP_VALID = re.compile(r'^0\d{1,4}-?\d{1,4}-?\d{4}$|^\+81-?\d{1,4}-?\d{1,4}-?\d{4}$')
    PHONE_EN_VALID = re.compile(r'^\+1-?\d{3}-?\d{3}-?\d{4}$|\(\d{3}\)\s?\d{3}-\d{4}$')
    
    # Postal code validation
    JP_ZIP_VALID = re.compile(r'^〒?\s*\d{3}-\d{4}$')
    US_ZIP_VALID = re.compile(r'^\d{5}(-\d{4})?$')
    
    # Entity priority for span collision resolution
    DEFAULT_PRIORITY = [
        "EMAIL_ADDRESS",
        "PHONE_NUMBER_JP",
        "PHONE_NUMBER",
        "JP_ZIP_CODE",
        "US_ZIP_CODE",
        "JP_ADDRESS",
        "LOCATION",
        "JP_PERSON",
        "PERSON",
        "DATE_OF_BIRTH_JP",
        "DATE",
        "JP_AGE",
        "JP_GENDER"
    ]
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize candidate verifier.
        
        Args:
            config: Configuration dict. If None, loads from config.yaml.
        """
        if config is None:
            config = load_config()
        
        self.config = config
        self._load_verification_config()
        self._load_allow_list()
    
    def _load_verification_config(self) -> None:
        """Load verification configuration."""
        verification_config = self.config.get("verification", {})
        
        # Score thresholds
        thresholds = verification_config.get("score_thresholds", {})
        self.mask_threshold = thresholds.get("mask", 0.7)
        self.review_threshold = thresholds.get("review", 0.5)
        
        # Section policies
        self.section_policies = verification_config.get("section_policies", {
            "education": {
                "person_penalty": -0.3,
                "address_penalty": -0.2
            },
            "experience": {
                "person_penalty": -0.3,
                "address_penalty": -0.2
            },
            "contact": {
                "person_boost": 0.2,
                "address_boost": 0.2
            }
        })
        
        # Entity priority
        extraction_config = self.config.get("candidate_extraction", {})
        self.entity_priority = extraction_config.get(
            "entity_priority", 
            self.DEFAULT_PRIORITY
        )
    
    def _load_allow_list(self) -> None:
        """Load allow list from config and dictionary file."""
        self.allow_list: Set[str] = set()
        self.allow_list_normalized: Set[str] = set()
        
        allow_config = self.config.get("allow_list", {})
        if not allow_config.get("enabled", False):
            return
        
        # Load from dictionary file
        dict_path = allow_config.get("dictionary_path", "")
        if dict_path:
            self._load_dictionary_file(dict_path)
        
        # Add additional terms from config
        additional = allow_config.get("additional_terms", [])
        for term in additional:
            self.allow_list.add(term)
            self.allow_list_normalized.add(self._normalize_text(term))
    
    def _load_dictionary_file(self, path: str) -> None:
        """Load allow list from dictionary file."""
        try:
            dict_file = Path(path)
            if not dict_file.is_absolute():
                dict_file = Path("/workspaces/pdfmasking") / path
            
            if dict_file.exists():
                with open(dict_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            self.allow_list.add(line)
                            self.allow_list_normalized.add(
                                self._normalize_text(line)
                            )
        except Exception:
            pass  # Silently ignore dictionary loading errors
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        return text.lower().strip().replace(' ', '').replace('　', '')
    
    def verify(self, candidates: List[Candidate]) -> List[VerificationResult]:
        """Verify candidates and apply suppression rules.
        
        Args:
            candidates: List of candidates to verify
            
        Returns:
            List of VerificationResult with status
        """
        results = []
        
        for candidate in candidates:
            result = self._verify_single(candidate)
            results.append(result)
        
        # Resolve span collisions
        results = self._resolve_collisions(results)
        
        return results
    
    def _verify_single(self, candidate: Candidate) -> VerificationResult:
        """Verify a single candidate.
        
        Args:
            candidate: Candidate to verify
            
        Returns:
            VerificationResult with adjusted score
        """
        score = candidate.score
        reasons = []
        
        # Check allow list first
        if self._is_in_allow_list(candidate.text):
            return VerificationResult(
                candidate=candidate,
                verified_score=0.0,
                status="exclude",
                reason="allow_list_match"
            )
        
        # Apply section-based policies
        score, section_reason = self._apply_section_policy(candidate, score)
        if section_reason:
            reasons.append(section_reason)
        
        # Format validation
        is_valid, valid_reason = self._validate_format(candidate)
        if not is_valid:
            return VerificationResult(
                candidate=candidate,
                verified_score=0.0,
                status="exclude",
                reason=valid_reason
            )
        if valid_reason:
            reasons.append(valid_reason)
        
        # Determine final status based on score
        if score >= self.mask_threshold:
            status = "mask"
        elif score >= self.review_threshold:
            status = "review"
        else:
            status = "exclude"
        
        reason = "; ".join(reasons) if reasons else "passed_verification"
        
        return VerificationResult(
            candidate=candidate,
            verified_score=score,
            status=status,
            reason=reason
        )
    
    def _is_in_allow_list(self, text: str) -> bool:
        """Check if text is in allow list."""
        if text in self.allow_list:
            return True
        
        normalized = self._normalize_text(text)
        return normalized in self.allow_list_normalized
    
    def _apply_section_policy(
        self, 
        candidate: Candidate, 
        score: float
    ) -> tuple[float, str]:
        """Apply section-based score adjustments.
        
        Args:
            candidate: Candidate to check
            score: Current score
            
        Returns:
            Tuple of (adjusted_score, reason)
        """
        section_type = candidate.section_type
        entity_type = candidate.entity_type
        
        policy = self.section_policies.get(section_type, {})
        
        # Person-like entities
        if entity_type in ["JP_PERSON", "PERSON"]:
            penalty = policy.get("person_penalty", 0)
            boost = policy.get("person_boost", 0)
            adjustment = penalty + boost
            
            if adjustment != 0:
                score = max(0.0, min(1.0, score + adjustment))
                if adjustment < 0:
                    return score, f"section_penalty:{section_type}"
                else:
                    return score, f"section_boost:{section_type}"
        
        # Address-like entities
        if entity_type in ["JP_ADDRESS", "LOCATION"]:
            penalty = policy.get("address_penalty", 0)
            boost = policy.get("address_boost", 0)
            adjustment = penalty + boost
            
            if adjustment != 0:
                score = max(0.0, min(1.0, score + adjustment))
                if adjustment < 0:
                    return score, f"section_penalty:{section_type}"
                else:
                    return score, f"section_boost:{section_type}"
        
        return score, ""
    
    def _validate_format(self, candidate: Candidate) -> tuple[bool, str]:
        """Validate candidate format.
        
        Args:
            candidate: Candidate to validate
            
        Returns:
            Tuple of (is_valid, reason)
        """
        entity_type = candidate.entity_type
        text = candidate.text.strip()
        
        # Email validation
        if entity_type == "EMAIL_ADDRESS":
            if self.EMAIL_VALID_PATTERN.match(text):
                return True, "format_valid:email"
            return False, "format_invalid:email"
        
        # Phone validation
        if entity_type == "PHONE_NUMBER_JP":
            # Remove spaces for validation
            text_clean = text.replace(' ', '').replace('　', '')
            if self.PHONE_JP_VALID.match(text_clean):
                return True, "format_valid:phone_jp"
            return False, "format_invalid:phone_jp"
        
        if entity_type == "PHONE_NUMBER":
            text_clean = text.replace(' ', '')
            if self.PHONE_EN_VALID.match(text_clean):
                return True, "format_valid:phone_en"
            return False, "format_invalid:phone_en"
        
        # Postal code validation
        if entity_type == "JP_ZIP_CODE":
            if self.JP_ZIP_VALID.match(text):
                return True, "format_valid:jp_zip"
            return False, "format_invalid:jp_zip"
        
        if entity_type == "US_ZIP_CODE":
            if self.US_ZIP_VALID.match(text):
                return True, "format_valid:us_zip"
            return False, "format_invalid:us_zip"
        
        # Date validation
        if entity_type in ["DATE_OF_BIRTH_JP", "DATE"]:
            if self._validate_date(text):
                return True, "format_valid:date"
            return False, "format_invalid:date"
        
        # Default: no specific validation
        return True, ""
    
    def _validate_date(self, text: str) -> bool:
        """Validate date format and reasonableness."""
        import re
        
        # Try to extract year
        year = None
        
        # ISO format: YYYY-MM-DD
        match = re.match(r'(\d{4})-(\d{2})-(\d{2})', text)
        if match:
            year = int(match.group(1))
            month = int(match.group(2))
            day = int(match.group(3))
            if 1 <= month <= 12 and 1 <= day <= 31:
                if 1900 <= year <= 2100:
                    return True
        
        # Japanese format: YYYY年MM月DD日
        match = re.match(r'(\d{4})年(\d{1,2})月(\d{1,2})日', text)
        if match:
            year = int(match.group(1))
            month = int(match.group(2))
            day = int(match.group(3))
            if 1 <= month <= 12 and 1 <= day <= 31:
                if 1900 <= year <= 2100:
                    return True
        
        # Slash format: YYYY/MM/DD
        match = re.match(r'(\d{4})/(\d{1,2})/(\d{1,2})', text)
        if match:
            year = int(match.group(1))
            month = int(match.group(2))
            day = int(match.group(3))
            if 1 <= month <= 12 and 1 <= day <= 31:
                if 1900 <= year <= 2100:
                    return True
        
        return False
    
    def _resolve_collisions(
        self, 
        results: List[VerificationResult]
    ) -> List[VerificationResult]:
        """Resolve span collisions between candidates.
        
        Args:
            results: List of verification results
            
        Returns:
            List with collisions resolved
        """
        # Filter to only mask/review candidates
        active = [r for r in results if r.status in ["mask", "review"]]
        excluded = [r for r in results if r.status == "exclude"]
        
        if not active:
            return results
        
        # Sort by start position
        active.sort(key=lambda r: r.candidate.start)
        
        resolved = []
        for result in active:
            overlaps = False
            for existing in resolved:
                if self._overlaps(result.candidate, existing.candidate):
                    overlaps = True
                    # Keep higher priority
                    if self._get_priority(result.candidate) < \
                       self._get_priority(existing.candidate):
                        # Replace existing with current
                        resolved.remove(existing)
                        resolved.append(result)
                        # Mark existing as excluded
                        excluded.append(VerificationResult(
                            candidate=existing.candidate,
                            verified_score=0.0,
                            status="exclude",
                            reason="collision_lower_priority"
                        ))
                    else:
                        # Mark current as excluded
                        excluded.append(VerificationResult(
                            candidate=result.candidate,
                            verified_score=0.0,
                            status="exclude",
                            reason="collision_lower_priority"
                        ))
                    break
            
            if not overlaps:
                resolved.append(result)
        
        return resolved + excluded
    
    def _overlaps(self, c1: Candidate, c2: Candidate) -> bool:
        """Check if two candidates overlap."""
        return not (c1.end <= c2.start or c2.end <= c1.start)
    
    def _get_priority(self, candidate: Candidate) -> int:
        """Get priority index for candidate (lower = higher priority)."""
        try:
            return self.entity_priority.index(candidate.entity_type)
        except ValueError:
            return len(self.entity_priority)
    
    def get_maskable_candidates(
        self, 
        results: List[VerificationResult]
    ) -> List[Candidate]:
        """Get candidates that should be masked.
        
        Args:
            results: Verification results
            
        Returns:
            List of candidates to mask
        """
        return [r.candidate for r in results if r.status == "mask"]
    
    def get_review_candidates(
        self, 
        results: List[VerificationResult]
    ) -> List[Candidate]:
        """Get candidates that need review.
        
        Args:
            results: Verification results
            
        Returns:
            List of candidates needing review
        """
        return [r.candidate for r in results if r.status == "review"]
