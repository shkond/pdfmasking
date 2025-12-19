"""Masking Result - Structured result for masking operations.

Provides immutable data classes for masking operation results.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EntityInfo:
    """Information about a single detected entity.
    
    Attributes:
        entity_type: Type of entity (e.g., "PERSON", "JP_ADDRESS")
        text: Original text that was detected
        score: Confidence score (0.0 - 1.0)
        start: Start position in original text
        end: End position in original text
    """
    entity_type: str
    text: str
    score: float
    start: int
    end: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "type": self.entity_type,
            "text": self.text,
            "score": self.score,
            "start": self.start,
            "end": self.end,
        }


@dataclass(frozen=True)
class MaskingStats:
    """Statistics about the masking operation.
    
    Attributes:
        total_entities: Total number of entities detected
        entities_by_type: Count of entities by type
    """
    total_entities: int
    entities_by_type: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class MaskingResult:
    """Result of a masking operation.
    
    Immutable result containing:
    - The masked text
    - List of detected entities
    - Statistics about the operation
    
    Attributes:
        masked_text: Text with PII replaced by masks
        entities: List of detected EntityInfo
        stats: MaskingStats with counts
    """
    masked_text: str
    entities: tuple[EntityInfo, ...] = field(default_factory=tuple)
    stats: MaskingStats = field(default_factory=lambda: MaskingStats(total_entities=0))

    @classmethod
    def from_anonymizer_result(
        cls,
        anonymized_text: str,
        original_text: str,
        analyzer_results: list
    ) -> "MaskingResult":
        """Create MaskingResult from Presidio analyzer results.
        
        Args:
            anonymized_text: Text after anonymization
            original_text: Original text before masking
            analyzer_results: List of RecognizerResult from analyzer
            
        Returns:
            MaskingResult instance
        """
        entities = []
        entities_by_type: dict[str, int] = {}
        
        for result in analyzer_results:
            entity_text = original_text[result.start:result.end]
            entities.append(EntityInfo(
                entity_type=result.entity_type,
                text=entity_text,
                score=result.score,
                start=result.start,
                end=result.end,
            ))
            entities_by_type[result.entity_type] = entities_by_type.get(result.entity_type, 0) + 1
        
        stats = MaskingStats(
            total_entities=len(entities),
            entities_by_type=entities_by_type,
        )
        
        return cls(
            masked_text=anonymized_text,
            entities=tuple(entities),
            stats=stats,
        )

    def to_entities_info(self) -> list[dict[str, Any]] | None:
        """Convert entities to list of dicts for backward compatibility.
        
        Returns:
            List of entity dicts if entities exist, None otherwise
        """
        if not self.entities:
            return None
        return [e.to_dict() for e in self.entities]
