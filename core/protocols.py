"""Protocol definitions for dependency abstraction.

Defines interfaces for external dependencies to enable:
- Dependency Injection
- Easy mocking in tests
- Clear layer boundaries
"""

from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LoggerProtocol(Protocol):
    """Protocol for logging operations.
    
    Implementations:
    - MaskingLogger (production)
    - Mock logger (tests)
    """
    
    def log(self, message: str) -> None:
        """Log a message."""
        ...
    
    def setup_file_handler(self, path: Path) -> None:
        """Set up file handler for logging."""
        ...


@runtime_checkable
class AnonymizerProtocol(Protocol):
    """Protocol for text anonymization.
    
    Wraps presidio_anonymizer.AnonymizerEngine to allow mocking.
    """
    
    def anonymize(
        self,
        *,
        text: str,
        analyzer_results: list,
        operators: dict
    ) -> Any:
        """Anonymize text based on analyzer results.
        
        Args:
            text: Text to anonymize
            analyzer_results: List of RecognizerResult
            operators: Dict of entity_type -> OperatorConfig
            
        Returns:
            AnonymizerResult with .text attribute
        """
        ...


@runtime_checkable
class TextExtractorProtocol(Protocol):
    """Protocol for text extraction from documents.
    
    Implementations:
    - TextExtractor (production - PDF, DOCX)
    - Mock extractor (tests)
    """
    
    def extract(self, file_path: str) -> str:
        """Extract text from a document.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Extracted text as string
            
        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If file format is not supported
        """
        ...


class NullLogger:
    """Null object pattern for logger - does nothing.
    
    Useful for tests where logging is not needed.
    """
    
    def log(self, message: str) -> None:
        """Do nothing."""
        pass
    
    def setup_file_handler(self, path: Path) -> None:
        """Do nothing."""
        pass
