"""Masking Service - Application layer service for PII masking.

This module provides MaskingService which orchestrates:
- Text extraction from documents
- PII detection and masking
- Logging of results

Layer: Application
Depends on: Domain layer (Masker), Infrastructure layer (Protocols)
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from presidio_anonymizer import AnonymizerEngine

from config import load_config
from core.masker import Masker, build_operators
from core.masking_result import MaskingResult
from core.processors.text import TextPreprocessor
from core.protocols import LoggerProtocol, TextExtractorProtocol


class MaskingService:
    """Application service for file-based PII masking.
    
    Orchestrates the complete masking workflow:
    1. Extract text from document
    2. Detect and mask PII
    3. Log results
    4. Save output
    
    Designed for dependency injection to enable testing without
    file I/O or external dependencies.
    
    Usage:
        from file_io.extractors import TextExtractor
        from masking_logging import MaskingLogger
        
        service = MaskingService(
            extractor=TextExtractor(),
            masker=Masker(),
            logger=MaskingLogger()
        )
        result = service.process_file(
            input_path=Path("document.pdf"),
            output_path=Path("output/document.txt"),
            log_path=Path("output/document_log.txt"),
            language="ja"
        )
    """
    
    def __init__(
        self,
        extractor: TextExtractorProtocol,
        masker: Masker,
        logger: LoggerProtocol
    ):
        """Initialize service with dependencies.
        
        Args:
            extractor: Text extraction implementation
            masker: Masker instance for PII detection/masking
            logger: Logger implementation
        """
        self.extractor = extractor
        self.masker = masker
        self.logger = logger
    
    def process_file(
        self,
        input_path: Path,
        output_path: Path | None = None,
        log_path: Path | None = None,
        language: str = "auto",
        verbose: bool = False
    ) -> MaskingResult | None:
        """Process a single file: extract, mask, log, and save.
        
        Args:
            input_path: Path to input document (PDF or Word)
            output_path: Path to save masked text (None = print to stdout)
            log_path: Path for masking log file (None = no file logging)
            language: Language code ("en", "ja", or "auto")
            verbose: If True, print detected entities
            
        Returns:
            MaskingResult if successful, None if extraction failed
        """
        try:
            # Setup logger for this file
            if log_path:
                self.logger.setup_file_handler(log_path)
            
            # 1) Extract text from document
            print(f"Extracting text from {input_path.name}...", file=sys.stderr)
            text = self.extractor.extract(str(input_path))
            
            if not text.strip():
                print(f"Warning: No text extracted from {input_path.name}.", file=sys.stderr)
                return None
            
            # 2) Mask PII
            print(f"Analyzing and masking PII (language: {language})...", file=sys.stderr)
            result = self.masker.mask(text, language=language, log_results=True)
            
            # 3) Show detected entities if verbose
            if verbose and result.entities:
                print(f"\n[{input_path.name}] Detected PII Entities:", file=sys.stderr)
                for i, entity in enumerate(result.entities, 1):
                    print(
                        f"{i}. {entity.entity_type}: '{entity.text}' (score: {entity.score:.2f})",
                        file=sys.stderr
                    )
                print(f"Total: {len(result.entities)} entities detected", file=sys.stderr)
            
            # 4) Output masked text
            if output_path:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(result.masked_text)
                print(f"Masked text saved to {output_path}", file=sys.stderr)
            else:
                print("\n=== Masked Text ===", file=sys.stderr)
                print(result.masked_text)
            
            return result
            
        except Exception as e:
            print(f"Error processing {input_path.name}: {e}", file=sys.stderr)
            return None
    
    def process_text(
        self,
        text: str,
        language: str = "auto",
        do_preprocess: bool = False
    ) -> MaskingResult:
        """Process text directly without file I/O.
        
        Useful for testing or API integrations.
        
        Args:
            text: Text to analyze and mask
            language: Language code
            do_preprocess: If True, normalize text before analysis
            
        Returns:
            MaskingResult with masked text and entity info
        """
        return self.masker.mask(text, language=language, do_preprocess=do_preprocess)


class MaskingServiceFactory:
    """Factory for creating MaskingService with default dependencies.
    
    Simplifies service creation for CLI and production use.
    """
    
    @staticmethod
    def create(
        config: dict[str, Any] | None = None,
        use_preprocessor: bool = False,
        use_ner: bool = False
    ) -> MaskingService:
        """Create a MaskingService with default production dependencies.
        
        Args:
            config: Configuration dict (default: load from config.yaml)
            use_preprocessor: If True, use structure-aware preprocessing
            use_ner: If True, enable NER with preprocessor
            
        Returns:
            Configured MaskingService
        """
        from file_io.extractors import TextExtractor
        from masking_logging import MaskingLogger
        
        config = config or load_config()
        
        return MaskingService(
            extractor=TextExtractor(),
            masker=Masker(
                anonymizer=AnonymizerEngine(),
                logger=MaskingLogger(),
                config=config
            ),
            logger=MaskingLogger()
        )
