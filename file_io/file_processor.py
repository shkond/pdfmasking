"""File processing utilities for batch PII masking.

Handles processing of individual files using the MaskingService layer.
Delegates all masking logic to the MaskingService.

Layer: Adapter (between CLI and Application layer)
"""

import sys
from pathlib import Path

from core.masking_service import MaskingServiceFactory


def process_file(
    input_path: Path,
    output_path: Path,
    log_path: Path,
    language: str,
    verbose: bool,
    use_preprocessor: bool = False,
    use_ner: bool = False
) -> None:
    """
    Process a single file: extract, mask, log, and save.
    
    Delegates to MaskingService for all masking operations.
    
    Args:
        input_path: Path to input document (PDF or Word)
        output_path: Path to save masked text
        log_path: Path for masking log file
        language: Language code ("en", "ja", or "auto")
        verbose: If True, print detected entities
        use_preprocessor: If True, use structure-aware TextPreprocessor pipeline
        use_ner: If True (with use_preprocessor), enable NER engines
    """
    try:
        # Create service with appropriate configuration
        service = MaskingServiceFactory.create(
            use_preprocessor=use_preprocessor,
            use_ner=use_ner
        )
        
        # Delegate to MaskingService
        service.process_file(
            input_path=input_path,
            output_path=output_path,
            log_path=log_path,
            language=language,
            verbose=verbose
        )
        
    except Exception as e:
        print(f"Error processing {input_path.name}: {e}", file=sys.stderr)
