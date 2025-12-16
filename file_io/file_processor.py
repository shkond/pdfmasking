"""File processing utilities for batch PII masking.

Handles processing of individual files and batch operations.
"""

import sys
from pathlib import Path

from core.masker import mask_pii_in_text
from .extractors import extract_text
from masking_logging import MaskingLogger


def process_file(
    input_path: Path,
    output_path: Path,
    log_path: Path,
    language: str,
    verbose: bool
) -> None:
    """
    Process a single file: extract, mask, log, and save.
    
    Args:
        input_path: Path to input document (PDF or Word)
        output_path: Path to save masked text
        log_path: Path for masking log file
        language: Language code ("en", "ja", or "auto")
        verbose: If True, print detected entities
    """
    try:
        # Setup logger for this file
        logger = MaskingLogger()
        logger.setup_file_handler(log_path)

        # 1) Extract text from document
        print(f"Extracting text from {input_path.name}...", file=sys.stderr)
        text = extract_text(str(input_path))

        if not text.strip():
            print(f"Warning: No text extracted from {input_path.name}.", file=sys.stderr)
            return

        # 2) Mask PII
        print(f"Analyzing and masking PII (language: {language})...", file=sys.stderr)
        masked, entities_info = mask_pii_in_text(text, language=language, verbose=verbose)

        # 3) Show detected entities if verbose
        if verbose and entities_info:
            print(f"\n[{input_path.name}] Detected PII Entities:", file=sys.stderr)
            for i, entity in enumerate(entities_info, 1):
                print(f"{i}. {entity['type']}: '{entity['text']}' (score: {entity['score']:.2f})", file=sys.stderr)
            print(f"Total: {len(entities_info)} entities detected", file=sys.stderr)

        # 4) Output masked text
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(masked)
            print(f"Masked text saved to {output_path}", file=sys.stderr)
        else:
            print("\n=== Masked Text ===", file=sys.stderr)
            print(masked)

    except Exception as e:
        print(f"Error processing {input_path.name}: {e}", file=sys.stderr)
