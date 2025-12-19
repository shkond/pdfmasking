"""File processing utilities for batch PII masking.

Handles processing of individual files and batch operations.

This module provides backward-compatible functions that delegate
to the new MaskingService architecture.

Layer: Adapter (between CLI and Application layer)
"""

import sys
from datetime import datetime
from pathlib import Path

from presidio_anonymizer import AnonymizerEngine

from config import load_config
from core.masker import Masker, build_operators
from core.protocols import LoggerProtocol
from file_io.extractors import TextExtractor, extract_text
from masking_logging import MaskingLogger


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
    
    This is a backward-compatible function that uses the new
    Masker class internally.
    
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
        # Setup logger for this file
        logger = MaskingLogger()
        logger.setup_file_handler(log_path)

        # 1) Extract text from document
        print(f"Extracting text from {input_path.name}...", file=sys.stderr)
        extractor = TextExtractor()
        text = extractor.extract(str(input_path))

        if not text.strip():
            print(f"Warning: No text extracted from {input_path.name}.", file=sys.stderr)
            return

        # 2) Mask PII
        if use_preprocessor:
            # New structure-aware pipeline
            masked, entities_info = _process_with_preprocessor(
                text, logger, use_ner, verbose
            )
        else:
            # Use new Masker class with DI
            print(f"Analyzing and masking PII (language: {language})...", file=sys.stderr)
            config = load_config()
            masker = Masker(
                anonymizer=AnonymizerEngine(),
                logger=logger,
                config=config
            )
            result = masker.mask(text, language=language, log_results=True)
            masked = result.masked_text
            entities_info = result.to_entities_info() if verbose else None

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


def _process_with_preprocessor(
    text: str,
    logger: LoggerProtocol,
    use_ner: bool,
    verbose: bool
) -> tuple[str, list[dict] | None]:
    """Process text using structure-aware TextPreprocessor pipeline.
    
    Args:
        text: Text to analyze and mask
        logger: Logger instance (supports LoggerProtocol)
        use_ner: Enable NER engines (GiNZA/Transformer)
        verbose: Return detected entities info
        
    Returns:
        Tuple of (masked_text, entities_info)
    """
    from core.processors.text import TextPreprocessor
    
    print("Analyzing PII with structure-aware preprocessor...", file=sys.stderr)
    
    config = load_config()
    preprocessor = TextPreprocessor(config, use_ner=use_ner)
    
    # Process text through pipeline
    segments, results = preprocessor.process(text)
    
    # Convert to Presidio RecognizerResult format
    recognizer_results = preprocessor.get_recognizer_results(results)
    
    # Log masked entities
    if recognizer_results:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.log(f"\n{'='*60}")
        logger.log(f"Masking Log (Preprocessor) - {timestamp}")
        logger.log(f"{'='*60}")
        for result in recognizer_results:
            entity_text = text[result.start:result.end]
            logger.log(
                f"[{result.entity_type}] \"{entity_text}\" "
                f"(score: {result.score:.2f}, pos: {result.start}-{result.end})"
            )
        logger.log(f"Total: {len(recognizer_results)} entities masked")
    
    # Build operators and anonymize
    operators = build_operators(config)
    anonymizer = AnonymizerEngine()
    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=recognizer_results,
        operators=operators
    )
    
    # Prepare verbose output
    entities_info = None
    if verbose and recognizer_results:
        entities_info = []
        for result in recognizer_results:
            entity_text = text[result.start:result.end]
            entities_info.append({
                "type": result.entity_type,
                "text": entity_text,
                "score": result.score,
                "start": result.start,
                "end": result.end,
            })
    
    return anonymized.text, entities_info

