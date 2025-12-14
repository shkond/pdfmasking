import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from analyzer_factory import create_analyzer
from document_extractors import extract_text

# Configure logging for masked entities
masking_logger = logging.getLogger("masking")
masking_logger.setLevel(logging.INFO)


def setup_logger(log_file_path: Path):
    """
    Set up the logger to write to the specified file.
    Removes existing handlers to switch log files dynamically.
    """
    # Remove existing handlers
    if masking_logger.hasHandlers():
        masking_logger.handlers.clear()

    # Add new handler
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    masking_logger.addHandler(file_handler)


# Entities to mask (exclude ORG, LOC, GPE to keep organization names and locations visible)
ENTITIES_TO_MASK = [
    # Standard Presidio entities
    "PERSON",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "DATE_TIME",
    "CREDIT_CARD",
    "IBAN_CODE",
    # Japanese-specific entities
    "PHONE_NUMBER_JP",
    "JP_ZIP_CODE",
    "DATE_OF_BIRTH_JP",
    "JP_PERSON",
]


def mask_pii_in_text(text: str, language: str = "en", verbose: bool = False) -> tuple:
    """
    Mask PII in text using language-specific analyzer.
    
    Args:
        text: Text to analyze and mask
        language: Language code ("en" or "ja")
        verbose: If True, return detected entities info
        
    Returns:
        Tuple of (masked_text, detected_entities_info)
    """
    # Create language-specific analyzer
    analyzer = create_analyzer(language)
    anonymizer = AnonymizerEngine()

    # Analyze text for PII - only specified entities
    results = analyzer.analyze(
        text=text,
        language=language,
        entities=ENTITIES_TO_MASK,  # Limit to specific entities (excludes ORG, LOC, GPE)
    )

    # Log masked entities to file
    if results:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        masking_logger.info(f"\n{'='*60}")
        masking_logger.info(f"Masking Log - {timestamp}")
        masking_logger.info(f"{'='*60}")
        for result in results:
            entity_text = text[result.start:result.end]
            masking_logger.info(
                f"[{result.entity_type}] \"{entity_text}\" "
                f"(score: {result.score:.2f}, pos: {result.start}-{result.end})"
            )
        masking_logger.info(f"Total: {len(results)} entities masked")

    # Prepare operators for masking
    operators = {
        # Mask all entities with "****"
        "DEFAULT": OperatorConfig("replace", {"new_value": "****"})
    }

    # Anonymize the text
    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators=operators,
    )
    
    # Prepare verbose output if requested
    entities_info = None
    if verbose and results:
        entities_info = []
        for result in results:
            entity_text = text[result.start:result.end]
            entities_info.append({
                "type": result.entity_type,
                "text": entity_text,
                "score": result.score,
                "start": result.start,
                "end": result.end,
            })
    
    return anonymized.text, entities_info


def process_file(input_path: Path, output_path: Path, log_path: Path, language: str, verbose: bool):
    """
    Process a single file: extract, mask, log, and save.
    """
    try:
        # Setup logger for this file
        setup_logger(log_path)
        
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


def main():
    parser = argparse.ArgumentParser(
        description="Mask PII in documents (PDF/Word) and output masked text.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mask English PDF (Single file)
  python main.py resume.pdf -o output.txt
  
  # Batch process all files in current directory (recursive)
  # Output will be saved to 'output/' folder
  python main.py

  # Mask Japanese resume
  python main.py resume.pdf --lang ja -o output.txt
        """
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        help="Path to input document (PDF or Word). If omitted, processes all compatible files in current directory recursively."
    )
    parser.add_argument(
        "-o", "--output",
        help="Output text file path (only for single file mode)"
    )
    parser.add_argument(
        "--lang",
        default="en",
        choices=["en", "ja"],
        help="Language code (default: en)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detected PII entities before masking"
    )
    args = parser.parse_args()

    if args.input_file:
        # Single file mode
        input_path = Path(args.input_file)
        if not input_path.exists():
            print(f"Error: File {input_path} not found.", file=sys.stderr)
            sys.exit(1)
        
        output_path = Path(args.output) if args.output else None
        # Use default log file for single mode if not specified (or should we use local?)
        # For backward compatibility/simplicity, let's just use a default log in current dir
        log_path = Path("masking_log.txt")
        
        process_file(input_path, output_path, log_path, args.lang, args.verbose)
        
    else:
        # Batch mode
        print("Running in Batch Mode...", file=sys.stderr)
        root_dir = Path(".")
        output_dir = root_dir / "output"
        output_dir.mkdir(exist_ok=True)
        
        # Extensions to look for
        extensions = ["*.pdf", "*.docx", "*.txt"]
        files_to_process = []
        
        for ext in extensions:
            files_to_process.extend(root_dir.glob(ext))
            
        # Filter out files in output directory and __pycache__ etc if needed
        # (rglob might catch created output files if we are not careful? No, we haven't created them yet within this run, but from previous runs yes)
        # Better to filter out anything inside the output directory we just created/referenced
        files_to_process = [
            f for f in files_to_process 
            if "output" not in f.parts 
            and ".git" not in f.parts
            and "__pycache__" not in f.parts
            and "venv" not in f.parts
            and f.name != "requirements.txt" # exclude text files that are not docs
            and f.name != "masking_log.txt"
        ]
        
        if not files_to_process:
            print("No compatible files found to process.", file=sys.stderr)
            return

        print(f"Found {len(files_to_process)} files.", file=sys.stderr)
        
        for input_path in files_to_process:
            # Determine output paths
            stem = input_path.stem
            # We treat all inputs as source, put results in output/
            # Name collision handling: if multiple files have same stem? 
            # For strict requirement "output folder created and masking result saved there", flat structure is simplest but risks collisions.
            # But "root directory all files" implies we might want to preserve structure or just handle flatness.
            # User said "folders all masking process output folder create", simplest is flat for now.
             
            processed_output_path = output_dir / f"{stem}.txt"
            processed_log_path = output_dir / f"{stem}_log.txt"
            
            process_file(input_path, processed_output_path, processed_log_path, args.lang, args.verbose)
            
        print(f"\nBatch processing complete. Results in '{output_dir.absolute()}'", file=sys.stderr)


if __name__ == "__main__":
    main()
