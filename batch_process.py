"""Batch process all PDF and Word files in the current directory."""

import os
import sys
from pathlib import Path

from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from analyzer_factory import create_analyzer
from document_extractors import extract_text


def mask_pii_in_text(text: str, language: str = "ja", use_ginza: bool = False) -> str:
    """
    Mask PII in text using language-specific analyzer.
    
    Args:
        text: Text to analyze and mask
        language: Language code ("en" or "ja")
        use_ginza: Whether to use GiNZA for NER
        
    Returns:
        Masked text
    """
    # Create language-specific analyzer
    analyzer = create_analyzer(language, use_ginza=use_ginza)
    anonymizer = AnonymizerEngine()

    # Analyze text for PII
    results = analyzer.analyze(
        text=text,
        language=language,
        entities=None,
    )

    # Prepare operators for masking
    operators = {
        "DEFAULT": OperatorConfig("replace", {"new_value": "****"})
    }

    # Anonymize the text
    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators=operators,
    )
    
    return anonymized.text


def process_directory(directory: str = ".", language: str = "ja", use_ginza: bool = False):
    """
    Process all PDF and Word files in the specified directory.
    
    Args:
        directory: Directory to process (default: current directory)
        language: Language code for analysis
        use_ginza: Whether to use GiNZA for NER
    """
    # Get all PDF and Word files
    path = Path(directory)
    pdf_files = list(path.glob("*.pdf"))
    docx_files = list(path.glob("*.docx"))
    
    all_files = pdf_files + docx_files
    
    if not all_files:
        print("No PDF or Word files found in the current directory.")
        return
    
    print(f"Found {len(all_files)} file(s) to process:")
    for f in all_files:
        print(f"  - {f.name}")
    print()
    
    # Process each file
    success_count = 0
    error_count = 0
    
    for file_path in all_files:
        try:
            print(f"Processing: {file_path.name}...", end=" ")
            
            # Extract text
            text = extract_text(str(file_path))
            
            if not text.strip():
                print("⚠ No text extracted, skipping")
                error_count += 1
                continue
            
            # Mask PII
            masked_text = mask_pii_in_text(text, language=language, use_ginza=use_ginza)
            
            # Create output filename: original_name.txt
            output_filename = file_path.stem + ".txt"
            output_path = path / output_filename
            
            # Write masked text
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(masked_text)
            
            print(f"✓ Saved to {output_filename}")
            success_count += 1
            
        except Exception as e:
            print(f"✗ Error: {e}")
            error_count += 1
    
    # Summary
    print()
    print("=" * 60)
    print(f"Processing complete!")
    print(f"  Success: {success_count} file(s)")
    print(f"  Errors:  {error_count} file(s)")
    print("=" * 60)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Batch process all PDF and Word files in a directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all files in current directory (Japanese, pattern-only)
  python batch_process.py
  
  # Process with GiNZA (requires Python 3.11/3.12)
  python batch_process.py --use-ginza
  
  # Process English documents
  python batch_process.py --lang en
  
  # Process files in specific directory
  python batch_process.py --dir /path/to/resumes
        """
    )
    
    parser.add_argument(
        "--dir",
        default=".",
        help="Directory to process (default: current directory)"
    )
    parser.add_argument(
        "--lang",
        default="ja",
        choices=["en", "ja"],
        help="Language code (default: ja)"
    )
    parser.add_argument(
        "--use-ginza",
        action="store_true",
        help="Use GiNZA for person/address detection (requires Python 3.11/3.12)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PDF/Word Batch PII Masking")
    print("=" * 60)
    print(f"Directory: {os.path.abspath(args.dir)}")
    print(f"Language:  {args.lang}")
    print(f"GiNZA:     {'Enabled' if args.use_ginza else 'Disabled (pattern-only)'}")
    print("=" * 60)
    print()
    
    try:
        process_directory(
            directory=args.dir,
            language=args.lang,
            use_ginza=args.use_ginza
        )
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
