import argparse
import sys

from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from analyzer_factory import create_analyzer
from document_extractors import extract_text


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

    # Analyze text for PII
    results = analyzer.analyze(
        text=text,
        language=language,
        entities=None,  # None = all entity types
    )

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


def main():
    parser = argparse.ArgumentParser(
        description="Mask PII in documents (PDF/Word) and output masked text.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mask English PDF
  python main.py resume.pdf -o output.txt
  
  # Mask Japanese resume (履歴書)
  python main.py resume.pdf --lang ja -o output.txt
  
  # Mask Word document with verbose output
  python main.py resume.docx --lang ja --verbose
  
Supported formats: .pdf, .docx
Supported languages: en (English), ja (Japanese)
        """
    )
    parser.add_argument(
        "input_file",
        help="Path to input document (PDF or Word)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output text file path (default: stdout)"
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

    try:
        # 1) Extract text from document
        print(f"Extracting text from {args.input_file}...", file=sys.stderr)
        text = extract_text(args.input_file)
        
        if not text.strip():
            print("Warning: No text extracted from document.", file=sys.stderr)
            return

        # 2) Mask PII
        print(f"Analyzing and masking PII (language: {args.lang})...", file=sys.stderr)
        masked, entities_info = mask_pii_in_text(text, language=args.lang, verbose=args.verbose)

        # 3) Show detected entities if verbose
        if args.verbose and entities_info:
            print("\n=== Detected PII Entities ===", file=sys.stderr)
            for i, entity in enumerate(entities_info, 1):
                print(f"{i}. {entity['type']}: '{entity['text']}' (score: {entity['score']:.2f})", file=sys.stderr)
            print(f"\nTotal: {len(entities_info)} entities detected\n", file=sys.stderr)
        elif args.verbose:
            print("\nNo PII entities detected.\n", file=sys.stderr)

        # 4) Output masked text
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(masked)
            print(f"Masked text saved to {args.output}", file=sys.stderr)
        else:
            print("\n=== Masked Text ===", file=sys.stderr)
            print(masked)
            
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
