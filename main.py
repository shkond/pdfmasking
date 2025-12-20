#!/usr/bin/env python3
"""PII Masking CLI - Mask personal information in PDF/Word documents.

This is the main entry point for the PII masking application.
Supports both single file processing and batch mode.

Target PII types (8):
- 名前 (Name)
- メールアドレス (Email)
- 郵便番号 (Zip code)
- 電話番号 (Phone)
- 生年月日 (Birth date)
- 住所 (Address)
- 性別 (Gender)
- 年齢 (Age)
"""

import argparse
import sys
from pathlib import Path

from file_io.file_processor import process_file
from recognizers import create_default_registry


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
        default="auto",
        choices=["en", "ja", "auto"],
        help="Language code: 'en' (English), 'ja' (Japanese), or 'auto' (both) (default: auto)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detected PII entities before masking"
    )
    parser.add_argument(
        "--use-preprocessor",
        action="store_true",
        help="Use structure-aware preprocessing pipeline (experimental)"
    )
    parser.add_argument(
        "--use-ner",
        action="store_true",
        help="Enable NER engines (GiNZA/Transformer) with preprocessor"
    )
    parser.add_argument(
        "--show-recognizers",
        action="store_true",
        help="Show registered recognizers and exit"
    )
    args = parser.parse_args()

    # Handle --show-recognizers
    if args.show_recognizers:
        registry = create_default_registry(use_ginza=True)
        print(registry.summary())
        sys.exit(0)

    if args.input_file:
        # Single file mode
        input_path = Path(args.input_file)
        if not input_path.exists():
            print(f"Error: File {input_path} not found.", file=sys.stderr)
            sys.exit(1)

        # Create output directory
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        stem = input_path.stem
        # Use specified output path or default to output/<stem>.txt
        output_path = Path(args.output) if args.output else output_dir / f"{stem}.txt"
        log_path = output_dir / f"{stem}_log.txt"

        process_file(
            input_path, output_path, log_path, args.lang, args.verbose,
            use_preprocessor=args.use_preprocessor, use_ner=args.use_ner
        )

    else:
        # Batch mode
        print("Running in Batch Mode...", file=sys.stderr)
        root_dir = Path(".")
        output_dir = root_dir / "output"
        output_dir.mkdir(exist_ok=True)

        # Extensions to look for (only PDF and DOCX)
        extensions = ["*.pdf", "*.docx"]
        files_to_process = []

        for ext in extensions:
            files_to_process.extend(root_dir.glob(ext))

        # Filter out files in output directory and __pycache__ etc
        files_to_process = [
            f for f in files_to_process
            if "output" not in f.parts
            and ".git" not in f.parts
            and "__pycache__" not in f.parts
            and "venv" not in f.parts
            and f.name != "requirements.txt"
            and f.name != "masking_log.txt"
        ]

        if not files_to_process:
            print("No compatible files found to process.", file=sys.stderr)
            return

        print(f"Found {len(files_to_process)} files.", file=sys.stderr)

        for input_path in files_to_process:
            # Determine output paths
            stem = input_path.stem
            processed_output_path = output_dir / f"{stem}.txt"
            processed_log_path = output_dir / f"{stem}_log.txt"

            process_file(
                input_path, processed_output_path, processed_log_path,
                args.lang, args.verbose,
                use_preprocessor=args.use_preprocessor, use_ner=args.use_ner
            )

        print(f"\nBatch processing complete. Results in '{output_dir.absolute()}'", file=sys.stderr)


if __name__ == "__main__":
    main()
