import argparse
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from analyzer_factory import create_analyzer, create_multilingual_analyzer
from document_extractors import extract_text

# Configure logging for masked entities
masking_logger = logging.getLogger("masking")
masking_logger.setLevel(logging.INFO)


def preprocess_text(text: str) -> str:
    """
    Preprocess text extracted from PDF to normalize whitespace and formatting.
    
    This helps improve context detection by:
    - Collapsing multiple newlines into single spaces
    - Normalizing whitespace
    - Removing extra spaces while preserving structure
    
    Args:
        text: Raw text extracted from PDF
        
    Returns:
        Normalized text with consistent formatting
    """
    # Replace multiple consecutive newlines with a single space
    text = re.sub(r'\n{2,}', ' ', text)
    # Replace single newlines with space
    text = re.sub(r'\n', ' ', text)
    # Collapse multiple spaces into single space
    text = re.sub(r' {2,}', ' ', text)
    # Normalize full-width spaces to regular spaces
    text = text.replace('\u3000', ' ')
    return text.strip()


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



def deduplicate_results(results, text):
    """
    Remove duplicate/overlapping entity detections.
    Keeps only the highest-scoring result for overlapping spans.
    
    Args:
        results: List of RecognizerResult objects
        text: The original text (for extracting entity text)
    
    Returns:
        List of deduplicated RecognizerResult objects
    """
    if not results:
        return results
    
    # Sort by score (descending), then by start position
    sorted_results = sorted(results, key=lambda x: (-x.score, x.start))
    
    # Keep track of which positions have been covered
    covered_positions = set()
    deduplicated = []
    
    for result in sorted_results:
        # Check if this result overlaps with already covered positions
        result_positions = set(range(result.start, result.end))
        if result_positions & covered_positions:
            # This result overlaps with a higher-scoring result, skip it
            continue
        
        # Add this result and mark its positions as covered
        deduplicated.append(result)
        covered_positions.update(result_positions)
    
    # Sort back by position for consistent ordering
    deduplicated.sort(key=lambda x: x.start)
    return deduplicated


def _merge_results(results_en, results_ja):
    """
    Merge results from English and Japanese analysis.
    
    Combines results from both languages and removes duplicates
    by keeping the higher-scoring result for overlapping spans.
    
    Args:
        results_en: RecognizerResult list from English analysis
        results_ja: RecognizerResult list from Japanese analysis
        
    Returns:
        Combined list of RecognizerResult objects
    """
    all_results = list(results_en) + list(results_ja)
    
    if not all_results:
        return all_results
    
    # Sort by score descending, then by span length (prefer longer matches)
    sorted_results = sorted(all_results, key=lambda x: (-x.score, -(x.end - x.start), x.start))
    
    # Remove overlapping results, keeping higher-scoring ones
    covered_positions = set()
    merged = []
    
    for result in sorted_results:
        result_positions = set(range(result.start, result.end))
        # Check for significant overlap (more than 50% of the smaller span)
        overlap = result_positions & covered_positions
        if len(overlap) > len(result_positions) * 0.5:
            # Significant overlap with a higher-scoring result, skip
            continue
        
        merged.append(result)
        covered_positions.update(result_positions)
    
    # Sort by position for consistent ordering
    merged.sort(key=lambda x: x.start)
    return merged


def mask_pii_in_text(text: str, language: str = "auto", verbose: bool = False, preprocess: bool = False) -> tuple:
    """
    Mask PII in text using language-specific or multilingual analyzer.
    
    When Transformer NER is enabled with dual-detection mode:
    - Runs both pattern-based and Transformer-based analysis
    - Only masks entities detected by BOTH recognizers
    
    Args:
        text: Text to analyze and mask
        language: Language code ("en", "ja", or "auto" for multilingual)
        verbose: If True, return detected entities info
        preprocess: If True, normalize text before analysis (recommended for PDF text)
        
    Returns:
        Tuple of (masked_text, detected_entities_info)
    """
    from analyzer_factory import load_config, get_transformer_config, get_entities_to_mask, get_entity_categories
    
    # Load configuration
    config = load_config()
    transformer_cfg = get_transformer_config(config)
    entities_to_mask = get_entities_to_mask(config)
    entity_categories = get_entity_categories(config)
    
    use_transformer = transformer_cfg.get("enabled", False)
    require_dual_detection = transformer_cfg.get("require_dual_detection", True)
    
    # Preprocess text if requested (useful for PDF-extracted text)
    if preprocess:
        text = preprocess_text(text)
    
    anonymizer = AnonymizerEngine()
    dual_scores = {}  # Will hold dual scores if available

    if language == "auto":
        # Multilingual mode: analyze in both English and Japanese
        if use_transformer and require_dual_detection:
            # Dual detection mode: run pattern-only and Transformer-only analyzers
            results, dual_scores = _dual_detection_analyze(
                text, transformer_cfg, entities_to_mask, entity_categories, language="auto"
            )
        else:
            # Standard mode
            analyzer = create_multilingual_analyzer(
                use_ginza=True, 
                use_transformer=use_transformer,
                transformer_config=transformer_cfg if use_transformer else None
            )
            
            # Analyze in both languages
            results_en = analyzer.analyze(
                text=text,
                language="en",
                entities=entities_to_mask,
            )
            results_ja = analyzer.analyze(
                text=text,
                language="ja",
                entities=entities_to_mask,
            )
            
            # Merge results from both languages
            results = _merge_results(results_en, results_ja)
    else:
        # Single language mode
        if use_transformer and require_dual_detection:
            # Dual detection mode
            results, dual_scores = _dual_detection_analyze(
                text, transformer_cfg, entities_to_mask, entity_categories, language=language
            )
        else:
            # Standard mode
            analyzer = create_analyzer(
                language=language,
                use_transformer=use_transformer,
                transformer_config=transformer_cfg if use_transformer else None
            )
            results = analyzer.analyze(
                text=text,
                language=language,
                entities=entities_to_mask,
            )
    
    # Deduplicate overlapping results
    results = deduplicate_results(results, text)

    # Log masked entities to file
    if results:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        masking_logger.info(f"\n{'='*60}")
        masking_logger.info(f"Masking Log - {timestamp}")
        if dual_scores:
            masking_logger.info("Mode: Dual Detection (Pattern + Transformer)")
        masking_logger.info(f"{'='*60}")
        for result in results:
            entity_text = text[result.start:result.end]
            # Check if we have dual scores for this entity
            if (result.start, result.end) in dual_scores:
                scores = dual_scores[(result.start, result.end)]
                p_type = scores.get('p_type', result.entity_type)
                t_type = scores.get('t_type', '?')
                masking_logger.info(
                    f"[{result.entity_type}] \"{entity_text}\" "
                    f"(pattern: {p_type}={scores['pattern']:.2f}, transformer: {t_type}={scores['transformer']:.2f}, pos: {result.start}-{result.end})"
                )

            else:
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


def _dual_detection_analyze(
    text: str, 
    transformer_cfg: dict, 
    entities_to_mask: list, 
    entity_categories: dict,
    language: str = "ja"
) -> tuple:
    """
    Dual detection mode: Only keep entities detected by BOTH pattern AND Transformer.
    
    This reduces false positives by requiring consensus between rule-based
    and ML-based recognizers.
    
    Args:
        text: Text to analyze
        transformer_cfg: Transformer configuration dict
        entities_to_mask: List of entity types to mask (from config)
        entity_categories: Dict of entity categories for type matching (from config)
        language: Language code ("en", "ja", or "auto")
        
    Returns:
        Tuple of (confirmed_results, dual_scores_dict)
        - confirmed_results: List of RecognizerResult objects detected by both
        - dual_scores_dict: Dict mapping (start, end) -> {"pattern": score, "transformer": score}
    """
    from recognizer_registry import create_default_registry
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    
    # Create pattern-only analyzer
    pattern_registry = create_default_registry(use_ginza=True, use_transformer=False)
    
    # Create Transformer-only analyzer  
    transformer_registry = create_default_registry(
        use_ginza=False, 
        use_transformer=True,
        transformer_config=transformer_cfg
    )
    
    # Setup NLP engine
    try:
        nlp_configuration = {
            "nlp_engine_name": "spacy",
            "models": [
                {"lang_code": "en", "model_name": "en_core_web_lg"},
                {"lang_code": "ja", "model_name": "ja_ginza"},
            ],
        }
        nlp_engine = NlpEngineProvider(nlp_configuration=nlp_configuration).create_engine()
        # Pattern analyzer uses default + custom recognizers
        pattern_analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en", "ja"])
        # Transformer analyzer: use empty registry to exclude all default recognizers (including SpacyRecognizer)
        from presidio_analyzer import RecognizerRegistry as PresidioRegistry
        empty_registry = PresidioRegistry()
        empty_registry.recognizers = []  # Clear all default recognizers
        empty_registry.supported_languages = ["en", "ja"]  # Match analyzer languages
        transformer_analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine, 
            supported_languages=["en", "ja"],
            registry=empty_registry
        )
    except Exception as e:
        print(f"Error creating analyzers: {e}")
        pattern_analyzer = AnalyzerEngine(supported_languages=["en", "ja"])
        from presidio_analyzer import RecognizerRegistry as PresidioRegistry
        empty_registry = PresidioRegistry()
        empty_registry.recognizers = []
        empty_registry.supported_languages = ["en", "ja"]
        transformer_analyzer = AnalyzerEngine(supported_languages=["en", "ja"], registry=empty_registry)

    
    # Apply recognizers
    pattern_registry.apply_to_analyzer(pattern_analyzer)
    
    # Get Transformer recognizers directly (don't use AnalyzerEngine to avoid default recognizers)
    transformer_recognizers = [
        config.recognizer for config in transformer_registry.configs 
        if config.type == "ner_transformer"
    ]

    # Analyze with both
    if language == "auto":
        # Multilingual: analyze in both languages
        pattern_results_en = pattern_analyzer.analyze(text=text, language="en", entities=entities_to_mask)
        pattern_results_ja = pattern_analyzer.analyze(text=text, language="ja", entities=entities_to_mask)
        pattern_results = list(pattern_results_en) + list(pattern_results_ja)
        
        # Call Transformer recognizers directly
        transformer_results = []
        for recognizer in transformer_recognizers:
            try:
                results = recognizer.analyze(text=text, entities=entities_to_mask)
                transformer_results.extend(results)
            except Exception as e:
                pass  # Skip on error
    else:
        pattern_results = list(pattern_analyzer.analyze(text=text, language=language, entities=entities_to_mask))
        
        # Call Transformer recognizers directly for specific language
        transformer_results = []
        for recognizer in transformer_recognizers:
            if recognizer.supported_language == language or language == "auto":
                try:
                    results = recognizer.analyze(text=text, entities=entities_to_mask)
                    transformer_results.extend(results)
                except Exception as e:
                    pass

    
    # Entity types that require dual detection (NER entities that can be detected by both pattern and Transformer)
    # All other entity types are pattern-only and should pass through without dual detection
    dual_detection_entity_types = set()
    for category, types in entity_categories.items():
        dual_detection_entity_types.update(types)
    
    # Find overlapping entities (detected by both) for NER entity types
    # Pattern-only entities pass through directly
    # An entity is considered a dual detection match if:
    # 1. The spans overlap significantly (>50%)
    # 2. The entity_type matches (PERSON must match PERSON, etc.)
    confirmed_results = []
    dual_scores = {}  # Maps (start, end) -> {"pattern": score, "transformer": score, "p_type": type, "t_type": type}
    
    for p_result in pattern_results:
        # Check if this entity type requires dual detection
        if p_result.entity_type not in dual_detection_entity_types:
            # Pattern-only entity (phone, zip, date, etc.) - include directly
            confirmed_results.append(p_result)
            continue
        
        p_span = set(range(p_result.start, p_result.end))
        
        for t_result in transformer_results:
            t_span = set(range(t_result.start, t_result.end))
            
            # Check for significant overlap (>50% of smaller span)
            overlap = p_span & t_span
            min_span_len = min(len(p_span), len(t_span))
            
            if min_span_len > 0 and len(overlap) >= min_span_len * 0.5:
                # CRITICAL: Also check entity_type matches
                # Map similar types (e.g., PERSON must match PERSON, LOCATION matches LOCATION)
                p_type_normalized = _normalize_entity_type(p_result.entity_type, entity_categories)
                t_type_normalized = _normalize_entity_type(t_result.entity_type, entity_categories)
                
                if p_type_normalized == t_type_normalized:
                    # Types match - this is a valid dual detection
                    dual_scores[(p_result.start, p_result.end)] = {
                        "pattern": p_result.score,
                        "transformer": t_result.score,
                        "p_type": p_result.entity_type,
                        "t_type": t_result.entity_type
                    }
                    # Use the pattern result's span (usually more accurate for boundaries)
                    confirmed_results.append(p_result)
                    break  # Only add once per pattern result
    
    return confirmed_results, dual_scores


def _normalize_entity_type(entity_type: str, entity_categories: dict) -> str:
    """
    Normalize entity types for comparison across different recognizers.
    
    Uses entity_categories from config.yaml to determine which entity types
    belong to the same category.
    
    Args:
        entity_type: The entity type to normalize
        entity_categories: Dict mapping category name to list of entity types
        
    Returns:
        Normalized category name (uppercase) or original entity_type if no match
    """
    for category, types in entity_categories.items():
        if entity_type in types:
            return category.upper()
    return entity_type







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
        "--show-recognizers",
        action="store_true",
        help="Show registered recognizers and exit"
    )
    args = parser.parse_args()

    # Handle --show-recognizers
    if args.show_recognizers:
        from recognizer_registry import create_default_registry
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
        
        process_file(input_path, output_path, log_path, args.lang, args.verbose)
        
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
