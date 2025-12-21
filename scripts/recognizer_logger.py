#!/usr/bin/env python3
"""
Recognizer Entity Logger
Processes all PDF and Word files in the root directory,
extracts text using file_io utilities, and logs detected entities
per individual recognizer to scripts/logs/<filename>.log.
"""

import os
import sys
import re
from pathlib import Path

# Add current directory to path so we can import local modules
sys.path.append(str(Path(__file__).parent.parent))

from core.analyzer import create_multilingual_analyzer
from config import load_config
from file_io.extractors import extract_text

def process_documents():
    # Setup directories
    root_dir = Path(".")
    log_dir = root_dir / "scripts" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config = load_config()

    # Check if transformers are actually available
    try:
        from recognizers.transformer_ner import TORCH_AVAILABLE
    except ImportError:
        TORCH_AVAILABLE = False

    print("Initializing Multilingual Analyzer...")
    # Use the project's centralized analyzer factory
    try:
        analyzer = create_multilingual_analyzer(
            use_ginza=True,
            use_transformer=TORCH_AVAILABLE,
            verbose=True
        )
    except Exception as e:
        print(f"Failed to initialize analyzer: {e}")
        return

    # Find PDF and Word files in root
    extensions = [".pdf", ".docx"]
    doc_files = []
    for ext in extensions:
        doc_files.extend(list(root_dir.glob(f"*{ext}")))
    
    # Also include .txt files as per previous requirement if needed, 
    # but user specifically asked for pdf and word.
    # I will stick to PDF and Word as requested.
    
    if not doc_files:
        print(f"No {', '.join(extensions)} files found in root directory.")
        return

    print(f"Found {len(doc_files)} files to process.")

    for doc_file in doc_files:
        print(f"Processing {doc_file.name}...")
        
        try:
            # Extract text using file_io utility
            text = extract_text(str(doc_file))
            
            if not text or not text.strip():
                print(f"  Warning: No text extracted from {doc_file.name}")
                continue

            # Detect language or assume Japanese for these resumes
            has_japanese = bool(re.search(r'[ぁ-んァ-ン一-龯]', text))
            lang = "ja" if has_japanese else "en"
            
            # Get NLP artifacts with the correctly configured engine
            try:
                nlp_artifacts = analyzer.nlp_engine.process_text(text, language=lang)
            except Exception as e:
                print(f"  Warning: NLP engine failed for {lang}: {e}")
                nlp_artifacts = None

            all_results = []

            # Iterate through all recognizers to get ALL detections
            recognizers = analyzer.get_recognizers()
            
            for recognizer in recognizers:
                # Check if recognizer supports the language
                supported_langs = recognizer.supported_language
                if isinstance(supported_langs, str):
                    supported_langs = [supported_langs]
                
                if lang not in supported_langs:
                    continue
                
                try:
                    # Run analysis
                    results = recognizer.analyze(text, recognizer.supported_entities, nlp_artifacts)
                    
                    if results:
                        for res in results:
                            all_results.append({
                                "recognizer": recognizer.name,
                                "entity_type": res.entity_type,
                                "start": res.start,
                                "end": res.end,
                                "score": res.score,
                                "text": text[res.start:res.end].replace('\n', ' ')
                            })
                except Exception as e:
                    print(f"  Warning: Recognizer {recognizer.name} failed: {e}")

            # Sort results by start position, then by score (desc), then by recognizer
            all_results.sort(key=lambda x: (x["start"], -x["score"], x["recognizer"]))

            # Write to log
            log_file = log_dir / f"{doc_file.stem}.log"
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(f"Source File: {doc_file.name}\n")
                f.write(f"Detected Language: {lang}\n")
                f.write("-" * 80 + "\n")
                f.write(f"{'RECOGNIZER':<35} | {'ENTITY TYPE':<20} | {'SCORE':<5} | {'TEXT'}\n")
                f.write("-" * 80 + "\n")
                
                if not all_results:
                    f.write("No entities detected.\n")
                else:
                    for res in all_results:
                        f.write(f"{res['recognizer']:<35} | {res['entity_type']:<20} | {res['score']:<5.2f} | {res['text']}\n")
            
            print(f"  Log saved to {log_file}")

        except Exception as e:
            print(f"Error processing {doc_file.name}: {e}")

if __name__ == "__main__":
    process_documents()
