"""Debug script to investigate dual detection for English Resume Sample.pdf"""

import sys
sys.path.insert(0, '/workspaces/pdfmasking')

from document_extractors import extract_text
from recognizer_registry import create_default_registry
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider


def debug_dual_detection():
    """Debug the dual detection logic for English Resume Sample.pdf"""
    
    # Extract text from PDF
    pdf_path = "/workspaces/pdfmasking/English Resume Sample.pdf"
    print("=" * 80)
    print("Extracting text from:", pdf_path)
    print("=" * 80)
    
    text = extract_text(pdf_path)
    print(f"Text length: {len(text)} characters")
    print()
    
    # Find Git and Docker occurrences
    git_positions = []
    docker_positions = []
    
    import re
    for match in re.finditer(r'\bGit\b', text):
        git_positions.append((match.start(), match.end()))
    for match in re.finditer(r'\bDocker\b', text):
        docker_positions.append((match.start(), match.end()))
    
    print(f"Found Git at positions: {git_positions}")
    print(f"Found Docker at positions: {docker_positions}")
    print()
    
    # Create pattern-only registry
    print("=" * 80)
    print("PATTERN-ONLY ANALYSIS")
    print("=" * 80)
    
    pattern_registry = create_default_registry(use_ginza=True, use_transformer=False)
    print(f"Pattern registry has {len(pattern_registry.configs)} recognizers")
    
    # Setup NLP engine
    nlp_configuration = {
        "nlp_engine_name": "spacy",
        "models": [
            {"lang_code": "en", "model_name": "en_core_web_lg"},
            {"lang_code": "ja", "model_name": "ja_ginza"},
        ],
    }
    nlp_engine = NlpEngineProvider(nlp_configuration=nlp_configuration).create_engine()
    
    pattern_analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en", "ja"])
    pattern_registry.apply_to_analyzer(pattern_analyzer)
    
    # Analyze
    pattern_results_en = pattern_analyzer.analyze(
        text=text, 
        language="en", 
        entities=["PERSON", "LOCATION", "ORGANIZATION"]
    )
    
    print(f"\nPattern PERSON/LOC/ORG results ({len(pattern_results_en)} total):")
    for result in sorted(pattern_results_en, key=lambda x: x.start):
        entity_text = text[result.start:result.end]
        print(f"  [{result.entity_type}] \"{entity_text}\" score={result.score:.2f} pos={result.start}-{result.end}")
    
    # Check Git and Docker specifically
    print("\n--- Checking Git/Docker in pattern results ---")
    for result in pattern_results_en:
        entity_text = text[result.start:result.end]
        if "Git" in entity_text or "Docker" in entity_text:
            print(f"  FOUND: [{result.entity_type}] \"{entity_text}\" score={result.score:.2f}")
    
    # Create transformer-only registry
    print("\n" + "=" * 80)
    print("TRANSFORMER-ONLY ANALYSIS")
    print("=" * 80)
    
    transformer_cfg = {
        "device": "cpu",
        "min_confidence": 0.8,
        "english_model": "dslim/bert-base-NER",
        "japanese_model": "knosing/japanese_ner_model"
    }
    
    transformer_registry = create_default_registry(
        use_ginza=False, 
        use_transformer=True,
        transformer_config=transformer_cfg
    )
    print(f"Transformer registry has {len(transformer_registry.configs)} recognizers")
    print(f"Transformer recognizers: {[c.type for c in transformer_registry.configs]}")
    
    transformer_analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en", "ja"])
    transformer_registry.apply_to_analyzer(transformer_analyzer)
    
    # Analyze
    transformer_results_en = transformer_analyzer.analyze(
        text=text, 
        language="en", 
        entities=["PERSON", "LOCATION", "ORGANIZATION"]
    )
    
    print(f"\nTransformer PERSON/LOC/ORG results ({len(transformer_results_en)} total):")
    for result in sorted(transformer_results_en, key=lambda x: x.start):
        entity_text = text[result.start:result.end]
        print(f"  [{result.entity_type}] \"{entity_text}\" score={result.score:.2f} pos={result.start}-{result.end}")
    
    # Check Git and Docker specifically
    print("\n--- Checking Git/Docker in transformer results ---")
    found_git_docker = False
    for result in transformer_results_en:
        entity_text = text[result.start:result.end]
        if "Git" in entity_text or "Docker" in entity_text:
            print(f"  FOUND: [{result.entity_type}] \"{entity_text}\" score={result.score:.2f}")
            found_git_docker = True
    
    if not found_git_docker:
        print("  NOT FOUND: Git/Docker not detected by Transformer as PERSON/LOC/ORG")
    
    # Show overlap analysis
    print("\n" + "=" * 80)
    print("OVERLAP ANALYSIS (Dual Detection)")
    print("=" * 80)
    
    for p_result in pattern_results_en:
        p_span = set(range(p_result.start, p_result.end))
        p_text = text[p_result.start:p_result.end]
        
        for t_result in transformer_results_en:
            t_span = set(range(t_result.start, t_result.end))
            t_text = text[t_result.start:t_result.end]
            
            overlap = p_span & t_span
            min_span_len = min(len(p_span), len(t_span))
            
            if min_span_len > 0 and len(overlap) >= min_span_len * 0.5:
                print(f"  MATCH: Pattern \"{p_text}\" ({p_result.entity_type}: {p_result.score:.2f}) <-> Transformer \"{t_text}\" ({t_result.entity_type}: {t_result.score:.2f})")


if __name__ == "__main__":
    debug_dual_detection()
