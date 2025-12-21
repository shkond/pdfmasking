"""Hybrid detection logic for entity-based routing.

This module implements the hybrid detection strategy:
- Transformer NER for PERSON, ADDRESS entities
- Pattern recognizers for PHONE, ZIP, DATE, etc.
"""

import warnings
from typing import Any

from presidio_analyzer import RecognizerResult


def hybrid_detection_analyze(
    text: str,
    transformer_entities: list[str],
    pattern_entities: list[str],
    language: str = "ja",
    app_config: dict[str, Any] | None = None,
    allow_list: list[str] | None = None
) -> list[RecognizerResult]:
    """
    Hybrid detection: Route entities to appropriate recognizers.
    
    - transformer_entities → Transformer NER (high-precision ML models)
    - pattern_entities → Pattern/GiNZA recognizers (rule-based)
    
    Args:
        text: Text to analyze
        transformer_entities: Entity types for Transformer NER
        pattern_entities: Entity types for Pattern recognizers
        language: Language code ("en", "ja", or "auto")
        app_config: Full application config (for ModelRegistry)
        allow_list: List of terms to exclude from PII detection
        
    Returns:
        List of RecognizerResult objects from both sources
    """
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    from recognizers.registry import create_default_registry

    all_results = []

    # Setup NLP engine for GiNZA/spaCy
    try:
        nlp_configuration = {
            "nlp_engine_name": "spacy",
            "models": [
                {"lang_code": "en", "model_name": "en_core_web_lg"},
                {"lang_code": "ja", "model_name": "ja_ginza"},
            ],
        }
        nlp_engine = NlpEngineProvider(nlp_configuration=nlp_configuration).create_engine()
    except Exception as e:
        warnings.warn(f"Failed to create NLP engine: {e}")
        nlp_engine = None

    # === Pattern Recognizers for pattern_entities ===
    if pattern_entities:
        pattern_registry = create_default_registry(
            use_ginza=True,
            use_transformer=False
        )
        
        if nlp_engine:
            pattern_analyzer = AnalyzerEngine(
                nlp_engine=nlp_engine,
                supported_languages=["en", "ja"]
            )
        else:
            pattern_analyzer = AnalyzerEngine(supported_languages=["en", "ja"])
        
        pattern_registry.apply_to_analyzer(pattern_analyzer)
        
        # Analyze with pattern recognizers
        if language == "auto":
            results_en = pattern_analyzer.analyze(
                text=text, language="en", entities=pattern_entities,
                allow_list=allow_list
            )
            results_ja = pattern_analyzer.analyze(
                text=text, language="ja", entities=pattern_entities,
                allow_list=allow_list
            )
            all_results.extend(list(results_en))
            all_results.extend(list(results_ja))
        else:
            results = pattern_analyzer.analyze(
                text=text, language=language, entities=pattern_entities,
                allow_list=allow_list
            )
            all_results.extend(list(results))

    # === Transformer NER for transformer_entities ===
    if transformer_entities:
        transformer_registry = create_default_registry(
            use_ginza=False,
            use_transformer=True,
            app_config=app_config
        )
        
        # Get ML recognizers directly (avoid AnalyzerEngine default recognizers)
        ml_recognizers = [
            config.recognizer for config in transformer_registry.configs
            if config.type in {"ner_transformer", "ner_gpt_masker"}
        ]
        
        for recognizer in ml_recognizers:
            # Match language
            if language == "auto" or recognizer.supported_language == language:
                try:
                    results = recognizer.analyze(
                        text=text,
                        entities=transformer_entities
                    )
                    all_results.extend(results)
                except Exception as e:
                    warnings.warn(f"ML analysis failed: {e}")

    return all_results
