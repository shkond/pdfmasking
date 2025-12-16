"""Dual detection logic for pattern + Transformer consensus.

This module implements the dual detection strategy:
Only mask entities detected by BOTH pattern AND Transformer recognizers.
"""

import warnings

from presidio_analyzer import RecognizerRegistry as PresidioRegistry


def normalize_entity_type(entity_type: str, entity_categories: dict[str, list]) -> str:
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


def dual_detection_analyze(
    text: str,
    transformer_cfg: dict,
    entities_to_mask: list,
    entity_categories: dict,
    language: str = "ja",
    app_config: dict = None
) -> tuple[list, dict]:
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
        app_config: Full application config (for ModelRegistry)
        
    Returns:
        Tuple of (confirmed_results, dual_scores_dict)
        - confirmed_results: List of RecognizerResult objects detected by both
        - dual_scores_dict: Dict mapping (start, end) -> {"pattern": score, "transformer": score}
    """
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    from recognizer_registry import create_default_registry

    # Create pattern-only analyzer (no app_config needed)
    pattern_registry = create_default_registry(use_ginza=True, use_transformer=False)

    # Create Transformer-only analyzer using ModelRegistry
    transformer_registry = create_default_registry(
        use_ginza=False,
        use_transformer=True,
        app_config=app_config
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
        # PresidioRegistry imported at module level
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
        # PresidioRegistry imported at module level
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
                warnings.warn(f"Transformer analysis failed for recognizer: {e}")
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
                    warnings.warn(f"Transformer analysis failed for recognizer: {e}")


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
                p_type_normalized = normalize_entity_type(p_result.entity_type, entity_categories)
                t_type_normalized = normalize_entity_type(t_result.entity_type, entity_categories)

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
