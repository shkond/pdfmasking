"""Factory for creating language-specific analyzer engines.

This module provides functions to create AnalyzerEngine instances
configured for different languages (English, Japanese, or multilingual).
"""

import warnings
from typing import Any

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider

from recognizers.registry import GINZA_AVAILABLE, create_default_registry


# Map spaCy / GiNZA NER labels to the entity labels used throughout this project.
# Without this, Presidio will keep the raw labels (e.g., "Person", "Postal_Address")
# which won't match our configured entities (e.g., "PERSON", "JP_ADDRESS").
_NER_MODEL_CONFIGURATION: dict[str, Any] = {
    "model_to_presidio_entity_mapping": {
        # spaCy (English)
        "PERSON": "PERSON",
        "PER": "PERSON",
        "ORG": "ORGANIZATION",
        "GPE": "LOCATION",
        "LOC": "LOCATION",
        "FAC": "LOCATION",
        "DATE": "DATE",
        # GiNZA / Japanese pipelines (labels vary by model)
        "Person": "PERSON",
        "Organization": "ORGANIZATION",
        "N_Organization": "ORGANIZATION",
        "Organization_Other": "ORGANIZATION",
        "Show_Organization": "ORGANIZATION",
        "Company": "ORGANIZATION",
        "Postal_Address": "JP_ADDRESS",
        "GPE_JP": "JP_ADDRESS",
        "Location": "JP_ADDRESS",
        "Date": "DATE",
    },
    # Keep defaults explicit to silence configuration warnings.
    "low_score_entity_names": [],
    "labels_to_ignore": [],
}


def create_japanese_analyzer(
    use_ginza: bool = True,
    use_transformer: bool = False,
    transformer_config: dict[str, Any] | None = None,
    verbose: bool = False
) -> AnalyzerEngine:
    """
    Create an AnalyzerEngine configured for Japanese text.
    
    This includes:
    - Pattern-based recognizers (phone, zip, date) - always available
    - GiNZA-based recognizers (person, address) - optional, requires GiNZA installation
    - Transformer-based recognizers - optional, requires torch and transformers
    
    Args:
        use_ginza: Whether to use GiNZA for NER (requires GiNZA installation)
        use_transformer: Whether to use Transformer-based NER
        transformer_config: Configuration for Transformer recognizers
        verbose: If True, print registry summary
    
    Returns:
        Configured AnalyzerEngine for Japanese
    """
    # Create registry
    registry = create_default_registry(
        use_ginza=use_ginza,
        use_transformer=use_transformer,
        transformer_config=transformer_config
    )

    if verbose:
        print(registry.summary())

    # Create NLP configuration for Japanese
    nlp_configuration = {
        "nlp_engine_name": "spacy",
        "ner_model_configuration": _NER_MODEL_CONFIGURATION,
        "models": [
            {"lang_code": "en", "model_name": "en_core_web_lg"},
            {"lang_code": "ja", "model_name": "ja_ginza" if use_ginza and GINZA_AVAILABLE else "en_core_web_lg"},
        ],
    }

    try:
        nlp_engine = NlpEngineProvider(nlp_configuration=nlp_configuration).create_engine()
    except (ImportError, OSError, ValueError) as e:
        warnings.warn(f"Failed to create Japanese NLP engine: {e}. Falling back to basic engine.")
        nlp_engine = None

    # Create analyzer with Japanese support
    if nlp_engine:
        analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine,
            supported_languages=["ja", "en"]
        )
    else:
        analyzer = AnalyzerEngine(supported_languages=["ja", "en"])

    # Apply recognizers from registry (Japanese only)
    registry.apply_to_analyzer(analyzer, language="ja")

    if verbose:
        ja_count = len(registry.get_by_language("ja"))
        print(f"✓ Japanese analyzer created with {ja_count} recognizers")

    return analyzer


def create_analyzer(
    language: str = "en",
    use_ginza: bool = True,
    use_transformer: bool = False,
    transformer_config: dict[str, Any] | None = None,
    verbose: bool = False
) -> AnalyzerEngine:
    """
    Create an AnalyzerEngine for the specified language.
    
    Args:
        language: Language code ("en" for English, "ja" for Japanese)
        use_ginza: Whether to use GiNZA for Japanese NER (default: True)
        use_transformer: Whether to use Transformer-based NER
        transformer_config: Configuration for Transformer recognizers
        verbose: If True, print registry summary
        
    Returns:
        Configured AnalyzerEngine
    """
    if language == "ja":
        return create_japanese_analyzer(
            use_ginza=use_ginza,
            use_transformer=use_transformer,
            transformer_config=transformer_config,
            verbose=verbose
        )
    # English analyzer with optional Transformer support
    elif use_transformer:
        registry = create_default_registry(
            use_ginza=False,
            use_transformer=True,
            transformer_config=transformer_config
        )
        analyzer = AnalyzerEngine()
        registry.apply_to_analyzer(analyzer, language="en")
        return analyzer
    else:
        nlp_configuration = {
            "nlp_engine_name": "spacy",
            "ner_model_configuration": _NER_MODEL_CONFIGURATION,
            "models": [
                {"lang_code": "en", "model_name": "en_core_web_lg"},
            ],
        }
        try:
            nlp_engine = NlpEngineProvider(nlp_configuration=nlp_configuration).create_engine()
            return AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
        except Exception:
            return AnalyzerEngine()


def create_multilingual_analyzer(
    use_ginza: bool = True,
    use_transformer: bool = False,
    transformer_config: dict[str, Any] | None = None,
    verbose: bool = False
) -> AnalyzerEngine:
    """
    Create an AnalyzerEngine that supports both English and Japanese text.
    
    This analyzer can process text in either language with a single instance,
    with all recognizers registered once (no double-registration).
    
    Args:
        use_ginza: Whether to use GiNZA for Japanese NER (default: True)
        use_transformer: Whether to use Transformer-based NER
        transformer_config: Configuration for Transformer recognizers
        verbose: If True, print registry summary
        
    Returns:
        Configured AnalyzerEngine supporting both 'en' and 'ja'
    """
    # Create registry with all recognizers
    registry = create_default_registry(
        use_ginza=use_ginza,
        use_transformer=use_transformer,
        transformer_config=transformer_config
    )

    if verbose:
        print("=== Multilingual Analyzer Configuration ===")
        print(registry.summary())

    # Configure NLP engine with both language models
    nlp_configuration = {
        "nlp_engine_name": "spacy",
        "ner_model_configuration": _NER_MODEL_CONFIGURATION,
        "models": [
            {"lang_code": "en", "model_name": "en_core_web_lg"},
            {"lang_code": "ja", "model_name": "ja_ginza" if use_ginza and GINZA_AVAILABLE else "en_core_web_lg"},
        ],
    }

    try:
        nlp_engine = NlpEngineProvider(nlp_configuration=nlp_configuration).create_engine()
    except Exception as e:
        warnings.warn(f"Failed to create multi-language NLP engine: {e}. Falling back to basic engine.")
        nlp_engine = None

    # Create analyzer with both language support
    if nlp_engine:
        analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine,
            supported_languages=["en", "ja"]
        )
    else:
        analyzer = AnalyzerEngine(supported_languages=["en", "ja"])

    # Apply ALL recognizers from registry (no double-registration)
    registry.apply_to_analyzer(analyzer)

    if verbose:
        print(f"✓ Multilingual analyzer created with {len(registry.configs)} recognizers")

    return analyzer
