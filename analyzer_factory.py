"""Factory for creating language-specific analyzer engines."""

import warnings
from pathlib import Path
from typing import Any

import yaml
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider

from recognizer_registry import GINZA_AVAILABLE, create_default_registry


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. Defaults to config.yaml in project root.
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    else:
        config_path = Path(config_path)

    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def get_transformer_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract transformer configuration from main config."""
    transformer = config.get("transformer", {})
    models = config.get("models", {})

    return {
        "enabled": transformer.get("enabled", False),
        "device": transformer.get("device", "cpu"),
        "min_confidence": transformer.get("min_confidence", 0.8),
        # Model Registry info
        "models_registry": models.get("registry", {}),
        "models_defaults": models.get("defaults", {}),
    }


def get_entities_to_mask(config: dict[str, Any]) -> list:
    """
    Get the list of entity types to mask from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of entity type strings
    """
    return config.get("entities_to_mask", [])


def get_entity_categories(config: dict[str, Any]) -> dict[str, list]:
    """
    Get entity categories for type normalization in Dual Detection.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dict mapping category name to list of entity types
    """
    return config.get("entity_categories", {})


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
        "models": [
            {"lang_code": "en", "model_name": "en_core_web_lg"},
            {"lang_code": "ja", "model_name": "ja_ginza" if use_ginza and GINZA_AVAILABLE else "en_core_web_lg"},
        ],
    }

    try:
        nlp_engine = NlpEngineProvider(nlp_configuration=nlp_configuration).create_engine()
    except Exception:
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

