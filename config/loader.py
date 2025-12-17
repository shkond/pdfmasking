"""Configuration loader for PII masking application.

This module handles loading and parsing of config.yaml settings.
"""

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. Defaults to config.yaml in project root.
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Look for config.yaml in project root (parent of config/)
        config_path = Path(__file__).parent.parent / "config.yaml"
    else:
        config_path = Path(config_path)

    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def get_transformer_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Extract transformer configuration from main config.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        Transformer-specific configuration dict with keys:
        - enabled: bool
        - device: str ("cpu" or "cuda")
        - min_confidence: float
    """
    transformer = config.get("transformer", {})

    return {
        "enabled": transformer.get("enabled", False),
        "device": transformer.get("device", "cpu"),
        "min_confidence": transformer.get("min_confidence", 0.8),
    }


def get_detection_strategy(config: dict[str, Any]) -> dict[str, list]:
    """
    Get detection strategy configuration.
    
    Defines which entities are handled by Transformer NER vs Pattern recognizers.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dict with keys:
        - transformer_entities: list of entity types for Transformer
        - pattern_entities: list of entity types for Pattern/GiNZA
    """
    strategy = config.get("detection_strategy", {})
    return {
        "transformer_entities": strategy.get("transformer_entities", [
            "JP_PERSON", "JP_ADDRESS", "PERSON", "LOCATION"
        ]),
        "pattern_entities": strategy.get("pattern_entities", [
            "PHONE_NUMBER_JP", "JP_ZIP_CODE", "DATE_OF_BIRTH_JP",
            "JP_AGE", "JP_GENDER", "EMAIL_ADDRESS"
        ]),
    }


def get_entities_to_mask(config: dict[str, Any]) -> list:
    """
    Get the list of entity types to mask from config.
    
    These are the 8 target PII types:
    - JP_PERSON / PERSON (名前)
    - EMAIL_ADDRESS (メールアドレス)
    - JP_ZIP_CODE (郵便番号)
    - PHONE_NUMBER_JP (電話番号)
    - DATE_OF_BIRTH_JP (生年月日)
    - JP_ADDRESS / LOCATION (住所)
    - JP_GENDER (性別)
    - JP_AGE (年齢)
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of entity type strings
    """
    return config.get("entities_to_mask", [])


def get_entity_categories(config: dict[str, Any]) -> dict[str, list]:
    """
    Get entity categories for type normalization in Dual Detection.
    
    Categories group equivalent entity types across different recognizers,
    e.g., PERSON and JP_PERSON are both in the "person" category.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dict mapping category name to list of entity types
    """
    return config.get("entity_categories", {})
