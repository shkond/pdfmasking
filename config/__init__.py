"""Configuration loading and management."""

from .loader import (
    get_detection_strategy,
    get_entities_to_mask,
    get_entity_categories,
    get_transformer_config,
    load_config,
)

__all__ = [
    "get_detection_strategy",
    "get_entities_to_mask",
    "get_entity_categories",
    "get_transformer_config",
    "load_config",
]
