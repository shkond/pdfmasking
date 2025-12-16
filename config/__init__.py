"""Configuration loading and management."""

from .loader import (
    load_config,
    get_transformer_config,
    get_entities_to_mask,
    get_entity_categories,
)

__all__ = [
    "load_config",
    "get_transformer_config",
    "get_entities_to_mask",
    "get_entity_categories",
]
