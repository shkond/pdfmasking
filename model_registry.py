"""Model Registry for centralized Transformer model management.

This module provides a registry pattern for managing Transformer models,
enabling:
- Centralized configuration in config.yaml
- Lazy loading of model instances
- Visibility into which models are available/loaded
"""

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Model information for visibility and debugging.
    
    Attributes:
        id: Unique model identifier (e.g., "bert_ner_en")
        name: Hugging Face model name
        language: Language code ("en" or "ja")
        entities: List of entity types this model can detect
        status: Current status ("available", "loaded", "error")
        description: Human-readable description
    """
    id: str
    name: str
    language: str
    entities: list[str]
    status: str = "available"
    description: str = ""


class ModelRegistry:
    """
    Registry for managing Transformer models from YAML configuration.
    
    Provides centralized model management with:
    - get(model_id): Get recognizer instance by ID (lazy loading)
    - get_for_language(language): Get default model for a language
    - list_models(): List all registered models with their status
    - summary(): Human-readable summary of the registry
    
    Example:
        >>> from model_registry import ModelRegistry
        >>> from analyzer_factory import load_config
        >>> 
        >>> config = load_config()
        >>> registry = ModelRegistry(config)
        >>> 
        >>> # List available models
        >>> for model in registry.list_models():
        ...     print(f"{model.id}: {model.name} ({model.language})")
        >>> 
        >>> # Get a specific model
        >>> recognizer = registry.get("bert_ner_en")
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize ModelRegistry from configuration.
        
        Args:
            config: Full application config (from config.yaml)
        """
        self._config = config
        self._transformer_config = config.get("transformer", {})
        self._models_config = config.get("models", {})
        self._model_registry = self._models_config.get("registry", {})
        self._defaults = self._models_config.get("defaults", {})

        # Instance cache (lazy loading)
        self._instances: dict[str, Any] = {}
        self._load_errors: dict[str, str] = {}

    def get(self, model_id: str) -> Any | None:
        """
        Get recognizer instance by model ID (lazy loading).
        
        Args:
            model_id: Model identifier from config.yaml
            
        Returns:
            TransformerNERRecognizer instance, or None if not found/error
        """
        if model_id not in self._model_registry:
            logger.warning(f"[ModelRegistry] Model not found: {model_id}")
            return None

        if model_id in self._instances:
            return self._instances[model_id]

        if model_id in self._load_errors:
            logger.warning(f"[ModelRegistry] Model previously failed to load: {model_id}")
            return None

        # Lazy load the recognizer
        try:
            recognizer = self._create_recognizer(model_id)
            self._instances[model_id] = recognizer
            logger.info(f"[ModelRegistry] Loaded model: {model_id}")
            return recognizer
        except Exception as e:
            self._load_errors[model_id] = str(e)
            logger.error(f"[ModelRegistry] Failed to load {model_id}: {e}")
            return None

    def get_config(self, model_id: str) -> dict[str, Any] | None:
        """
        Get model configuration by ID.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model configuration dict, or None if not found
        """
        return self._model_registry.get(model_id)

    def get_default_model_id(self, language: str) -> str | None:
        """
        Get default model ID for a language.
        
        Args:
            language: Language code ("en" or "ja")
            
        Returns:
            Model ID, or None if no default
        """
        return self._defaults.get(language)

    def get_for_language(self, language: str) -> Any | None:
        """
        Get default recognizer for a language.
        
        Args:
            language: Language code ("en" or "ja")
            
        Returns:
            TransformerNERRecognizer instance, or None if not found
        """
        model_id = self.get_default_model_id(language)
        if model_id:
            return self.get(model_id)
        return None

    def list_model_ids(self) -> list[str]:
        """
        List all registered model IDs.
        
        Returns:
            List of model IDs
        """
        return list(self._model_registry.keys())

    def list_models(self) -> list[ModelInfo]:
        """
        List all registered models with their info.
        
        Returns:
            List of ModelInfo objects
        """
        models = []
        for model_id, model_cfg in self._model_registry.items():
            status = "loaded" if model_id in self._instances else \
                     "error" if model_id in self._load_errors else "available"

            models.append(ModelInfo(
                id=model_id,
                name=model_cfg.get("model_name", ""),
                language=model_cfg.get("language", ""),
                entities=model_cfg.get("entities", []),
                status=status,
                description=model_cfg.get("description", ""),
            ))
        return models

    def list_models_dict(self) -> dict[str, ModelInfo]:
        """
        List all registered models as a dictionary.
        
        Returns:
            Dict mapping model_id to ModelInfo
        """
        return {m.id: m for m in self.list_models()}

    def summary(self) -> str:
        """
        Generate human-readable summary of the registry.
        
        Returns:
            Formatted string showing all models and their status
        """
        lines = ["=== Model Registry ==="]

        if not self._model_registry:
            lines.append("  (no models registered)")
            return "\n".join(lines)

        for model in self.list_models():
            status_icon = {
                "available": "○",
                "loaded": "●",
                "error": "✗"
            }.get(model.status, "?")

            lines.append(f"\n[{model.id}] {status_icon} {model.status}")
            lines.append(f"  Model: {model.name}")
            lines.append(f"  Language: {model.language}")
            lines.append(f"  Entities: {', '.join(model.entities)}")
            if model.description:
                lines.append(f"  Description: {model.description}")

        # Show defaults
        lines.append("\n--- Defaults ---")
        for lang, model_id in self._defaults.items():
            lines.append(f"  {lang}: {model_id}")

        lines.append(f"\nTotal: {len(self._model_registry)} models registered")

        return "\n".join(lines)

    def _create_recognizer(self, model_id: str) -> Any:
        """
        Create a recognizer instance for the given model ID.
        
        Args:
            model_id: Model identifier
            
        Returns:
            TransformerNERRecognizer instance
            
        Raises:
            ValueError: If model not found
            ImportError: If required libraries not available
        """
        model_cfg = self._model_registry.get(model_id)
        if not model_cfg:
            raise ValueError(f"Model not found in registry: {model_id}")

        # Import here to avoid circular imports and allow graceful degradation
        try:
            from recognizers.transformer_ner import create_transformer_recognizer
        except ImportError as e:
            raise ImportError(
                f"Cannot create Transformer recognizer: {e}. "
                "Install torch and transformers: pip install torch transformers"
            )

        language = model_cfg.get("language", "en")

        # Build transformer config with model-specific settings
        transformer_config = {
            "min_confidence": self._transformer_config.get("min_confidence", 0.8),
            "device": self._transformer_config.get("device", "cpu"),
            "label_mapping": {
                language: model_cfg.get("label_mapping", {})
            }
        }

        return create_transformer_recognizer(
            model_config=model_cfg,
            language=language,
            transformer_config=transformer_config,
            model_id=model_id
        )
