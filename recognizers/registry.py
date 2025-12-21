"""Centralized registry for managing recognizers with clear categorization."""

import warnings
from dataclasses import dataclass
from typing import Any, Literal

from presidio_analyzer import AnalyzerEngine, EntityRecognizer

# Import directly from submodules to avoid circular import
from recognizers.japanese_patterns import (
    JapaneseAddressRecognizer,
    JapaneseAgeRecognizer,
    JapaneseBirthDateRecognizer,
    JapaneseGenderRecognizer,
    JapaneseNameRecognizer,
    JapanesePhoneRecognizer,
    JapaneseZipCodeRecognizer,
)

# Try to import GiNZA recognizers (optional)
GINZA_AVAILABLE = False
try:
    import spacy

    from recognizers.japanese_ner import GinzaAddressRecognizer, GinzaPersonRecognizer
    GINZA_AVAILABLE = True
except ImportError:
    pass

# Try to import Transformer recognizers
TRANSFORMER_AVAILABLE = False
try:
    from recognizers.transformer_ner import (
        TransformerNERRecognizer,
    )
    TRANSFORMER_AVAILABLE = True
except ImportError:
    pass

RecognizerType = Literal["pattern", "ner_ginza", "ner_presidio", "ner_transformer"]


@dataclass
class RecognizerConfig:
    """
    Recognizer configuration with metadata for visibility.
    
    Attributes:
        recognizer: The EntityRecognizer instance
        type: Classification of recognizer ("pattern", "ner_ginza", "ner_presidio", "ner_transformer")
        language: Supported language code
        entity_type: Entity type string this recognizer detects
        description: Human-readable description
        requires_nlp: Whether this recognizer requires NLP artifacts
    """
    recognizer: EntityRecognizer
    type: RecognizerType
    language: str
    entity_type: str
    description: str
    requires_nlp: bool = False

    def __repr__(self):
        return f"<{self.type.upper()}:{self.language}:{self.entity_type}>"


class RecognizerRegistry:
    """
    Centralized registry for managing recognizers with clear categorization.
    
    Provides methods to:
    - Register recognizers with metadata
    - Filter by type or language
    - Apply to AnalyzerEngine
    - Generate summary for debugging
    """

    def __init__(self):
        self.configs: list[RecognizerConfig] = []

    def register(self, config: RecognizerConfig):
        """Register a recognizer with metadata."""
        self.configs.append(config)

    def get_by_type(self, recognizer_type: RecognizerType) -> list[RecognizerConfig]:
        """Get all recognizers of a specific type."""
        return [c for c in self.configs if c.type == recognizer_type]

    def get_by_language(self, language: str) -> list[RecognizerConfig]:
        """Get all recognizers for a specific language."""
        return [c for c in self.configs if c.language == language]

    def apply_to_analyzer(
        self,
        analyzer: AnalyzerEngine,
        language: str | None = None,
        types: list[RecognizerType] | None = None
    ):
        """
        Apply registered recognizers to an analyzer with filtering.
        
        Args:
            analyzer: AnalyzerEngine instance
            language: Filter by language (None = all)
            types: Filter by recognizer types (None = all)
        """
        configs = self.configs

        if language:
            configs = [c for c in configs if c.language == language]
        if types:
            configs = [c for c in configs if c.type in types]

        for config in configs:
            analyzer.registry.add_recognizer(config.recognizer)

    def summary(self) -> str:
        """Generate a human-readable summary of registered recognizers."""
        lines = ["Recognizer Registry Summary:"]
        for rtype in ["pattern", "ner_ginza", "ner_presidio", "ner_transformer"]:
            configs = self.get_by_type(rtype)
            if configs:
                lines.append(f"\n{rtype.upper()}:")
                for cfg in configs:
                    lines.append(f"  - [{cfg.language}] {cfg.entity_type}: {cfg.description}")
        lines.append(f"\nTotal: {len(self.configs)} recognizers")
        return "\n".join(lines)


def create_default_registry(
    use_ginza: bool = True,
    use_transformer: bool = False,
    transformer_config: dict[str, Any] | None = None,
    app_config: dict[str, Any] | None = None
) -> RecognizerRegistry:
    """
    Create a registry with all available recognizers.
    
    Args:
        use_ginza: Whether to include GiNZA-based recognizers
        use_transformer: Whether to include Transformer-based recognizers
        transformer_config: Configuration for Transformer recognizers (legacy, use app_config)
        app_config: Full application config (from config.yaml). If provided, 
                    transformer settings are read from app_config["transformer"]
        
    Returns:
        RecognizerRegistry with all recognizers registered
    """
    registry = RecognizerRegistry()

    # Load config if not provided
    if app_config is None:
        from config import load_config
        app_config = load_config()

    # Get transformer config from app_config
    transformer_section = app_config.get("transformer", {})

    # === Pattern-based recognizers (Japanese) ===
    registry.register(RecognizerConfig(
        recognizer=JapanesePhoneRecognizer(),
        type="pattern",
        language="ja",
        entity_type="PHONE_NUMBER_JP",
        description="Japanese phone numbers (正規表現)",
        requires_nlp=False
    ))

    registry.register(RecognizerConfig(
        recognizer=JapaneseZipCodeRecognizer(),
        type="pattern",
        language="ja",
        entity_type="JP_ZIP_CODE",
        description="Japanese postal codes (〒XXX-XXXX)",
        requires_nlp=False
    ))

    registry.register(RecognizerConfig(
        recognizer=JapaneseBirthDateRecognizer(),
        type="pattern",
        language="ja",
        entity_type="DATE_OF_BIRTH_JP",
        description="Japanese birth dates (生年月日パターン)",
        requires_nlp=False
    ))

    registry.register(RecognizerConfig(
        recognizer=JapaneseNameRecognizer(),
        type="pattern",
        language="ja",
        entity_type="JP_PERSON",
        description="Japanese names (コンテキストベース)",
        requires_nlp=False
    ))

    registry.register(RecognizerConfig(
        recognizer=JapaneseAgeRecognizer(),
        type="pattern",
        language="ja",
        entity_type="JP_AGE",
        description="Age mentions (XX歳)",
        requires_nlp=False
    ))

    registry.register(RecognizerConfig(
        recognizer=JapaneseGenderRecognizer(),
        type="pattern",
        language="ja",
        entity_type="JP_GENDER",
        description="Gender (性別: 男/女)",
        requires_nlp=False
    ))

    registry.register(RecognizerConfig(
        recognizer=JapaneseAddressRecognizer(),
        type="pattern",
        language="ja",
        entity_type="JP_ADDRESS",
        description="Japanese addresses (都道府県パターン)",
        requires_nlp=False
    ))

    # === GiNZA-based recognizers (if available) ===
    if use_ginza and GINZA_AVAILABLE:
        registry.register(RecognizerConfig(
            recognizer=GinzaPersonRecognizer(),
            type="ner_ginza",
            language="ja",
            entity_type="JP_PERSON",
            description="Person names via GiNZA NER",
            requires_nlp=True
        ))

        registry.register(RecognizerConfig(
            recognizer=GinzaAddressRecognizer(),
            type="ner_ginza",
            language="ja",
            entity_type="JP_ADDRESS",
            description="Addresses via GiNZA NER",
            requires_nlp=True
        ))

    # === Transformer-based recognizers (if available and enabled) ===
    if use_transformer and TRANSFORMER_AVAILABLE:
        # Get models registry and defaults from app_config
        models_config = app_config.get("models", {})
        models_registry = models_config.get("registry", {})
        models_defaults = models_config.get("defaults", {})

        # English Transformer recognizer
        en_model_id = models_defaults.get("en")
        if en_model_id and en_model_id in models_registry:
            en_model_config = models_registry[en_model_id]
            registry.register(RecognizerConfig(
                recognizer=create_transformer_recognizer(
                    model_config=en_model_config,
                    language="en",
                    transformer_config=transformer_section,
                    model_id=en_model_id
                ),
                type="ner_transformer",
                language="en",
                entity_type=",".join(en_model_config.get("entities", [])),
                description=en_model_config.get("description", f"English NER via {en_model_id}"),
                requires_nlp=False
            ))

        # Japanese Transformer recognizer
        ja_model_id = models_defaults.get("ja")
        if ja_model_id and ja_model_id in models_registry:
            ja_model_config = models_registry[ja_model_id]
            registry.register(RecognizerConfig(
                recognizer=create_transformer_recognizer(
                    model_config=ja_model_config,
                    language="ja",
                    transformer_config=transformer_section,
                    model_id=ja_model_id
                ),
                type="ner_transformer",
                language="ja",
                entity_type=",".join(ja_model_config.get("entities", [])),
                description=ja_model_config.get("description", f"Japanese NER via {ja_model_id}"),
                requires_nlp=False
            ))
    elif use_transformer and not TRANSFORMER_AVAILABLE:
        warnings.warn(
            "Transformer recognizers requested but 'torch' and 'transformers' libraries are not available. "
            "Install them with: pip install torch transformers"
        )

    return registry


def create_transformer_recognizer(
    model_config: dict,
    language: str,
    transformer_config: dict,
    model_id: str | None = None
) -> TransformerNERRecognizer:
    """
    設定駆動のTransformer認識器ファクトリー
    
    Args:
        model_config: モデル設定 (model_name, tokenizer_name, entities, label_mapping)
        language: 言語コード ("en" or "ja")
        transformer_config: Transformer全体設定 (min_confidence, device)
        model_id: モデルID（ロギング・識別用、省略可）
        
    Returns:
        設定に基づいたTransformerNERRecognizer
    """
    # Get label mapping from model_config first, then fallback to transformer_config
    label_mapping = model_config.get("label_mapping")
    if label_mapping is None:
        label_mapping = transformer_config.get("label_mapping", {}).get(language, {})

    model_name = model_config.get("model_name")
    if model_id:
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"[{model_id}] Creating recognizer: {model_name}")

    return TransformerNERRecognizer(
        model_name=model_name,
        tokenizer_name=model_config.get("tokenizer_name"),
        supported_language=language,
        supported_entities=model_config.get("entities", []),
        min_confidence=transformer_config.get("min_confidence", 0.8),
        device=transformer_config.get("device", "cpu"),
        label_mapping=label_mapping
    )
