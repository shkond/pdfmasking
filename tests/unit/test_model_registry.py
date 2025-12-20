"""Tests for ModelRegistry."""

import pytest

from model_registry import ModelInfo, ModelRegistry


class TestModelRegistry:
    """Tests for ModelRegistry functionality."""

    @pytest.fixture
    def sample_config(self):
        """Sample config matching the structure in config.yaml."""
        return {
            "transformer": {
                "enabled": True,
                "device": "cpu",
                "min_confidence": 0.8,
            },
            "models": {
                "registry": {
                    "bert_ner_en": {
                        "type": "transformer",
                        "model_name": "dslim/bert-base-NER",
                        "tokenizer_name": None,
                        "language": "en",
                        "entities": ["PERSON", "LOCATION", "ORGANIZATION"],
                        "label_mapping": {
                            "B-PER": "PERSON",
                            "I-PER": "PERSON",
                            "B-LOC": "LOCATION",
                            "I-LOC": "LOCATION",
                            "B-ORG": "ORGANIZATION",
                            "I-ORG": "ORGANIZATION",
                        },
                        "description": "English NER - dslim/bert-base-NER (CoNLL-2003)",
                    },
                    "knosing_ner_ja": {
                        "type": "transformer",
                        "model_name": "knosing/japanese_ner_model",
                        "tokenizer_name": "tohoku-nlp/bert-base-japanese-v3",
                        "language": "ja",
                        "entities": ["JP_PERSON", "JP_ADDRESS", "JP_ORGANIZATION"],
                        "label_mapping": {
                            "B-PER": "JP_PERSON",
                            "I-PER": "JP_PERSON",
                            "B-LOC": "JP_ADDRESS",
                            "I-LOC": "JP_ADDRESS",
                        },
                        "description": "Japanese NER - knosing/japanese_ner_model",
                    },
                },
                "defaults": {
                    "en": "bert_ner_en",
                    "ja": "knosing_ner_ja",
                },
            },
        }

    def test_list_model_ids(self, sample_config):
        """Test that list_model_ids returns all registered model IDs."""
        registry = ModelRegistry(sample_config)
        model_ids = registry.list_model_ids()

        assert len(model_ids) == 2
        assert "bert_ner_en" in model_ids
        assert "knosing_ner_ja" in model_ids

    def test_list_models(self, sample_config):
        """Test that list_models returns ModelInfo objects."""
        registry = ModelRegistry(sample_config)
        models = registry.list_models()

        assert len(models) == 2
        assert all(isinstance(m, ModelInfo) for m in models)

        # Check English model
        en_model = next(m for m in models if m.id == "bert_ner_en")
        assert en_model.name == "dslim/bert-base-NER"
        assert en_model.language == "en"
        assert "PERSON" in en_model.entities
        assert en_model.status == "available"

        # Check Japanese model
        ja_model = next(m for m in models if m.id == "knosing_ner_ja")
        assert ja_model.name == "knosing/japanese_ner_model"
        assert ja_model.language == "ja"
        assert "JP_PERSON" in ja_model.entities

    def test_get_default_model_id(self, sample_config):
        """Test getting default model ID for a language."""
        registry = ModelRegistry(sample_config)

        en_model_id = registry.get_default_model_id("en")
        ja_model_id = registry.get_default_model_id("ja")
        unknown_model_id = registry.get_default_model_id("fr")

        assert en_model_id == "bert_ner_en"
        assert ja_model_id == "knosing_ner_ja"
        assert unknown_model_id is None

    def test_get_config(self, sample_config):
        """Test getting model configuration by ID."""
        registry = ModelRegistry(sample_config)

        en_config = registry.get_config("bert_ner_en")
        ja_config = registry.get_config("knosing_ner_ja")
        unknown_config = registry.get_config("unknown_model")

        assert en_config is not None
        assert en_config["model_name"] == "dslim/bert-base-NER"

        assert ja_config is not None
        assert ja_config["model_name"] == "knosing/japanese_ner_model"

        assert unknown_config is None

    def test_summary_includes_model_info(self, sample_config):
        """Test that summary includes model details."""
        registry = ModelRegistry(sample_config)
        summary = registry.summary()

        # Check that summary contains key information
        assert "Model Registry" in summary
        assert "bert_ner_en" in summary
        assert "dslim/bert-base-NER" in summary
        assert "knosing_ner_ja" in summary
        assert "en" in summary
        assert "ja" in summary
        assert "2 models registered" in summary

    def test_list_models_dict(self, sample_config):
        """Test that list_models_dict returns dict mapping ID to ModelInfo."""
        registry = ModelRegistry(sample_config)
        models_dict = registry.list_models_dict()

        assert isinstance(models_dict, dict)
        assert "bert_ner_en" in models_dict
        assert "knosing_ner_ja" in models_dict
        assert isinstance(models_dict["bert_ner_en"], ModelInfo)

    def test_empty_registry(self):
        """Test behavior with empty configuration."""
        empty_config = {"transformer": {}, "models": {}}
        registry = ModelRegistry(empty_config)

        assert registry.list_model_ids() == []
        assert registry.list_models() == []
        assert registry.get_default_model_id("en") is None
        assert "(no models registered)" in registry.summary()

    def test_get_nonexistent_model(self, sample_config):
        """Test getting a model that doesn't exist."""
        registry = ModelRegistry(sample_config)

        result = registry.get("nonexistent_model")
        assert result is None


class TestModelInfo:
    """Tests for ModelInfo dataclass."""

    def test_model_info_creation(self):
        """Test creating ModelInfo with all fields."""
        info = ModelInfo(
            id="test_model",
            name="test/model-name",
            language="en",
            entities=["PERSON", "LOCATION"],
            status="available",
            description="Test model description"
        )

        assert info.id == "test_model"
        assert info.name == "test/model-name"
        assert info.language == "en"
        assert info.entities == ["PERSON", "LOCATION"]
        assert info.status == "available"
        assert info.description == "Test model description"

    def test_model_info_defaults(self):
        """Test ModelInfo with default values."""
        info = ModelInfo(
            id="test_model",
            name="test/model-name",
            language="en",
            entities=["PERSON"]
        )

        assert info.status == "available"
        assert info.description == ""
