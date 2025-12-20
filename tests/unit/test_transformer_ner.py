"""Tests for Transformer NER recognizers."""

import pytest


class TestTransformerNERRecognizer:
    """Tests for TransformerNERRecognizer functionality."""

    def test_transformer_available(self):
        """Test that Transformer recognizers are available when torch is installed."""
        from recognizers import TRANSFORMER_AVAILABLE

        # Should be True since user confirmed torch is installed
        assert TRANSFORMER_AVAILABLE, "TRANSFORMER_AVAILABLE should be True when torch/transformers are installed"

    def test_create_english_recognizer(self):
        """Test creating English Transformer recognizer via config-driven factory."""
        from recognizers import create_transformer_recognizer

        model_config = {
            "model_name": "dslim/bert-base-NER",
            "entities": ["PERSON", "LOCATION", "ORGANIZATION"]
        }
        transformer_config = {
            "min_confidence": 0.8,
            "device": "cpu",
            "label_mapping": {
                "en": {
                    "B-PER": "PERSON",
                    "I-PER": "PERSON",
                    "B-LOC": "LOCATION",
                    "I-LOC": "LOCATION",
                    "B-ORG": "ORGANIZATION",
                    "I-ORG": "ORGANIZATION"
                }
            }
        }

        recognizer = create_transformer_recognizer(
            model_config=model_config,
            language="en",
            transformer_config=transformer_config
        )

        assert recognizer.supported_language == "en"
        assert "PERSON" in recognizer.supported_entities
        assert "LOCATION" in recognizer.supported_entities
        assert recognizer.min_confidence == 0.8

    def test_create_japanese_recognizer(self):
        """Test creating Japanese Transformer recognizer via config-driven factory."""
        from recognizers import create_transformer_recognizer

        model_config = {
            "model_name": "knosing/japanese_ner_model",
            "tokenizer_name": "tohoku-nlp/bert-base-japanese-v3",
            "entities": ["JP_PERSON", "JP_ADDRESS", "JP_ORGANIZATION"]
        }
        transformer_config = {
            "min_confidence": 0.8,
            "device": "cpu",
            "label_mapping": {
                "ja": {
                    "B-PER": "JP_PERSON",
                    "I-PER": "JP_PERSON",
                    "B-LOC": "JP_ADDRESS",
                    "I-LOC": "JP_ADDRESS",
                    "B-ORG": "JP_ORGANIZATION",
                    "I-ORG": "JP_ORGANIZATION"
                }
            }
        }

        recognizer = create_transformer_recognizer(
            model_config=model_config,
            language="ja",
            transformer_config=transformer_config
        )

        assert recognizer.supported_language == "ja"
        assert "JP_PERSON" in recognizer.supported_entities
        assert "JP_ADDRESS" in recognizer.supported_entities
        assert recognizer.min_confidence == 0.8

    def test_registry_with_transformer(self):
        """Test that registry correctly includes Transformer recognizers."""
        from recognizers.registry import create_default_registry

        registry = create_default_registry(
            use_ginza=False,
            use_transformer=True
        )

        # Should have Transformer recognizers
        transformer_configs = registry.get_by_type("ner_transformer")
        assert len(transformer_configs) == 2, f"Expected 2 Transformer recognizers, got {len(transformer_configs)}"

        # Check languages
        languages = {c.language for c in transformer_configs}
        assert "en" in languages
        assert "ja" in languages

    def test_config_loading(self):
        """Test that configuration loading works correctly."""
        from config import get_transformer_config, load_config

        config = load_config()
        transformer_cfg = get_transformer_config(config)

        # Check default values from config.yaml
        assert "enabled" in transformer_cfg
        assert "device" in transformer_cfg
        assert "min_confidence" in transformer_cfg
        # Model Registry keys (new pattern)
        assert "models_registry" in transformer_cfg
        assert "models_defaults" in transformer_cfg

        # Config file has enabled: False (Transformer disabled by default)
        assert transformer_cfg["enabled"] == False
        assert transformer_cfg["min_confidence"] == 0.8

        # Check models_defaults contains language mappings
        assert "en" in transformer_cfg["models_defaults"]
        assert "ja" in transformer_cfg["models_defaults"]

    def test_summary_includes_transformer(self):
        """Test that registry summary includes Transformer section."""
        from recognizers.registry import create_default_registry

        registry = create_default_registry(
            use_ginza=False,
            use_transformer=True
        )

        summary = registry.summary()
        assert "NER_TRANSFORMER:" in summary
        assert "dslim/bert-base-NER" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

