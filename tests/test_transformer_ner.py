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
        """Test creating English Transformer recognizer."""
        from recognizers import create_english_transformer_recognizer
        
        recognizer = create_english_transformer_recognizer(
            model_name="dslim/bert-base-NER",
            min_confidence=0.8
        )
        
        assert recognizer.supported_language == "en"
        assert "PERSON" in recognizer.supported_entities
        assert "LOCATION" in recognizer.supported_entities
        assert recognizer.min_confidence == 0.8
    
    def test_create_japanese_recognizer(self):
        """Test creating Japanese Transformer recognizer."""
        from recognizers import create_japanese_transformer_recognizer
        
        recognizer = create_japanese_transformer_recognizer(
            model_name="knosing/japanese_ner_model",
            min_confidence=0.8
        )
        
        assert recognizer.supported_language == "ja"
        assert "JP_PERSON" in recognizer.supported_entities
        assert "JP_ADDRESS" in recognizer.supported_entities
        assert recognizer.min_confidence == 0.8
    
    def test_registry_with_transformer(self):
        """Test that registry correctly includes Transformer recognizers."""
        from recognizer_registry import create_default_registry
        
        registry = create_default_registry(
            use_ginza=False,
            use_transformer=True,
            transformer_config={
                "device": "cpu",
                "min_confidence": 0.8,
                "english_model": "dslim/bert-base-NER",
                "japanese_model": "knosing/japanese_ner_model"
            }
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
        from analyzer_factory import load_config, get_transformer_config
        
        config = load_config()
        transformer_cfg = get_transformer_config(config)
        
        # Check default values from config.yaml
        assert "enabled" in transformer_cfg
        assert "device" in transformer_cfg
        assert "min_confidence" in transformer_cfg
        assert "english_model" in transformer_cfg
        assert "japanese_model" in transformer_cfg
        
        # Config file has enabled: false by default
        assert transformer_cfg["enabled"] == False
        assert transformer_cfg["min_confidence"] == 0.8
    
    def test_summary_includes_transformer(self):
        """Test that registry summary includes Transformer section."""
        from recognizer_registry import create_default_registry
        
        registry = create_default_registry(
            use_ginza=False,
            use_transformer=True,
            transformer_config={"device": "cpu", "min_confidence": 0.8}
        )
        
        summary = registry.summary()
        assert "NER_TRANSFORMER:" in summary
        assert "dslim/bert-base-NER" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
