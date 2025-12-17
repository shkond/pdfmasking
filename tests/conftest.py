"""Shared fixtures for pdfmasking tests."""

import shutil
import tempfile
from pathlib import Path

import pytest

from masking_logging import MaskingLogger


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)

    # Close all logger handlers before cleanup
    masking_logger = MaskingLogger().logger
    for handler in masking_logger.handlers[:]:
        handler.close()
        masking_logger.removeHandler(handler)

    # Clean up temp directory, ignore errors on Windows
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_config():
    """Sample config matching the structure in config.yaml."""
    return {
        "transformer": {
            "enabled": True,
            "device": "cpu",
            "min_confidence": 0.8,
        },
        "detection_strategy": {
            "transformer_entities": ["JP_PERSON", "JP_ADDRESS", "PERSON", "LOCATION"],
            "pattern_entities": ["PHONE_NUMBER_JP", "JP_ZIP_CODE", "DATE_OF_BIRTH_JP", "JP_AGE", "JP_GENDER", "EMAIL_ADDRESS"],
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
                        "人名": "JP_PERSON",
                        "地名": "JP_ADDRESS",
                        "法人名": "JP_ORGANIZATION",
                        "その他の組織名": "JP_ORGANIZATION",
                        "政治的組織名": "JP_ORGANIZATION",
                        "施設名": "JP_ADDRESS",
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


@pytest.fixture
def masking_logger_instance():
    """Get a masking logger instance."""
    return MaskingLogger()


def setup_logger(log_path):
    """Setup logger handler for tests."""
    logger_instance = MaskingLogger()
    logger_instance.setup_file_handler(log_path)
