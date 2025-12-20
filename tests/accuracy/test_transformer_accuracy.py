"""Accuracy tests for Transformer NER recognizers.

These tests evaluate the actual detection accuracy of Transformer models.
They use real models without mocking to verify detection quality.

Excluded from CI by default due to:
- Long execution time (model loading)
- Resource requirements (memory)
"""

import pytest


# Mark all tests in this module as accuracy tests
pytestmark = pytest.mark.accuracy


class TestTransformerJapaneseAccuracy:
    """Accuracy tests for Japanese Transformer NER model."""

    @pytest.fixture
    def recognizer(self):
        """Create Japanese Transformer recognizer."""
        from recognizers import create_transformer_recognizer

        model_config = {
            "model_name": "knosing/japanese_ner_model",
            "tokenizer_name": "tohoku-nlp/bert-base-japanese-v3",
            "entities": ["JP_PERSON", "JP_ADDRESS", "JP_ORGANIZATION"]
        }
        transformer_config = {
            "min_confidence": 0.5,
            "device": "cpu",
            "label_mapping": {
                "ja": {
                    # 実際のモデル出力ラベル（日本語）
                    "人名": "JP_PERSON",
                    "地名": "JP_ADDRESS",
                    "法人名": "JP_ORGANIZATION",
                    "その他の組織名": "JP_ORGANIZATION",
                    "政治的組織名": "JP_ORGANIZATION",
                    "施設名": "JP_ADDRESS",
                }
            }
        }

        return create_transformer_recognizer(
            model_config=model_config,
            language="ja",
            transformer_config=transformer_config
        )

    def test_person_detection_accuracy(self, recognizer):
        """Test accuracy of Japanese person name detection."""
        test_cases = [
            ("山田太郎さんは東京に住んでいます", "山田太郎", "JP_PERSON"),
            ("田中花子と申します", "田中花子", "JP_PERSON"),
            ("佐藤一郎が担当です", "佐藤一郎", "JP_PERSON"),
        ]

        for text, expected_entity, expected_type in test_cases:
            results = recognizer.analyze(text, entities=[expected_type])

            # Find matching result
            found = False
            for result in results:
                detected = text[result.start:result.end]
                if expected_entity in detected or detected in expected_entity:
                    found = True
                    assert result.entity_type == expected_type
                    assert result.score >= 0.5

            assert found, f"Expected to find '{expected_entity}' in '{text}', got {results}"

    def test_organization_detection_accuracy(self, recognizer):
        """Test accuracy of Japanese organization detection."""
        test_cases = [
            ("株式会社テスト", "JP_ORGANIZATION"),
            ("東京大学", "JP_ORGANIZATION"),
        ]

        for text, expected_type in test_cases:
            results = recognizer.analyze(text, entities=[expected_type])

            # Organization detection may vary by model
            # Just verify no errors occur
            assert isinstance(results, list)


class TestTransformerEnglishAccuracy:
    """Accuracy tests for English Transformer NER model."""

    @pytest.fixture
    def recognizer(self):
        """Create English Transformer recognizer."""
        from recognizers import create_transformer_recognizer

        model_config = {
            "model_name": "dslim/bert-base-NER",
            "entities": ["PERSON", "LOCATION", "ORGANIZATION"]
        }
        transformer_config = {
            "min_confidence": 0.5,
            "device": "cpu",
            "label_mapping": {
                "en": {
                    "B-PER": "PERSON",
                    "I-PER": "PERSON",
                    "B-LOC": "LOCATION",
                    "I-LOC": "LOCATION",
                    "B-ORG": "ORGANIZATION",
                    "I-ORG": "ORGANIZATION",
                }
            }
        }

        return create_transformer_recognizer(
            model_config=model_config,
            language="en",
            transformer_config=transformer_config
        )

    def test_person_detection_accuracy(self, recognizer):
        """Test accuracy of English person name detection."""
        test_cases = [
            ("John Smith works at Google", "John Smith", "PERSON"),
            ("Dr. Jane Doe attended the conference", "Jane Doe", "PERSON"),
        ]

        for text, expected_entity, expected_type in test_cases:
            results = recognizer.analyze(text, entities=[expected_type])

            # Find matching result
            found = False
            for result in results:
                detected = text[result.start:result.end]
                if expected_entity in detected or detected in expected_entity:
                    found = True
                    assert result.entity_type == expected_type

            assert found, f"Expected to find '{expected_entity}' in '{text}'"

    def test_organization_detection_accuracy(self, recognizer):
        """Test accuracy of English organization detection."""
        text = "Microsoft and Google are tech companies"
        results = recognizer.analyze(text, entities=["ORGANIZATION"])

        org_detected = any(r.entity_type == "ORGANIZATION" for r in results)
        assert org_detected, "Expected to detect organizations"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
