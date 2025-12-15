"""Transformer-based NER recognizers using Hugging Face models."""

from typing import List, Optional, Dict, Tuple
import warnings

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from presidio_analyzer import EntityRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import NlpArtifacts


class TransformerNERRecognizer(EntityRecognizer):
    """
    Transformer NER認識器
    
    サポートモデル:
    - dslim/bert-base-NER (英語)
    - knosing/japanese_ner_model (日本語)
    """
    
    # ラベルマッピング: モデルのラベル → Presidioエンティティタイプ
    LABEL_MAPPINGS = {
        "en": {
            # dslim/bert-base-NER labels
            "B-PER": "PERSON",
            "I-PER": "PERSON",
            "B-LOC": "LOCATION", 
            "I-LOC": "LOCATION",
            "B-ORG": "ORGANIZATION",
            "I-ORG": "ORGANIZATION",
            "B-MISC": "MISC",
            "I-MISC": "MISC",
        },
        "ja": {
            # knosing/japanese_ner_model labels (adjust based on actual model output)
            "B-PER": "JP_PERSON",
            "I-PER": "JP_PERSON",
            "B-PERSON": "JP_PERSON",
            "I-PERSON": "JP_PERSON",
            "B-LOC": "JP_ADDRESS",
            "I-LOC": "JP_ADDRESS",
            "B-LOCATION": "JP_ADDRESS",
            "I-LOCATION": "JP_ADDRESS",
            "B-ORG": "JP_ORGANIZATION",
            "I-ORG": "JP_ORGANIZATION",
            "B-ORGANIZATION": "JP_ORGANIZATION",
            "I-ORGANIZATION": "JP_ORGANIZATION",
            # Additional common Japanese NER labels
            "B-人名": "JP_PERSON",
            "I-人名": "JP_PERSON",
            "B-地名": "JP_ADDRESS",
            "I-地名": "JP_ADDRESS",
            "B-組織": "JP_ORGANIZATION",
            "I-組織": "JP_ORGANIZATION",
        }
    }
    
    def __init__(
        self,
        model_name: str,
        supported_language: str = "en",
        supported_entities: Optional[List[str]] = None,
        min_confidence: float = 0.8,
        device: str = "cpu",
        tokenizer_name: Optional[str] = None
    ):
        """
        Args:
            model_name: Hugging Faceモデル名
            supported_language: "en" or "ja"
            supported_entities: 検出対象エンティティ(None=全て)
            min_confidence: 最小信頼度スコア
            device: "cpu" or "cuda"
            tokenizer_name: トークナイザー名(Noneの場合はmodel_nameと同じ)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("torch and transformers are required for TransformerNERRecognizer")
        
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name or model_name
        self.min_confidence = min_confidence
        self.device = device
        self._model = None
        self._tokenizer = None
        
        # デフォルトエンティティ設定
        if supported_entities is None:
            if supported_language == "en":
                supported_entities = ["PERSON", "LOCATION", "ORGANIZATION"]
            else:
                supported_entities = ["JP_PERSON", "JP_ADDRESS", "JP_ORGANIZATION"]
        
        super().__init__(
            supported_entities=supported_entities,
            supported_language=supported_language,
        )
    
    def load(self) -> None:
        """モデルとトークナイザーの遅延読み込み"""
        if self._model is None:
            # Fast tokenizerを優先して使用 (offset_mapping対応)
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.tokenizer_name, 
                    use_fast=True
                )
            except Exception:
                # Fallback to slow tokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            
            self._model = AutoModelForTokenClassification.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()
    
    def analyze(
        self, text: str, entities: List[str], nlp_artifacts: Optional[NlpArtifacts] = None
    ) -> List[RecognizerResult]:
        """
        テキストを解析してエンティティを検出
        
        Args:
            text: 解析対象テキスト
            entities: 検出対象エンティティリスト
            nlp_artifacts: 未使用(Transformer内部で処理)
            
        Returns:
            RecognizerResultのリスト
        """
        self.load()
        
        # 要求されたエンティティのみフィルタ
        requested_entities = set(entities) & set(self.supported_entities)
        if not requested_entities:
            return []
        
        # Check if tokenizer supports offset_mapping
        try:
            # トークン化
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                return_offsets_mapping=True,
                padding=False
            )
            offset_mapping = inputs.pop("offset_mapping")[0]
            use_offset_mapping = True
        except Exception:
            # Fallback: tokenize without offset_mapping
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=False
            )
            offset_mapping = None
            use_offset_mapping = False
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 推論
        with torch.no_grad():
            outputs = self._model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits[0], dim=-1)
        
        # ラベル抽出
        label_ids = predictions.argmax(dim=-1).cpu().numpy()
        scores = predictions.max(dim=-1).values.cpu().numpy()
        
        # エンティティの構築
        if use_offset_mapping:
            entities_found = self._build_entities(
                label_ids, scores, offset_mapping, text
            )
        else:
            # Fallback: build entities using token-based approach
            entities_found = self._build_entities_from_tokens(
                label_ids, scores, text
            )
        
        # フィルタリング
        results = []
        for entity in entities_found:
            if (entity["entity_type"] in requested_entities and 
                entity["score"] >= self.min_confidence):
                results.append(RecognizerResult(
                    entity_type=entity["entity_type"],
                    start=entity["start"],
                    end=entity["end"],
                    score=entity["score"]
                ))
        
        return results

    
    def _build_entities(
        self,
        label_ids: List[int],
        scores: List[float],
        offset_mapping: List[Tuple[int, int]],
        text: str
    ) -> List[Dict]:
        """
        BIOタグからエンティティを構築
        
        Returns:
            [{"entity_type": str, "start": int, "end": int, "score": float}, ...]
        """
        entities = []
        current_entity = None
        label_map = self.LABEL_MAPPINGS.get(self.supported_language, {})
        
        for i, (label_id, score, (token_start, token_end)) in enumerate(
            zip(label_ids, scores, offset_mapping)
        ):
            # 特殊トークン([CLS], [SEP], [PAD])をスキップ
            if token_start == token_end == 0:
                continue
            
            label = self._model.config.id2label.get(label_id, "O")
            
            if label.startswith("B-"):
                # 新しいエンティティの開始
                if current_entity:
                    entities.append(current_entity)
                
                entity_type = label_map.get(label, None)
                if entity_type:
                    current_entity = {
                        "entity_type": entity_type,
                        "start": int(token_start),
                        "end": int(token_end),
                        "score": float(score),
                        "token_count": 1
                    }
                else:
                    current_entity = None
            
            elif label.startswith("I-") and current_entity:
                # 既存エンティティの継続
                entity_type = label_map.get(label, None)
                if entity_type == current_entity["entity_type"]:
                    current_entity["end"] = int(token_end)
                    current_entity["score"] = (
                        (current_entity["score"] * current_entity["token_count"] + float(score)) /
                        (current_entity["token_count"] + 1)
                    )
                    current_entity["token_count"] += 1
            
            elif label == "O":
                # エンティティ外
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        # 最後のエンティティを追加
        if current_entity:
            entities.append(current_entity)
        
        return entities
    
    def _build_entities_from_tokens(
        self,
        label_ids: List[int],
        scores: List[float],
        text: str
    ) -> List[Dict]:
        """
        トークンベースでエンティティを構築 (offset_mapping非対応時のフォールバック)
        
        テキスト全体をエンティティとして返す簡易実装
        """
        label_map = self.LABEL_MAPPINGS.get(self.supported_language, {})
        entities = []
        current_entity = None
        
        # トークンを取得
        tokens = self._tokenizer.tokenize(text)
        
        for i, (label_id, score) in enumerate(zip(label_ids[1:-1], scores[1:-1])):  # Skip [CLS] and [SEP]
            if i >= len(tokens):
                break
            
            label = self._model.config.id2label.get(label_id, "O")
            
            if label.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                
                entity_type = label_map.get(label, None)
                if entity_type:
                    current_entity = {
                        "entity_type": entity_type,
                        "tokens": [tokens[i]],
                        "score": float(score),
                        "token_count": 1
                    }
                else:
                    current_entity = None
            
            elif label.startswith("I-") and current_entity:
                entity_type = label_map.get(label, None)
                if entity_type == current_entity["entity_type"]:
                    current_entity["tokens"].append(tokens[i])
                    current_entity["score"] = (
                        (current_entity["score"] * current_entity["token_count"] + float(score)) /
                        (current_entity["token_count"] + 1)
                    )
                    current_entity["token_count"] += 1
            
            elif label == "O":
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        if current_entity:
            entities.append(current_entity)
        
        # Convert token-based entities to character-based  
        result_entities = []
        for entity in entities:
            # Reconstruct the entity text from tokens
            entity_text = self._tokenizer.convert_tokens_to_string(entity["tokens"])
            # Find the entity in the original text
            start_pos = text.find(entity_text)
            if start_pos >= 0:
                result_entities.append({
                    "entity_type": entity["entity_type"],
                    "start": start_pos,
                    "end": start_pos + len(entity_text),
                    "score": entity["score"]
                })
        
        return result_entities



def create_english_transformer_recognizer(
    model_name: str = "dslim/bert-base-NER",
    min_confidence: float = 0.8,
    device: str = "cpu"
) -> TransformerNERRecognizer:
    """
    英語用Transformer認識器
    
    デフォルトモデル: dslim/bert-base-NER
    """
    return TransformerNERRecognizer(
        model_name=model_name,
        supported_language="en",
        supported_entities=["PERSON", "LOCATION", "ORGANIZATION"],
        min_confidence=min_confidence,
        device=device
    )


def create_japanese_transformer_recognizer(
    model_name: str = "knosing/japanese_ner_model",
    tokenizer_name: str = "tohoku-nlp/bert-base-japanese-v3",
    min_confidence: float = 0.8,
    device: str = "cpu"
) -> TransformerNERRecognizer:
    """
    日本語用Transformer認識器
    
    デフォルトモデル: knosing/japanese_ner_model
    トークナイザー: tohoku-nlp/bert-base-japanese-v3
    """
    return TransformerNERRecognizer(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        supported_language="ja",
        supported_entities=["JP_PERSON", "JP_ADDRESS", "JP_ORGANIZATION"],
        min_confidence=min_confidence,
        device=device
    )
