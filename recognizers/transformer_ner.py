"""Transformer-based NER recognizers using Hugging Face models."""


try:
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer
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
    
    ラベルマッピングは設定ファイル (config.yaml) から読み込まれます。
    """

    def __init__(
        self,
        model_name: str,
        supported_language: str = "en",
        supported_entities: list[str] | None = None,
        min_confidence: float = 0.8,
        device: str = "cpu",
        tokenizer_name: str | None = None,
        label_mapping: dict[str, str] | None = None
    ):
        """
        Args:
            model_name: Hugging Faceモデル名
            supported_language: "en" or "ja"
            supported_entities: 検出対象エンティティ(None=設定から取得)
            min_confidence: 最小信頼度スコア
            device: "cpu" or "cuda"
            tokenizer_name: トークナイザー名(Noneの場合はmodel_nameと同じ)
            label_mapping: BIOタグ→エンティティタイプのマッピング (config.yamlから渡される)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("torch and transformers are required for TransformerNERRecognizer")

        self.model_name = model_name
        self.tokenizer_name = tokenizer_name or model_name
        self.min_confidence = min_confidence
        self.device = device
        self._model = None
        self._tokenizer = None
        self.label_mapping = label_mapping or {}

        # supported_entities が指定されていない場合はエラー (設定から渡すべき)
        if supported_entities is None:
            raise ValueError("supported_entities must be provided (from config.yaml)")

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
        self, text: str, entities: list[str], nlp_artifacts: NlpArtifacts | None = None
    ) -> list[RecognizerResult]:
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
        label_ids: list[int],
        scores: list[float],
        offset_mapping: list[tuple[int, int]],
        text: str
    ) -> list[dict]:
        """
        BIOタグおよび非BIO形式ラベルからエンティティを構築
        
        対応ラベル形式:
        - BIO形式: B-PER, I-PER, B-LOC, I-LOC など
        - 非BIO形式: 人名, 地名, 法人名 など（日本語モデル用）
        
        Returns:
            [{"entity_type": str, "start": int, "end": int, "score": float}, ...]
        """
        entities = []
        current_entity = None
        label_map = self.label_mapping

        for i, (label_id, score, (token_start, token_end)) in enumerate(
            zip(label_ids, scores, offset_mapping)
        ):
            # 特殊トークン([CLS], [SEP], [PAD])をスキップ
            if token_start == token_end == 0:
                continue

            label = self._model.config.id2label.get(label_id, "O")

            # BIO形式か非BIO形式かを判定
            is_bio_begin = label.startswith("B-")
            is_bio_inside = label.startswith("I-")
            is_outside = label == "O"
            is_non_bio_entity = not is_bio_begin and not is_bio_inside and not is_outside

            if is_bio_begin:
                # BIO形式: 新しいエンティティの開始
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

            elif is_bio_inside and current_entity:
                # BIO形式: 既存エンティティの継続
                entity_type = label_map.get(label, None)
                if entity_type == current_entity["entity_type"]:
                    current_entity["end"] = int(token_end)
                    current_entity["score"] = (
                        (current_entity["score"] * current_entity["token_count"] + float(score)) /
                        (current_entity["token_count"] + 1)
                    )
                    current_entity["token_count"] += 1

            elif is_non_bio_entity:
                # 非BIO形式（日本語モデル: 人名, 地名, etc.）
                entity_type = label_map.get(label, None)
                if entity_type:
                    if current_entity and current_entity["entity_type"] == entity_type:
                        # 同じエンティティタイプ → 継続
                        current_entity["end"] = int(token_end)
                        current_entity["score"] = (
                            (current_entity["score"] * current_entity["token_count"] + float(score)) /
                            (current_entity["token_count"] + 1)
                        )
                        current_entity["token_count"] += 1
                    else:
                        # 異なるエンティティタイプ または 新規 → 新しいエンティティ開始
                        if current_entity:
                            entities.append(current_entity)
                        current_entity = {
                            "entity_type": entity_type,
                            "start": int(token_start),
                            "end": int(token_end),
                            "score": float(score),
                            "token_count": 1
                        }
                else:
                    # マッピングにないラベル → エンティティ終了
                    if current_entity:
                        entities.append(current_entity)
                        current_entity = None

            elif is_outside:
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
        label_ids: list[int],
        scores: list[float],
        text: str
    ) -> list[dict]:
        """
        トークンベースでエンティティを構築 (offset_mapping非対応時のフォールバック)
        
        対応ラベル形式:
        - BIO形式: B-PER, I-PER, B-LOC, I-LOC など
        - 非BIO形式: 人名, 地名, 法人名 など（日本語モデル用）
        """
        label_map = self.label_mapping
        entities = []
        current_entity = None

        # トークンを取得
        tokens = self._tokenizer.tokenize(text)

        for i, (label_id, score) in enumerate(zip(label_ids[1:-1], scores[1:-1])):  # Skip [CLS] and [SEP]
            if i >= len(tokens):
                break

            label = self._model.config.id2label.get(label_id, "O")

            # BIO形式か非BIO形式かを判定
            is_bio_begin = label.startswith("B-")
            is_bio_inside = label.startswith("I-")
            is_outside = label == "O"
            is_non_bio_entity = not is_bio_begin and not is_bio_inside and not is_outside

            if is_bio_begin:
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

            elif is_bio_inside and current_entity:
                entity_type = label_map.get(label, None)
                if entity_type == current_entity["entity_type"]:
                    current_entity["tokens"].append(tokens[i])
                    current_entity["score"] = (
                        (current_entity["score"] * current_entity["token_count"] + float(score)) /
                        (current_entity["token_count"] + 1)
                    )
                    current_entity["token_count"] += 1

            elif is_non_bio_entity:
                # 非BIO形式（日本語モデル: 人名, 地名, etc.）
                entity_type = label_map.get(label, None)
                if entity_type:
                    if current_entity and current_entity["entity_type"] == entity_type:
                        # 同じエンティティタイプ → 継続
                        current_entity["tokens"].append(tokens[i])
                        current_entity["score"] = (
                            (current_entity["score"] * current_entity["token_count"] + float(score)) /
                            (current_entity["token_count"] + 1)
                        )
                        current_entity["token_count"] += 1
                    else:
                        # 異なるエンティティタイプ または 新規 → 新しいエンティティ開始
                        if current_entity:
                            entities.append(current_entity)
                        current_entity = {
                            "entity_type": entity_type,
                            "tokens": [tokens[i]],
                            "score": float(score),
                            "token_count": 1
                        }
                else:
                    if current_entity:
                        entities.append(current_entity)
                        current_entity = None

            elif is_outside:
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
            # For Japanese (and CJK), tokenizer may add spaces between characters
            # Try matching without the spaces first
            start_pos = text.find(entity_text)
            if start_pos < 0:
                # Try without spaces (for Japanese)
                entity_text_no_space = entity_text.replace(" ", "")
                start_pos = text.find(entity_text_no_space)
                if start_pos >= 0:
                    entity_text = entity_text_no_space
            
            if start_pos >= 0:
                result_entities.append({
                    "entity_type": entity["entity_type"],
                    "start": start_pos,
                    "end": start_pos + len(entity_text),
                    "score": entity["score"]
                })

        return result_entities



def create_transformer_recognizer(
    model_config: dict,
    language: str,
    transformer_config: dict,
    model_id: str | None = None
) -> TransformerNERRecognizer:
    """
    設定駆動のTransformer認識器ファクトリー
    
    Args:
        model_config: モデル設定 (model_name, tokenizer_name, entities)
        language: 言語コード ("en" or "ja")
        transformer_config: Transformer全体設定 (min_confidence, device, label_mapping)
        model_id: モデルID（ロギング・識別用、省略可）
        
    Returns:
        設定に基づいたTransformerNERRecognizer
    """
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

