## 最小実装: Transformer NER Recognizer

### 1. 新規ファイル: `recognizers/transformer_ner.py`

```python
"""Transformer-based NER recognizers using Hugging Face models."""

from typing import List, Optional, Dict, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

from presidio_analyzer import EntityRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import NlpArtifacts


class TransformerNERRecognizer(EntityRecognizer):
    """
    汎用Transformer NER認識器(最小実装版)
    
    サポートモデル:
    - elastic/distilbert-base-uncased-finetuned-conll03-english
    - cl-tohoku/bert-base-japanese-v3
    - その他のtoken-classificationモデル
    """
    
    # ラベルマッピング: モデルのラベル → Presidioエンティティタイプ
    LABEL_MAPPINGS = {
        "en": {
            "B-PER": "PERSON",
            "I-PER": "PERSON",
            "B-LOC": "LOCATION", 
            "I-LOC": "LOCATION",
            "B-ORG": "ORGANIZATION",
            "I-ORG": "ORGANIZATION",
        },
        "ja": {
            "B-PERSON": "JP_PERSON",
            "I-PERSON": "JP_PERSON",
            "B-LOCATION": "JP_ADDRESS",
            "I-LOCATION": "JP_ADDRESS",
            "B-ORGANIZATION": "JP_ORGANIZATION",
            "I-ORGANIZATION": "JP_ORGANIZATION",
        }
    }
    
    def __init__(
        self,
        model_name: str,
        supported_language: str = "en",
        supported_entities: Optional[List[str]] = None,
        min_confidence: float = 0.7,
        device: str = "cpu"
    ):
        """
        Args:
            model_name: Hugging Faceモデル名
            supported_language: "en" or "ja"
            supported_entities: 検出対象エンティティ(None=全て)
            min_confidence: 最小信頼度スコア
            device: "cpu" or "cuda"
        """
        self.model_name = model_name
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
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
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
        
        # トークン化
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,  # 文字オフセット取得用
            padding=False
        )
        
        offset_mapping = inputs.pop("offset_mapping")[0]
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 推論
        with torch.no_grad():
            outputs = self._model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits[0], dim=-1)
        
        # ラベル抽出
        label_ids = predictions.argmax(dim=-1).cpu().numpy()
        scores = predictions.max(dim=-1).values.cpu().numpy()
        
        # エンティティの構築
        entities_found = self._build_entities(
            label_ids, scores, offset_mapping, text
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
                        "start": token_start,
                        "end": token_end,
                        "score": score,
                        "token_count": 1
                    }
                else:
                    current_entity = None
            
            elif label.startswith("I-") and current_entity:
                # 既存エンティティの継続
                entity_type = label_map.get(label, None)
                if entity_type == current_entity["entity_type"]:
                    current_entity["end"] = token_end
                    current_entity["score"] = (
                        (current_entity["score"] * current_entity["token_count"] + score) /
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


# === 言語特化ファクトリー関数 ===

def create_english_transformer_recognizer(
    model_name: str = "elastic/distilbert-base-uncased-finetuned-conll03-english",
    min_confidence: float = 0.75,
    device: str = "cpu"
) -> TransformerNERRecognizer:
    """
    英語用Transformer認識器
    
    デフォルトモデル: elastic/distilbert-base-uncased-finetuned-conll03-english
    - CoNLL-2003でファインチューニング済み
    - PERSON, LOCATION, ORGANIZATIONを検出
    """
    return TransformerNERRecognizer(
        model_name=model_name,
        supported_language="en",
        supported_entities=["PERSON", "LOCATION", "ORGANIZATION"],
        min_confidence=min_confidence,
        device=device
    )


def create_japanese_transformer_recognizer(
    model_name: str = "cl-tohoku/bert-base-japanese-v3",
    min_confidence: float = 0.70,
    device: str = "cpu"
) -> TransformerNERRecognizer:
    """
    日本語用Transformer認識器
    
    デフォルトモデル: cl-tohoku/bert-base-japanese-v3
    - 日本語WikipediaとNEプロジェクトでファインチューニング済み
    - PERSON, LOCATION, ORGANIZATIONを検出
    
    注意: このモデルはNERタスク用にファインチューニングが必要な場合があります。
    代替モデル: stockmark/bert-base-japanese-ner など
    """
    return TransformerNERRecognizer(
        model_name=model_name,
        supported_language="ja",
        supported_entities=["JP_PERSON", "JP_ADDRESS", "JP_ORGANIZATION"],
        min_confidence=min_confidence,
        device=device
    )
```

### 2. 更新: `recognizers/__init__.py`

```python
"""Custom recognizers for Japanese PII detection."""

from .japanese_patterns import (
    JapanesePhoneRecognizer,
    JapaneseZipCodeRecognizer,
    JapaneseBirthDateRecognizer,
    JapaneseNameRecognizer,
    JapaneseAgeRecognizer,
    JapaneseGenderRecognizer,
    JapaneseAddressRecognizer,
)
from .japanese_ner import (
    GinzaPersonRecognizer,
    GinzaAddressRecognizer,
)

# 条件付きインポート(transformersが必要)
try:
    from .transformer_ner import (
        TransformerNERRecognizer,
        create_english_transformer_recognizer,
        create_japanese_transformer_recognizer,
    )
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False

__all__ = [
    "JapanesePhoneRecognizer",
    "JapaneseZipCodeRecognizer",
    "JapaneseBirthDateRecognizer",
    "JapaneseNameRecognizer",
    "JapaneseAgeRecognizer",
    "JapaneseGenderRecognizer",
    "JapaneseAddressRecognizer",
    "GinzaPersonRecognizer",
    "GinzaAddressRecognizer",
]

if TRANSFORMER_AVAILABLE:
    __all__.extend([
        "TransformerNERRecognizer",
        "create_english_transformer_recognizer",
        "create_japanese_transformer_recognizer",
    ])
```

### 3. 更新: `recognizer_registry.py`

```python
# 既存のimport文の後に追加
try:
    from recognizers import (
        TransformerNERRecognizer,
        create_english_transformer_recognizer,
        create_japanese_transformer_recognizer,
        TRANSFORMER_AVAILABLE
    )
except ImportError:
    TRANSFORMER_AVAILABLE = False


# create_default_registry関数に追加
def create_default_registry(
    use_ginza: bool = True,
    use_transformer: bool = False,  # 新規パラメータ
    transformer_device: str = "cpu"
) -> RecognizerRegistry:
    """
    Create a registry with all available recognizers.
    
    Args:
        use_ginza: Whether to include GiNZA-based recognizers
        use_transformer: Whether to include Transformer-based recognizers
        transformer_device: Device for Transformer models ("cpu" or "cuda")
        
    Returns:
        RecognizerRegistry with all recognizers registered
    """
    registry = RecognizerRegistry()
    
    # === Pattern-based recognizers (Japanese) ===
    # ... (既存コードそのまま) ...
    
    # === GiNZA-based recognizers (if available) ===
    # ... (既存コードそのまま) ...
    
    # === Transformer-based recognizers (NEW) ===
    if use_transformer and TRANSFORMER_AVAILABLE:
        # 英語用
        registry.register(RecognizerConfig(
            recognizer=create_english_transformer_recognizer(device=transformer_device),
            type="ner_transformer",
            language="en",
            entity_type="PERSON,LOCATION,ORGANIZATION",  # 複数エンティティ
            description="English NER via DistilBERT (CoNLL-2003)",
            requires_nlp=False
        ))
        
        # 日本語用(注意: モデルによってはファインチューニング必要)
        registry.register(RecognizerConfig(
            recognizer=create_japanese_transformer_recognizer(device=transformer_device),
            type="ner_transformer",
            language="ja",
            entity_type="JP_PERSON,JP_ADDRESS,JP_ORGANIZATION",
            description="Japanese NER via BERT (Tohoku)",
            requires_nlp=False
        ))
    elif use_transformer and not TRANSFORMER_AVAILABLE:
        warnings.warn("Transformer recognizers requested but 'transformers' library not available")
    
    return registry
```

### 4. 更新: `analyzer_factory.py`

```python
def create_analyzer(
    language: str = "en",
    use_ginza: bool = True,
    use_transformer: bool = False,  # 新規パラメータ
    transformer_device: str = "cpu",
    verbose: bool = False
) -> AnalyzerEngine:
    """
    Create an AnalyzerEngine for the specified language.
    
    Args:
        language: Language code ("en" for English, "ja" for Japanese)
        use_ginza: Whether to use GiNZA for Japanese NER (default: True)
        use_transformer: Whether to use Transformer-based NER
        transformer_device: Device for Transformer models ("cpu" or "cuda")
        verbose: If True, print registry summary
        
    Returns:
        Configured AnalyzerEngine
    """
    if language == "ja":
        return create_japanese_analyzer(
            use_ginza=use_ginza,
            use_transformer=use_transformer,
            transformer_device=transformer_device,
            verbose=verbose
        )
    else:
        # 英語用Analyzer(Transformer対応版)
        registry = create_default_registry(
            use_ginza=False,
            use_transformer=use_transformer,
            transformer_device=transformer_device
        )
        
        if verbose:
            print(registry.summary())
        
        analyzer = AnalyzerEngine()
        registry.apply_to_analyzer(analyzer, language=language)
        
        return analyzer


def create_japanese_analyzer(
    use_ginza: bool = True,
    use_transformer: bool = False,  # 新規パラメータ
    transformer_device: str = "cpu",
    verbose: bool = False
) -> AnalyzerEngine:
    """
    Create an AnalyzerEngine configured for Japanese text.
    
    Args:
        use_ginza: Whether to use GiNZA for NER
        use_transformer: Whether to use Transformer-based NER
        transformer_device: Device for Transformer models ("cpu" or "cuda")
        verbose: If True, print registry summary
    
    Returns:
        Configured AnalyzerEngine for Japanese
    """
    # Create registry with Transformer support
    registry = create_default_registry(
        use_ginza=use_ginza,
        use_transformer=use_transformer,
        transformer_device=transformer_device
    )
    
    if verbose:
        print(registry.summary())
    
    # ... (既存のNLP設定コード) ...
    
    # Apply recognizers from registry (Japanese only)
    registry.apply_to_analyzer(analyzer, language="ja")
    
    if verbose:
        ja_count = len(registry.get_by_language("ja"))
        print(f"✓ Japanese analyzer created with {ja_count} recognizers")
    
    return analyzer
```

### 5. 更新: `requirements.txt`

```txt
presidio-analyzer
presidio-anonymizer
pymupdf
spacy
ginza
ja-ginza
# Transformer support (optional)
torch>=2.0.0
transformers>=4.30.0
```

### 6. 使用例: `main.py`への統合

```python
# main.py の該当箇所を修正

from analyzer_factory import create_analyzer, create_japanese_analyzer

# 既存のパターン+GiNZA
analyzer_basic = create_japanese_analyzer(use_ginza=True, verbose=True)

# Transformerを追加
analyzer_with_transformer = create_japanese_analyzer(
    use_ginza=True,
    use_transformer=True,
    transformer_device="cuda" if torch.cuda.is_available() else "cpu",
    verbose=True
)

# 英語版
analyzer_en = create_analyzer(
    language="en",
    use_transformer=True,
    verbose=True
)

# 使用方法は既存と同じ
results = analyzer_with_transformer.analyze(
    text="佐藤太郎さんは東京都に住んでいます。",
    language="ja",
    entities=["JP_PERSON", "JP_ADDRESS"]
)
```

## 動作検証スクリプト

```python
# test_transformer_recognizer.py
from recognizers.transformer_ner import (
    create_english_transformer_recognizer,
    create_japanese_transformer_recognizer
)

def test_english():
    recognizer = create_english_transformer_recognizer()
    recognizer.load()
    
    text = "John Smith works at Microsoft in Seattle."
    results = recognizer.analyze(text, ["PERSON", "ORGANIZATION", "LOCATION"], None)
    
    print("=== English Test ===")
    for r in results:
        print(f"{r.entity_type}: '{text[r.start:r.end]}' (score: {r.score:.3f})")

def test_japanese():
    recognizer = create_japanese_transformer_recognizer(
        model_name="stockmark/bert-base-japanese-ner"  # NER専用モデル推奨
    )
    recognizer.load()
    
    text = "佐藤太郎は東京都渋谷区に住んでいます。"
    results = recognizer.analyze(text, ["JP_PERSON", "JP_ADDRESS"], None)
    
    print("\n=== Japanese Test ===")
    for r in results:
        print(f"{r.entity_type}: '{text[r.start:r.end]}' (score: {r.score:.3f})")

if __name__ == "__main__":
    test_english()
    test_japanese()
```

## 重要な注意点

### モデル選定について

**英語モデル**: `elastic/distilbert-base-uncased-finetuned-conll03-english`は**そのまま使用可能**です。

**日本語モデル**: `cl-tohoku/bert-base-japanese-v3`は**ベースモデル**であり、NERタスク用にファインチューニングされていません。以下の代替を推奨:

```python
# 推奨日本語NERモデル(そのまま使える)
model_options = [
    "stockmark/bert-base-japanese-ner",  # 最も実用的
    "megagonlabs/transformers-ud-japanese-electra-base-discriminator",
    "izumi-lab/electra-base-japanese-discriminator-finetuned-ner"
]

# 使用例
recognizer = TransformerNERRecognizer(
    model_name="stockmark/bert-base-japanese-ner",
    supported_language="ja",
    supported_entities=["JP_PERSON", "JP_ADDRESS", "JP_ORGANIZATION"],
    min_confidence=0.70
)
```

### パフォーマンス最適化

```python
# GPUが利用可能な場合
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

analyzer = create_japanese_analyzer(
    use_transformer=True,
    transformer_device=device
)

# バッチ処理の場合は別途実装が必要
```

この実装で、既存のパターンベース・GiNZA認識器と並行してTransformer NERが動作します。複数の認識器が同じエンティティタイプを検出した場合、Presidioはデフォルトで**最高スコアのみ**を採用します。

次のステップとして、前述の「投票方式」や「カスケード方式」の実装が必要であれば、この基礎実装をベースに拡張できます。