# Transformer NER デバッグログ

このドキュメントは、Transformer NER をPresidio Analyzerに統合する際に発見した問題点、原因特定のプロセス、および解決策を記録します。

---

## 問題1: Transformer が Git/Docker を PERSON として検出

### 症状
- `dslim/bert-base-NER` モデルを使用した Dual Detection で、Git や Docker が PERSON としてマスキングされる
- ログには `pattern: PERSON=0.85, transformer: PERSON=0.85` と表示される

### 調査プロセス

#### Step 1: Transformer の生ラベル確認
`debug_transformer_labels.py` を作成し、Transformer モデルの生出力を確認:

```python
# 結果: 文脈がある場合
"Git is a version control system." → Git は "O" (非固有表現) ✅
"Docker containers are useful."   → Docker は "O" (非固有表現) ✅

# 結果: PDF のスキルリストコンテキスト
"Microsoft Azure, Docker, GitHub" → Docker は "B-ORG" (組織) ✅
```

**発見**: Transformer は正しく Docker を ORGANIZATION として認識している！

#### Step 2: Presidio 経由の結果確認
`debug_dual_detection.py` で Pattern と Transformer の結果を分離して確認:

```
Pattern認識器 (SpaCy en_core_web_lg):
  [PERSON] "Docker" score=0.85

Transformer認識器:
  [PERSON] "Docker" score=0.85  ← なぜ PERSON になる？
```

**発見**: Transformer も PERSON を返している。矛盾がある！

#### Step 3: AnalyzerEngine のデフォルト認識器問題
```python
empty_registry = PresidioRegistry()
empty_registry.recognizers = []  # 空にしたはず

transformer_analyzer = AnalyzerEngine(registry=empty_registry)
print(len(transformer_analyzer.registry.recognizers))  # → 25件！
```

**発見**: AnalyzerEngine は空の registry を渡しても、起動時にデフォルト認識器（SpacyRecognizer 含む）を自動追加する！

### 根本原因
1. `AnalyzerEngine` 作成時に SpacyRecognizer が自動的に追加される
2. SpacyRecognizer が Docker を PERSON として検出
3. Transformer の結果（ORGANIZATION）と SpaCy の結果（PERSON）が混在
4. 位置が重なるため、Pattern(PERSON) と Transformer経由SpaCy(PERSON) が一致と判定される

### 解決策
Transformer 認識器を直接呼び出し、AnalyzerEngine を経由しない:

```python
# 修正前: AnalyzerEngine 経由（デフォルト認識器が混入）
transformer_analyzer = AnalyzerEngine(registry=empty_registry)
results = transformer_analyzer.analyze(text=text, entities=entities)

# 修正後: 直接呼び出し（クリーンな結果）
transformer_recognizers = [c.recognizer for c in registry.configs if c.type == "ner_transformer"]
transformer_results = []
for recognizer in transformer_recognizers:
    results = recognizer.analyze(text=text, entities=entities)
    transformer_results.extend(results)
```

---

## 問題2: entity_type マッチングの欠如

### 症状
- Pattern が PERSON、Transformer が ORGANIZATION として検出しても、両方一致と判定
- 位置の重なりだけでマッチング判定していた

### 根本原因
```python
# 問題のあるコード
if min_span_len > 0 and len(overlap) >= min_span_len * 0.5:
    # 位置重なりのみ確認、entity_type は無視
    confirmed_results.append(p_result)
```

### 解決策
entity_type の正規化と一致確認を追加:

```python
def _normalize_entity_type(entity_type: str) -> str:
    """エンティティタイプを共通カテゴリに正規化"""
    type_mapping = {
        "PERSON": "PERSON", "PER": "PERSON", "JP_PERSON": "PERSON",
        "LOCATION": "LOCATION", "LOC": "LOCATION", "JP_ADDRESS": "LOCATION",
        "ORGANIZATION": "ORGANIZATION", "ORG": "ORGANIZATION",
    }
    return type_mapping.get(entity_type, entity_type)

# マッチング時
p_type_normalized = _normalize_entity_type(p_result.entity_type)
t_type_normalized = _normalize_entity_type(t_result.entity_type)

if p_type_normalized == t_type_normalized:
    # 位置と entity_type の両方が一致した場合のみマッチ
    confirmed_results.append(p_result)
```

---

## 問題3: ENTITIES_TO_MASK に Transformer のエンティティタイプがない

### 症状
- Transformer が ORGANIZATION を検出しても、analyze() の結果に含まれない

### 根本原因
```python
ENTITIES_TO_MASK = [
    "PERSON",
    "PHONE_NUMBER",
    # ... ORGANIZATION が含まれていない！
]
```

`analyzer.analyze(entities=ENTITIES_TO_MASK)` で ORGANIZATION をリクエストしていない。

### 解決策
```python
ENTITIES_TO_MASK = [
    "PERSON",
    "LOCATION",       # 追加
    "ORGANIZATION",   # 追加
    # ...
]
```

---

## 問題4: 日本語トークナイザーの offset_mapping エラー

### 症状
```
return_offset_mapping is not available when using Python tokenizers
```

### 根本原因
- `knosing/japanese_ner_model` は slow tokenizer のみ提供
- Slow tokenizer は `return_offsets_mapping=True` をサポートしない

### 解決策
1. Fast tokenizer を優先してロード
2. フォールバックとして slow tokenizer + `_build_entities_from_tokens()` を実装

```python
def load(self):
    try:
        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, use_fast=True  # Fast を優先
        )
    except Exception:
        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, use_fast=False  # Fallback
        )
```

---

## デバッグに有用だったスクリプト

| スクリプト | 目的 |
|-----------|------|
| `debug_transformer_labels.py` | Transformer モデルの生ラベル出力確認 |
| `debug_dual_detection.py` | Pattern/Transformer の分離結果確認 |
| `debug_pdf_context.py` | PDF内の実際のコンテキストでのラベル確認 |

---

## 学んだ教訓

1. **PresidioのAnalyzerEngineはデフォルト認識器を自動追加する** - registry=None でも空にならない
2. **生のモデル出力と Presidio 経由の出力は異なる可能性がある** - 常に両方確認
3. **entity_type のマッチングは明示的に行う必要がある** - 位置重なりだけでは不十分
4. **ENTITIES_TO_MASK にないエンティティタイプは検出されない** - analyze() のフィルタリング
5. **日本語Tokenizerは Fast/Slow で機能が異なる** - offset_mapping のサポート有無

---

## 最終結果

| 項目 | 修正前 | 修正後 |
|-----|-------|-------|
| English Resume マスキング数 | 82 件 | 8 件 |
| Docker/Git 検出 | PERSON として誤検出 | 正しく除外 |
| entity_type 確認 | なし（位置のみ） | 必須 |
