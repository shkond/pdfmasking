現在の実装を分析し、TDDのテストリストを作成します。

## 現在実装されている機能の分類

### 1. **Transformer NER Recognizer** (`transformer_ner.py`)
- モデル管理機能
  - モデルの遅延読み込み (`load()`)
  - CPU/GPUデバイス選択
  - Fast/Slow Tokenizer自動切り替え
- エンティティ検出機能
  - BIOタグベースのエンティティ抽出
  - offset_mapping対応による文字位置特定
  - offset_mapping非対応時のフォールバック処理
  - 信頼度スコア計算（平均スコア）
- ラベルマッピング機能
  - 設定ファイル駆動のBIOタグ→エンティティ変換
  - 英語モデル対応 (dslim/bert-base-NER)
  - 日本語モデル対応 (knosing/japanese_ner_model)
- 結果フィルタリング
  - 最小信頼度閾値適用
  - 要求されたエンティティタイプのみ返却

### 2. **GiNZA NER Recognizers** (`japanese_ner.py`)
#### 2-1. GinzaPersonRecognizer
- spaCyのPERSONエンティティ検出
- コンテキストベースの信頼度調整
  - 周辺コンテキストワード検出（"氏名"、"ふりがな"等）
  - コンテキストウィンドウ（前後50文字）
  - コンテキスト発見時の信頼度ブースト (0.6→0.9)

#### 2-2. GinzaAddressRecognizer
- spaCyのLOCエンティティ検出
- コンテキストベースの信頼度調整
  - 周辺コンテキストワード検出（"住所"、"現住所"等）
  - コンテキストウィンドウ（前後50文字）
  - コンテキスト発見時の信頼度ブースト (0.6→0.9)

***

## TDDテストリスト（第1イテレーション用）

### **カテゴリA: Transformer NER単体機能**
```
[ ] A1. モデル読み込み成功（英語モデル）
[ ] A2. モデル読み込み成功（日本語モデル）
[ ] A3. トークナイザーFast/Slowフォールバック
[ ] A4. BIOタグ→エンティティ変換（B-PERからPERSON）
[ ] A5. BIOタグ継続（I-PERでエンティティ連結）
[ ] A6. offset_mappingによる文字位置特定
[ ] A7. offset_mapping非対応時のフォールバック動作
[ ] A8. 信頼度閾値未満のエンティティ除外
[ ] A9. 複数エンティティの平均スコア計算
[ ] A10. サポート外言語のエンティティ無視
[ ] A11. 設定ファイルからのラベルマッピング読み込み
[ ] A12. 特殊トークン（[CLS], [SEP]）のスキップ
[ ] A13. 最大長512トークン制限時の切り捨て
```

### **カテゴリB: GiNZA Recognizer単体機能**
```
[ ] B1. GiNZA PERSONエンティティ検出
[ ] B2. GiNZA LOCエンティティ検出
[ ] B3. コンテキストワード検出による信頼度ブースト（人名）
[ ] B4. コンテキストワード検出による信頼度ブースト（住所）
[ ] B5. コンテキストウィンドウ外の場合の基本信頼度
[ ] B6. 複数コンテキストワードの優先順位
[ ] B7. nlp_artifacts未提供時の空リスト返却
[ ] B8. 要求されていないエンティティタイプの無視
```

### **カテゴリC: 統合テスト（マスキングパイプライン）**
```
[ ] C1. Standard Mode: パターンベースのみでマスキング
[ ] C2. Dual Detection Mode: パターン+Transformer合意でマスキング
[ ] C3. Dual Detection Mode: 片方のみ検出時はマスキングしない
[ ] C4. 日本語8種PII全タイプのマスキング成功
[ ] C5. 英語PII（EMAIL, PERSON）のマスキング成功
[ ] C6. 重複エンティティの重複排除
[ ] C7. 部分重複エンティティのマージ
[ ] C8. ログ出力の正常動作
```

### **カテゴリD: エッジケース・エラーハンドリング**
```
[ ] D1. 空テキスト入力時の処理
[ ] D2. 特殊文字のみのテキスト
[ ] D3. torch/transformers未インストール時のエラーメッセージ
[ ] D4. 無効なモデル名指定時のエラー
[ ] D5. GPU利用可能だがメモリ不足時のフォールバック
[ ] D6. supported_entities未指定時のValueError
[ ] D7. 設定ファイル欠損時のデフォルト動作
```

***

## テスト分離方針

### **精度テスト（カテゴリA, B）**
- 各Recognizerの検出精度を個別にテスト
- モックを使わず実際のモデルで実行
- 疑似コード例:
```python
def test_transformer_ner_person_detection():
    text = "山田太郎さんは東京に住んでいます"
    recognizer = TransformerNERRecognizer(
        model_name="knosing/japanese_ner_model",
        supported_language="ja",
        supported_entities=["PERSON"],
        min_confidence=0.5
    )
    results = recognizer.analyze(text, entities=["PERSON"])
    
    assert len(results) == 1
    assert results[0].entity_type == "PERSON"
    assert text[results[0].start:results[0].end] == "山田太郎"
    assert results[0].score >= 0.5
```

### **統合テスト（カテゴリC）**
- システム全体の動作確認
- マスキング結果の正確性
- 疑似コード例:
```python
def test_dual_detection_mode_masking():
    config = load_config()
    config["transformer"]["require_dual_detection"] = True
    
    text = "氏名: 田中花子\nメール: hanako@example.com"
    masker = PIIMasker(config)
    masked_text = masker.mask_pii_in_text(text)
    
    # パターン+Transformerの両方で検出された場合のみマスク
    assert "田中花子" not in masked_text or "[MASKED_JP_PERSON]" in masked_text
```

### **システム稼働テスト（カテゴリC, D）**
- エラーハンドリング
- ログ出力
- 疑似コード例:
```python
def test_system_health_check():
    masker = PIIMasker()
    health_status = masker.check_dependencies()
    
    assert health_status["torch_available"] is True
    assert health_status["ginza_loaded"] is True
    assert len(health_status["recognizers"]) >= 10
```

***
