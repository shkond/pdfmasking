# 精度判定テスト (Accuracy Tests)

このディレクトリには、PII検出の精度を評価するためのテストが含まれています。

## 概要

精度テストは **CI除外** として設計されています。理由:

1. **実行時間**: 実際のTransformer/GiNZAモデルを使用するため時間がかかる
2. **リソース**: GPUメモリやCPUを大量に消費する
3. **目的**: 日常的なコード変更ではなく、モデル精度の評価が目的

## テストファイル

| ファイル | 内容 |
|---------|------|
| `test_transformer_accuracy.py` | Transformerモデルの検出精度評価 |
| `test_ginza_accuracy.py` | GiNZAモデルの検出精度評価 |
| `test_pdf_accuracy.py` | PDF処理の精度評価 (document.pdf) |

## 実行方法

```bash
# 全精度テスト実行
pytest tests/accuracy -v

# 特定ファイルのみ
pytest tests/accuracy/test_transformer_accuracy.py -v

# マーカーによる選択
pytest -m accuracy -v
```

## 評価指標

精度テストでは以下を評価:

- **検出率 (Recall)**: 期待されるPIIがすべて検出されたか
- **精度 (Precision)**: 誤検出がないか
- **位置精度**: 開始・終了位置が正確か
- **信頼度スコア**: 適切なスコアが付与されているか

## CI除外設定

`pyproject.toml`にてデフォルトで除外:

```toml
[tool.pytest.ini_options]
testpaths = ["tests/unit", "tests/integration"]
```
