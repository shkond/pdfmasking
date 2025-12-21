# Japanese Resume PII Masking System

日本語履歴書の個人情報マスキングシステム

## 概要

このシステムは、Presidio を中心に Pattern / GiNZA(spaCy) /（任意で）機械学習モデルを組み合わせて、日本語の履歴書（PDF・Word）から個人情報を自動的に検出・マスキングします。

機械学習（ML）モデルは設定で切り替えできます:

- Transformer NER（TokenClassification）
- GPT PII Masker（CausalLM: cameltech/japanese-gpt-1b-PII-masking）

## 対応している個人情報

主に以下を対象にしています（設定で増減可能）:

- **氏名・ふりがな** (`JP_PERSON`)
- **住所** (`JP_ADDRESS`)
- **電話番号** (`PHONE_NUMBER_JP`)
- **郵便番号** (`JP_ZIP_CODE`)
- **生年月日** (`DATE_OF_BIRTH_JP`)
- **メールアドレス** (`EMAIL_ADDRESS`)
- **年齢** (`JP_AGE`)
- **性別** (`JP_GENDER`)

任意で以下も設定に含めています:

- **組織名** (`JP_ORGANIZATION`)
- **顧客ID** (`CUSTOMER_ID_JP`)

## インストール

### 1. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

**注意**: GiNZA のインストールには約500MBのダウンロードが必要で、数分かかる場合があります。

### 2. GiNZA モデルの自動インストール

初回実行時に GiNZA モデルが自動的にダウンロードされます。手動でインストールする場合:

```bash
python -m pip install ginza ja-ginza
```

## 使い方

### 基本的な使い方

```bash
# 日本語 PDF をマスキング
python main.py resume.pdf --lang ja -o output.txt

# Word 文書をマスキング
python main.py resume.docx --lang ja -o output.txt

# 検出された個人情報を表示（verbose モード）
python main.py resume.pdf --lang ja --verbose -o output.txt
```

### コマンドラインオプション

- `input_file`: 入力ファイル（PDF または Word）
- `-o, --output`: 出力ファイルパス（省略時は標準出力）
- `--lang`: 言語コード（`en` または `ja`、デフォルト: `en`）
- `-v, --verbose`: 検出された個人情報を表示

### 使用例

#### 例1: 日本語履歴書のマスキング

```bash
python main.py 履歴書.pdf --lang ja -o masked_resume.txt
```

#### 例2: 検出内容の確認

```bash
python main.py 履歴書.pdf --lang ja --verbose
```

出力例:
```
=== Detected PII Entities ===
1. JP_PERSON: '山田太郎' (score: 0.90)
2. JP_ADDRESS: '東京都渋谷区' (score: 0.90)
3. PHONE_NUMBER_JP: '03-1234-5678' (score: 0.70)
4. JP_ZIP_CODE: '150-0001' (score: 0.60)
5. DATE_OF_BIRTH_JP: '1990/01/01' (score: 0.60)

Total: 5 entities detected
```

#### 例3: Word 文書の処理

```bash
python main.py resume.docx --lang ja -o output.txt
```

## 技術詳細

### アーキテクチャ

詳細は [doc/ARCHITECTURE.md](doc/ARCHITECTURE.md) を参照してください。

```
pdfmasking/
├── main.py                    # CLIエントリーポイント
├── config.yaml                # 設定ファイル
├── config/                    # 設定管理（単一ソース）
│   ├── __init__.py            # load_config, get_* をエクスポート
│   └── loader.py              # 設定読み込み関数
├── core/                      # ドメイン・アプリケーション層
│   ├── masker.py              # Masker (ドメイン層、DI対応)
│   ├── masking_service.py     # MaskingService (アプリケーション層)
│   ├── analyzer.py            # Analyzer作成ファクトリ
│   └── processors/            # テキスト・結果処理
├── file_io/                   # アダプター層
│   ├── extractors.py          # PDF/Word抽出
│   └── file_processor.py      # MaskingServiceへの委譲
├── masking_logging/           # ロギング
├── recognizers/               # PII認識器
│   ├── registry.py            # RecognizerRegistry
│   ├── japanese_patterns.py   # パターンベース
│   ├── japanese_ner.py        # GiNZA NER
│   ├── gpt_pii_masker.py       # GPT PII masker (CausalLM, span recovery)
│   └── transformer_ner.py     # Transformer NER
└── tests/
```

### マスキング対象（8種類）

| エンティティ | コード |
|-------------|--------|
| 名前 | `JP_PERSON`, `PERSON` |
| メールアドレス | `EMAIL_ADDRESS` |
| 郵便番号 | `JP_ZIP_CODE` |
| 電話番号 | `PHONE_NUMBER_JP` |
| 生年月日 | `DATE_OF_BIRTH_JP` |
| 住所 | `JP_ADDRESS`, `LOCATION` |
| 性別 | `JP_GENDER` |
| 年齢 | `JP_AGE` |

※ `JP_ORGANIZATION` / `CUSTOMER_ID_JP` も設定に含めています（必要に応じて有効化）。

### MLモデル切替（Transformer / GPT）

ML経路は `config.yaml` の `transformer.enabled` で有効化されます。
（歴史的命名の都合で `transformer` というキー名ですが、GPT PII masker の有効化もこのフラグで制御されています）

切替は `models.defaults.ja` を変更します:

```yaml
transformer:
	enabled: true

models:
	defaults:
		ja: gpt_pii_masker_ja   # または knosing_ner_ja
```

GPT PII masker は生成系モデルのため、Presidio互換の `start/end` を満たすためにタグ→元文への位置復元を行います。
復元に失敗した候補は誤マスク防止のため破棄し、`masking` ログに理由付きで記録します。

### パターンベース認識器

正規表現とコンテキスト語を使用:

- **電話番号**: `0\d{1,4}-\d{1,4}-\d{4}`, `0[789]0-\d{4}-\d{4}`
- **郵便番号**: `\d{3}-\d{4}`
- **生年月日**: `YYYY/MM/DD`, `YYYY年MM月DD日`, `令和X年X月X日` など

### GiNZA ベース認識器

日本語 NLP による固有表現抽出:

- **氏名**: GiNZA の PERSON エンティティ
- **住所**: GiNZA の LOC エンティティ

コンテキスト語（「氏名」「住所」など）が近くにある場合、信頼度スコアを向上。

### カスタマイズ

`config.yaml` で以下をカスタマイズ可能:

- 信頼度スコアの閾値
- コンテキスト語のリスト
- マスキングパターン（デフォルト: `****`）

## 対応フォーマット

- **PDF**: `.pdf`
- **Word**: `.docx`

## 対応言語

- **日本語**: `--lang ja`
- **英語**: `--lang en`（デフォルト）

## トラブルシューティング

### GiNZA のインストールエラー

```bash
python -m pip install --upgrade pip
python -m pip install ginza ja-ginza
```

### テキストが抽出されない

- PDF が画像ベースの場合、OCR が必要です
- Word 文書が破損していないか確認してください

### 検出精度が低い

- `config.yaml` でコンテキスト語を追加
- 信頼度スコアの閾値を調整

## ライセンス

このプロジェクトは MIT ライセンスの下で公開されています。

## 依存ライブラリ

- presidio-analyzer
- presidio-anonymizer
- pdfminer.six
- python-docx
- spacy
- ginza
- ja-ginza
- torch（ML使用時）
- transformers（ML使用時）
