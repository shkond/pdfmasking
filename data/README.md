# IT技術職履歴書 - 機械学習学習データセット

機械学習の個人情報抽出タスク用の、IT技術職履歴書データセットです。

## 📊 データセット概要

- **言語**: 英語 5件、日本語 5件（計10件）
- **職種**: フロントエンドエンジニア、バックエンドエンジニア、データエンジニア、インフラエンジニア、セキュリティエンジニア
- **用途**: 機械学習の個人情報抽出（NER: Named Entity Recognition）学習データ
- **特徴**: 実在企業名・大学名を使用したフィクショナルなデータ

## 📁 ファイル構成

### 1. resume_data.json
10件の履歴書データ（構造化JSON形式）

**構造**:
```json
{
  "id": "EN_001",
  "language": "English",
  "job_title": "Frontend Engineer",
  "personal_info": {
    "name": "Jane Smith",
    "email": "jane_smith859@outlook.com",
    "address": "456 Main Ave, New York, NY",
    "postal_code": "23434",
    "phone": "+1-892-958-9935",
    "date_of_birth": "1986-12-14"
  },
  "education": {...},
  "work_experience": [...],
  "technologies": [...],
  "certifications": [...]
}
```

**個人情報フィールド（抽出対象）**:
1. **name** - 氏名（PERSON_NAME）
2. **email** - メールアドレス（EMAIL）
3. **address** - 住所（ADDRESS）
4. **postal_code** - 郵便番号（POSTAL_CODE）
5. **phone** - 電話番号（PHONE）
6. **date_of_birth** - 生年月日（DATE_OF_BIRTH）

### 2. annotations.json
アノテーション例 - 各フィールドのエンティティ型情報付き

**構造**:
```json
{
  "resume_id": "EN_001",
  "language": "English",
  "annotations": {
    "名前 (name)": {
      "value": "Jane Smith",
      "type": "PERSON_NAME",
      "japanese_label": "名前",
      "english_label": "Name"
    },
    ...
  }
}
```

### 3. normalized_data.json
正規化済みデータ - 各フィールドの正規化版と言語別の処理

**各フィールドの正規化内容**:

#### 名前 (Name)
- **英語**: First/Last name に分割
- **日本語**: 文字数カウント、フォーマット識別
- **例**:
  ```json
  "name": {
    "original": "Jane Smith",
    "format": "WESTERN",
    "first_name": "Jane",
    "last_name": "Smith"
  }
  ```

#### メールアドレス (Email)
- ローカルパート抽出
- ドメイン分割
- TLD（トップレベルドメイン）抽出
- **例**:
  ```json
  "email": {
    "original": "jane_smith859@outlook.com",
    "local_part": "jane_smith859",
    "domain": "outlook",
    "tld": "com"
  }
  ```

#### 住所 (Address)
- **英語**: ストリート、シティ、州に分割
- **日本語**: 都道府県、市区町村、番地の構造識別
- **例** (英語):
  ```json
  "address": {
    "original": "456 Main Ave, New York, NY",
    "street": "456 Main Ave",
    "city": "New York",
    "state": "NY"
  }
  ```

#### 郵便番号 (Postal Code)
- フォーマット自動検出
- **英語**: 5桁形式（例: 23434）
- **日本語**: 7桁形式 (XXX-XXXX)（例: 100-0001）
- **例** (日本語):
  ```json
  "postal_code": {
    "original": "150-0001",
    "format": "JAPAN_7_DIGIT",
    "prefix": "150",
    "suffix": "0001"
  }
  ```

#### 電話番号 (Phone)
- 国番号抽出
- 市外局番抽出
- **英語**: +1-XXX-XXX-XXXX 形式
- **日本語**: 0XX-XXXX-XXXX 形式
- **例** (英語):
  ```json
  "phone": {
    "original": "+1-892-958-9935",
    "country_code": "+1",
    "area_code": "892",
    "number": "958-9935"
  }
  ```

#### 生年月日 (Date of Birth)
- ISO 8601形式で標準化
- 年月日分割
- 年齢自動計算
- 日本の年号に変換
- **例**:
  ```json
  "date_of_birth": {
    "original": "1986-12-14",
    "iso_format": "1986-12-14",
    "year": 1986,
    "month": 12,
    "day": 14,
    "age": 39,
    "japanese_era": "昭和60"
  }
  ```

### 4. generate_pdf.py
PDF生成スクリプト

**機能**:
- 全10件の履歴書をPDF形式で生成
- 言語別に適切なページサイズを自動選択
  - 英語: Letter (8.5" x 11")
  - 日本語: A4 (210mm x 297mm)
- `resumes_pdf/` ディレクトリに自動保存

## 🚀 使用方法

### ステップ1: 依存ライブラリのインストール

```bash
pip install reportlab
```

### ステップ2: PDF生成

```bash
python generate_pdf.py
```

### ステップ3: 生成確認

```
resumes_pdf/
├── EN_001_Frontend_Engineer.pdf
├── EN_002_Backend_Engineer.pdf
├── EN_003_Data_Engineer.pdf
├── EN_004_Infrastructure_Engineer.pdf
├── EN_005_Security_Engineer.pdf
├── JP_001_フロントエンドエンジニア.pdf
├── JP_002_バックエンドエンジニア.pdf
├── JP_003_データエンジニア.pdf
├── JP_004_インフラエンジニア.pdf
└── JP_005_セキュリティエンジニア.pdf
```

## 📝 データセット統計

| 言語 | 件数 | 職種 | 企業数 |
|------|------|------|--------|
| English | 5 | 5種類 | 複数 |
| Japanese | 5 | 5種類 | 複数 |
| **合計** | **10** | **5種類** | **20+** |

## 🎯 機械学習での利用例

### 個人情報抽出（NER）

```python
import json
import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding

# Load data
with open('resume_data.json', 'r', encoding='utf-8') as f:
    resumes = json.load(f)

with open('annotations.json', 'r', encoding='utf-8') as f:
    annotations = json.load(f)

# Create training data
training_data = []
for resume, annotation in zip(resumes, annotations):
    # Extract text and entities
    # Create training example
    pass

# Train spaCy NER model
nlp = spacy.load("en_core_web_sm")
# ... training code ...
```

### 正規化データの利用

```python
import json

# Load normalized data
with open('normalized_data.json', 'r', encoding='utf-8') as f:
    normalized = json.load(f)

# Use for validation/standardization
for resume in normalized:
    name_info = resume['normalized_personal_info']['name']
    email_info = resume['normalized_personal_info']['email']
    # ... process normalized data ...
```

## 📊 対応エンティティ型

| タイプ | 説明 | 例 |
|--------|------|-----|
| PERSON_NAME | 個人の氏名 | Jane Smith, 田中太郎 |
| EMAIL | メールアドレス | jane_smith859@outlook.com |
| ADDRESS | 住所 | 456 Main Ave, New York, NY |
| POSTAL_CODE | 郵便番号 | 23434, 150-0001 |
| PHONE | 電話番号 | +1-892-958-9935, 090-1234-5678 |
| DATE_OF_BIRTH | 生年月日 | 1986-12-14 |

## ⚖️ ライセンス・利用規約

このデータセットは機械学習研究・教育目的での使用を想定しています。
全てのデータはフィクショナルです。

## 💡 注意事項

1. **フィクショナルなデータ**: 全て架空の人物です
2. **実名・実住所なし**: 個人情報を特定できません
3. **言語別処理**: 英語と日本語で異なる正規化ルールを適用しています
4. **拡張可能**: 追加の職種やフィールドの拡張が可能です

## 🔧 カスタマイズ

### データセット拡張

`resume_data.json` を編集して以下を追加可能:
- 新しい職種
- 追加の技術スタック
- より詳細な職務経歴

### 正規化ルール変更

`generate_pdf.py` または後処理スクリプトで正規化ルールをカスタマイズ可能

## 📧 サポート

このデータセットに関する質問や改善提案は、プロジェクトの Discussions にてお願いします。

---

**作成日**: 2025年12月17日
**バージョン**: 1.0
