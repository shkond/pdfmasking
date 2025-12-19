# IT技術職履歴書 - 機械学習学習データセット

**作成日**: 2025年12月17日  
**バージョン**: 1.0  
**用途**: 機械学習の個人情報抽出（NER）学習データ

---

## 📊 データセット概要

| 項目 | 内容 |
|------|------|
| **総件数** | 10件 |
| **言語** | 英語 5件、日本語 5件 |
| **職種** | 5種類（各言語で1件ずつ） |
| **企業数** | 20以上（実在企業名使用） |
| **技術スタック** | 32種類以上 |
| **抽出対象フィールド** | 6項目 |

---

## 🎯 個人情報抽出対象フィールド（6項目）

| # | フィールド | 英語例 | 日本語例 | エンティティ型 |
|----|-----------|--------|---------|----------------|
| 1 | 名前 | Jane Smith | 田中太郎 | PERSON_NAME |
| 2 | メールアドレス | jane_smith859@outlook.com | tanaka_taro789@gmail.com | EMAIL |
| 3 | 住所 | 456 Main Ave, New York, NY | 東京都渋谷区1-2-3 | ADDRESS |
| 4 | 郵便番号 | 23434 | 150-0001 | POSTAL_CODE |
| 5 | 電話番号 | +1-892-958-9935 | 090-1234-5678 | PHONE |
| 6 | 生年月日 | 1986-12-14 | 1986-12-14 | DATE_OF_BIRTH |

---

## 📁 生成ファイル（6個）

### 1. **resume_data.json** (13,423 bytes)
10件の履歴書データを構造化JSON形式で含む

**構成**:
- Personal Info (6フィールド)
- Education (大学、学位、卒業年)
- Work Experience (複数社の職歴)
- Technologies (技術スタック)
- Certifications (資格・認定)

### 2. **annotations.json** (12,257 bytes)
アノテーション例 - 各フィールドのエンティティ型情報付き

**特徴**:
- 各フィールドの値とエンティティ型を対応
- 日本語ラベルと英語ラベル併記
- 機械学習訓練用フォーマット

### 3. **normalized_data.json** (13,279 bytes)
正規化済みデータ - 言語別の処理済みデータ

**正規化内容**:

#### 名前 (Name)
- **英語**: First/Last name に分割
- **日本語**: フルネーム、文字数

#### メールアドレス (Email)
- ローカルパート抽出
- ドメイン分割
- TLD (トップレベルドメイン)

#### 住所 (Address)
- **英語**: Street / City / State に分割
- **日本語**: 都道府県・市区町村・番地

#### 郵便番号 (Postal Code)
- **英語**: 5桁形式 (例: 23434)
- **日本語**: 7桁形式 XXX-XXXX (例: 150-0001)
- プレフィックス・サフィックス分割

#### 電話番号 (Phone)
- **英語**: +1-XXX-XXX-XXXX 形式
  - 国番号: +1
  - 市外局番: XXX
  - 番号: XXX-XXXX
- **日本語**: 0XX-XXXX-XXXX 形式
  - 国番号: +81 (正規化後)
  - 市外局番: 0XX
  - 番号: XXXX-XXXX

#### 生年月日 (Date of Birth)
- ISO 8601形式: YYYY-MM-DD
- 年・月・日に分割
- 年齢自動計算
- 日本の年号に変換 (例: 昭和60)

### 4. **generate_pdf.py** (6,357 bytes)
PDF生成スクリプト

**機能**:
- 10件全ての履歴書をPDF形式で生成
- 言語別に適切なページサイズを自動選択
  - 英語: Letter (8.5" × 11")
  - 日本語: A4 (210mm × 297mm)
- `resumes_pdf/` ディレクトリに保存

**依存ライブラリ**: reportlab

**実行方法**:
```bash
pip install reportlab
python generate_pdf.py
```

### 5. **usage_examples.py** (5,529 bytes)
使用例スクリプト - 6つの利用例デモンストレーション

**含まれる例**:
1. 言語別メール抽出
2. 正規化データの処理
3. 職種分布分析
4. 技術スタック分析
5. アノテーション構造の確認
6. 英語・日本語の比較

**実行方法**:
```bash
python usage_examples.py
```

### 6. **README.md** (7,608 bytes)
完全なドキュメント

**内容**:
- データセット概要
- ファイル構成の詳細説明
- 使用方法ガイド
- 統計情報
- ML利用例
- カスタマイズ方法

---

## 👔 職種別分布（5種類、各言語1件ずつ）

### 英語版 (English)
1. **Frontend Engineer** - React, Vue.js, TypeScript, Webpack
2. **Backend Engineer** - Python, Java, FastAPI, Spring Boot
3. **Data Engineer** - Python, Spark, Airflow, SQL
4. **Infrastructure Engineer** - AWS, Kubernetes, Docker, Terraform
5. **Security Engineer** - Linux, Penetration Testing, Cryptography, SIEM

### 日本語版 (Japanese)
1. **フロントエンドエンジニア** - React, Vue.js, TypeScript, Webpack
2. **バックエンドエンジニア** - Python, Java, FastAPI, Spring Boot
3. **データエンジニア** - Python, Spark, Airflow, SQL
4. **インフラエンジニア** - AWS, Kubernetes, Docker, Terraform
5. **セキュリティエンジニア** - Linux, ペネトレーションテスト, 暗号化, SIEM

---

## 💼 企業名（実在企業、架空の従業員）

### 日本企業
Sony, Nintendo, Toyota, Honda, Panasonic, NTT Data, NEC, Fujitsu, IBM Japan, CyberAgent, DeNA, Rakuten, Mercari, SmartNews, Sansan

### グローバル企業
Google, Microsoft, Amazon, Apple, Meta, IBM, Oracle, Cisco, VMware, Atlassian, GitHub, Docker, HashiCorp, Stripe, Shopify

---

## 🎓 大学名（実在大学、架空の卒業生）

### 日本の大学
東京大学, 京都大学, 大阪大学, 東工大, 名古屋大学, 早稲田大学, 慶應義塾大学, 上智大学

### 海外の大学
Stanford University, MIT, UC Berkeley, Carnegie Mellon University, University of Cambridge, University of Oxford, ETH Zurich, NUS

---

## 📈 統計情報

| 指標 | 値 |
|------|-----|
| 総ファイルサイズ | 58,453 bytes |
| 含まれる技術スタック | 32+ 個 |
| 含まれる企業 | 20+ 個 |
| 含まれる大学 | 10+ 個 |
| 平均職歴件数 | 2 件/人 |
| 平均技術スタック数 | 5 個/人 |
| 平均資格数 | 3 個/人 |

---

## 🚀 使用開始ガイド

### ステップ 1: PDF生成
```bash
pip install reportlab
python generate_pdf.py
```
→ `resumes_pdf/` ディレクトリに10個のPDFが生成されます

### ステップ 2: データ構造の確認
```bash
python usage_examples.py
```
→ 6つの利用例が実行され、データ構造を確認できます

### ステップ 3: 機械学習での利用
- `resume_data.json` - テキスト処理、NERタスク用
- `annotations.json` - アノテーション情報、ラベル付け
- `normalized_data.json` - 正規化データ、特徴抽出

---

## 🔍 機械学習での活用例

### 1. 個人情報抽出 (NER)
```
入力: 履歴書テキスト
出力: [PERSON_NAME: "Jane Smith"], [EMAIL: "jane_smith@example.com"], ...
```

### 2. テキスト分類
```
職種分類: Frontend Engineer, Backend Engineer, ...
言語分類: English, Japanese
```

### 3. 個人情報マスキング
```
Presidio, spaCy NER 等を使用した個人情報の自動マスキング
```

### 4. データ標準化
- 電話番号の正規化
- 住所の分解
- 名前の抽出

---

## ⚖️ 注意事項

### 📌 フィクショナルなデータ
- **全て架空の人物です** - 実在しません
- **実名・実住所なし** - 個人を特定できません
- **架空のメールアドレス・電話番号** - 実在しません
- **実在企業・大学の名前を使用** - ただしデータは架空

### 🔒 プライバシー
- **個人情報保護法準拠** - フィクショナルなデータのみ
- **機械学習学習用** - 教育研究目的向け
- **配布制限なし** - 自由に利用・配布可能

---

## 📝 ライセンス

このデータセットはパブリックドメインです。自由に利用、配布、修正できます。

---

## 🔧 カスタマイズ・拡張

### データセット拡張
`resume_data.json` を編集して追加可能:
- 新しい職種 (SRE, DevOps, ML Engineer など)
- 追加の技術スタック
- より詳細な職務経歴
- 他の個人情報フィールド

### 言語追加
`generate_pdf.py` を修正して対応:
- 中国語、韓国語など
- 言語別の正規化ルール

### 正規化ルールのカスタマイズ
`normalized_data.json` の生成ロジックを編集:
- 地域別の住所形式
- 国別の電話番号形式
- 言語別の日付フォーマット

---

## 📧 サポート

このデータセットに関する質問や改善提案については、プロジェクトのDiscussionsをご利用ください。

---

**作成日**: 2025年12月17日  
**最終更新**: 2025年12月17日  
**バージョン**: 1.0  
**ステータス**: ✅ 完成
