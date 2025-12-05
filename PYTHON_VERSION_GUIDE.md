# Python 3.14 互換性問題の解決方法

## 問題

Python 3.14では、spaCyとpydantic v1の互換性問題により、Presidioが正常に動作しません。

**エラーメッセージ:**
```
pydantic.v1.errors.ConfigError: unable to infer type for attribute "REGEX"
```

## 推奨される解決策

### 方法1: Python 3.12を使用する（推奨）

1. **Python 3.12のインストール**
   - https://www.python.org/downloads/ から Python 3.12.x をダウンロード
   - インストール時に「Add Python to PATH」にチェック

2. **仮想環境の作成**
   ```bash
   cd c:\Users\kondo\Desktop\pdfmasking
   
   # Python 3.12で仮想環境を作成
   py -3.12 -m venv venv312
   
   # 仮想環境を有効化
   venv312\Scripts\activate
   
   # 依存パッケージをインストール
   pip install presidio-analyzer presidio-anonymizer pdfminer.six python-docx
   ```

3. **バッチ処理の実行**
   ```bash
   # 仮想環境が有効化されていることを確認
   python batch_process.py
   ```

### 方法2: Conda環境を使用する

```bash
# Conda環境を作成（Python 3.12）
conda create -n pdfmasking python=3.12
conda activate pdfmasking

# 依存パッケージをインストール
pip install presidio-analyzer presidio-anonymizer pdfminer.six python-docx

# バッチ処理の実行
cd c:\Users\kondo\Desktop\pdfmasking
python batch_process.py
```

### 方法3: Docker を使用する（上級者向け）

```dockerfile
# Dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install presidio-analyzer presidio-anonymizer pdfminer.six python-docx

COPY . .
CMD ["python", "batch_process.py"]
```

## 使用方法（Python 3.12環境で）

### 基本的な使い方

```bash
# 仮想環境を有効化
venv312\Scripts\activate

# カレントディレクトリのすべてのPDF/Wordファイルを処理
python batch_process.py
```

### 実行例

```
============================================================
PDF/Word Batch PII Masking
============================================================
Directory: C:\Users\kondo\Desktop\pdfmasking
Language:  ja
GiNZA:     Disabled (pattern-only)
============================================================

Found 3 file(s) to process:
  - resume1.pdf
  - resume2.docx
  - 履歴書.pdf

Processing: resume1.pdf... ✓ Saved to resume1.txt
Processing: resume2.docx... ✓ Saved to resume2.txt
Processing: 履歴書.pdf... ✓ Saved to 履歴書.txt

============================================================
Processing complete!
  Success: 3 file(s)
  Errors:  0 file(s)
============================================================
```

## 出力ファイル

各ファイルは元のファイル名で `.txt` 拡張子で保存されます:

- `resume.pdf` → `resume.txt`
- `履歴書.docx` → `履歴書.txt`
- `document.pdf` → `document.txt`

## マスキングされる情報

- ✓ 電話番号 (03-1234-5678, 090-1234-5678)
- ✓ 郵便番号 (150-0001)
- ✓ 生年月日 (1990/01/01, 1990年1月1日, 令和5年12月1日)
- ✓ メールアドレス (example@email.com)

## トラブルシューティング

### Q: Python 3.12が見つからない
A: `py -0` コマンドでインストール済みのPythonバージョンを確認できます

### Q: 仮想環境の有効化ができない
A: PowerShellの実行ポリシーを変更する必要があるかもしれません:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Q: GiNZAを使いたい
A: Python 3.12環境で以下を実行:
```bash
pip install ginza ja-ginza
python batch_process.py --use-ginza
```
（注意: Rustコンパイラが必要です）

## 参考リンク

- Python 3.12 ダウンロード: https://www.python.org/downloads/
- 仮想環境の使い方: https://docs.python.org/ja/3/tutorial/venv.html
- Presidio ドキュメント: https://microsoft.github.io/presidio/
