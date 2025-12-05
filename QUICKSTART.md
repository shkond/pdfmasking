# Quick Start Guide - Japanese Resume PII Masking

## Installation

### For Python 3.11 or 3.12 (Recommended)

```bash
cd c:\Users\kondo\Desktop\pdfmasking
pip install -r requirements.txt
```

### For Python 3.14 (Pattern-Only Mode)

```bash
cd c:\Users\kondo\Desktop\pdfmasking
pip install presidio-analyzer presidio-anonymizer pdfminer.six python-docx
```

## Usage

### Basic Command

```bash
# Mask Japanese PDF resume
python main.py resume.pdf --lang ja -o output.txt

# Mask Word document
python main.py resume.docx --lang ja -o output.txt

# Show what will be masked (verbose)
python main.py resume.pdf --lang ja --verbose
```

### Python 3.14 Workaround

If you're using Python 3.14, you'll need to disable GiNZA:

1. Edit `main.py` line 24:
   ```python
   # Change this:
   analyzer = create_analyzer(language)
   
   # To this:
   analyzer = create_analyzer(language, use_ginza=False)
   ```

2. Run normally:
   ```bash
   python main.py resume.pdf --lang ja -o output.txt
   ```

## What Gets Masked

### With Pattern-Only Mode (Python 3.14)
- ✓ Phone numbers (03-1234-5678, 090-1234-5678)
- ✓ Postal codes (150-0001)
- ✓ Birth dates (1990/01/01, 1990年1月1日, 令和5年12月1日)
- ✓ Email addresses (example@email.com)

### With GiNZA (Python 3.11/3.12)
- ✓ All of the above, plus:
- ✓ Person names (山田太郎)
- ✓ Addresses (東京都渋谷区...)

## Testing

Create a test file `test_resume.txt`:

```
履歴書

氏名: 山田太郎
生年月日: 1990年1月1日
住所: 〒150-0001 東京都渋谷区神宮前1-2-3
TEL: 03-1234-5678
携帯: 090-9876-5432
Email: yamada.taro@example.com
```

Run:
```bash
python main.py test_resume.txt --lang ja --verbose
```

Expected output:
```
=== Detected PII Entities ===
1. DATE_OF_BIRTH_JP: '1990年1月1日' (score: 0.70)
2. JP_ZIP_CODE: '150-0001' (score: 0.60)
3. PHONE_NUMBER_JP: '03-1234-5678' (score: 0.70)
4. PHONE_NUMBER_JP: '090-9876-5432' (score: 0.80)
5. EMAIL_ADDRESS: 'yamada.taro@example.com' (score: 0.90)

Total: 5 entities detected
```

## Troubleshooting

### Error: "unable to infer type for attribute REGEX"
- **Cause**: Python 3.14 incompatibility with spaCy
- **Solution**: Use Python 3.11/3.12 OR use pattern-only mode (see above)

### No entities detected
- Check language is set to `ja`: `--lang ja`
- Try verbose mode to see what's happening: `--verbose`
- Verify text extraction worked (check if output file has content)

### GiNZA installation fails
- **Requires**: Rust compiler for SudachiPy
- **Solution**: Use pattern-only mode OR install Rust from https://rustup.rs

## Next Steps

1. Test with your actual resume files
2. Adjust confidence thresholds in `config.yaml` if needed
3. Add custom context words for your specific resume format
4. Consider downgrading to Python 3.12 for full GiNZA support
