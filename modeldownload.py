import os

from transformers import AutoModelForTokenClassification, AutoTokenizer

# ダウンロードしたいモデルのリスト
MODELS = [
    "knosing/japanese_ner_model",
    "dslim/bert-base-NER",
]

SAVE_BASE_DIR = "./models"

for model_name in MODELS:
    print(f"\n{'='*60}")
    print(f"Downloading: {model_name}")
    print(f"{'='*60}")

    # モデル名からディレクトリ名を生成
    model_dir = model_name.replace("/", "_")
    save_dir = os.path.join(SAVE_BASE_DIR, model_dir)
    os.makedirs(save_dir, exist_ok=True)

    try:
        # ダウンロード
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)

        # 保存
        tokenizer.save_pretrained(save_dir)
        model.save_pretrained(save_dir)

        print(f"✓ Saved to: {save_dir}")
    except Exception as e:
        print(f"✗ Error: {e}")

print("\n" + "="*60)
print("All downloads completed!")
