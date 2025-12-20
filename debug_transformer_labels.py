"""Debug script to show raw NER labels from Transformer models."""

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


def analyze_with_raw_labels(text: str, model_name: str = "dslim/bert-base-NER"):
    """
    Show raw NER labels for each token in the text.
    
    Args:
        text: Text to analyze
        model_name: Hugging Face model name
    """
    print(f"Model: {model_name}")
    print(f"Text: {text}")
    print("-" * 70)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    model.eval()

    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        return_offsets_mapping=True,
        padding=False
    )

    offset_mapping = inputs.pop("offset_mapping")[0]
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits[0], dim=-1)

    label_ids = predictions.argmax(dim=-1).cpu().numpy()
    scores = predictions.max(dim=-1).values.cpu().numpy()

    # Print results
    print(f"{'Token':<20} {'Label':<15} {'Score':<10} {'Text':<20}")
    print("=" * 70)

    for i, (token, label_id, score, (start, end)) in enumerate(
        zip(tokens, label_ids, scores, offset_mapping)
    ):
        label = model.config.id2label.get(label_id, "O")
        original_text = text[start:end] if start != end else "[SPECIAL]"

        # Highlight non-O labels
        if label != "O":
            print(f"\033[91m{token:<20} {label:<15} {score:.4f}     {original_text:<20}\033[0m")
        else:
            print(f"{token:<20} {label:<15} {score:.4f}     {original_text:<20}")

    print()


def main():
    # Test with English Resume Sample keywords
    test_texts = [
        "Git Docker Jupyter Notebook Python JavaScript",
        "TARO YAMAMOTO is a software engineer.",
        "Git is a version control system.",
        "Docker containers are useful.",
        "Dean is the head of the department.",
    ]

    for text in test_texts:
        analyze_with_raw_labels(text, "dslim/bert-base-NER")
        print("\n")


if __name__ == "__main__":
    main()
