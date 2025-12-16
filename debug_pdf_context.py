"""Debug to check exact context where Git/Docker appear in PDF"""

import sys

sys.path.insert(0, '/workspaces/pdfmasking')

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from document_extractors import extract_text


def analyze_pdf_context():
    """Check the actual context where Git and Docker appear in the PDF"""

    pdf_path = "/workspaces/pdfmasking/English Resume Sample.pdf"
    text = extract_text(pdf_path)

    print("=" * 80)
    print("Checking context where Git and Docker appear in PDF")
    print("=" * 80)

    # Find Git occurrences
    import re
    git_matches = list(re.finditer(r'.{30}Git.{30}', text, re.DOTALL))
    docker_matches = list(re.finditer(r'.{30}Docker.{30}', text, re.DOTALL))

    print("\n--- Git contexts in PDF ---")
    for i, match in enumerate(git_matches):
        context = match.group(0).replace('\n', '\\n')
        print(f"{i+1}. \"{context}\"")
        print(f"   Position: {match.start()} - {match.end()}")

    print("\n--- Docker contexts in PDF ---")
    for i, match in enumerate(docker_matches):
        context = match.group(0).replace('\n', '\\n')
        print(f"{i+1}. \"{context}\"")
        print(f"   Position: {match.start()} - {match.end()}")

    # Now test these exact contexts with Transformer
    print("\n" + "=" * 80)
    print("Testing exact PDF contexts with Transformer")
    print("=" * 80)

    model_name = "dslim/bert-base-NER"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    model.eval()

    # Extract exact context around first Git occurrence
    if git_matches:
        git_context = git_matches[0].group(0)
        print("\n--- Testing exact Git context ---")
        print(f"Context: \"{git_context.replace(chr(10), ' ')}\"")
        analyze_text_with_labels(git_context, tokenizer, model)

    if docker_matches:
        docker_context = docker_matches[0].group(0)
        print("\n--- Testing exact Docker context ---")
        print(f"Context: \"{docker_context.replace(chr(10), ' ')}\"")
        analyze_text_with_labels(docker_context, tokenizer, model)

    # Also test a clean version without newlines
    print("\n" + "=" * 80)
    print("Testing with clean text (newlines replaced with spaces)")
    print("=" * 80)

    if git_matches:
        git_clean = git_matches[0].group(0).replace('\n', ' ')
        print("\n--- Clean Git context ---")
        print(f"Context: \"{git_clean}\"")
        analyze_text_with_labels(git_clean, tokenizer, model)


def analyze_text_with_labels(text, tokenizer, model):
    """Show raw NER labels for the text"""
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

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits[0], dim=-1)

    label_ids = predictions.argmax(dim=-1).cpu().numpy()
    scores = predictions.max(dim=-1).values.cpu().numpy()

    print(f"{'Token':<15} {'Label':<12} {'Score':<8} {'Original':<20}")
    print("-" * 60)

    for token, label_id, score, (start, end) in zip(tokens, label_ids, scores, offset_mapping):
        label = model.config.id2label.get(label_id, "O")
        original_text = text[start:end] if start != end else "[SPEC]"

        # Only show non-O or Git/Docker related tokens
        if label != "O" or "git" in token.lower() or "dock" in token.lower():
            print(f"\033[91m{token:<15} {label:<12} {score:.4f}   {original_text:<20}\033[0m")


if __name__ == "__main__":
    analyze_pdf_context()
