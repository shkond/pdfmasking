"""Text preprocessing utilities for PDF/document text.

Normalizes extracted text for better PII detection accuracy.
"""

import re


def preprocess_text(text: str) -> str:
    """
    Preprocess text extracted from PDF to normalize whitespace and formatting.
    
    This helps improve context detection by:
    - Collapsing multiple newlines into single spaces
    - Normalizing whitespace
    - Removing extra spaces while preserving structure
    
    Args:
        text: Raw text extracted from PDF
        
    Returns:
        Normalized text with consistent formatting
    """
    # Replace multiple consecutive newlines with a single space
    text = re.sub(r'\n{2,}', ' ', text)
    # Replace single newlines with space
    text = re.sub(r'\n', ' ', text)
    # Collapse multiple spaces into single space
    text = re.sub(r' {2,}', ' ', text)
    # Normalize full-width spaces to regular spaces
    text = text.replace('\u3000', ' ')
    return text.strip()
