"""Text extraction utilities for multiple document formats.

Supports:
- PDF files (.pdf)
- Word documents (.docx)
"""

import os

from pdfminer.high_level import extract_text as pdf_extract_text


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text as string
    """
    return pdf_extract_text(file_path)


def extract_text_from_docx(file_path: str) -> str:
    """
    Extract text from a Word document (.docx).
    
    Args:
        file_path: Path to the Word document
        
    Returns:
        Extracted text as string
    """
    try:
        from docx import Document
    except ImportError:
        raise ImportError(
            "python-docx is required for Word document support. "
            "Install it with: pip install python-docx"
        )

    doc = Document(file_path)

    # Extract text from paragraphs
    paragraphs = [para.text for para in doc.paragraphs]

    # Extract text from tables
    table_text = []
    for table in doc.tables:
        for row in table.rows:
            row_text = [cell.text for cell in row.cells]
            table_text.append(" | ".join(row_text))

    # Combine all text
    all_text = "\n".join(paragraphs + table_text)
    return all_text


def extract_text(file_path: str) -> str:
    """
    Extract text from a document, auto-detecting the format.
    
    Supported formats:
    - PDF (.pdf)
    - Word (.docx)
    
    Args:
        file_path: Path to the document
        
    Returns:
        Extracted text as string
        
    Raises:
        ValueError: If file format is not supported
        FileNotFoundError: If file does not exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    else:
        raise ValueError(
            f"Unsupported file format: {ext}. "
            "Supported formats: .pdf, .docx"
        )
