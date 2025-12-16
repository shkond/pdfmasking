"""File input/output utilities.

This package handles document extraction and file processing:
- extractors: PDF and Word document text extraction
- file_processor: Batch file processing
"""

from .extractors import extract_text, extract_text_from_pdf, extract_text_from_docx
from .file_processor import process_file

__all__ = [
    "extract_text",
    "extract_text_from_pdf",
    "extract_text_from_docx",
    "process_file",
]
