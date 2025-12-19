"""Structure restoration for PDF-extracted text.

Reconstructs text into lines, blocks, and sections for improved PII detection.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from config import load_config


@dataclass
class TextSegment:
    """Represents a text segment with section context."""
    section_id: str               # Unique section identifier
    section_type: str             # contact/education/experience/skills/unknown
    line_text: str                # Line text content
    char_start: int               # Start position in original text
    char_end: int                 # End position in original text
    line_number: int              # Line number within document


@dataclass
class SectionHeading:
    """Detected section heading information."""
    heading_text: str             # Original heading text
    section_type: str             # Classified section type
    char_start: int               # Start position
    char_end: int                 # End position
    line_number: int              # Line number


class StructureRestorer:
    """Restores document structure from raw extracted text."""
    
    # Default section headings (can be overridden by config)
    DEFAULT_SECTION_HEADINGS = {
        "contact": {
            "ja": ["連絡先", "住所", "基本情報", "個人情報"],
            "en": ["Contact", "Personal", "Personal Information"]
        },
        "education": {
            "ja": ["学歴"],
            "en": ["Education", "Academic", "EDUCATION"]
        },
        "experience": {
            "ja": ["職歴", "職務経歴", "経歴", "職務経歴書サマリー"],
            "en": ["Experience", "Work Experience", "Employment", 
                   "WORK EXPERIENCE", "PROFESSIONAL SUMMARY"]
        },
        "skills": {
            "ja": ["スキル", "技術", "技術スキル"],
            "en": ["Skills", "Technical Skills", "TECHNICAL SKILLS"]
        },
        "certifications": {
            "ja": ["資格", "認定"],
            "en": ["Certifications", "Licenses", "CERTIFICATIONS"]
        }
    }
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize structure restorer.
        
        Args:
            config: Configuration dict. If None, loads from config.yaml.
        """
        if config is None:
            config = load_config()
        
        self.config = config
        self.section_headings = self._load_section_headings()
    
    def _load_section_headings(self) -> Dict[str, Dict[str, List[str]]]:
        """Load section headings from config or use defaults."""
        structure_config = self.config.get("structure_restoration", {})
        headings = structure_config.get("section_headings", None)
        
        if headings is None:
            return self.DEFAULT_SECTION_HEADINGS
        
        return headings
    
    def restore(self, raw_text: str) -> List[TextSegment]:
        """Restore structure from raw text.
        
        Processes text through:
        1. Whitespace normalization
        2. Line splitting
        3. Section detection
        4. Segment creation with section context
        
        Args:
            raw_text: Raw text extracted from PDF/DOCX
            
        Returns:
            List of TextSegment objects with section context
        """
        # Step 1: Normalize whitespace while preserving structure
        normalized_text = self._normalize_text(raw_text)
        
        # Step 2: Split into lines
        lines = self._split_into_lines(normalized_text)
        
        # Step 3: Detect section headings
        section_boundaries = self._detect_sections(lines, normalized_text)
        
        # Step 4: Create segments with section context
        segments = self._create_segments(lines, section_boundaries, normalized_text)
        
        return segments
    
    def _normalize_text(self, text: str) -> str:
        """Normalize whitespace and formatting.
        
        Args:
            text: Raw text
            
        Returns:
            Normalized text with consistent formatting
        """
        # Normalize full-width spaces to regular spaces
        text = text.replace('\u3000', ' ')
        
        # Collapse multiple spaces into single space (but preserve newlines)
        text = re.sub(r'[ \t]{2,}', ' ', text)
        
        # Normalize multiple newlines to double newline (paragraph break)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def _split_into_lines(self, text: str) -> List[str]:
        """Split text into lines.
        
        Args:
            text: Normalized text
            
        Returns:
            List of non-empty lines
        """
        lines = text.split('\n')
        return [line.strip() for line in lines if line.strip()]
    
    def _detect_sections(
        self, 
        lines: List[str], 
        full_text: str
    ) -> List[SectionHeading]:
        """Detect section headings in text.
        
        Args:
            lines: List of text lines
            full_text: Full normalized text
            
        Returns:
            List of detected section headings sorted by position
        """
        headings = []
        current_pos = 0
        
        for line_num, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Find position of this line in full text
            line_start = full_text.find(line_stripped, current_pos)
            if line_start == -1:
                continue
            line_end = line_start + len(line_stripped)
            
            # Check if this line is a section heading
            section_type = self._classify_heading(line_stripped)
            if section_type:
                headings.append(SectionHeading(
                    heading_text=line_stripped,
                    section_type=section_type,
                    char_start=line_start,
                    char_end=line_end,
                    line_number=line_num
                ))
            
            current_pos = line_end
        
        return headings
    
    def _classify_heading(self, text: str) -> Optional[str]:
        """Classify text as a section heading type.
        
        Args:
            text: Line text to check
            
        Returns:
            Section type if heading detected, None otherwise
        """
        text_lower = text.lower().strip()
        text_stripped = text.strip()
        
        for section_type, lang_headings in self.section_headings.items():
            # Check Japanese headings
            for heading in lang_headings.get("ja", []):
                if heading in text_stripped:
                    return section_type
            
            # Check English headings (case-insensitive for lowercase check)
            for heading in lang_headings.get("en", []):
                if heading.lower() in text_lower or heading == text_stripped:
                    return section_type
        
        return None
    
    def _create_segments(
        self,
        lines: List[str],
        section_boundaries: List[SectionHeading],
        full_text: str
    ) -> List[TextSegment]:
        """Create text segments with section context.
        
        Args:
            lines: List of text lines
            section_boundaries: Detected section headings
            full_text: Full normalized text
            
        Returns:
            List of TextSegment objects
        """
        segments = []
        current_pos = 0
        
        # Determine section for each line
        current_section_type = "header"  # Default for content before first heading
        current_section_id = "section_0"
        section_count = 0
        
        for line_num, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            # Find position of this line in full text
            line_start = full_text.find(line_stripped, current_pos)
            if line_start == -1:
                continue
            line_end = line_start + len(line_stripped)
            
            # Check if this line is a section heading
            for heading in section_boundaries:
                if heading.line_number == line_num:
                    section_count += 1
                    current_section_type = heading.section_type
                    current_section_id = f"section_{section_count}"
                    break
            
            segments.append(TextSegment(
                section_id=current_section_id,
                section_type=current_section_type,
                line_text=line_stripped,
                char_start=line_start,
                char_end=line_end,
                line_number=line_num
            ))
            
            current_pos = line_end
        
        return segments
    
    def get_section_text(
        self, 
        segments: List[TextSegment], 
        section_type: str
    ) -> str:
        """Get combined text for a specific section type.
        
        Args:
            segments: List of text segments
            section_type: Section type to extract (e.g., 'contact', 'education')
            
        Returns:
            Combined text from all matching sections
        """
        matching_segments = [s for s in segments if s.section_type == section_type]
        return '\n'.join(s.line_text for s in matching_segments)
    
    def get_sections_summary(self, segments: List[TextSegment]) -> Dict[str, int]:
        """Get summary of detected sections.
        
        Args:
            segments: List of text segments
            
        Returns:
            Dict mapping section type to line count
        """
        summary = {}
        for segment in segments:
            section_type = segment.section_type
            summary[section_type] = summary.get(section_type, 0) + 1
        return summary
