"""Allow list management for excluding terms from PII detection.

This module parses dictionary files (.dic format) and provides
allow lists to Presidio analyzer to prevent false positives.
"""

import re
from pathlib import Path
from typing import Any


def parse_dictionary(path: str | Path) -> list[str]:
    """
    Parse a .dic dictionary file and extract all terms including aliases.
    
    Format supported:
    - Simple term: "Python"
    - With alias: "AI/alias[AI|Artificial Intelligence]"
    - With js suffix: "node.js/js[node]"
    
    Args:
        path: Path to the dictionary file
        
    Returns:
        List of all terms (main terms and aliases)
    """
    terms = []
    path = Path(path)
    
    if not path.exists():
        return terms
    
    # Pattern to extract aliases: /alias[term1|term2|term3]
    alias_pattern = re.compile(r'/alias\[([^\]]+)\]')
    # Pattern to extract js variants: /js[term]
    js_pattern = re.compile(r'/js\[([^\]]+)\]')
    
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Extract main term (before any / suffix)
            main_term = re.split(r'/(?:alias|js)\[', line)[0].strip()
            if main_term:
                terms.append(main_term)
            
            # Extract aliases
            alias_match = alias_pattern.search(line)
            if alias_match:
                aliases = alias_match.group(1).split('|')
                terms.extend(a.strip() for a in aliases if a.strip())
            
            # Extract js variants
            js_match = js_pattern.search(line)
            if js_match:
                js_terms = js_match.group(1).split('|')
                terms.extend(t.strip() for t in js_terms if t.strip())
    
    # Remove duplicates while preserving order
    seen = set()
    unique_terms = []
    for term in terms:
        if term not in seen:
            seen.add(term)
            unique_terms.append(term)
    
    return unique_terms


def get_allow_list(config: dict[str, Any]) -> list[str]:
    """
    Get the allow list from configuration.
    
    Reads the dictionary file path from config and parses it,
    then combines with any additional terms specified in config.
    
    Args:
        config: Application configuration dictionary
        
    Returns:
        List of terms to exclude from PII detection
    """
    allow_list_cfg = config.get("allow_list", {})
    
    if not allow_list_cfg.get("enabled", False):
        return []
    
    terms = []
    
    # Parse dictionary file if specified
    dict_path = allow_list_cfg.get("dictionary_path")
    if dict_path:
        # Resolve relative paths from project root
        path = Path(dict_path)
        if not path.is_absolute():
            path = Path(__file__).parent.parent / dict_path
        terms.extend(parse_dictionary(path))
    
    # Add any additional terms from config
    additional = allow_list_cfg.get("additional_terms", [])
    if additional:
        terms.extend(additional)
    
    return terms
