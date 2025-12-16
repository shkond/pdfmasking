"""Result processing utilities for entity detection results.

Handles deduplication and merging of RecognizerResult objects.
"""

from typing import List


def deduplicate_results(results, text: str):
    """
    Remove duplicate/overlapping entity detections.
    Keeps only the highest-scoring result for overlapping spans.
    
    Args:
        results: List of RecognizerResult objects
        text: The original text (for extracting entity text)
    
    Returns:
        List of deduplicated RecognizerResult objects
    """
    if not results:
        return results
    
    # Sort by score (descending), then by start position
    sorted_results = sorted(results, key=lambda x: (-x.score, x.start))
    
    # Keep track of which positions have been covered
    covered_positions = set()
    deduplicated = []
    
    for result in sorted_results:
        # Check if this result overlaps with already covered positions
        result_positions = set(range(result.start, result.end))
        if result_positions & covered_positions:
            # This result overlaps with a higher-scoring result, skip it
            continue
        
        # Add this result and mark its positions as covered
        deduplicated.append(result)
        covered_positions.update(result_positions)
    
    # Sort back by position for consistent ordering
    deduplicated.sort(key=lambda x: x.start)
    return deduplicated


def merge_results(results_en, results_ja):
    """
    Merge results from English and Japanese analysis.
    
    Combines results from both languages and removes duplicates
    by keeping the higher-scoring result for overlapping spans.
    
    Args:
        results_en: RecognizerResult list from English analysis
        results_ja: RecognizerResult list from Japanese analysis
        
    Returns:
        Combined list of RecognizerResult objects
    """
    all_results = list(results_en) + list(results_ja)
    
    if not all_results:
        return all_results
    
    # Sort by score descending, then by span length (prefer longer matches)
    sorted_results = sorted(all_results, key=lambda x: (-x.score, -(x.end - x.start), x.start))
    
    # Remove overlapping results, keeping higher-scoring ones
    covered_positions = set()
    merged = []
    
    for result in sorted_results:
        result_positions = set(range(result.start, result.end))
        # Check for significant overlap (more than 50% of the smaller span)
        overlap = result_positions & covered_positions
        if len(overlap) > len(result_positions) * 0.5:
            # Significant overlap with a higher-scoring result, skip
            continue
        
        merged.append(result)
        covered_positions.update(result_positions)
    
    # Sort by position for consistent ordering
    merged.sort(key=lambda x: x.start)
    return merged
