"""PII Masker - Main masking functionality.

This module provides the PIIMasker class and mask_pii_in_text function
for detecting and masking personally identifiable information.
"""

from datetime import datetime
from typing import Any

from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from config import get_detection_strategy, get_entities_to_mask, get_entity_categories, get_transformer_config, load_config
from core.allow_list import get_allow_list
from core.analyzer import create_analyzer, create_multilingual_analyzer
from core.processors.hybrid_detection import hybrid_detection_analyze
from core.processors.result import deduplicate_results, merge_results
from core.processors.text import preprocess_text
from masking_logging import MaskingLogger


def build_operators(config: dict[str, Any]) -> dict:
    """
    Build anonymizer operators from config.
    
    Reads masking.entity_masks from config.yaml to create entity-specific
    mask patterns (e.g., phone: ***-****-****, zip: ***-****).
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dict of entity_type -> OperatorConfig for AnonymizerEngine
    """
    masking_cfg = config.get("masking", {})
    default_mask = masking_cfg.get("default_mask", "****")
    entity_masks = masking_cfg.get("entity_masks", {})
    
    operators = {
        "DEFAULT": OperatorConfig("replace", {"new_value": default_mask})
    }
    for entity_type, mask in entity_masks.items():
        operators[entity_type] = OperatorConfig("replace", {"new_value": mask})
    return operators

class PIIMasker:
    """
    PII Masker for detecting and masking personal information.
    
    Targets 8 PII types:
    - 名前 (PERSON, JP_PERSON)
    - メールアドレス (EMAIL_ADDRESS)
    - 郵便番号 (JP_ZIP_CODE)
    - 電話番号 (PHONE_NUMBER_JP)
    - 生年月日 (DATE_OF_BIRTH_JP)
    - 住所 (JP_ADDRESS, LOCATION)
    - 性別 (JP_GENDER)
    - 年齢 (JP_AGE)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the masker with configuration.
        
        Args:
            config: Configuration dictionary. If None, loads from config.yaml.
        """
        self.config = config or load_config()
        self.transformer_cfg = get_transformer_config(self.config)
        self.entities_to_mask = get_entities_to_mask(self.config)
        self.entity_categories = get_entity_categories(self.config)
        self.anonymizer = AnonymizerEngine()
        self.logger = MaskingLogger()

    def mask(
        self,
        text: str,
        language: str = "auto",
        verbose: bool = False,
        do_preprocess: bool = False
    ) -> tuple[str, list[dict[str, Any]] | None]:
        """
        Mask PII in text.
        
        Args:
            text: Text to analyze and mask
            language: Language code ("en", "ja", or "auto" for multilingual)
            verbose: If True, return detected entities info
            do_preprocess: If True, normalize text before analysis
            
        Returns:
            Tuple of (masked_text, detected_entities_info)
        """
        return mask_pii_in_text(
            text,
            language=language,
            verbose=verbose,
            preprocess=do_preprocess,
            config=self.config
        )


def mask_pii_in_text(
    text: str,
    language: str = "auto",
    verbose: bool = False,
    preprocess: bool = False,
    config: dict[str, Any] | None = None
) -> tuple[str, list[dict[str, Any]] | None]:
    """
    Mask PII in text using hybrid detection strategy.
    
    Uses Transformer NER for PERSON/ADDRESS entities and
    Pattern recognizers for PHONE/ZIP/DATE/etc.
    
    Args:
        text: Text to analyze and mask
        language: Language code ("en", "ja", or "auto" for multilingual)
        verbose: If True, return detected entities info
        preprocess: If True, normalize text before analysis (recommended for PDF text)
        config: Configuration dictionary. If None, loads from config.yaml.
        
    Returns:
        Tuple of (masked_text, detected_entities_info)
    """
    # Load configuration
    if config is None:
        config = load_config()

    transformer_cfg = get_transformer_config(config)
    detection_strategy = get_detection_strategy(config)
    
    use_transformer = transformer_cfg.get("enabled", False)
    transformer_entities = detection_strategy.get("transformer_entities", [])
    pattern_entities = detection_strategy.get("pattern_entities", [])
    
    # Load allow list
    allow_list = get_allow_list(config)

    # Preprocess text if requested (useful for PDF-extracted text)
    if preprocess:
        text = preprocess_text(text)

    anonymizer = AnonymizerEngine()

    # Use hybrid detection when Transformer is enabled
    if use_transformer:
        results = hybrid_detection_analyze(
            text=text,
            transformer_entities=transformer_entities,
            pattern_entities=pattern_entities,
            language=language,
            app_config=config,
            allow_list=allow_list
        )
    else:
        # Transformer disabled - use only pattern recognizers
        all_entities = transformer_entities + pattern_entities
        
        if language == "auto":
            analyzer = create_multilingual_analyzer(use_ginza=True, use_transformer=False)
            results_en = analyzer.analyze(text=text, language="en", entities=all_entities, allow_list=allow_list)
            results_ja = analyzer.analyze(text=text, language="ja", entities=all_entities, allow_list=allow_list)
            results = merge_results(results_en, results_ja)
        else:
            analyzer = create_analyzer(language=language, use_transformer=False)
            results = analyzer.analyze(text=text, language=language, entities=all_entities, allow_list=allow_list)

    # Deduplicate overlapping results
    results = deduplicate_results(results, text)

    # Log masked entities to file
    logger = MaskingLogger()
    if results:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.log(f"\n{'='*60}")
        logger.log(f"Masking Log - {timestamp}")
        logger.log(f"{'='*60}")
        for result in results:
            entity_text = text[result.start:result.end]
            logger.log(
                f"[{result.entity_type}] \"{entity_text}\" "
                f"(score: {result.score:.2f}, pos: {result.start}-{result.end})"
            )
        logger.log(f"Total: {len(results)} entities masked")


    # Prepare operators for masking (from config.yaml)
    operators = build_operators(config)


    # Anonymize the text
    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators=operators,
    )

    # Prepare verbose output if requested
    entities_info = None
    if verbose and results:
        entities_info = []
        for result in results:
            entity_text = text[result.start:result.end]
            entities_info.append({
                "type": result.entity_type,
                "text": entity_text,
                "score": result.score,
                "start": result.start,
                "end": result.end,
            })

    return anonymized.text, entities_info
