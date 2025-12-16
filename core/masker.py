"""PII Masker - Main masking functionality.

This module provides the PIIMasker class and mask_pii_in_text function
for detecting and masking personally identifiable information.
"""

from datetime import datetime
from typing import Any

from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from config import get_entities_to_mask, get_entity_categories, get_transformer_config, load_config
from core.analyzer import create_analyzer, create_multilingual_analyzer
from core.processors.dual_detection import dual_detection_analyze
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
    Mask PII in text using language-specific or multilingual analyzer.
    
    When Transformer NER is enabled with dual-detection mode:
    - Runs both pattern-based and Transformer-based analysis
    - Only masks entities detected by BOTH recognizers
    
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
    entities_to_mask = get_entities_to_mask(config)
    entity_categories = get_entity_categories(config)

    use_transformer = transformer_cfg.get("enabled", False)
    require_dual_detection = transformer_cfg.get("require_dual_detection", True)

    # Preprocess text if requested (useful for PDF-extracted text)
    if preprocess:
        text = preprocess_text(text)

    anonymizer = AnonymizerEngine()
    dual_scores = {}  # Will hold dual scores if available

    if language == "auto":
        # Multilingual mode: analyze in both English and Japanese
        if use_transformer and require_dual_detection:
            # Dual detection mode: run pattern-only and Transformer-only analyzers
            results, dual_scores = dual_detection_analyze(
                text, transformer_cfg, entities_to_mask, entity_categories, language="auto"
            )
        else:
            # Standard mode
            analyzer = create_multilingual_analyzer(
                use_ginza=True,
                use_transformer=use_transformer,
                transformer_config=transformer_cfg if use_transformer else None
            )

            # Analyze in both languages
            results_en = analyzer.analyze(
                text=text,
                language="en",
                entities=entities_to_mask,
            )
            results_ja = analyzer.analyze(
                text=text,
                language="ja",
                entities=entities_to_mask,
            )

            # Merge results from both languages
            results = merge_results(results_en, results_ja)
    # Single language mode
    elif use_transformer and require_dual_detection:
        # Dual detection mode
        results, dual_scores = dual_detection_analyze(
            text, transformer_cfg, entities_to_mask, entity_categories, language=language
        )
    else:
        # Standard mode
        analyzer = create_analyzer(
            language=language,
            use_transformer=use_transformer,
            transformer_config=transformer_cfg if use_transformer else None
        )
        results = analyzer.analyze(
            text=text,
            language=language,
            entities=entities_to_mask,
        )

    # Deduplicate overlapping results
    results = deduplicate_results(results, text)

    # Log masked entities to file
    logger = MaskingLogger()
    if results:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.log(f"\n{'='*60}")
        logger.log(f"Masking Log - {timestamp}")
        if dual_scores:
            logger.log("Mode: Dual Detection (Pattern + Transformer)")
        logger.log(f"{'='*60}")
        for result in results:
            entity_text = text[result.start:result.end]
            # Check if we have dual scores for this entity
            if (result.start, result.end) in dual_scores:
                scores = dual_scores[(result.start, result.end)]
                p_type = scores.get('p_type', result.entity_type)
                t_type = scores.get('t_type', '?')
                logger.log(
                    f"[{result.entity_type}] \"{entity_text}\" "
                    f"(pattern: {p_type}={scores['pattern']:.2f}, transformer: {t_type}={scores['transformer']:.2f}, pos: {result.start}-{result.end})"
                )

            else:
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
