"""Factory for creating language-specific analyzer engines."""

from typing import Optional
import warnings

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider

from recognizers import (
    JapanesePhoneRecognizer,
    JapaneseZipCodeRecognizer,
    JapaneseBirthDateRecognizer,
    JapaneseNameRecognizer,
)

# Try to import GiNZA recognizers (optional)
GINZA_AVAILABLE = False
try:
    import spacy
    from recognizers import GinzaPersonRecognizer, GinzaAddressRecognizer
    GINZA_AVAILABLE = True
except ImportError:
    warnings.warn(
        "GiNZA is not available. Person name and address recognition will be limited. "
        "To install GiNZA, run: pip install ginza ja-ginza"
    )


def create_japanese_analyzer(use_ginza: bool = True) -> AnalyzerEngine:
    """
    Create an AnalyzerEngine configured for Japanese text.
    
    This includes:
    - Pattern-based recognizers (phone, zip, date) - always available
    - GiNZA-based recognizers (person, address) - optional, requires GiNZA installation
    
    Args:
        use_ginza: Whether to use GiNZA for NER (requires GiNZA installation)
    
    Returns:
        Configured AnalyzerEngine for Japanese
    """
    # Create NLP configuration for Japanese
    # Even without GiNZA, we need to register 'ja' as a supported language
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    
    nlp_configuration = {
        "nlp_engine_name": "spacy",
        "models": [
            {"lang_code": "en", "model_name": "en_core_web_lg"},
            {"lang_code": "ja", "model_name": "en_core_web_lg"},  # Fallback to English model for Japanese
        ],
    }
    
    try:
        nlp_engine = NlpEngineProvider(nlp_configuration=nlp_configuration).create_engine()
    except Exception:
        # If NLP engine creation fails, create analyzer without NLP engine
        nlp_engine = None
    
    # Create analyzer with Japanese support
    if nlp_engine:
        analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine,
            supported_languages=["ja", "en"]
        )
    else:
        analyzer = AnalyzerEngine(supported_languages=["ja", "en"])
    
    # Register pattern-based recognizers (always available)
    analyzer.registry.add_recognizer(JapanesePhoneRecognizer())
    analyzer.registry.add_recognizer(JapaneseZipCodeRecognizer())
    analyzer.registry.add_recognizer(JapaneseBirthDateRecognizer())
    analyzer.registry.add_recognizer(JapaneseNameRecognizer())  # Context-based name detection
    
    # Register GiNZA-based recognizers if available and requested
    if use_ginza and GINZA_AVAILABLE:
        try:
            # Load GiNZA model
            try:
                nlp = spacy.load("ja_ginza")
            except OSError:
                print("GiNZA model not found. Installing...")
                print("This may take several minutes and requires ~500MB download.")
                import subprocess
                subprocess.run(
                    ["python", "-m", "pip", "install", "ginza", "ja-ginza"],
                    check=True
                )
                nlp = spacy.load("ja_ginza")
            
            # Create NLP engine provider with GiNZA
            nlp_configuration = {
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "ja", "model_name": "ja_ginza"}],
            }
            
            nlp_engine = NlpEngineProvider(nlp_configuration=nlp_configuration).create_engine()
            
            # Update analyzer with GiNZA NLP engine
            analyzer.nlp_engine = nlp_engine
            
            # Register GiNZA-based recognizers
            analyzer.registry.add_recognizer(GinzaPersonRecognizer())
            analyzer.registry.add_recognizer(GinzaAddressRecognizer())
            
            print("✓ GiNZA-based person and address recognition enabled")
            
        except Exception as e:
            warnings.warn(
                f"Failed to initialize GiNZA: {e}. "
                "Person and address recognition will be limited to pattern matching."
            )
    elif use_ginza and not GINZA_AVAILABLE:
        warnings.warn(
            "GiNZA requested but not available. Install with: pip install ginza ja-ginza"
        )
    
    return analyzer


def create_analyzer(language: str = "en", use_ginza: bool = True) -> AnalyzerEngine:
    """
    Create an AnalyzerEngine for the specified language.
    
    Args:
        language: Language code ("en" for English, "ja" for Japanese)
        use_ginza: Whether to use GiNZA for Japanese NER (default: True)
        
    Returns:
        Configured AnalyzerEngine
    """
    if language == "ja":
        return create_japanese_analyzer(use_ginza=use_ginza)
    else:
        # Default English analyzer
        return AnalyzerEngine()

