"""GiNZA-based NER recognizers for Japanese person names and addresses."""


from presidio_analyzer import EntityRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import NlpArtifacts


class GinzaPersonRecognizer(EntityRecognizer):
    """
    Recognizer for Japanese person names using GiNZA NER.
    
    Uses spaCy's PERSON entity type from GiNZA model.
    Applies context-aware scoring based on resume-specific keywords.
    """

    EXPECTED_CONFIDENCE_LEVEL = 0.6
    CONTEXT_WORDS = ["氏名", "ふりがな", "フリガナ", "名前", "お名前", "姓名"]
    CONTEXT_BOOST_SCORE = 0.9
    CONTEXT_WINDOW = 50  # characters before/after entity

    def __init__(
        self,
        supported_language: str = "ja",
        supported_entity: str = "JP_PERSON",
        context_words: list[str] | None = None,
    ):
        self.context_words = context_words if context_words else self.CONTEXT_WORDS
        super().__init__(
            supported_entities=[supported_entity],
            supported_language=supported_language,
        )

    def load(self) -> None:
        """Load is not needed for this recognizer."""
        pass

    def analyze(
        self, text: str, entities: list[str], nlp_artifacts: NlpArtifacts
    ) -> list[RecognizerResult]:
        """
        Analyze text for Japanese person names using GiNZA.
        
        Args:
            text: The text to analyze
            entities: List of entities to look for
            nlp_artifacts: NLP artifacts from spaCy/GiNZA
            
        Returns:
            List of RecognizerResult objects
        """
        results = []

        if not nlp_artifacts or not nlp_artifacts.entities:
            return results

        for entity in nlp_artifacts.entities:
            # Only process PERSON entities
            if entity.label_ != "PERSON":
                continue

            # Check if this entity type is requested
            if self.supported_entities[0] not in entities:
                continue

            # Calculate confidence score based on context
            score = self._calculate_score(text, entity.start_char, entity.end_char)

            results.append(
                RecognizerResult(
                    entity_type=self.supported_entities[0],
                    start=entity.start_char,
                    end=entity.end_char,
                    score=score,
                )
            )

        return results

    def _calculate_score(self, text: str, start: int, end: int) -> float:
        """
        Calculate confidence score based on context.
        
        Higher score if context words are found nearby.
        """
        # Extract context window
        context_start = max(0, start - self.CONTEXT_WINDOW)
        context_end = min(len(text), end + self.CONTEXT_WINDOW)
        context = text[context_start:context_end]

        # Check for context words
        for word in self.context_words:
            if word in context:
                return self.CONTEXT_BOOST_SCORE

        return self.EXPECTED_CONFIDENCE_LEVEL


class GinzaAddressRecognizer(EntityRecognizer):
    """
    Recognizer for Japanese addresses using GiNZA NER.
    
    Uses spaCy's LOC (location) entity type from GiNZA model.
    Applies context-aware scoring based on resume-specific keywords.
    """

    EXPECTED_CONFIDENCE_LEVEL = 0.6
    CONTEXT_WORDS = ["住所", "現住所", "連絡先", "都道府県", "市区町村", "番地", "所在地"]
    CONTEXT_BOOST_SCORE = 0.9
    CONTEXT_WINDOW = 50  # characters before/after entity

    def __init__(
        self,
        supported_language: str = "ja",
        supported_entity: str = "JP_ADDRESS",
        context_words: list[str] | None = None,
    ):
        self.context_words = context_words if context_words else self.CONTEXT_WORDS
        super().__init__(
            supported_entities=[supported_entity],
            supported_language=supported_language,
        )

    def load(self) -> None:
        """Load is not needed for this recognizer."""
        pass

    def analyze(
        self, text: str, entities: list[str], nlp_artifacts: NlpArtifacts
    ) -> list[RecognizerResult]:
        """
        Analyze text for Japanese addresses using GiNZA.
        
        Args:
            text: The text to analyze
            entities: List of entities to look for
            nlp_artifacts: NLP artifacts from spaCy/GiNZA
            
        Returns:
            List of RecognizerResult objects
        """
        results = []

        if not nlp_artifacts or not nlp_artifacts.entities:
            return results

        for entity in nlp_artifacts.entities:
            # Process LOC (location) entities as addresses
            if entity.label_ != "LOC":
                continue

            # Check if this entity type is requested
            if self.supported_entities[0] not in entities:
                continue

            # Calculate confidence score based on context
            score = self._calculate_score(text, entity.start_char, entity.end_char)

            results.append(
                RecognizerResult(
                    entity_type=self.supported_entities[0],
                    start=entity.start_char,
                    end=entity.end_char,
                    score=score,
                )
            )

        return results

    def _calculate_score(self, text: str, start: int, end: int) -> float:
        """
        Calculate confidence score based on context.
        
        Higher score if context words are found nearby.
        """
        # Extract context window
        context_start = max(0, start - self.CONTEXT_WINDOW)
        context_end = min(len(text), end + self.CONTEXT_WINDOW)
        context = text[context_start:context_end]

        # Check for context words
        for word in self.context_words:
            if word in context:
                return self.CONTEXT_BOOST_SCORE

        return self.EXPECTED_CONFIDENCE_LEVEL
