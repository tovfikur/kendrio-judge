import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import spacy
from datetime import datetime
from ..LegalKnowledgeBaseModule import LegalKnowledgeBase, LegalStatute


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_lg")
    logger.info("Loaded spaCy model successfully")
except OSError:
    logger.warning("Failed to load full spaCy model, trying smaller model")
    try:
        nlp = spacy.load("en_core_web_sm")
        logger.info("Loaded smaller spaCy model")
    except OSError:
        logger.error("Failed to load spaCy model - defaulting to basic processing")
        nlp = None

@dataclass
class StatutoryInterpretation:
    """Represents the interpretation of a statute."""
    statute_id: str
    plain_meaning: str
    elements: List[str]
    ambiguities: List[str]
    contextual_analysis: Dict[str, Any]
    confidence: float
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)

class StatutoryInterpreter:
    """Interprets statutes using plain meaning, context, and legislative intent."""

    def __init__(self):
        """Initialize the statutory interpreter with knowledge base access."""
        self.kb_manager = LegalKnowledgeBase()

    def interpret_statute(self, statute_id: str, context: Dict[str, Any]) -> StatutoryInterpretation:
        """
        Interpret a statute based on its text and case context.

        Args:
            statute_id: Unique statute identifier
            context: Case context including facts and jurisdiction

        Returns:
            StatutoryInterpretation object
        """
        try:
            # Retrieve statute
            statute = self.kb_manager.get_statute(statute_id)
            if not statute:
                raise ValueError(f"Statute {statute_id} not found")

            # Extract plain meaning
            plain_meaning = self._extract_plain_meaning(statute)

            # Identify statutory elements
            elements = self._identify_elements(statute)

            # Detect ambiguities
            ambiguities = self._detect_ambiguities(statute)

            # Perform contextual analysis
            contextual_analysis = self._perform_contextual_analysis(statute, context)

            # Calculate confidence
            confidence = self._calculate_confidence(elements, ambiguities, contextual_analysis)

            interpretation = StatutoryInterpretation(
                statute_id=statute_id,
                plain_meaning=plain_meaning,
                elements=elements,
                ambiguities=ambiguities,
                contextual_analysis=contextual_analysis,
                confidence=confidence,
                metadata={
                    "interpreted_at": datetime.now().isoformat(),
                    "jurisdiction": statute.jurisdiction,
                    "category": statute.category
                }
            )

            return interpretation

        except Exception as e:
            error_details = {"error": str(e), "statute_id": statute_id}
            raise

    def _extract_plain_meaning(self, statute: LegalStatute) -> str:
        """Extract the plain meaning of the statute text."""
        if not nlp:
            return statute.text[:200]  # Fallback: truncate text

        statute_doc = nlp(statute.text)
        # Extract main clause (simplified)
        for sent in statute_doc.sents:
            return sent.text[:200]  # Return first sentence as plain meaning
        return statute.text[:200]

    def _identify_elements(self, statute: LegalStatute) -> List[str]:
        """Identify legal elements or requirements in the statute."""
        elements = []
        if statute.sections:
            for section_text in statute.sections.values():
                section_doc = nlp(section_text) if nlp else None
                if section_doc:
                    for sent in section_doc.sents:
                        if any(token.lemma_ in ["require", "must", "shall"] for token in sent):
                            elements.append(sent.text)
                else:
                    elements.append(section_text[:100])
        else:
            statute_doc = nlp(statute.text) if nlp else None
            if statute_doc:
                for sent in statute_doc.sents:
                    if any(token.lemma_ in ["require", "must", "shall"] for token in sent):
                        elements.append(sent.text)
            else:
                elements.append(statute.text[:100])
        return elements[:5]  # Limit to 5 elements

    def _detect_ambiguities(self, statute: LegalStatute) -> List[str]:
        """Detect potential ambiguities in the statute text."""
        if not nlp:
            return []

        statute_doc = nlp(statute.text)
        ambiguities = []

        # Simple ambiguity detection: look for vague terms or complex sentences
        vague_terms = ["reasonable", "appropriate", "sufficient"]
        for sent in statute_doc.sents:
            if len(sent) > 50:  # Long sentences may be ambiguous
                ambiguities.append(f"Complex sentence: {sent.text[:100]}")
            for token in sent:
                if token.text.lower() in vague_terms:
                    ambiguities.append(f"Vague term '{token.text}' in: {sent.text[:100]}")

        return ambiguities[:3]  # Limit to 3 ambiguities

    def _perform_contextual_analysis(self, statute: LegalStatute, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the statute in the context of case facts and jurisdiction."""
        case_facts = context.get("facts", {}).get("established_facts", [])
        fact_texts = [fact.get("claim", {}).get("text", "") for fact in case_facts]

        related_cases = self.kb_manager.search_case_law(
            query=statute.text,
            jurisdiction=statute.jurisdiction,
            limit=3
        )

        analysis = {
            "relevant_facts": [],
            "related_precedents": [case.to_dict() for case in related_cases],
            "legislative_intent": "Not implemented"  # Placeholder
        }

        if nlp:
            statute_doc = nlp(statute.text)
            for fact_text in fact_texts:
                fact_doc = nlp(fact_text)
                if statute_doc.similarity(fact_doc) > 0.6:
                    analysis["relevant_facts"].append(fact_text)

        return analysis

    def _calculate_confidence(self, elements: List[str], ambiguities: List[str], 
                           contextual_analysis: Dict[str, Any]) -> float:
        """Calculate confidence in the statutory interpretation."""
        element_score = len(elements) / 5.0 if elements else 0.5
        ambiguity_score = 1.0 - (len(ambiguities) / 3.0) if ambiguities else 1.0
        precedent_score = len(contextual_analysis["related_precedents"]) / 3.0
        return (element_score * 0.4) + (ambiguity_score * 0.3) + (precedent_score * 0.3)