import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import spacy
from datetime import datetime
from ..LegalKnowledgeBaseModule import *
from ..EvidenceAnalysisEngine import *
from ..precedent_matching.precedent_matcher import PrecedentMatcher


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
class ArgumentEvaluation:
    """Represents the evaluation of a legal argument."""
    argument_id: str
    score: float
    strengths: List[str]
    weaknesses: List[str]
    supporting_facts: List[Dict[str, Any]]
    supporting_precedents: List[Dict[str, Any]]
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)

class ArgumentEvaluator:
    """Evaluates the strength of legal arguments based on facts and law."""

    def __init__(self):
        """Initialize the argument evaluator with dependencies."""
        self.kb_manager = LegalKnowledgeBase()
        self.factual_analyzer = FactualAnalyzer()
        self.precedent_matcher = PrecedentMatcher()


    def evaluate_argument(self, argument: Dict[str, Any], opposing_argument: Optional[Dict[str, Any]], 
                        context: Dict[str, Any]) -> ArgumentEvaluation:
        """
        Evaluate a legal argument based on facts, law, and precedents.

        Args:
            argument: Dictionary containing argument details (id, description)
            opposing_argument: Optional opposing argument for comparison
            context: Case context including facts and precedents

        Returns:
            ArgumentEvaluation object
        """
        try:
            argument_id = argument.get("id", "arg_unnamed")
            argument_text = argument.get("description", "")

            # Retrieve case facts and precedents
            case_facts = context.get("facts", {})
            precedents = context.get("precedents", [])

            # Analyze argument strength
            fact_support = self._analyze_fact_support(argument_text, case_facts)
            precedent_support = self._analyze_precedent_support(argument_text, precedents)
            legal_support = self._analyze_legal_support(argument_text, case_facts.get("case_id"))

            # Compare with opposing argument if provided
            opposing_score = self._compare_with_opposing(argument_text, opposing_argument) if opposing_argument else 0.0

            # Calculate overall score
            score = self._calculate_argument_score(fact_support, precedent_support, legal_support, opposing_score)

            # Identify strengths and weaknesses
            strengths = self._identify_strengths(fact_support, precedent_support, legal_support)
            weaknesses = self._identify_weaknesses(fact_support, precedent_support, legal_support)

            evaluation = ArgumentEvaluation(
                argument_id=argument_id,
                score=score,
                strengths=strengths,
                weaknesses=weaknesses,
                supporting_facts=fact_support["supporting_facts"],
                supporting_precedents=[p.to_dict() for p in precedent_support["supporting_precedents"]],
                metadata={
                    "evaluated_at": datetime.now().isoformat(),
                    "case_id": case_facts.get("case_id", "unknown")
                }
            )

            return evaluation

        except Exception as e:
            error_details = {"error": str(e), "argument_id": argument.get("id", "unknown")}
            raise

    def _analyze_fact_support(self, argument_text: str, case_facts: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how well facts support the argument."""
        if not nlp:
            return {"supporting_facts": [], "score": 0.5}

        argument_doc = nlp(argument_text)
        supporting_facts = []
        fact_scores = []

        for fact in case_facts.get("established_facts", []):
            fact_text = fact.get("claim", {}).get("text", "")
            fact_doc = nlp(fact_text) if fact_text else None
            if fact_doc:
                similarity = argument_doc.similarity(fact_doc)
                if similarity > 0.6:
                    supporting_facts.append(fact)
                    fact_scores.append(similarity)

        score = sum(fact_scores) / len(fact_scores) if fact_scores else 0.5
        return {"supporting_facts": supporting_facts, "score": score}

    def _analyze_precedent_support(self, argument_text: str, precedents: List[CaseLaw]) -> Dict[str, Any]:
        """Analyze how well precedents support the argument."""
        if not nlp:
            return {"supporting_precedents": [], "score": 0.5}

        argument_doc = nlp(argument_text)
        supporting_precedents = []
        precedent_scores = []

        for precedent in precedents:
            precedent_summary = precedent.summary
            precedent_doc = nlp(precedent_summary) if precedent_summary else None
            if precedent_doc:
                similarity = argument_doc.similarity(precedent_doc)
                if similarity > 0.6:
                    supporting_precedents.append(precedent)
                    precedent_scores.append(similarity)

        score = sum(precedent_scores) / len(precedent_scores) if precedent_scores else 0.5
        return {"supporting_precedents": supporting_precedents, "score": score}

    def _analyze_legal_support(self, argument_text: str, case_id: str) -> Dict[str, Any]:
        """Analyze how well legal provisions support the argument."""
        statutes = self.kb_manager.search_statutes(query=argument_text, limit=3)
        principles = self.kb_manager.search_legal_principles(query=argument_text, limit=3)

        if not nlp:
            return {"supporting_laws": [], "score": 0.5}

        argument_doc = nlp(argument_text)
        supporting_laws = []
        law_scores = []

        for statute in statutes:
            statute_doc = nlp(statute.text) if statute.text else None
            if statute_doc:
                similarity = argument_doc.similarity(statute_doc)
                if similarity > 0.6:
                    supporting_laws.append(statute.to_dict())
                    law_scores.append(similarity)

        for principle in principles:
            principle_doc = nlp(principle.description) if principle.description else None
            if principle_doc:
                similarity = argument_doc.similarity(principle_doc)
                if similarity > 0.6:
                    supporting_laws.append(principle.to_dict())
                    law_scores.append(similarity)

        score = sum(law_scores) / len(law_scores) if law_scores else 0.5
        return {"supporting_laws": supporting_laws, "score": score}

    def _compare_with_opposing(self, argument_text: str, opposing_argument: Dict[str, Any]) -> float:
        """Compare argument strength against an opposing argument."""
        if not nlp or not opposing_argument:
            return 0.0

        argument_doc = nlp(argument_text)
        opposing_text = opposing_argument.get("description", "")
        opposing_doc = nlp(opposing_text) if opposing_text else None

        if opposing_doc:
            similarity = argument_doc.similarity(opposing_doc)
            # Lower similarity to opposing argument increases strength
            return 1.0 - similarity
        return 0.0

    def _calculate_argument_score(self, fact_support: Dict[str, Any], precedent_support: Dict[str, Any], 
                               legal_support: Dict[str, Any], opposing_score: float) -> float:
        """Calculate overall argument score."""
        fact_score = fact_support["score"]
        precedent_score = precedent_support["score"]
        legal_score = legal_support["score"]
        return (fact_score * 0.4) + (precedent_score * 0.3) + (legal_score * 0.2) + (opposing_score * 0.1)

    def _identify_strengths(self, fact_support: Dict[str, Any], precedent_support: Dict[str, Any], 
                          legal_support: Dict[str, Any]) -> List[str]:
        """Identify strengths of the argument."""
        strengths = []
        if fact_support["score"] > 0.7:
            strengths.append(f"Strong factual support ({len(fact_support['supporting_facts'])} relevant facts)")
        if precedent_support["score"] > 0.7:
            strengths.append(f"Strong precedent support ({len(precedent_support['supporting_precedents'])} cases)")
        if legal_support["score"] > 0.7:
            strengths.append(f"Strong legal support ({len(legal_support['supporting_laws'])} provisions)")
        return strengths

    def _identify_weaknesses(self, fact_support: Dict[str, Any], precedent_support: Dict[str, Any], 
                           legal_support: Dict[str, Any]) -> List[str]:
        """Identify weaknesses of the argument."""
        weaknesses = []
        if fact_support["score"] < 0.5:
            weaknesses.append("Limited factual support")
        if precedent_support["score"] < 0.5:
            weaknesses.append("Limited precedent support")
        if legal_support["score"] < 0.5:
            weaknesses.append("Limited legal support")
        return weaknesses