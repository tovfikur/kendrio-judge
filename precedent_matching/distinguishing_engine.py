import logging
from typing import Dict, List, Any
from datetime import datetime
import spacy
from ..LegalKnowledgeBaseModule import LegalKnowledgeBase, CaseLaw
from ..EvidenceAnalysisEngine import FactualAnalyzer
from ..LegalReasoningModule.legal_reasoning import LegalReasoner

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load spaCy model for text similarity
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

class DistinguishingEngine:
    """Identifies meaningful distinctions between the current case and precedents."""

    def __init__(self):
        """Initialize the distinguishing engine with dependencies."""
        self.kb_manager = LegalKnowledgeBase()
        self.factual_analyzer = FactualAnalyzer()
        self.legal_reasoner = LegalReasoner()

    def distinguish_precedent(self, case_id: str, precedent_id: str) -> Dict[str, Any]:
        """
        Identify distinctions between a case and a precedent.

        Args:
            case_id: Unique case identifier
            precedent_id: Unique precedent identifier

        Returns:
            Dictionary containing distinctions and their legal significance
        """
        try:
            # Retrieve case and precedent details
            case_facts = self.factual_analyzer.analyze_facts(case_id)
            precedent = self.kb_manager.get_case_law(precedent_id)
            if not precedent:
                raise ValueError(f"Precedent {precedent_id} not found")

            # Evaluate factual distinctions
            precedent_facts = self._get_precedent_facts(precedent_id)
            factual_distinctions = self.evaluate_factual_distinctions(case_facts, precedent_facts)

            # Evaluate legal context differences
            legal_differences = self.evaluate_legal_context_differences(case_id, precedent_id)

            # Generate distinguishing argument
            argument = self.generate_distinguishing_argument(case_id, precedent_id)

            distinctions = {
                "case_id": case_id,
                "precedent_id": precedent_id,
                "factual_distinctions": factual_distinctions,
                "legal_differences": legal_differences,
                "distinguishing_argument": argument,
                "significance": self._assess_significance(factual_distinctions, legal_differences),
                "analysis_date": datetime.now().isoformat()
            }


            return distinctions

        except Exception as e:
            error_details = {"error": str(e), "case_id": case_id, "precedent_id": precedent_id}
            raise

    def evaluate_factual_distinctions(self, case_facts: Dict[str, Any], precedent_facts: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Evaluate factual distinctions between case and precedent facts.

        Args:
            case_facts: Facts of the current case
            precedent_facts: Facts of the precedent case

        Returns:
            List of dictionaries containing factual distinctions
        """
        try:
            distinctions = []
            case_fact_texts = [f.get("claim", {}).get("text", "") for f in case_facts.get("established_facts", [])]
            precedent_fact_texts = [f.get("claim", {}).get("text", "") for f in precedent_facts.get("established_facts", [])]

            if not nlp:
                # Fallback: keyword-based comparison
                case_words = set(" ".join(case_fact_texts).lower().split())
                precedent_words = set(" ".join(precedent_fact_texts).lower().split())
                unique_case = case_words - precedent_words
                unique_precedent = precedent_words - case_words
                if unique_case:
                    distinctions.append({
                        "fact_type": "unique_to_case",
                        "description": f"Case has unique facts: {', '.join(list(unique_case)[:5])}",
                        "significance": "moderate"
                    })
                if unique_precedent:
                    distinctions.append({
                        "fact_type": "unique_to_precedent",
                        "description": f"Precedent has unique facts: {', '.join(list(unique_precedent)[:5])}",
                        "significance": "moderate"
                    })
                return distinctions

            # NLP-based comparison
            for case_fact in case_fact_texts:
                case_doc = nlp(case_fact) if case_fact else None
                if not case_doc:
                    continue
                min_similarity = 1.0
                for precedent_fact in precedent_fact_texts:
                    precedent_doc = nlp(precedent_fact) if precedent_fact else None
                    if precedent_doc:
                        similarity = case_doc.similarity(precedent_doc)
                        min_similarity = min(min_similarity, similarity)
                if min_similarity < 0.6:
                    distinctions.append({
                        "fact_type": "unique_to_case",
                        "description": f"Case fact not closely matched: {case_fact[:100]}",
                        "significance": "high" if min_similarity < 0.4 else "moderate"
                    })

            for precedent_fact in precedent_fact_texts:
                precedent_doc = nlp(precedent_fact) if precedent_fact else None
                if not precedent_doc:
                    continue
                min_similarity = 1.0
                for case_fact in case_fact_texts:
                    case_doc = nlp(case_fact) if case_fact else None
                    if case_doc:
                        similarity = precedent_doc.similarity(case_doc)
                        min_similarity = min(min_similarity, similarity)
                if min_similarity < 0.6:
                    distinctions.append({
                        "fact_type": "unique_to_precedent",
                        "description": f"Precedent fact not closely matched: {precedent_fact[:100]}",
                        "significance": "high" if min_similarity < 0.4 else "moderate"
                    })

            return distinctions

        except Exception as e:
            error_details = {"error": str(e), "case_id": case_facts.get("case_id", "unknown")}
            raise

    def evaluate_legal_context_differences(self, case_id: str, precedent_id: str) -> List[Dict[str, Any]]:
        """
        Evaluate differences in legal context between case and precedent.

        Args:
            case_id: Unique case identifier
            precedent_id: Unique precedent identifier

        Returns:
            List of dictionaries containing legal context differences
        """
        try:
            differences = []
            case_context = self._get_case_context(case_id)
            precedent = self.kb_manager.get_case_law(precedent_id)
            if not precedent:
                raise ValueError(f"Precedent {precedent_id} not found")

            # Compare jurisdictions
            case_jurisdiction = case_context.get("jurisdiction", "federal")
            precedent_jurisdiction = precedent.jurisdiction
            if case_jurisdiction != precedent_jurisdiction:
                differences.append({
                    "difference_type": "jurisdiction",
                    "description": f"Case jurisdiction ({case_jurisdiction}) differs from precedent ({precedent_jurisdiction})",
                    "significance": "high"
                })

            # Compare legal principles
            case_principles = case_context.get("legal_principles", [])
            precedent_principles = precedent.legal_principles or []
            unique_case_principles = set(case_principles) - set(precedent_principles)
            unique_precedent_principles = set(precedent_principles) - set(case_principles)
            if unique_case_principles:
                differences.append({
                    "difference_type": "legal_principles",
                    "description": f"Case applies unique principles: {', '.join(unique_case_principles)}",
                    "significance": "moderate"
                })
            if unique_precedent_principles:
                differences.append({
                    "difference_type": "legal_principles",
                    "description": f"Precedent applies unique principles: {', '.join(unique_precedent_principles)}",
                    "significance": "moderate"
                })

            # Compare court levels (placeholder)
            case_court = case_context.get("court", "unknown")
            precedent_court = precedent.court
            if case_court != precedent_court:
                differences.append({
                    "difference_type": "court_level",
                    "description": f"Case court ({case_court}) differs from precedent court ({precedent_court})",
                    "significance": "low"
                })

            return differences

        except Exception as e:
            error_details = {"error": str(e), "case_id": case_id, "precedent_id": precedent_id}
            raise

    def generate_distinguishing_argument(self, case_id: str, precedent_id: str) -> Dict[str, Any]:
        """
        Generate a structured argument for distinguishing a case from a precedent.

        Args:
            case_id: Unique case identifier
            precedent_id: Unique precedent identifier

        Returns:
            Dictionary containing structured distinguishing argument
        """
        try:
            # Get distinctions
            distinctions = self.distinguish_precedent(case_id, precedent_id)
            factual_distinctions = distinctions["factual_distinctions"]
            legal_differences = distinctions["legal_differences"]

            # Build argument
            argument_premises = []
            for fact in factual_distinctions:
                argument_premises.append({
                    "type": "factual_distinction",
                    "content": fact["description"],
                    "significance": fact["significance"]
                })
            for diff in legal_differences:
                argument_premises.append({
                    "type": "legal_difference",
                    "content": diff["description"],
                    "significance": diff["significance"]
                })

            conclusion = "The precedent is distinguishable due to significant factual and legal differences."
            if not argument_premises:
                conclusion = "No significant distinctions found; precedent is closely analogous."

            argument = {
                "case_id": case_id,
                "precedent_id": precedent_id,
                "premises": argument_premises,
                "conclusion": conclusion,
                "confidence": self._calculate_argument_confidence(argument_premises),
                "generated_at": datetime.now().isoformat()
            }

            return argument

        except Exception as e:
            error_details = {"error": str(e), "case_id": case_id, "precedent_id": precedent_id}
            raise

    def _get_precedent_facts(self, precedent_id: str) -> Dict[str, Any]:
        """Retrieve facts for a precedent (placeholder)."""
        # Would integrate with factual_analysis.py or parse precedent summary
        return {
            "established_facts": [
                {"claim": {"text": "Precedent fact 1: Contract was breached."}},
                {"claim": {"text": "Precedent fact 2: Damages were awarded."}}
            ]
        }

    def _get_case_context(self, case_id: str) -> Dict[str, Any]:
        """Retrieve case context (placeholder)."""
        # Would call case_processor.get_case_summary(case_id)
        return {
            "case_id": case_id,
            "jurisdiction": "federal",
            "court": "district",
            "legal_principles": ["principle_1", "principle_2"],
            "summary": "Sample case summary"
        }

    def _assess_significance(self, factual_distinctions: List[Dict[str, Any]], 
                           legal_differences: List[Dict[str, Any]]) -> str:
        """Assess overall significance of distinctions."""
        high_significance = any(d["significance"] == "high" for d in factual_distinctions + legal_differences)
        moderate_significance = any(d["significance"] == "moderate" for d in factual_distinctions + legal_differences)
        if high_significance:
            return "high"
        elif moderate_significance:
            return "moderate"
        return "low"

    def _calculate_argument_confidence(self, premises: List[Dict[str, Any]]) -> float:
        """Calculate confidence in the distinguishing argument."""
        significance_scores = {"high": 0.9, "moderate": 0.6, "low": 0.3}
        scores = [significance_scores.get(p["significance"], 0.5) for p in premises]
        return sum(scores) / len(scores) if scores else 0.5