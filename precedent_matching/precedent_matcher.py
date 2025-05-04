import logging
from typing import List, Dict, Any, Optional
import spacy
from datetime import datetime
from ..LegalKnowledgeBaseModule import LegalKnowledgeBase, CaseLaw

# Placeholder imports for dependencies (to be replaced with actual implementations)
from citation_analyzer import CitationAnalyzer
from distinguishing_engine import DistinguishingEngine
from precedent_weight_calculator import PrecedentWeightCalculator

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

class PrecedentMatcher:
    """Identifies and ranks relevant legal precedents for a given case."""

    def __init__(self):
        """Initialize the precedent matcher with dependencies."""
        self.kb_manager = LegalKnowledgeBase()
        self.citation_analyzer = CitationAnalyzer()
        self.distinguishing_engine = DistinguishingEngine()
        self.weight_calculator = PrecedentWeightCalculator()

    def find_relevant_precedents(self, case_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Find relevant precedents for a given case.

        Args:
            case_id: Unique case identifier
            limit: Maximum number of precedents to return (default: 20)

        Returns:
            List of dictionaries containing precedent details and similarity scores
        """
        try:
            # Retrieve case context and facts (placeholder for case_processor integration)
            case_context = self._get_case_context(case_id)
            case_facts = self._get_case_facts(case_id)
            jurisdiction = case_context.get("jurisdiction", "federal")

            # Combine case summary and facts for query
            query_text = case_context.get("summary", "") + " ".join(
                [fact.get("claim", {}).get("text", "") for fact in case_facts.get("established_facts", [])]
            )

            # Search for precedents in knowledge base
            precedents = self.kb_manager.search_case_law(
                query=query_text,
                jurisdiction=jurisdiction,
                limit=limit * 2  # Oversample to allow for filtering
            )

            # Calculate similarity and weight for each precedent
            results = []
            for precedent in precedents:
                similarity, factors = self.calculate_case_similarity(case_id, precedent.id)
                weight, weight_factors = self.weight_calculator.calculate_precedent_weight(
                    precedent.id, jurisdiction, case_context
                )

                # Combine similarity and weight
                combined_score = (similarity * 0.6) + (weight * 0.4)

                # Get distinctions to adjust relevance
                distinctions = self.distinguishing_engine.distinguish_precedent(case_id, precedent.id)
                distinction_penalty = len(distinctions.get("distinctions", [])) * 0.1
                final_score = max(0.0, combined_score - distinction_penalty)

                results.append({
                    "precedent_id": precedent.id,
                    "title": precedent.title,
                    "citation": precedent.citation,
                    "similarity_score": final_score,
                    "similarity_factors": factors,
                    "weight": weight,
                    "weight_factors": weight_factors,
                    "distinctions": distinctions,
                    "details": precedent.to_dict()
                })

            # Sort by final score and limit results
            results.sort(key=lambda x: x["similarity_score"], reverse=True)
            results = results[:limit]


            return results

        except Exception as e:
            error_details = {"error": str(e), "case_id": case_id}
            raise

    def calculate_case_similarity(self, case_id: str, precedent_id: str) -> tuple[float, Dict[str, Any]]:
        """
        Calculate similarity between a case and a precedent.

        Args:
            case_id: Unique case identifier
            precedent_id: Unique precedent identifier

        Returns:
            Tuple of (similarity score, dictionary of similarity factors)
        """
        try:
            # Retrieve case and precedent details
            case_context = self._get_case_context(case_id)
            precedent = self.kb_manager.get_case_law(precedent_id)
            if not precedent:
                raise ValueError(f"Precedent {precedent_id} not found")

            # Combine case facts and summary for comparison
            case_facts = self._get_case_facts(case_id)
            case_text = case_context.get("summary", "") + " ".join(
                [fact.get("claim", {}).get("text", "") for fact in case_facts.get("established_facts", [])]
            )
            precedent_text = precedent.summary or precedent.full_text[:1000]  # Limit precedent text

            # Calculate text similarity
            similarity = 0.5
            factors = {"text_similarity": 0.5, "legal_principles": 0.5, "citation_influence": 0.5}
            
            if nlp:
                case_doc = nlp(case_text)
                precedent_doc = nlp(precedent_text)
                similarity = case_doc.similarity(precedent_doc) if case_doc and precedent_doc else 0.5
                factors["text_similarity"] = similarity
            else:
                # Fallback: keyword overlap
                case_words = set(case_text.lower().split())
                precedent_words = set(precedent_text.lower().split())
                overlap = len(case_words.intersection(precedent_words)) / len(case_words.union(precedent_words))
                similarity = overlap
                factors["text_similarity"] = overlap

            # Adjust similarity based on legal principles
            if precedent.legal_principles:
                case_principles = case_context.get("legal_principles", [])
                common_principles = len(set(precedent.legal_principles).intersection(set(case_principles)))
                principle_score = common_principles / len(precedent.legal_principles) if precedent.legal_principles else 0.5
                factors["legal_principles"] = principle_score
                similarity = (similarity * 0.7) + (principle_score * 0.3)

            # Adjust based on citation influence
            citation_analysis = self.citation_analyzer.analyze_citation_network(precedent_id)
            citation_score = citation_analysis.get("influence_score", 0.5)
            factors["citation_influence"] = citation_score
            similarity = (similarity * 0.8) + (citation_score * 0.2)

            return similarity, factors

        except Exception as e:
            error_details = {"error": str(e), "case_id": case_id, "precedent_id": precedent_id}
            raise

    def get_precedent_details(self, precedent_id: str) -> Dict[str, Any]:
        """
        Retrieve detailed information about a precedent.

        Args:
            precedent_id: Unique precedent identifier

        Returns:
            Dictionary containing precedent details
        """
        try:
            precedent = self.kb_manager.get_case_law(precedent_id)
            if not precedent:
                raise ValueError(f"Precedent {precedent_id} not found")

            # Get citation analysis
            citation_analysis = self.citation_analyzer.analyze_citation_network(precedent_id)

            # Get weight
            weight, weight_factors = self.weight_calculator.calculate_precedent_weight(
                precedent_id, precedent.jurisdiction, {}
            )

            details = precedent.to_dict()
            details.update({
                "citation_analysis": citation_analysis,
                "weight": weight,
                "weight_factors": weight_factors,
                "retrieved_at": datetime.now().isoformat()
            })


            return details

        except Exception as e:
            error_details = {"error": str(e), "precedent_id": precedent_id}
            raise

    def get_binding_precedents(self, case_id: str) -> List[Dict[str, Any]]:
        """
        Identify binding precedents based on jurisdiction and court hierarchy.

        Args:
            case_id: Unique case identifier

        Returns:
            List of dictionaries containing binding precedent details
        """
        try:
            # Retrieve case context
            case_context = self._get_case_context(case_id)
            jurisdiction = case_context.get("jurisdiction", "federal")

            # Get jurisdiction details
            jurisdiction_info = self.kb_manager.get_jurisdiction(jurisdiction)
            if not jurisdiction_info:
                raise ValueError(f"Jurisdiction {jurisdiction} not found")

            # Search for precedents in the same jurisdiction
            precedents = self.kb_manager.search_case_law(
                query=case_context.get("summary", ""),
                jurisdiction=jurisdiction,
                limit=50
            )

            binding_precedents = []
            for precedent in precedents:
                # Check if precedent is from a superior court
                court_weight = self.weight_calculator.evaluate_court_hierarchy_weight(
                    precedent.court, jurisdiction
                )
                if court_weight > 0.7:  # Threshold for binding precedents (e.g., higher courts)
                    weight, weight_factors = self.weight_calculator.calculate_precedent_weight(
                        precedent.id, jurisdiction, case_context
                    )
                    similarity, factors = self.calculate_case_similarity(case_id, precedent.id)

                    binding_precedents.append({
                        "precedent_id": precedent.id,
                        "title": precedent.title,
                        "citation": precedent.citation,
                        "court": precedent.court,
                        "similarity_score": similarity,
                        "similarity_factors": factors,
                        "weight": weight,
                        "weight_factors": weight_factors,
                        "details": precedent.to_dict()
                    })

            # Sort by weight and similarity
            binding_precedents.sort(key=lambda x: (x["weight"], x["similarity_score"]), reverse=True)


            return binding_precedents

        except Exception as e:
            error_details = {"error": str(e), "case_id": case_id}
            raise

    def _get_case_context(self, case_id: str) -> Dict[str, Any]:
        """Retrieve case context from case processor (placeholder)."""
        # Would call case_processor.get_case_summary(case_id)
        return {
            "case_id": case_id,
            "jurisdiction": "federal",
            "summary": "Sample case summary",
            "legal_principles": ["principle_1", "principle_2"]
        }

    def _get_case_facts(self, case_id: str) -> Dict[str, Any]:
        """Retrieve case facts from factual analysis (placeholder)."""
        # Would call factual_analysis.analyze_facts(case_id)
        return {
            "established_facts": [
                {"claim": {"text": "Plaintiff claims breach of contract."}},
                {"claim": {"text": "Defendant failed to deliver services."}}
            ]
        }