import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field
import spacy
import json
from datetime import datetime
from .issue_identification import IssueIdentifier, LegalIssue
from .statutory_interpretation import StatutoryInterpreter
from .legal_argument_evaluator import ArgumentEvaluator
from ..EvidenceAnalysisEngine import FactualAnalyzer
from ..precedent_matching.precedent_matcher import PrecedentMatcher
from ..LegalKnowledgeBaseModule import LegalKnowledgeBase, LegalStatute, LegalPrinciple, CaseLaw, LegalTest


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
class LegalAnalysis:
    """Represents the complete legal analysis for a case."""
    case_id: str
    legal_issues: List[Dict[str, Any]] = field(default_factory=list)
    applied_laws: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_chains: List[Dict[str, Any]] = field(default_factory=list)
    conclusions: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

class LegalReasoner:
    """Implements core reasoning algorithms to apply legal principles to factual scenarios."""

    def __init__(self):
        """Initialize the legal reasoner with its components."""
        self.issue_identifier = IssueIdentifier()
        self.statutory_interpreter = StatutoryInterpreter()
        self.argument_evaluator = ArgumentEvaluator()
        self.factual_analyzer = FactualAnalyzer()
        self.precedent_matcher = PrecedentMatcher()
        self.kb_manager = LegalKnowledgeBase()

    def analyze_legal_issues(self, case_id: str) -> LegalAnalysis:
        """
        Analyze all legal issues in a case.

        Args:
            case_id: Unique case identifier

        Returns:
            LegalAnalysis object containing complete analysis
        """
        try:
            # Get case context
            case_context = self._get_case_context(case_id)

            # Identify legal issues
            legal_issues = self.issue_identifier.identify_legal_issues(case_id)

            # Get factual analysis
            factual_analysis = self.factual_analyzer.analyze_facts(case_id)

            # Initialize analysis
            analysis = LegalAnalysis(case_id=case_id, metadata={"analysis_date": datetime.now().isoformat()})

            # Process each legal issue
            for issue in legal_issues:
                issue_analysis = self._analyze_single_issue(issue, factual_analysis, case_context)
                analysis.legal_issues.append(issue_analysis)

                # Apply relevant laws and precedents
                applied_law = self.apply_law_to_facts(issue_analysis["legal_provision"], factual_analysis)
                analysis.applied_laws.append(applied_law)

                # Generate reasoning chain
                reasoning = self._create_reasoning_chain(issue_analysis, applied_law, factual_analysis)
                analysis.reasoning_chains.append(reasoning)

                # Add conclusions
                analysis.conclusions.append({
                    "issue_id": issue.id,
                    "conclusion": issue_analysis["conclusion"],
                    "confidence": issue_analysis["confidence"]
                })

            return analysis

        except Exception as e:
            error_details = {"error": str(e), "case_id": case_id}
            raise

    def apply_law_to_facts(self, legal_provision: Dict[str, Any], case_facts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply a legal provision to case facts.

        Args:
            legal_provision: Statute or principle
            case_facts: Established case facts

        Returns:
            Dictionary containing application analysis and conclusions
        """
        try:
            # Extract statute or principle ID
            statute_id = legal_provision.get("statutes", [{}])[0].get("id") if legal_provision.get("statutes") else None
            principle_id = legal_provision.get("principles", [{}])[0].get("id") if legal_provision.get("principles") else None

            # Interpret the legal provision
            interpretation = None
            if statute_id:
                interpretation = self.statutory_interpreter.interpret_statute(statute_id, {"facts": case_facts})
            elif principle_id:
                principle = self.kb_manager.get_legal_principle(principle_id)
                interpretation = {
                    "plain_meaning": principle.description if principle else "",
                    "elements": legal_provision.get("elements", []),
                    "ambiguities": [],
                    "contextual_analysis": {"relevant_facts": [], "related_precedents": []},
                    "confidence": 0.5
                }

            # Match facts to legal elements
            elements_analysis = self._match_facts_to_elements(
                legal_provision.get("elements", []),
                case_facts.get("established_facts", [])
            )

            # Find relevant precedents with fallback
            try:
                precedents = self.precedent_matcher.find_relevant_precedents(
                    case_facts.get("case_id", ""), limit=5
                )
            except Exception as e:
                logger.warning(f"Precedent matching failed: {str(e)}. Using fallback precedents.")
                precedents = self.kb_manager.search_case_law(
                    query="",
                    jurisdiction=case_facts.get("jurisdiction", "federal"),
                    limit=5
                )

            # Synthesize application
            application = {
                "provision_id": statute_id or principle_id or "unknown",
                "interpretation": interpretation.to_dict() if interpretation else {},
                "elements_analysis": elements_analysis,
                "relevant_precedents": [p["details"] if isinstance(p, dict) else p.to_dict() for p in precedents],
                "conclusion": self._derive_conclusion(elements_analysis, precedents),
                "confidence": self._calculate_confidence(elements_analysis, precedents)
            }

            return application

        except Exception as e:
            error_details = {"error": str(e), "provision_id": legal_provision.get("id", "unknown")}
            raise

    def evaluate_competing_theories(self, theory_list: List[Dict[str, Any]], 
                                 case_facts: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Evaluate competing legal theories.

        Args:
            theory_list: List of competing legal theories
            case_facts: Established case facts

        Returns:
            List of evaluated theories with strengths/weaknesses
        """
        evaluated_theories = []

        try:
            for theory in theory_list:
                # Get precedents with fallback
                try:
                    precedents = self.precedent_matcher.find_relevant_precedents(
                        case_facts.get("case_id", ""), limit=5
                    )
                except Exception as e:
                    logger.warning(f"Precedent matching failed: {str(e)}. Using fallback precedents.")
                    precedents = self.kb_manager.search_case_law(
                        query=theory.get("description", ""),
                        jurisdiction=case_facts.get("jurisdiction", "federal"),
                        limit=5
                    )

                evaluation = self.argument_evaluator.evaluate_argument(
                    theory,
                    None,  # No opposing argument for initial evaluation
                    {
                        "facts": case_facts,
                        "precedents": precedents
                    }
                )

                evaluated_theories.append({
                    "theory_id": theory.get("id", "unknown"),
                    "description": theory.get("description", ""),
                    "evaluation": evaluation.to_dict(),
                    "strength_score": evaluation.score,
                    "weaknesses": evaluation.weaknesses,
                    "supporting_facts": evaluation.supporting_facts
                })

            # Rank theories
            evaluated_theories.sort(key=lambda x: x["strength_score"], reverse=True)


            return evaluated_theories

        except Exception as e:
            error_details = {"error": str(e), "theory_count": len(theory_list)}
            raise

    def apply_legal_test(self, test_id: str, case_facts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply a legal test to case facts.

        Args:
            test_id: Identifier for legal test
            case_facts: Established case facts

        Returns:
            Dictionary containing test application and results
        """
        try:
            # Get legal test from knowledge base
            legal_test = self.kb_manager.get_legal_test(test_id)
            if not legal_test:
                raise ValueError(f"Legal test {test_id} not found")

            # Apply each step of the test
            results = []
            for step in legal_test.steps:
                step_result = self._apply_test_step(step, case_facts)
                results.append(step_result)

            # Synthesize test outcome
            outcome = {
                "test_id": test_id,
                "name": legal_test.name,
                "steps": results,
                "overall_result": self._calculate_test_outcome(results),
                "confidence": self._calculate_test_confidence(results),
                "supporting_facts": [f for r in results for f in r.get("supporting_facts", [])]
            }

            return outcome

        except Exception as e:
            error_details = {"error": str(e), "test_id": test_id}
            raise

    def _get_case_context(self, case_id: str) -> Dict[str, Any]:
        """Retrieve case context from case processor."""
        # Placeholder: Would call case_processor.get_case_summary(case_id)
        return {"case_id": case_id, "jurisdiction": "federal", "legal_issues": [], "summary": "Sample case summary"}

    def _analyze_single_issue(self, issue: LegalIssue, factual_analysis: Dict[str, Any], 
                           case_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single legal issue."""
        provisions = self.issue_identifier.map_issues_to_legal_provisions([issue], case_context["jurisdiction"])
        issue_provision = provisions.get(issue.id, {})

        # Apply law to facts
        application = self.apply_law_to_facts(issue_provision, factual_analysis)

        return {
            "issue_id": issue.id,
            "description": issue.description,
            "legal_provision": issue_provision,
            "application": application,
            "conclusion": application["conclusion"],
            "confidence": application["confidence"]
        }

    def _match_facts_to_elements(self, elements: List[str], facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Match case facts to legal elements."""
        matches = []

        for element in elements:
            element_doc = nlp(element) if nlp else None
            best_match = None
            best_score = 0.0

            for fact in facts:
                fact_text = fact.get("claim", {}).get("text", "")
                fact_doc = nlp(fact_text) if nlp and fact_text else None

                if element_doc and fact_doc:
                    similarity = element_doc.similarity(fact_doc)
                    if similarity > best_score and similarity > 0.5:
                        best_score = similarity
                        best_match = fact
                else:
                    # Fallback: keyword matching
                    element_words = set(element.lower().split())
                    fact_words = set(fact_text.lower().split())
                    overlap = len(element_words.intersection(fact_words)) / len(element_words)
                    if overlap > best_score and overlap > 0.3:
                        best_score = overlap
                        best_match = fact

            matches.append({
                "element": element,
                "matched_fact": best_match,
                "confidence": best_score
            })

        return matches

    def _derive_conclusion(self, elements_analysis: List[Dict[str, Any]], 
                         precedents: List[Any]) -> str:
        """Derive conclusion from elements analysis and precedents."""
        satisfied_elements = sum(1 for ea in elements_analysis if ea["matched_fact"] and ea["confidence"] > 0.5)
        total_elements = len(elements_analysis)

        # Simple majority rule for conclusion
        if satisfied_elements / total_elements >= 0.5:
            return "Legal standard met based on factual findings"
        else:
            return "Legal standard not met based on factual findings"

    def _calculate_confidence(self, elements_analysis: List[Dict[str, Any]], 
                           precedents: List[Any]) -> float:
        """Calculate confidence in the analysis."""
        element_confidences = [ea["confidence"] for ea in elements_analysis if ea["matched_fact"]]
        precedent_weights = []
        for p in precedents:
            if isinstance(p, dict):
                precedent_weights.append(p.get("similarity_score", 0.5))
            else:
                precedent_weights.append(getattr(p, "similarity_score", 0.5))

        avg_element_conf = sum(element_confidences) / len(element_confidences) if element_confidences else 0.5
        avg_precedent_conf = sum(precedent_weights) / len(precedent_weights) if precedent_weights else 0.5

        return (avg_element_conf * 0.6) + (avg_precedent_conf * 0.4)

    def _create_reasoning_chain(self, issue_analysis: Dict[str, Any], applied_law: Dict[str, Any], 
                             factual_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create a structured reasoning chain."""
        premises = [
            {"type": "fact", "content": f["claim"]["text"]} 
            for f in factual_analysis.get("established_facts", [])
        ]
        premises.extend([
            {"type": "law", "content": law["interpretation"]} 
            for law in applied_law.get("elements_analysis", [])
        ])

        return {
            "issue_id": issue_analysis["issue_id"],
            "premises": premises,
            "conclusion": issue_analysis["conclusion"],
            "confidence": issue_analysis["confidence"]
        }

    def _apply_test_step(self, step: str, case_facts: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a single step of a legal test."""
        step_doc = nlp(step) if nlp else None
        matching_facts = []
        confidence = 0.0

        for fact in case_facts.get("established_facts", []):
            fact_text = fact.get("claim", {}).get("text", "")
            fact_doc = nlp(fact_text) if nlp and fact_text else None

            if step_doc and fact_doc:
                similarity = step_doc.similarity(fact_doc)
                if similarity > 0.5:
                    matching_facts.append(fact)
                    confidence = max(confidence, similarity)
            else:
                # Fallback: keyword matching
                step_words = set(step.lower().split())
                fact_words = set(fact_text.lower().split())
                overlap = len(step_words.intersection(fact_words)) / len(step_words)
                if overlap > 0.3:
                    matching_facts.append(fact)
                    confidence = max(confidence, overlap)

        return {
            "step_description": step,
            "matching_facts": matching_facts,
            "confidence": confidence,
            "result": "met" if matching_facts else "not met"
        }

    def _calculate_test_outcome(self, step_results: List[Dict[str, Any]]) -> str:
        """Calculate overall test outcome."""
        met_steps = sum(1 for r in step_results if r["result"] == "met")
        return "passed" if met_steps / len(step_results) >= 0.5 else "failed"

    def _calculate_test_confidence(self, step_results: List[Dict[str, Any]]) -> float:
        """Calculate confidence in test results."""
        confidences = [r["confidence"] for r in step_results]
        return sum(confidences) / len(confidences) if confidences else 0.5