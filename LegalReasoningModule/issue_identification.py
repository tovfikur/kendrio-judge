import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import spacy
from datetime import datetime
from ..LegalKnowledgeBaseModule import LegalKnowledgeBase, LegalStatute, LegalPrinciple


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
class LegalIssue:
    """Represents an identified legal issue in a case."""
    id: str
    description: str
    category: str
    relevant_facts: List[Dict[str, Any]]
    confidence: float
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)

class IssueIdentifier:
    """Identifies legal issues from case facts and context."""

    def __init__(self):
        """Initialize the issue identifier with knowledge base access."""
        self.kb_manager = LegalKnowledgeBase()

    def identify_legal_issues(self, case_id: str) -> List[LegalIssue]:
        """
        Identify legal issues in a case based on facts and context.

        Args:
            case_id: Unique case identifier

        Returns:
            List of LegalIssue objects
        """
        try:
            # Retrieve case context and facts (placeholder for case_processor integration)
            case_context = self._get_case_context(case_id)
            case_facts = self._get_case_facts(case_id)

            issues = []
            fact_texts = [fact.get("claim", {}).get("text", "") for fact in case_facts.get("established_facts", [])]

            # Process each fact to identify potential legal issues
            for idx, fact_text in enumerate(fact_texts):
                if not fact_text:
                    continue

                # Use NLP to analyze fact text
                fact_doc = nlp(fact_text) if nlp else None
                issue_description = fact_text[:200]  # Truncate for description
                issue_category = self._categorize_issue(fact_text, case_context.get("jurisdiction", "federal"))

                # Search knowledge base for relevant statutes and principles
                statutes = self.kb_manager.search_statutes(
                    query=fact_text,
                    jurisdiction=case_context.get("jurisdiction"),
                    limit=3
                )
                principles = self.kb_manager.search_legal_principles(
                    query=fact_text,
                    jurisdiction=case_context.get("jurisdiction"),
                    limit=3
                )

                # Calculate confidence based on matches
                confidence = self._calculate_issue_confidence(fact_doc, statutes, principles)

                issue = LegalIssue(
                    id=f"issue_{case_id}_{idx}",
                    description=issue_description,
                    category=issue_category,
                    relevant_facts=[fact for fact in case_facts.get("established_facts", []) if fact.get("claim", {}).get("text") == fact_text],
                    confidence=confidence,
                    metadata={
                        "identified_at": datetime.now().isoformat(),
                        "related_statutes": [s.id for s in statutes],
                        "related_principles": [p.id for p in principles]
                    }
                )
                issues.append(issue)

            return issues

        except Exception as e:
            error_details = {"error": str(e), "case_id": case_id}
            raise

    def map_issues_to_legal_provisions(self, issues: List[LegalIssue], jurisdiction: str) -> Dict[str, Dict[str, Any]]:
        """
        Map identified issues to relevant legal provisions (statutes, principles).

        Args:
            issues: List of LegalIssue objects
            jurisdiction: Jurisdiction for legal provisions

        Returns:
            Dictionary mapping issue IDs to legal provisions
        """
        try:
            provisions = {}
            for issue in issues:
                # Search for relevant statutes and principles
                statutes = self.kb_manager.search_statutes(
                    query=issue.description,
                    jurisdiction=jurisdiction,
                    category=issue.category,
                    limit=2
                )
                principles = self.kb_manager.search_legal_principles(
                    query=issue.description,
                    jurisdiction=jurisdiction,
                    category=issue.category,
                    limit=2
                )

                # Combine provisions
                provision_data = {
                    "statutes": [s.to_dict() for s in statutes],
                    "principles": [p.to_dict() for p in principles],
                    "elements": self._extract_elements(statutes, principles)
                }
                provisions[issue.id] = provision_data


            return provisions

        except Exception as e:
            error_details = {"error": str(e), "issue_count": len(issues)}
            raise

    def _get_case_context(self, case_id: str) -> Dict[str, Any]:
        """Retrieve case context from case processor (placeholder)."""
        # Would call case_processor.get_case_summary(case_id)
        return {"case_id": case_id, "jurisdiction": "federal", "case_type": "civil"}

    def _get_case_facts(self, case_id: str) -> Dict[str, Any]:
        """Retrieve case facts from factual analysis (placeholder)."""
        # Would call factual_analysis.analyze_facts(case_id)
        return {
            "established_facts": [
                {"claim": {"text": "Plaintiff claims breach of contract due to non-delivery."}},
                {"claim": {"text": "Defendant failed to provide services as agreed."}}
            ]
        }

    def _categorize_issue(self, fact_text: str, jurisdiction: str) -> str:
        """Categorize an issue based on fact text and jurisdiction."""
        # Simple keyword-based categorization
        fact_lower = fact_text.lower()
        if "contract" in fact_lower or "agreement" in fact_lower:
            return "contract_law"
        elif "injury" in fact_lower or "damage" in fact_lower:
            return "tort_law"
        else:
            return "general"

    def _calculate_issue_confidence(self, fact_doc: Any, statutes: List[LegalStatute], 
                                 principles: List[LegalPrinciple]) -> float:
        """Calculate confidence score for an identified issue."""
        if not fact_doc or not nlp:
            return 0.5  # Default confidence without NLP

        statute_scores = []
        principle_scores = []

        for statute in statutes:
            statute_doc = nlp(statute.text) if statute.text else None
            if statute_doc:
                statute_scores.append(fact_doc.similarity(statute_doc))

        for principle in principles:
            principle_doc = nlp(principle.description) if principle.description else None
            if principle_doc:
                principle_scores.append(fact_doc.similarity(principle_doc))

        avg_statute_score = sum(statute_scores) / len(statute_scores) if statute_scores else 0.5
        avg_principle_score = sum(principle_scores) / len(principle_scores) if principle_scores else 0.5
        return (avg_statute_score * 0.6) + (avg_principle_score * 0.4)

    def _extract_elements(self, statutes: List[LegalStatute], principles: List[LegalPrinciple]) -> List[str]:
        """Extract legal elements from statutes and principles."""
        elements = []
        for statute in statutes:
            if statute.sections:
                for section_text in statute.sections.values():
                    elements.append(section_text[:100])  # Truncate for brevity
        for principle in principles:
            elements.append(principle.description[:100])
        return elements