#EvidenceAnalysisEngine.py

"""
Evidence Analysis Engine for AI Judge System
Processes and categorizes evidence submitted in legal cases.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
import spacy
from collections import defaultdict

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

# Define constants for evidence categories
EVIDENCE_CATEGORIES = {
    "TESTIMONY": "Statements from witnesses or parties",
    "DOCUMENT": "Written records, contracts, or communications",
    "PHYSICAL": "Tangible objects or items",
    "EXPERT": "Expert opinions or analysis",
    "DIGITAL": "Electronic records, emails, or digital media",
    "DEMONSTRATIVE": "Visual representations or reconstructions",
    "CIRCUMSTANTIAL": "Indirect evidence requiring inference",
    "CHARACTER": "Evidence about a person's character",
    "HEARSAY": "Second-hand information",
    "OTHER": "Evidence not fitting other categories"
}

@dataclass
class EvidenceItem:
    """Represents a single piece of evidence with metadata."""
    id: str
    content: str
    source: Optional[str] = None
    category: Optional[str] = None
    admissibility: Optional[bool] = None
    admissibility_issues: Optional[Dict[str, Any]] = None
    reliability_score: Optional[float] = None
    reliability_factors: Optional[Dict[str, Any]] = None
    relevance_scores: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

@dataclass
class LegalIssue:
    """Represents a legal issue in a case."""
    id: str
    description: str
    category: str
    elements: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)

@dataclass
class EvidenceProfile:
    """Collection of processed evidence for a case with analysis."""
    case_id: str
    evidence_items: List[EvidenceItem] = field(default_factory=list)
    summary: Optional[Dict[str, Any]] = None
    factual_disputes: List[Dict[str, Any]] = field(default_factory=list)
    consistency_analysis: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

class EvidenceValidator:
    """Validates evidence according to legal standards of admissibility."""
    
    def __init__(self):
        """Initialize the evidence validator."""
        pass
    
    def validate_evidence(self, evidence_item: Dict[str, Any], jurisdiction: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate evidence according to admissibility rules.
        
        Args:
            evidence_item: Evidence submission
            jurisdiction: Applicable jurisdiction
            
        Returns:
            Tuple of (validity status, validation issues if any)
        """
        # Check if the evidence has required fields
        validation_issues = {}
        
        # Basic structural validation
        if "content" not in evidence_item or not evidence_item["content"]:
            validation_issues["missing_content"] = "Evidence must have content"
        
        if "source" not in evidence_item or not evidence_item["source"]:
            validation_issues["missing_source"] = "Evidence should have a source"
        
        # Check admissibility based on jurisdiction rules
        admissibility, admissibility_issues = self.check_admissibility(evidence_item, jurisdiction)
        
        if admissibility_issues:
            validation_issues["admissibility"] = admissibility_issues
        
        # Check for privileged information
        if "content" in evidence_item and evidence_item["content"]:
            privileged_info = self.identify_privileged_information(evidence_item["content"])
            if privileged_info:
                validation_issues["privileged_information"] = privileged_info
                
        # Check for hearsay
        if "content" in evidence_item and evidence_item["content"]:
            hearsay_issues = self.identify_hearsay(evidence_item["content"])
            if hearsay_issues:
                validation_issues["hearsay"] = hearsay_issues
        
        # Evidence is valid if there are no issues
        is_valid = len(validation_issues) == 0
        
        return is_valid, validation_issues
    
    def check_admissibility(self, evidence_item: Dict[str, Any], jurisdiction: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if evidence is admissible under given jurisdiction.
        
        Args:
            evidence_item: Evidence submission
            jurisdiction: Applicable jurisdiction
            
        Returns:
            Tuple of (admissibility status, admissibility issues and rules)
        """
        # Default to admissible
        is_admissible = True
        issues = {}
        
        # Check evidence type-specific admissibility rules
        evidence_type = evidence_item.get("category", "").upper()
        
        # Apply jurisdiction-specific rules
        if jurisdiction.lower() == "federal":
            # Federal Rules of Evidence checks
            if evidence_type == "HEARSAY":
                # Check for hearsay exceptions
                exceptions = self._check_hearsay_exceptions(evidence_item)
                if not exceptions:
                    is_admissible = False
                    issues["hearsay"] = "Hearsay evidence without applicable exception"
                else:
                    issues["hearsay_exception"] = exceptions
            
            elif evidence_type == "CHARACTER":
                # Character evidence limitations in federal court
                is_admissible = False
                issues["character_evidence"] = "Character evidence generally not admissible to prove conduct"
                
                # Check for exceptions
                if evidence_item.get("purpose") in ["impeachment", "character in issue"]:
                    is_admissible = True
                    issues.pop("character_evidence")
                    issues["character_exception"] = "Admissible for limited purpose"
        
        # Common rules across jurisdictions
        if "privileged" in evidence_item and evidence_item["privileged"]:
            is_admissible = False
            issues["privilege"] = "Evidence contains privileged information"
        
        # Return admissibility result with applicable rules
        applicable_rules = self._get_applicable_rules(evidence_type, jurisdiction)
        
        return is_admissible, {
            "is_admissible": is_admissible,
            "issues": issues,
            "applicable_rules": applicable_rules
        }
    
    def _check_hearsay_exceptions(self, evidence_item: Dict[str, Any]) -> Optional[str]:
        """
        Check if hearsay evidence meets any exceptions.
        
        Args:
            evidence_item: Evidence submission
            
        Returns:
            Exception name if applicable, None otherwise
        """
        content = evidence_item.get("content", "").lower()
        metadata = evidence_item.get("metadata", {})
        
        # Check for common hearsay exceptions
        if "excited utterance" in metadata.get("context", "").lower():
            return "Excited Utterance"
        
        if "business" in content and "record" in content:
            return "Business Record"
        
        if "medical" in content and ("diagnosis" in content or "treatment" in content):
            return "Medical Diagnosis/Treatment"
        
        if "state of mind" in metadata.get("purpose", "").lower():
            return "State of Mind"
        
        # No exception found
        return None
    
    def _get_applicable_rules(self, evidence_type: str, jurisdiction: str) -> List[str]:
        """
        Get applicable evidence rules for the given type and jurisdiction.
        
        Args:
            evidence_type: Category of evidence
            jurisdiction: Applicable jurisdiction
            
        Returns:
            List of applicable rule references
        """
        rules = []
        
        # Federal Rules of Evidence references
        if jurisdiction.lower() == "federal":
            if evidence_type == "HEARSAY":
                rules.append("FRE 801-804: Hearsay and Exceptions")
            elif evidence_type == "EXPERT":
                rules.append("FRE 702-705: Expert Testimony")
            elif evidence_type == "CHARACTER":
                rules.append("FRE 404: Character Evidence")
            elif evidence_type == "DOCUMENT":
                rules.append("FRE 1001-1008: Authentication and Best Evidence")
        
        # Add general relevance rule
        rules.append("Relevance: Evidence must be relevant to an issue in the case")
        
        return rules
    
    def identify_hearsay(self, text_content: str) -> List[Dict[str, Any]]:
        """
        Identify potential hearsay in text content.
        
        Args:
            text_content: Text to check for hearsay
            
        Returns:
            List of potential hearsay instances with context
        """
        hearsay_instances = []
        
        # Simple heuristic: Look for reporting verbs and quotations
        hearsay_markers = [
            "said", "told", "stated", "mentioned", "claimed", "asserted",
            "according to", "heard from", "reported that"
        ]
        
        if nlp:
            doc = nlp(text_content)
            
            # Look for sentences with reporting structure
            for sent in doc.sents:
                sent_text = sent.text.lower()
                
                # Check for hearsay markers
                if any(marker in sent_text for marker in hearsay_markers):
                    # Check if there's a reporting structure (someone said something)
                    for token in sent:
                        if token.lemma_ in ["say", "tell", "state", "claim", "mention", "assert", "report"]:
                            hearsay_instances.append({
                                "text": sent.text,
                                "reporting_verb": token.text,
                                "confidence": 0.8
                            })
                            break
                
                # Check for quotations which might indicate hearsay
                if '"' in sent_text or "'" in sent_text:
                    hearsay_instances.append({
                        "text": sent.text,
                        "quotation": True,
                        "confidence": 0.6
                    })
        else:
            # Fallback if NLP not available
            sentences = text_content.split('.')
            for sentence in sentences:
                sentence = sentence.strip().lower()
                if any(marker in sentence for marker in hearsay_markers):
                    hearsay_instances.append({
                        "text": sentence,
                        "confidence": 0.5
                    })
        
        return hearsay_instances
    
    def identify_privileged_information(self, text_content: str) -> List[Dict[str, Any]]:
        """
        Identify potentially privileged information in text.
        
        Args:
            text_content: Text to check for privileged information
            
        Returns:
            List of potentially privileged information segments
        """
        privileged_instances = []
        
        # Check for common privilege indicators
        privilege_indicators = {
            "attorney-client": ["attorney", "lawyer", "counsel", "legal advice", "privileged", "confidential"],
            "doctor-patient": ["doctor", "physician", "medical", "diagnosis", "patient", "confidential health"],
            "spousal": ["spouse", "husband", "wife", "married", "marital communication"],
            "clergy": ["priest", "rabbi", "imam", "clergy", "confession", "spiritual advice"]
        }
        
        if nlp:
            doc = nlp(text_content)
            
            # Analyze by sentence
            for sent in doc.sents:
                sent_text = sent.text.lower()
                
                # Check each privilege type
                for privilege_type, indicators in privilege_indicators.items():
                    if any(indicator in sent_text for indicator in indicators):
                        privileged_instances.append({
                            "text": sent.text,
                            "privilege_type": privilege_type,
                            "confidence": 0.7
                        })
                        break
        else:
            # Fallback without NLP
            sentences = text_content.split('.')
            for sentence in sentences:
                sentence = sentence.strip().lower()
                
                for privilege_type, indicators in privilege_indicators.items():
                    if any(indicator in sentence for indicator in indicators):
                        privileged_instances.append({
                            "text": sentence,
                            "privilege_type": privilege_type,
                            "confidence": 0.5
                        })
                        break
        
        return privileged_instances

class FactualAnalyzer:
    """Analyzes factual claims and evidence to establish factual basis for legal reasoning."""
    
    def __init__(self):
        """Initialize the factual analyzer."""
        pass
    
    def analyze_facts(self, case_id: str, evidence_items: List[EvidenceItem]) -> Dict[str, Any]:
        """
        Analyze the facts presented in a case.
        
        Args:
            case_id: Unique case identifier
            evidence_items: List of evidence items
            
        Returns:
            FactualAnalysis object (analysis of case facts)
        """
        # Extract all factual claims from evidence
        factual_claims = self._extract_factual_claims(evidence_items)
        
        # Group related claims
        claim_groups = self._group_related_claims(factual_claims)
        
        # Identify contradictions
        contradictions = self._identify_contradictions(claim_groups)
        
        # Create fact summary
        fact_summary = self._create_fact_summary(claim_groups, contradictions)
        
        return {
            "case_id": case_id,
            "factual_claims": factual_claims,
            "claim_groups": claim_groups,
            "contradictions": contradictions,
            "fact_summary": fact_summary
        }
    
    def _extract_factual_claims(self, evidence_items: List[EvidenceItem]) -> List[Dict[str, Any]]:
        """
        Extract factual claims from evidence items.
        
        Args:
            evidence_items: List of evidence items
            
        Returns:
            List of extracted factual claims
        """
        factual_claims = []
        
        for item in evidence_items:
            # Process the content of the evidence item
            if nlp:
                doc = nlp(item.content)
                
                # Extract sentences that contain factual claims
                for sent in doc.sents:
                    # Simple heuristic: sentences with entities, dates, or numbers
                    # are likely to contain factual claims
                    has_entities = len(sent.ents) > 0
                    has_numbers = any(token.like_num for token in sent)
                    
                    if has_entities or has_numbers:
                        # Extract entities
                        entities = [
                            {"text": ent.text, "type": ent.label_} 
                            for ent in sent.ents
                        ]
                        
                        factual_claims.append({
                            "text": sent.text,
                            "source_evidence_id": item.id,
                            "source_type": item.category if item.category else "UNKNOWN",
                            "entities": entities,
                            "reliability": item.reliability_score if item.reliability_score else 0.5
                        })
            else:
                # Fallback without NLP
                sentences = item.content.split('.')
                for sentence in sentences:
                    if sentence.strip():
                        factual_claims.append({
                            "text": sentence.strip(),
                            "source_evidence_id": item.id,
                            "source_type": item.category if item.category else "UNKNOWN",
                            "reliability": item.reliability_score if item.reliability_score else 0.5
                        })
        
        return factual_claims
    
    def _group_related_claims(self, factual_claims: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group related factual claims.
        
        Args:
            factual_claims: List of factual claims
            
        Returns:
            Dictionary of topic groups with related claims
        """
        claim_groups = defaultdict(list)
        
        if nlp:
            # Create embeddings for claims
            claim_texts = [claim["text"] for claim in factual_claims]
            docs = list(nlp.pipe(claim_texts))
            
            # Simple topic extraction using named entities
            for i, doc in enumerate(docs):
                # Use the main entities as topics
                main_entities = [ent.text for ent in doc.ents]
                
                if main_entities:
                    # Use the most frequent entity type as the topic
                    entity_counts = defaultdict(int)
                    for ent in doc.ents:
                        entity_counts[ent.label_] += 1
                    
                    main_topic = max(entity_counts.items(), key=lambda x: x[1])[0]
                    
                    # Add the claim to this topic group
                    claim_groups[main_topic].append(factual_claims[i])
                else:
                    # No entities found, use a generic topic
                    claim_groups["GENERAL"].append(factual_claims[i])
        else:
            # Without NLP, use source types as grouping
            for claim in factual_claims:
                source_type = claim.get("source_type", "UNKNOWN")
                claim_groups[source_type].append(claim)
        
        return dict(claim_groups)
    
    def _identify_contradictions(self, claim_groups: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Identify contradictions between factual claims.
        
        Args:
            claim_groups: Grouped factual claims
            
        Returns:
            List of identified contradictions
        """
        contradictions = []
        
        # For each group, compare claims for contradictions
        for topic, claims in claim_groups.items():
            if len(claims) < 2:
                continue
            
            # Compare each pair of claims
            for i in range(len(claims)):
                for j in range(i + 1, len(claims)):
                    claim1 = claims[i]
                    claim2 = claims[j]
                    
                    # Check for potential contradictions
                    contradiction_score = self._calculate_contradiction_score(claim1["text"], claim2["text"])
                    
                    if contradiction_score > 0.7:  # Threshold for contradiction
                        contradictions.append({
                            "claim1": claim1,
                            "claim2": claim2,
                            "topic": topic,
                            "contradiction_score": contradiction_score
                        })
        
        return contradictions
    
    def _calculate_contradiction_score(self, text1: str, text2: str) -> float:
        """
        Calculate a contradiction score between two text statements.
        
        Args:
            text1: First text statement
            text2: Second text statement
            
        Returns:
            Contradiction score (0-1, higher means more likely contradiction)
        """
        if nlp:
            # Use spaCy to calculate semantic similarity
            doc1 = nlp(text1)
            doc2 = nlp(text2)
            
            # Calculate semantic similarity
            similarity = doc1.similarity(doc2)
            
            # Check for negation words that might indicate contradiction
            negation_terms = ["not", "never", "no", "didn't", "doesn't", "won't", "can't", "cannot"]
            has_negation1 = any(term in text1.lower().split() for term in negation_terms)
            has_negation2 = any(term in text2.lower().split() for term in negation_terms)
            
            # If statements are somewhat similar but one has negation and the other doesn't,
            # this suggests a contradiction
            if similarity > 0.5 and (has_negation1 != has_negation2):
                return 0.8  # High likelihood of contradiction
            
            # If statements are very similar but have negations, might be contradiction
            elif similarity > 0.7 and has_negation1 and has_negation2:
                return 0.6  # Moderate likelihood of contradiction
            
            # If statements are very dissimilar, probably not contradicting
            elif similarity < 0.3:
                return 0.1  # Low likelihood of contradiction
            
            # Default moderate score
            return 0.3
        else:
            # Basic heuristic without NLP
            # Check for direct opposition with negation words
            negation_terms = ["not", "never", "no", "didn't", "doesn't", "won't", "can't", "cannot"]
            has_negation1 = any(term in text1.lower().split() for term in negation_terms)
            has_negation2 = any(term in text2.lower().split() for term in negation_terms)
            
            # Check for common words to see if they're talking about the same thing
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            common_words = words1.intersection(words2)
            
            # If they share several words but one has negation and the other doesn't,
            # this might be a contradiction
            if len(common_words) > 3 and (has_negation1 != has_negation2):
                return 0.7
            
            return 0.2  # Low default score without NLP
    
    def _create_fact_summary(self, claim_groups: Dict[str, List[Dict[str, Any]]], 
                            contradictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a summary of established facts from claim groups.
        
        Args:
            claim_groups: Grouped factual claims
            contradictions: Identified contradictions
            
        Returns:
            Dictionary with fact summary
        """
        # Initialize fact summary
        fact_summary = {
            "established_facts": [],
            "disputed_facts": [],
            "topics": {}
        }
        
        # Find disputed fact topics based on contradictions
        disputed_topics = set(contradiction["topic"] for contradiction in contradictions)
        
        # Process each topic
        for topic, claims in claim_groups.items():
            # Skip empty claim groups
            if not claims:
                continue
                
            # Sort claims by reliability
            sorted_claims = sorted(claims, key=lambda x: x.get("reliability", 0), reverse=True)
            
            # Check if this topic has contradictions
            if topic in disputed_topics:
                # Add to disputed facts
                fact_summary["disputed_facts"].append({
                    "topic": topic,
                    "claims": sorted_claims
                })
                
                fact_summary["topics"][topic] = {
                    "status": "disputed",
                    "reliability": sum(claim.get("reliability", 0) for claim in claims) / len(claims)
                }
            else:
                # Add most reliable claim as established fact
                fact_summary["established_facts"].append({
                    "topic": topic,
                    "claim": sorted_claims[0]
                })
                
                fact_summary["topics"][topic] = {
                    "status": "established",
                    "reliability": sorted_claims[0].get("reliability", 0)
                }
        
        return fact_summary
    
    def identify_factual_disputes(self, plaintiff_claims: List[Dict[str, Any]], 
                                defendant_claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Identify factual disputes between plaintiff and defendant claims.
        
        Args:
            plaintiff_claims: List of plaintiff's factual assertions
            defendant_claims: List of defendant's factual assertions
            
        Returns:
            List of identified factual disputes
        """
        disputes = []
        
        # Group claims by topic
        plaintiff_topics = self._group_claims_by_topic(plaintiff_claims)
        defendant_topics = self._group_claims_by_topic(defendant_claims)
        
        # Find common topics
        common_topics = set(plaintiff_topics.keys()).intersection(set(defendant_topics.keys()))
        
        # For each common topic, check for disputes
        for topic in common_topics:
            plaintiff_topic_claims = plaintiff_topics[topic]
            defendant_topic_claims = defendant_topics[topic]
            
            # Compare claims to find disputes
            for p_claim in plaintiff_topic_claims:
                for d_claim in defendant_topic_claims:
                    contradiction_score = self._calculate_contradiction_score(p_claim["text"], d_claim["text"])
                    
                    if contradiction_score > 0.6:  # Threshold for dispute
                        disputes.append({
                            "topic": topic,
                            "plaintiff_claim": p_claim,
                            "defendant_claim": d_claim,
                            "contradiction_score": contradiction_score
                        })
        
        return disputes
    
    def _group_claims_by_topic(self, claims: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group claims by topic.
        
        Args:
            claims: List of claims
            
        Returns:
            Dictionary mapping topics to lists of related claims
        """
        topics = defaultdict(list)
        
        if nlp:
            # Use NLP to extract topics
            for claim in claims:
                doc = nlp(claim["text"])
                
                # Use entities as topics
                if doc.ents:
                    # Use most significant entity as topic
                    main_ent = max(doc.ents, key=lambda e: len(e.text))
                    topic = f"{main_ent.label_}:{main_ent.text}"
                    topics[topic].append(claim)
                else:
                    # Fallback to using noun chunks
                    noun_chunks = list(doc.noun_chunks)
                    if noun_chunks:
                        topic = str(max(noun_chunks, key=lambda nc: len(nc.text)))
                        topics[topic].append(claim)
                    else:
                        # Fallback to first few words
                        topic = " ".join(claim["text"].split()[:3])
                        topics[topic].append(claim)
        else:
            # Fallback without NLP
            for claim in claims:
                # Use first few words as topic
                words = claim["text"].split()
                topic = " ".join(words[:min(3, len(words))])
                topics[topic].append(claim)
        
        return dict(topics)
    
    def evaluate_testimony_consistency(self, testimony_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate consistency across testimony evidence.
        
        Args:
            testimony_items: List of testimony evidence
            
        Returns:
            Dictionary with consistency evaluation and contradictions
        """
        # Group testimony by source
        source_testimony = defaultdict(list)
        for item in testimony_items:
            source = item.get("source", "unknown")
            source_testimony[source].append(item)
        
        # Check internal consistency for each source
        internal_consistency = {}
        for source, items in source_testimony.items():
            if len(items) < 2:
                internal_consistency[source] = {
                    "score": 1.0,  # Only one statement, so consistent by default
                    "contradictions": []
                }
                continue
            
            # Check for contradictions within this source's testimony
            contradictions = []
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    item1 = items[i]
                    item2 = items[j]
                    
                    contradiction_score = self._calculate_contradiction_score(
                        item1.get("content", ""), 
                        item2.get("content", "")
                    )
                    
                    if contradiction_score > 0.7:
                        contradictions.append({
                            "statement1": item1.get("content", ""),
                            "statement2": item2.get("content", ""),
                            "score": contradiction_score
                        })
            
            # Calculate consistency score (inverse of contradiction rate)
            total_comparisons = (len(items) * (len(items) - 1)) / 2
            consistency_score = 1.0 - (len(contradictions) / total_comparisons if total_comparisons > 0 else 0)
            
            internal_consistency[source] = {
                "score": consistency_score,
                "contradictions": contradictions
            }
        
        # Check cross-source consistency
        cross_source_contradictions = []
        sources = list(source_testimony.keys())
        
        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                source1 = sources[i]
                source2 = sources[j]
                
                # Compare each testimony from source1 with each from source2
                for item1 in source_testimony[source1]:
                    for item2 in source_testimony[source2]:
                        contradiction_score = self._calculate_contradiction_score(
                            item1.get("content", ""),
                            item2.get("content", "")
                        )
                        
                        if contradiction_score > 0.7:
                            cross_source_contradictions.append({
                                "source1": source1,
                                "statement1": item1.get("content", ""),
                                "source2": source2,
                                "statement2": item2.get("content", ""),
                                "score": contradiction_score
                            })
        
        # Calculate overall consistency score
        overall_score = sum(src["score"] for src in internal_consistency.values()) / len(internal_consistency) if internal_consistency else 0.0
        
        return {
            "overall_consistency": overall_score,
            "internal_consistency": internal_consistency,
            "cross_source_contradictions": cross_source_contradictions
        }
    
    def create_fact_summary(self, evidence_profile: EvidenceProfile, case_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a summary of established facts from the evidence profile.

        Args:
            evidence_profile: Processed evidence profile containing evidence items and analysis
            case_context: Case details and legal issues

        Returns:
            Dictionary containing summarized facts, their strength, and relevance to legal issues
        """
        fact_summary = {
            "case_id": evidence_profile.case_id,
            "established_facts": [],
            "disputed_facts": [],
            "legal_issue_relevance": {},
            "summary_metrics": {
                "total_facts": 0,
                "avg_fact_strength": 0.0,
                "facts_per_issue": {}
            }
        }

        # Extract facts from evidence items
        facts = []
        for item in evidence_profile.evidence_items:
            if item.admissibility and item.content:  # Only consider admissible evidence
                # Use NLP if available to extract factual claims
                if nlp:
                    doc = nlp(item.content)
                    for sent in doc.sents:
                        facts.append({
                            "text": sent.text,
                            "evidence_id": item.id,
                            "weight": item.reliability_score or 0.5,
                            "category": item.category,
                            "relevance_scores": item.relevance_scores or {},
                            "source_role": item.metadata.get("source_role", "unknown")
                        })
                else:
                    # Fallback: Treat content as a single fact
                    facts.append({
                        "text": item.content,
                        "evidence_id": item.id,
                        "weight": item.reliability_score or 0.5,
                        "category": item.category,
                        "relevance_scores": item.relevance_scores or {},
                        "source_role": item.metadata.get("source_role", "unknown")
                    })

        # Classify facts as established or disputed
        for fact in facts:
            # Check if fact is disputed based on factual disputes in evidence profile
            is_disputed = any(
                dispute["claim_1"]["text"] == fact["text"] or dispute["claim_2"]["text"] == fact["text"]
                for dispute in evidence_profile.factual_disputes
            )

            fact_entry = {
                "text": fact["text"],
                "evidence_id": fact["evidence_id"],
                "strength": fact["weight"],
                "category": fact["category"],
                "relevance": fact["relevance_scores"],
                "source_role": fact["source_role"]
            }

            if is_disputed:
                fact_summary["disputed_facts"].append(fact_entry)
            else:
                fact_summary["established_facts"].append(fact_entry)

        # Map facts to legal issues
        legal_issues = case_context.get("legal_issues", [])
        for issue in legal_issues:
            issue_facts = []
            for fact in fact_summary["established_facts"] + fact_summary["disputed_facts"]:
                if issue.id in fact["relevance"] and fact["relevance"][issue.id] > 0.3:  # Threshold for relevance
                    issue_facts.append({
                        "text": fact["text"],
                        "strength": fact["strength"],
                        "relevance_score": fact["relevance"][issue.id],
                        "disputed": fact in fact_summary["disputed_facts"]
                    })
            fact_summary["legal_issue_relevance"][issue.id] = {
                "description": issue.description,
                "facts": issue_facts,
                "total_facts": len(issue_facts),
                "avg_relevance": (
                    sum(f["relevance_score"] for f in issue_facts) / len(issue_facts)
                    if issue_facts else 0.0
                )
            }
            fact_summary["summary_metrics"]["facts_per_issue"][issue.id] = len(issue_facts)

        # Calculate summary metrics
        all_facts = fact_summary["established_facts"] + fact_summary["disputed_facts"]
        fact_summary["summary_metrics"]["total_facts"] = len(all_facts)
        fact_summary["summary_metrics"]["avg_fact_strength"] = (
            sum(f["strength"] for f in all_facts) / len(all_facts) if all_facts else 0.0
        )

        return fact_summary
    

    def _create_event_timeline(self, established_facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create a timeline of events from established facts.
        
        Args:
            established_facts: List of established facts
            
        Returns:
            List of events with timestamps
        """
        timeline = []
        
        if nlp:
            for fact in established_facts:
                doc = nlp(fact["claim"]["text"])
                
                # Extract dates and associated events
                for ent in doc.ents:
                    if ent.label_ == "DATE":
                        timeline.append({
                            "date": ent.text,
                            "event": fact["claim"]["text"],
                            "topic": fact["topic"],
                            "reliability": fact["claim"].get("reliability", 0.5)
                        })
        
        # Sort timeline by date (basic heuristic, assumes dates are in recognizable format)
        timeline.sort(key=lambda x: x["date"], reverse=False)
        
        return timeline

class EvidenceWeighter:
    """Assigns weights to evidence based on type, credibility, and relevance."""
    
    def __init__(self):
        """Initialize the evidence weighter."""
        # Define baseline weights for different evidence types
        self.type_weights = {
            "TESTIMONY": 0.7,
            "DOCUMENT": 0.9,
            "PHYSICAL": 0.8,
            "EXPERT": 0.85,
            "DIGITAL": 0.75,
            "DEMONSTRATIVE": 0.6,
            "CIRCUMSTANTIAL": 0.5,
            "CHARACTER": 0.4,
            "HEARSAY": 0.3,
            "OTHER": 0.5
        }
    
    def weight_evidence(self, evidence_item: EvidenceItem, case_context: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate the weight of an evidence item.
        
        Args:
            evidence_item: Evidence item to weight
            case_context: Case details and legal issues
            
        Returns:
            Tuple of (weight, factors affecting weight)
        """
        # Initialize factors dictionary
        factors = {}
        
        # Base weight from evidence type
        base_weight = self.type_weights.get(evidence_item.category.upper() if evidence_item.category else "OTHER", 0.5)
        factors["type_weight"] = base_weight
        
        # Adjust for reliability
        reliability_adjustment = self._calculate_reliability_adjustment(evidence_item)
        factors["reliability_adjustment"] = reliability_adjustment
        
        # Adjust for relevance to case issues
        relevance_adjustment = self._calculate_relevance_adjustment(evidence_item, case_context)
        factors["relevance_adjustment"] = relevance_adjustment
        
        # Adjust for source credibility
        source_credibility = self._evaluate_source_credibility(evidence_item)
        factors["source_credibility"] = source_credibility
        
        # Combine weights
        final_weight = (
            base_weight * 
            (1 + reliability_adjustment) * 
            (1 + relevance_adjustment) * 
            source_credibility
        )
        
        # Ensure weight is between 0 and 1
        final_weight = min(max(final_weight, 0.0), 1.0)
        
        return final_weight, factors
    
    def _calculate_reliability_adjustment(self, evidence_item: EvidenceItem) -> float:
        """
        Calculate reliability adjustment for evidence.
        
        Args:
            evidence_item: Evidence item
            
        Returns:
            Reliability adjustment factor (-0.5 to 0.5)
        """
        if evidence_item.reliability_score is not None:
            # Use provided reliability score if available
            return evidence_item.reliability_score - 0.5
        
        # Basic heuristic based on evidence characteristics
        adjustment = 0.0
        
        # Penalize for hearsay
        if evidence_item.category == "HEARSAY":
            adjustment -= 0.2
        
        # Boost for verifiable sources
        if evidence_item.source and any(keyword in evidence_item.source.lower() for keyword in ["official", "government", "certified"]):
            adjustment += 0.2
            
        # Penalize for missing metadata
        if not evidence_item.metadata:
            adjustment -= 0.1
            
        return adjustment
    
    def _calculate_relevance_adjustment(self, evidence_item: EvidenceItem, case_context: Dict[str, Any]) -> float:
        """
        Calculate relevance adjustment for evidence.
        
        Args:
            evidence_item: Evidence item
            case_context: Case details and legal issues
            
        Returns:
            Relevance adjustment factor (-0.5 to 0.5)
        """
        if not case_context.get("legal_issues"):
            return 0.0
            
        adjustment = 0.0
        
        # Calculate relevance scores if not provided
        if evidence_item.relevance_scores is None:
            relevance_scores = self.evaluate_evidence_relevance(evidence_item, case_context["legal_issues"])
            evidence_item.relevance_scores = relevance_scores
            
        # Average relevance scores across legal issues
        if evidence_item.relevance_scores:
            avg_relevance = sum(evidence_item.relevance_scores.values()) / len(evidence_item.relevance_scores)
            adjustment = (avg_relevance - 0.5)  # Convert to adjustment factor
            
        return adjustment
    
    def _evaluate_source_credibility(self, evidence_item: EvidenceItem) -> float:
        """
        Evaluate the credibility of the evidence source.
        
        Args:
            evidence_item: Evidence item
            
        Returns:
            Credibility score (0.0 to 1.0)
        """
        credibility = 0.5  # Default neutral score
        
        if evidence_item.source:
            source_lower = evidence_item.source.lower()
            
            # Boost for authoritative sources
            if any(keyword in source_lower for keyword in ["court", "government", "official", "expert"]):
                credibility += 0.3
                
            # Penalize for anonymous or unclear sources
            if "anonymous" in source_lower or "unknown" in source_lower:
                credibility -= 0.2
                
            # Adjust based on metadata
            if evidence_item.metadata.get("verified", False):
                credibility += 0.2
                
        return min(max(credibility, 0.0), 1.0)
    
    def evaluate_witness_credibility(self, witness_data: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate witness credibility.
        
        Args:
            witness_data: Witness information and testimony
            
        Returns:
            Tuple of (credibility score, credibility factors)
        """
        credibility = 0.5
        factors = {}
        
        # Check witness background
        if witness_data.get("background"):
            background = witness_data["background"].lower()
            if any(keyword in background for keyword in ["expert", "professional", "official"]):
                credibility += 0.2
                factors["background"] = "Professional or expert status"
            if "criminal" in background or "conviction" in background:
                credibility -= 0.2
                factors["background"] = "Criminal history"
        
        # Check consistency with other testimony
        if witness_data.get("related_testimony"):
            consistency_score = self._evaluate_testimony_consistency(witness_data["related_testimony"])
            credibility += (consistency_score - 0.5)
            factors["consistency"] = f"Consistency score: {consistency_score}"
        
        return min(max(credibility, 0.0), 1.0), factors
    
    def _evaluate_testimony_consistency(self, related_testimony: List[Dict[str, Any]]) -> float:
        """
        Evaluate consistency of testimony with related statements.
        
        Args:
            related_testimony: List of related testimony items
            
        Returns:
            Consistency score (0.0 to 1.0)
        """
        if not related_testimony:
            return 0.5
            
        consistency_scores = []
        
        for i, testimony in enumerate(related_testimony):
            for j in range(i + 1, len(related_testimony)):
                score = 1.0 - self._calculate_contradiction_score(
                    testimony.get("content", ""),
                    related_testimony[j].get("content", "")
                )
                consistency_scores.append(score)
        
        return sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.5
    
    def evaluate_document_reliability(self, document_data: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate document reliability.
        
        Args:
            document_data: Document metadata and content
            
        Returns:
            Tuple of (reliability score, reliability factors)
        """
        reliability = 0.5
        factors = {}
        
        # Check document authenticity
        if document_data.get("metadata", {}).get("authenticated", False):
            reliability += 0.3
            factors["authentication"] = "Document is authenticated"
            
        # Check document origin
        if document_data.get("source"):
            source = document_data["source"].lower()
            if any(keyword in source for keyword in ["official", "government", "court"]):
                reliability += 0.2
                factors["source"] = "Official source"
                
        # Check for tampering indicators
        if document_data.get("metadata", {}).get("tampering_detected", False):
            reliability -= 0.3
            factors["tampering"] = "Potential tampering detected"
            
        return min(max(reliability, 0.0), 1.0), factors
    
    def evaluate_evidence_relevance(self, evidence_item: EvidenceItem, legal_issues: List[LegalIssue]) -> Dict[str, float]:
        """
        Evaluate evidence relevance to legal issues.
        
        Args:
            evidence_item: Evidence item
            legal_issues: List of legal issues
            
        Returns:
            Dictionary mapping issue IDs to relevance scores
        """
        relevance_scores = {}
        
        if nlp:
            evidence_doc = nlp(evidence_item.content)
            
            for issue in legal_issues:
                # Combine issue description and elements for comparison
                issue_text = f"{issue.description} {' '.join(issue.elements)}"
                issue_doc = nlp(issue_text)
                
                # Calculate semantic similarity
                similarity = evidence_doc.similarity(issue_doc)
                
                # Adjust based on keyword matches
                issue_keywords = set(issue_text.lower().split())
                evidence_keywords = set(evidence_item.content.lower().split())
                keyword_overlap = len(issue_keywords.intersection(evidence_keywords)) / len(issue_keywords) if issue_keywords else 0.0
                
                # Combine similarity and keyword overlap
                relevance_score = (similarity * 0.7) + (keyword_overlap * 0.3)
                
                relevance_scores[issue.id] = min(max(relevance_score, 0.0), 1.0)
        else:
            # Fallback without NLP
            for issue in legal_issues:
                issue_text = f"{issue.description} {' '.join(issue.elements)}".lower()
                evidence_text = evidence_item.content.lower()
                
                # Simple keyword-based relevance
                issue_words = set(issue_text.split())
                evidence_words = set(evidence_text.split())
                overlap = len(issue_words.intersection(evidence_words))
                
                # Normalize by issue length
                relevance_score = overlap / len(issue_words) if issue_words else 0.0
                relevance_scores[issue.id] = min(max(relevance_score, 0.0), 1.0)
        
        return relevance_scores
    
    def generate_evidence_weight_profile(self, case_id: str, evidence_items: List[EvidenceItem], 
                                       case_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate weight profile for all evidence items.
        
        Args:
            case_id: Unique case identifier
            evidence_items: List of evidence items
            case_context: Case details and legal issues
            
        Returns:
            Dictionary with weighted evidence profile
        """
        weight_profile = []
        
        for item in evidence_items:
            weight, factors = self.weight_evidence(item, case_context)
            
            weight_profile.append({
                "evidence_id": item.id,
                "weight": weight,
                "factors": factors,
                "category": item.category,
                "reliability_score": item.reliability_score,
                "relevance_scores": item.relevance_scores
            })
        
        return {
            "case_id": case_id,
            "evidence_weights": weight_profile,
            "summary": {
                "total_items": len(weight_profile),
                "avg_weight": sum(w["weight"] for w in weight_profile) / len(weight_profile) if weight_profile else 0.0,
                "categories": {cat: 0 for cat in EVIDENCE_CATEGORIES}
            }
        }

class EvidenceProcessor:
    """Main class for processing and analyzing evidence."""
    
    def __init__(self):
        """Initialize the evidence processor with its components."""
        self.validator = EvidenceValidator()
        self.factual_analyzer = FactualAnalyzer()
        self.weighter = EvidenceWeighter()
    
    def process_evidence(self, case_id: str, evidence_items: List[Dict[str, Any]], 
                       jurisdiction: str, case_context: Dict[str, Any]) -> EvidenceProfile:
        """
        Process evidence items for a case.
        
        Args:
            case_id: Unique case identifier
            evidence_items: List of evidence submissions
            jurisdiction: Applicable jurisdiction
            case_context: Case details and legal issues
            
        Returns:
            EvidenceProfile object with processed evidence
        """
        processed_items = []
        
        # Process each evidence item
        for idx, item in enumerate(evidence_items):
            # Validate evidence
            is_valid, validation_issues = self.validator.validate_evidence(item, jurisdiction)
            
            # Create EvidenceItem
            evidence_item = EvidenceItem(
                id=f"{case_id}_evidence_{idx}",
                content=item.get("content", ""),
                source=item.get("source"),
                category=self.categorize_evidence(item)[0],
                admissibility=is_valid,
                admissibility_issues=validation_issues,
                metadata=item.get("metadata", {})
            )
            
            # Assign weights
            weight, weight_factors = self.weighter.weight_evidence(evidence_item, case_context)
            evidence_item.reliability_score = weight
            evidence_item.reliability_factors = weight_factors
            
            # Calculate relevance
            if case_context.get("legal_issues"):
                evidence_item.relevance_scores = self.weighter.evaluate_evidence_relevance(
                    evidence_item, 
                    case_context["legal_issues"]
                )
            
            processed_items.append(evidence_item)
        
        # Perform factual analysis
        factual_analysis = self.factual_analyzer.analyze_facts(case_id, processed_items)
        
        # Create evidence profile
        profile = EvidenceProfile(
            case_id=case_id,
            evidence_items=processed_items,
            summary=self.get_evidence_summary(case_id, processed_items),
            factual_disputes=self.factual_analyzer.identify_factual_disputes(
                [c for c in factual_analysis["factual_claims"] if c.get("source_role", "").lower() == "plaintiff"],
                [c for c in factual_analysis["factual_claims"] if c.get("source_role", "").lower() == "defendant"]
            ),
            consistency_analysis=self.factual_analyzer.evaluate_testimony_consistency(
                [item.to_dict() for item in processed_items if item.category == "TESTIMONY"]
            )
        )
        
        return profile
    
    def categorize_evidence(self, evidence_item: Dict[str, Any]) -> Tuple[str, float]:
        """
        Categorize an evidence item.
        
        Args:
            evidence_item: Evidence submission
            
        Returns:
            Tuple of (category, confidence score)
        """
        content = evidence_item.get("content", "").lower()
        metadata = evidence_item.get("metadata", {})
        
        # Check metadata for explicit category
        if metadata.get("category"):
            category = metadata["category"].upper()
            if category in EVIDENCE_CATEGORIES:
                return category, 0.95
                
        # Keyword-based categorization
        category_scores = defaultdict(float)
        
        # Define keywords for each category
        category_keywords = {
            "TESTIMONY": ["witness", "testimony", "statement", "deposition", "sworn"],
            "DOCUMENT": ["contract", "agreement", "letter", "record", "document"],
            "PHYSICAL": ["object", "item", "evidence", "exhibit", "weapon"],
            "EXPERT": ["expert", "specialist", "analyst", "report", "opinion"],
            "DIGITAL": ["email", "video", "audio", "digital", "electronic"],
            "DEMONSTRATIVE": ["diagram", "chart", "model", "reconstruction"],
            "CIRCUMSTANTIAL": ["circumstance", "indirect", "implication"],
            "CHARACTER": ["character", "reputation", "history", "behavior"],
            "HEARSAY": ["heard", "said", "told", "reported"]
        }
        
        # Score based on keywords
        for category, keywords in category_keywords.items():
            for keyword in keywords:
                if keyword in content:
                    category_scores[category] += 0.2
                if metadata.get("description", "").lower().find(keyword) != -1:
                    category_scores[category] += 0.1               

        # Check for hearsay specifically
        hearsay_issues = self.validator.identify_hearsay(content)
        if metadata.get("context", "").lower() == "witness testimony":
            category_scores["TESTIMONY"] += 0.5
        else:
            hearsay_issues = self.validator.identify_hearsay(content)
            if hearsay_issues:
                category_scores["HEARSAY"] += 0.4
        
        if metadata.get("authenticated", False):
            category_scores["DOCUMENT"] += 0.4
        
        # If no strong matches, default to OTHER
        if not category_scores:
            return "OTHER", 0.5
            
        # Get category with highest score
        best_category = max(category_scores.items(), key=lambda x: x[1])[0]
        confidence = min(category_scores[best_category], 0.9)
        
        return best_category, confidence
    
    def get_evidence_summary(self, case_id: str, evidence_items: List[EvidenceItem]) -> Dict[str, Any]:
        """
        Generate a summary of evidence for a case.
        
        Args:
            case_id: Unique case identifier
            evidence_items: List of evidence items
            
        Returns:
            Dictionary with summarized evidence profile
        """
        category_counts = defaultdict(int)
        admissibility_counts = {"admissible": 0, "inadmissible": 0}
        avg_reliability = 0.0
        total_items = len(evidence_items)
        
        for item in evidence_items:
            category_counts[item.category] += 1
            if item.admissibility:
                admissibility_counts["admissible"] += 1
            else:
                admissibility_counts["inadmissible"] += 1
            avg_reliability += item.reliability_score or 0.5
        
        avg_reliability = avg_reliability / total_items if total_items > 0 else 0.0
        
        return {
            "case_id": case_id,
            "total_items": total_items,
            "category_distribution": dict(category_counts),
            "admissibility": admissibility_counts,
            "average_reliability": avg_reliability,
            "issues": [
                {
                    "evidence_id": item.id,
                    "issues": item.admissibility_issues
                } for item in evidence_items if item.admissibility_issues
            ]
        }

if __name__ == "__main__":
    # Example usage
    processor = EvidenceProcessor()
    
    # Sample evidence items
    sample_evidence = [
        {
            "content": "John Smith stated that he saw the defendant at the crime scene on July 15, 2023.",
            "source": "Witness John Smith",
            "metadata": {"context": "Witness testimony"}
        },
        {
            "content": "Contract signed between ABC Corp and XYZ Inc on January 1, 2023.",
            "source": "Court Records",
            "metadata": {"authenticated": True}
        }
    ]
    
    # Sample case context
    case_context = {
        "legal_issues": [
            LegalIssue(
                id="issue_1",
                description="Whether the defendant was present at the crime scene",
                category="Criminal",
                elements=["presence", "time", "location"]
            ),
            LegalIssue(
                id="issue_2",
                description="Whether the contract was valid",
                category="Contract",
                elements=["agreement", "consideration", "capacity"]
            )
        ]
    }
    
    # Process evidence
    profile = processor.process_evidence(
        case_id="case_001",
        evidence_items=sample_evidence,
        jurisdiction="federal",
        case_context=case_context
    )
    
    print("Evidence Profile:")
    print(profile.to_json())