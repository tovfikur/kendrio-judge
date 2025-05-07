"""
Legal Reasoning Module for the AI Judge System

This module applies legal principles, standards, and tests to case facts and evidence profiles
to formulate legal conclusions and determine applicable precedents.

Dependencies:
- Evidence Analysis Engine: For evidence profile input
- Legal Knowledge Base: For retrieving legal rules, tests, and principles
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any
import json
import numpy as np
from enum import Enum



# Import dependencies from other modules
# These would be actual imports in a real implementation
# For now we'll define stub classes to represent external dependencies

# Stub for EvidenceProfile from Evidence Analysis Engine
@dataclass
class EvidenceProfile:
    case_id: str
    evidence_items: List[Any]
    factual_disputes: List[Dict]
    consistency_analysis: Dict

# Stub for LegalKnowledgeBase access
class LegalKnowledgeBase:
    def search_case_law(self, query, filters):
        pass
    
    def semantic_search(self, query_vector, collection):
        pass
    
    def get_legal_test(self, test_id):
        pass

# Data Structures for Legal Reasoning Module
class BurdenStandard(str, Enum):
    PREPONDERANCE = "PREPONDERANCE"
    CLEAR_AND_CONVINCING = "CLEAR_AND_CONVINCING"
    BEYOND_REASONABLE_DOUBT = "BEYOND_REASONABLE_DOUBT"

class ClaimType(str, Enum):
    CIVIL = "CIVIL"
    CRIMINAL = "CRIMINAL"
    ADMINISTRATIVE = "ADMINISTRATIVE"

class Party(str, Enum):
    PLAINTIFF = "PLAINTIFF"
    DEFENDANT = "DEFENDANT"
    PROSECUTION = "PROSECUTION"

@dataclass
class LegalIssue:
    id: str                    # Unique identifier
    description: str           # Issue statement
    claim_type: ClaimType      # CIVIL/CRIMINAL/ADMIN
    elements: List[str]        # Required elements to prove
    burden_standard: BurdenStandard  # PREPONDERANCE/CLEAR_AND_CONVINCING/BEYOND_REASONABLE_DOUBT
    burden_party: Party        # PLAINTIFF/DEFENDANT/PROSECUTION
    
@dataclass
class LegalRule:
    id: str                    # Unique identifier
    rule_text: str             # The legal rule statement
    source_type: str           # STATUTE/CASELAW/REGULATION
    source_id: str             # Reference to source in knowledge base
    jurisdiction: str          # Applicable jurisdiction
    
@dataclass
class ElementAnalysis:
    element_id: str              # The element being analyzed
    element_text: str            # Description of the element
    satisfied: bool              # Whether element is satisfied
    confidence_score: float      # 0.0-1.0
    supporting_evidence: List[str]  # IDs of supporting evidence
    contradicting_evidence: List[str]  # IDs of contradicting evidence
    reasoning_chain: List[str]   # Step-by-step reasoning
    
@dataclass
class BurdenAnalysis:
    standard: BurdenStandard   # PREPONDERANCE/CLEAR_AND_CONVINCING/BEYOND_REASONABLE_DOUBT
    party: Party               # PLAINTIFF/DEFENDANT/PROSECUTION
    met: bool                  # Whether burden is met
    confidence_score: float    # 0.0-1.0
    explanation: str           # Explanation of burden analysis
    
@dataclass
class ReasoningChain:
    steps: List[str]           # Sequential reasoning steps
    confidence: float          # Overall confidence in reasoning
    fallback_explanation: str  # Human-readable explanation
    
@dataclass
class ReasoningProfile:
    case_id: str                 # Unique case identifier
    legal_issues: List[LegalIssue]  # Identified legal issues
    applicable_rules: List[LegalRule]  # Rules applied to case
    element_analyses: Dict[str, ElementAnalysis]  # Analysis by element
    burden_analyses: Dict[str, BurdenAnalysis]  # Analysis by issue
    reasoning_chains: Dict[str, ReasoningChain]  # Reasoning by issue
    conclusion: Dict[str, bool]  # Outcome per issue
    overall_confidence: float    # 0.0-1.0
    explanation: str             # Human-readable explanation


# Core Component Classes

class IssueIdentifier:
    """Identifies legal issues relevant to a case based on factual claims and evidence."""
    
    def __init__(self, knowledge_base: LegalKnowledgeBase):
        self.knowledge_base = knowledge_base
        # Potential additional attributes:
        # - Pattern matchers for common legal issues
        # - ML classifier for issue categorization
        
    def identify_issues(self, evidence_profile: EvidenceProfile, case_context: Dict) -> List[LegalIssue]:
        """Identify legal issues from evidence and case context."""
        issues = []
        
        # Extract claim types from case context
        case_type = case_context.get("case_type", "CIVIL")
        jurisdiction = case_context.get("jurisdiction", "FEDERAL")
        
        # Search knowledge base for relevant legal issue templates
        # This is a simplified version - a real implementation would use 
        # more advanced matching techniques and semantic search
        query_vector = self._generate_issue_query_vector(evidence_profile, case_context)
        potential_issues = self.knowledge_base.semantic_search(
            query_vector, collection="legal_issues", top_k=5
        )
        
        # For each potential issue, check if evidence suggests its relevance
        for issue_template in potential_issues:
            if self._is_issue_relevant(issue_template, evidence_profile):
                # Create tailored issue for this case
                issue = self._create_issue_from_template(issue_template, case_context)
                issues.append(issue)
        
        return issues
    
    def _generate_issue_query_vector(self, evidence_profile: EvidenceProfile, case_context: Dict) -> List[float]:
        """Generate vector representation for querying relevant issues."""
        # In a real implementation, this would construct a semantic embedding
        # based on evidence items and case context
        # Stub implementation returns dummy vector
        return [0.1, 0.3, 0.5, 0.7]
    
    def _is_issue_relevant(self, issue_template: Dict, evidence_profile: EvidenceProfile) -> bool:
        """Determine if an issue is relevant based on evidence."""
        # Check if key facts align with issue requirements
        # Simplified implementation
        return True  # Would be actual logic in real system
    
    def _create_issue_from_template(self, issue_template: Dict, case_context: Dict) -> LegalIssue:
        """Create a case-specific legal issue from a template."""
        # In a real system, this would adapt generic issue templates to the specific case
        return LegalIssue(
            id=f"issue_{issue_template['id']}_{case_context['case_id']}",
            description=issue_template.get("description", "Unknown issue"),
            claim_type=ClaimType(case_context.get("case_type", "CIVIL")),
            elements=issue_template.get("elements", []),
            burden_standard=BurdenStandard(issue_template.get("burden_standard", "PREPONDERANCE")),
            burden_party=Party(issue_template.get("burden_party", "PLAINTIFF"))
        )


class RuleExtractor:
    """Extracts applicable legal rules for identified issues."""
    
    def __init__(self, knowledge_base: LegalKnowledgeBase):
        self.knowledge_base = knowledge_base
        
    def extract_rules(self, 
                      issues: List[LegalIssue], 
                      case_context: Dict) -> List[LegalRule]:
        """Extract relevant legal rules for the identified issues."""
        rules = []
        
        jurisdiction = case_context.get("jurisdiction", "FEDERAL")
        
        for issue in issues:
            # Query knowledge base for rules related to this issue
            issue_rules = self._find_rules_for_issue(issue, jurisdiction)
            rules.extend(issue_rules)
            
            # Also find relevant tests that might apply
            relevant_tests = self._find_tests_for_issue(issue, jurisdiction)
            for test in relevant_tests:
                test_rules = self._convert_test_to_rules(test)
                rules.extend(test_rules)
        
        # Remove duplicates while preserving order
        unique_rules = []
        rule_ids = set()
        for rule in rules:
            if rule.id not in rule_ids:
                unique_rules.append(rule)
                rule_ids.add(rule.id)
        
        return unique_rules
    
    def _find_rules_for_issue(self, issue: LegalIssue, jurisdiction: str) -> List[LegalRule]:
        """Query knowledge base for rules applicable to a specific issue."""
        # In real implementation, would use more advanced querying
        query = f"rules related to {issue.description} in {jurisdiction}"
        filters = {
            "jurisdiction": jurisdiction,
            "claim_type": issue.claim_type
        }
        
        # This would be an actual KB query in real implementation
        # Using stub data for illustration
        rules = [
            LegalRule(
                id=f"rule_1_for_{issue.id}",
                rule_text=f"Rule applying to {issue.description}",
                source_type="STATUTE",
                source_id="statute_123",
                jurisdiction=jurisdiction
            )
        ]
        
        return rules
    
    def _find_tests_for_issue(self, issue: LegalIssue, jurisdiction: str) -> List[Dict]:
        """Find legal tests applicable to an issue."""
        # In real implementation, would search knowledge base
        # Using stub data for illustration
        return [
            {
                "id": f"test_1_for_{issue.id}",
                "name": f"Test for {issue.description}",
                "steps": ["Step 1", "Step 2", "Step 3"]
            }
        ]
    
    def _convert_test_to_rules(self, test: Dict) -> List[LegalRule]:
        """Convert a legal test into a set of rules."""
        rules = []
        
        for i, step in enumerate(test.get("steps", [])):
            rule = LegalRule(
                id=f"{test['id']}_step_{i+1}",
                rule_text=step,
                source_type="TEST",
                source_id=test["id"],
                jurisdiction="COMMON" # Tests often apply across jurisdictions
            )
            rules.append(rule)
            
        return rules


class ElementAnalyzer:
    """Analyzes each element of legal issues against evidence."""
    
    def __init__(self):
        # Could include ML models for evidence-element matching
        pass
    
    def analyze_elements(self, 
                         issues: List[LegalIssue],
                         rules: List[LegalRule],
                         evidence_profile: EvidenceProfile) -> Dict[str, ElementAnalysis]:
        """Analyze each element of the legal issues against available evidence."""
        element_analyses = {}
        
        # For each issue, analyze each required element
        for issue in issues:
            for element in issue.elements:
                element_id = f"{issue.id}_element_{element}"
                
                # Find relevant rules for this element
                element_rules = self._find_rules_for_element(element, rules)
                
                # Find supporting and contradicting evidence
                supporting_evidence, contradicting_evidence = self._evaluate_evidence_for_element(
                    element, element_rules, evidence_profile
                )
                
                # Generate reasoning chain
                reasoning_chain = self._generate_reasoning_for_element(
                    element, element_rules, supporting_evidence, contradicting_evidence
                )
                
                # Determine if element is satisfied
                satisfied, confidence = self._determine_element_satisfaction(
                    supporting_evidence, contradicting_evidence, issue.burden_standard
                )
                
                # Create the analysis
                analysis = ElementAnalysis(
                    element_id=element_id,
                    element_text=element,
                    satisfied=satisfied,
                    confidence_score=confidence,
                    supporting_evidence=[e.id for e in supporting_evidence],
                    contradicting_evidence=[e.id for e in contradicting_evidence],
                    reasoning_chain=reasoning_chain
                )
                
                element_analyses[element_id] = analysis
                
        return element_analyses
    
    def _find_rules_for_element(self, element: str, rules: List[LegalRule]) -> List[LegalRule]:
        """Find relevant rules that apply to a specific element."""
        # Simplified implementation - would use more sophisticated matching in reality
        return [rule for rule in rules if element.lower() in rule.rule_text.lower()]
    
    def _evaluate_evidence_for_element(self, 
                                       element: str, 
                                       rules: List[LegalRule],
                                       evidence_profile: EvidenceProfile) -> Tuple[List, List]:
        """Evaluate evidence items for relevance to this element."""
        supporting = []
        contradicting = []
        
        # In a real system, this would do sophisticated analysis of evidence items
        # for their support or contradiction of the element
        
        # This is stub implementation that simulates finding evidence
        for evidence_item in evidence_profile.evidence_items:
            # Check relevance to this element using relevance scores
            if hasattr(evidence_item, 'relevance_scores') and element in evidence_item.relevance_scores:
                relevance = evidence_item.relevance_scores[element]
                if relevance > 0.7:  # Supporting threshold
                    supporting.append(evidence_item)
                elif relevance < 0.3:  # Contradicting threshold
                    contradicting.append(evidence_item)
        
        return supporting, contradicting
    
    def _generate_reasoning_for_element(self,
                                        element: str,
                                        rules: List[LegalRule],
                                        supporting_evidence: List,
                                        contradicting_evidence: List) -> List[str]:
        """Generate step-by-step reasoning for element analysis."""
        reasoning = []
        
        # Add element definition
        reasoning.append(f"Element to prove: {element}")
        
        # Add applicable rules
        for rule in rules:
            reasoning.append(f"Applied rule: {rule.rule_text}")
        
        # Add supporting evidence summary
        if supporting_evidence:
            reasoning.append(f"Found {len(supporting_evidence)} items of supporting evidence:")
            for i, evidence in enumerate(supporting_evidence[:3]):  # Limit to top 3 for brevity
                reasoning.append(f"  - Supporting evidence {i+1}: {getattr(evidence, 'id', 'unknown')}")
        else:
            reasoning.append("No supporting evidence found.")
            
        # Add contradicting evidence summary
        if contradicting_evidence:
            reasoning.append(f"Found {len(contradicting_evidence)} items of contradicting evidence:")
            for i, evidence in enumerate(contradicting_evidence[:3]):  # Limit to top 3
                reasoning.append(f"  - Contradicting evidence {i+1}: {getattr(evidence, 'id', 'unknown')}")
        else:
            reasoning.append("No contradicting evidence found.")
        
        return reasoning
    
    def _determine_element_satisfaction(self,
                                       supporting_evidence: List,
                                       contradicting_evidence: List,
                                       burden_standard: BurdenStandard) -> Tuple[bool, float]:
        """Determine if element is satisfied based on evidence and burden."""
        # Simple weighted calculation - real implementation would be more sophisticated
        supporting_weight = len(supporting_evidence) 
        contradicting_weight = len(contradicting_evidence) * 1.5  # Contradicting evidence weighted more
        
        # Different thresholds based on burden of proof
        threshold = 0.5  # Default - preponderance
        if burden_standard == BurdenStandard.CLEAR_AND_CONVINCING:
            threshold = 0.7
        elif burden_standard == BurdenStandard.BEYOND_REASONABLE_DOUBT:
            threshold = 0.9
            
        # Calculate net evidence weight
        if supporting_weight + contradicting_weight == 0:
            # No evidence either way
            return False, 0.0
            
        net_support = supporting_weight / (supporting_weight + contradicting_weight)
        confidence = abs(net_support - 0.5) * 2  # Convert to 0-1 confidence scale
        
        return net_support >= threshold, confidence


class BurdenAssessor:
    """Assesses whether the burden of proof is met for each legal issue."""
    
    def assess_burdens(self,
                       issues: List[LegalIssue],
                       element_analyses: Dict[str, ElementAnalysis]) -> Dict[str, BurdenAnalysis]:
        """Assess whether burden of proof is met for each issue."""
        burden_analyses = {}
        
        for issue in issues:
            # Get analyses for all elements of this issue
            issue_elements = [ea for ea_id, ea in element_analyses.items() 
                             if ea_id.startswith(issue.id)]
            
            # Determine if burden is met based on elements and standard
            met, confidence = self._evaluate_burden(issue, issue_elements)
            
            # Generate explanation
            explanation = self._generate_burden_explanation(issue, issue_elements, met)
            
            # Create the analysis
            analysis = BurdenAnalysis(
                standard=issue.burden_standard,
                party=issue.burden_party,
                met=met,
                confidence_score=confidence,
                explanation=explanation
            )
            
            burden_analyses[issue.id] = analysis
            
        return burden_analyses
    
    def _evaluate_burden(self, 
                         issue: LegalIssue, 
                         element_analyses: List[ElementAnalysis]) -> Tuple[bool, float]:
        """Evaluate if burden of proof is met for an issue."""
        # For burden to be met, generally all elements must be satisfied
        all_satisfied = all(ea.satisfied for ea in element_analyses)
        
        # Calculate confidence as average of element confidences
        if element_analyses:
            avg_confidence = sum(ea.confidence_score for ea in element_analyses) / len(element_analyses)
        else:
            avg_confidence = 0.0
            
        # Apply burden standard modifiers
        # Higher standards reduce confidence
        if issue.burden_standard == BurdenStandard.CLEAR_AND_CONVINCING:
            avg_confidence *= 0.9
        elif issue.burden_standard == BurdenStandard.BEYOND_REASONABLE_DOUBT:
            avg_confidence *= 0.8
            
        return all_satisfied, avg_confidence
    
    def _generate_burden_explanation(self, 
                                    issue: LegalIssue, 
                                    element_analyses: List[ElementAnalysis],
                                    burden_met: bool) -> str:
        """Generate explanation of burden analysis."""
        party = issue.burden_party.value.capitalize()
        standard = issue.burden_standard.value.replace("_", " ").capitalize()
        
        if not element_analyses:
            return f"No elements analyzed for issue {issue.id}."
        
        satisfied_elements = [ea for ea in element_analyses if ea.satisfied]
        
        explanation = (
            f"The {party} has the burden to prove {len(element_analyses)} elements "
            f"under the '{standard}' standard. "
        )
        
        if burden_met:
            explanation += (
                f"All {len(element_analyses)} required elements were satisfied, "
                f"meeting the burden of proof."
            )
        else:
            satisfied_count = len(satisfied_elements)
            explanation += (
                f"Only {satisfied_count} out of {len(element_analyses)} required elements "
                f"were satisfied, failing to meet the burden of proof."
            )
            
        return explanation


class ReasoningEngine:
    """Constructs logical reasoning chains and generates conclusions."""
    
    def generate_reasoning(self,
                          issues: List[LegalIssue],
                          rules: List[LegalRule],
                          element_analyses: Dict[str, ElementAnalysis],
                          burden_analyses: Dict[str, BurdenAnalysis]) -> Tuple[Dict[str, ReasoningChain], Dict[str, bool], float, str]:
        """Generate reasoning chains and conclusions for each issue."""
        reasoning_chains = {}
        conclusions = {}
        
        # Process each issue
        for issue in issues:
            # Get relevant analyses
            burden_analysis = burden_analyses.get(issue.id)
            issue_element_analyses = {k: v for k, v in element_analyses.items() 
                                    if k.startswith(issue.id)}
            
            # Generate reasoning chain
            chain = self._construct_reasoning_chain(issue, issue_element_analyses, burden_analysis)
            reasoning_chains[issue.id] = chain
            
            # Determine conclusion for this issue
            # Generally, conclusion matches whether burden was met
            if burden_analysis:
                conclusions[issue.id] = burden_analysis.met
            else:
                conclusions[issue.id] = False
        
        # Calculate overall confidence 
        if reasoning_chains:
            overall_confidence = sum(rc.confidence for rc in reasoning_chains.values()) / len(reasoning_chains)
        else:
            overall_confidence = 0.0
            
        # Generate overall explanation
        explanation = self._generate_overall_explanation(issues, conclusions, reasoning_chains)
        
        return reasoning_chains, conclusions, overall_confidence, explanation
    
    def _construct_reasoning_chain(self,
                                  issue: LegalIssue,
                                  element_analyses: Dict[str, ElementAnalysis],
                                  burden_analysis: Optional[BurdenAnalysis]) -> ReasoningChain:
        """Construct a reasoning chain for an issue."""
        steps = []
        
        # Start with issue description
        steps.append(f"Issue: {issue.description}")
        
        # Add burden standard
        if burden_analysis:
            steps.append(
                f"Required standard of proof: {burden_analysis.standard.value.replace('_', ' ').capitalize()} "
                f"(burden on {burden_analysis.party.value.capitalize()})"
            )
        
        # Add element analyses
        for i, (element_id, analysis) in enumerate(element_analyses.items()):
            element_result = "satisfied" if analysis.satisfied else "not satisfied"
            steps.append(f"Element {i+1}: {analysis.element_text} - {element_result}")
            
            # Add key points from element reasoning
            steps.extend([f"  • {s}" for s in analysis.reasoning_chain if "evidence" in s.lower()])
        
        # Add burden conclusion
        if burden_analysis:
            burden_result = "met" if burden_analysis.met else "not met"
            steps.append(f"Burden of proof: {burden_result}")
            steps.append(f"Explanation: {burden_analysis.explanation}")
        
        # Calculate confidence
        if element_analyses and burden_analysis:
            # Weighted average of element and burden confidences
            element_confidences = [ea.confidence_score for ea in element_analyses.values()]
            avg_element_confidence = sum(element_confidences) / len(element_confidences)
            confidence = (avg_element_confidence * 0.7) + (burden_analysis.confidence_score * 0.3)
        elif element_analyses:
            element_confidences = [ea.confidence_score for ea in element_analyses.values()]
            confidence = sum(element_confidences) / len(element_confidences)
        elif burden_analysis:
            confidence = burden_analysis.confidence_score
        else:
            confidence = 0.0
            
        # Generate fallback explanation (simplified version of steps)
        fallback = (
            f"Analysis of {issue.description}: "
            f"{'All' if burden_analysis and burden_analysis.met else 'Not all'} "
            f"required elements were proven."
        )
        
        return ReasoningChain(
            steps=steps,
            confidence=confidence,
            fallback_explanation=fallback
        )
    
    def _generate_overall_explanation(self,
                                     issues: List[LegalIssue],
                                     conclusions: Dict[str, bool],
                                     reasoning_chains: Dict[str, ReasoningChain]) -> str:
        """Generate an overall explanation of conclusions."""
        if not issues:
            return "No legal issues identified for analysis."
            
        explanations = []
        explanations.append(f"Analysis of {len(issues)} legal issues:")
        
        for issue in issues:
            conclusion = conclusions.get(issue.id, False)
            chain = reasoning_chains.get(issue.id)
            
            if chain:
                explanations.append(
                    f"- {issue.description}: "
                    f"{'PROVEN' if conclusion else 'NOT PROVEN'} "
                    f"(confidence: {chain.confidence:.2f})"
                )
            else:
                explanations.append(
                    f"- {issue.description}: INSUFFICIENT INFORMATION"
                )
                
        return "\n".join(explanations)


class LegalReasoningModule:
    """Main orchestrator for legal reasoning processes."""
    
    def __init__(self, knowledge_base: LegalKnowledgeBase):
        """Initialize reasoning module with dependencies."""
        self.knowledge_base = knowledge_base
        
        # Initialize component modules
        self.issue_identifier = IssueIdentifier(knowledge_base)
        self.rule_extractor = RuleExtractor(knowledge_base)
        self.element_analyzer = ElementAnalyzer()
        self.burden_assessor = BurdenAssessor()
        self.reasoning_engine = ReasoningEngine()
        
    def process_case(self, 
                     evidence_profile: EvidenceProfile, 
                     case_context: Dict) -> ReasoningProfile:
        """Process a case to generate a legal reasoning profile."""
        # Step 1: Identify legal issues
        issues = self.issue_identifier.identify_issues(evidence_profile, case_context)
        
        # Step 2: Extract applicable rules
        rules = self.rule_extractor.extract_rules(issues, case_context)
        
        # Step 3: Analyze elements against evidence
        element_analyses = self.element_analyzer.analyze_elements(issues, rules, evidence_profile)
        
        # Step 4: Assess burden of proof
        burden_analyses = self.burden_assessor.assess_burdens(issues, element_analyses)
        
        # Step 5: Generate reasoning chains and conclusions
        reasoning_chains, conclusions, overall_confidence, explanation = (
            self.reasoning_engine.generate_reasoning(
                issues, rules, element_analyses, burden_analyses
            )
        )
        
        # Create and return the reasoning profile
        profile = ReasoningProfile(
            case_id=evidence_profile.case_id,
            legal_issues=issues,
            applicable_rules=rules,
            element_analyses=element_analyses,
            burden_analyses=burden_analyses,
            reasoning_chains=reasoning_chains,
            conclusion=conclusions,
            overall_confidence=overall_confidence,
            explanation=explanation
        )
        
        return profile
    
    def to_json(self, reasoning_profile: ReasoningProfile) -> str:
        """Convert reasoning profile to JSON for storage or transmission."""
        # This is a simplified implementation - a real version would need
        # proper serialization of complex objects
        
        # Convert to dict first, handling custom objects
        profile_dict = {
            "case_id": reasoning_profile.case_id,
            "legal_issues": [self._issue_to_dict(issue) for issue in reasoning_profile.legal_issues],
            "applicable_rules": [self._rule_to_dict(rule) for rule in reasoning_profile.applicable_rules],
            "element_analyses": {k: self._element_analysis_to_dict(v) 
                                for k, v in reasoning_profile.element_analyses.items()},
            "burden_analyses": {k: self._burden_analysis_to_dict(v) 
                               for k, v in reasoning_profile.burden_analyses.items()},
            "reasoning_chains": {k: self._chain_to_dict(v) 
                                for k, v in reasoning_profile.reasoning_chains.items()},
            "conclusion": reasoning_profile.conclusion,
            "overall_confidence": reasoning_profile.overall_confidence,
            "explanation": reasoning_profile.explanation
        }
        
        return json.dumps(profile_dict, indent=2)
    
    def _issue_to_dict(self, issue: LegalIssue) -> Dict:
        """Convert LegalIssue to dict."""
        return {
            "id": issue.id,
            "description": issue.description,
            "claim_type": issue.claim_type.value,
            "elements": issue.elements,
            "burden_standard": issue.burden_standard.value,
            "burden_party": issue.burden_party.value
        }

    def _rule_to_dict(self, rule: LegalRule) -> Dict:
        """Convert LegalRule to dict."""
        return {
            "id": rule.id,
            "rule_text": rule.rule_text,
            "source_type": rule.source_type,
            "source_id": rule.source_id,
            "jurisdiction": rule.jurisdiction
        }

    def _element_analysis_to_dict(self, analysis: ElementAnalysis) -> Dict:
        """Convert ElementAnalysis to dict."""
        return {
            "element_id": analysis.element_id,
            "element_text": analysis.element_text,
            "satisfied": analysis.satisfied,
            "confidence_score": analysis.confidence_score,
            "supporting_evidence": analysis.supporting_evidence,
            "contradicting_evidence": analysis.contradicting_evidence,
            "reasoning_chain": analysis.reasoning_chain
        }

    def _burden_analysis_to_dict(self, analysis: BurdenAnalysis) -> Dict:
        """Convert BurdenAnalysis to dict."""
        return {
            "standard": analysis.standard.value,
            "party": analysis.party.value,
            "met": analysis.met,
            "confidence_score": analysis.confidence_score,
            "explanation": analysis.explanation
        }

    def _chain_to_dict(self, chain: ReasoningChain) -> Dict:
        """Convert ReasoningChain to dict."""
        return {
            "steps": chain.steps,
            "confidence": chain.confidence,
            "fallback_explanation": chain.fallback_explanation
        }