import unittest
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json
from enum import Enum

# Override dataclasses so module under test picks these definitions
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
    id: str
    description: str
    claim_type: ClaimType
    elements: List[str]
    burden_standard: BurdenStandard
    burden_party: Party

@dataclass
class LegalRule:
    id: str
    rule_text: str
    source_type: str
    source_id: str
    jurisdiction: str

@dataclass
class ElementAnalysis:
    element_id: str
    element_text: str
    satisfied: bool
    confidence_score: float
    supporting_evidence: List[str]
    contradicting_evidence: List[str]
    reasoning_chain: List[str]

@dataclass
class BurdenAnalysis:
    standard: BurdenStandard
    party: Party
    met: bool
    confidence_score: float
    explanation: str

@dataclass
class ReasoningChain:
    steps: List[str]
    confidence: float
    fallback_explanation: str

@dataclass
class ReasoningProfile:
    case_id: str
    legal_issues: List[LegalIssue]
    applicable_rules: List[LegalRule]
    element_analyses: Dict[str, ElementAnalysis]
    burden_analyses: Dict[str, BurdenAnalysis]
    reasoning_chains: Dict[str, ReasoningChain]
    conclusion: Dict[str, bool]
    overall_confidence: float
    explanation: str

# Now import the module under test
import LegalReasoningModule as LRM

class DummyKB:
    def semantic_search(self, query_vector, collection, top_k=None):
        # Return one dummy issue template
        return [{
            'id': 'tmpl1',
            'description': 'Negligence',
            'elements': ['Duty', 'Breach'],
            'burden_standard': 'PREPONDERANCE',
            'burden_party': 'PLAINTIFF'
        }]

class DummyEvidenceItem:
    def __init__(self, id, relevance_scores):
        self.id = id
        self.relevance_scores = relevance_scores

class TestIssueIdentifier(unittest.TestCase):
    def test_identify_issues(self):
        kb = DummyKB()
        identifier = LRM.IssueIdentifier(kb)
        profile = LRM.EvidenceProfile(
            case_id='C1', evidence_items=[], factual_disputes=[], consistency_analysis={}
        )
        context = {'case_id':'C1','case_type':'CIVIL','jurisdiction':'FEDERAL'}
        issues = identifier.identify_issues(profile, context)
        self.assertIsInstance(issues, list)
        self.assertEqual(len(issues), 1)
        issue = issues[0]
        self.assertIsInstance(issue, LRM.LegalIssue)
        self.assertEqual(issue.description, 'Negligence')
        self.assertEqual(issue.claim_type, LRM.ClaimType.CIVIL)

class TestRuleExtractor(unittest.TestCase):
    def test_extract_rules(self):
        kb = DummyKB()
        extractor = LRM.RuleExtractor(kb)
        issue = LRM.LegalIssue(
            id='issue1', description='Negligence', claim_type=LRM.ClaimType.CIVIL,
            elements=['Duty'], burden_standard=LRM.BurdenStandard.PREPONDERANCE, burden_party=LRM.Party.PLAINTIFF
        )
        rules = extractor.extract_rules([issue], {'jurisdiction':'FEDERAL'})
        self.assertTrue(len(rules) >= 1)
        for rule in rules:
            self.assertIsInstance(rule, LRM.LegalRule)
            self.assertTrue(rule.id.startswith('rule_') or rule.id.startswith('test_'))

class TestElementAnalyzer(unittest.TestCase):
    def test_analyze_elements(self):
        analyzer = LRM.ElementAnalyzer()
        issue = LRM.LegalIssue(
            id='issue1', description='Negligence', claim_type=LRM.ClaimType.CIVIL,
            elements=['Duty'], burden_standard=LRM.BurdenStandard.PREPONDERANCE, burden_party=LRM.Party.PLAINTIFF
        )
        # rule_text contains element lower-case to match
        rule = LRM.LegalRule(id='r1', rule_text='Duty of care', source_type='STATUTE', source_id='s1', jurisdiction='FEDERAL')
        ev1 = DummyEvidenceItem('e1', {'Duty': 0.8})
        ev2 = DummyEvidenceItem('e2', {'Duty': 0.2})
        profile = LRM.EvidenceProfile(case_id='C1', evidence_items=[ev1, ev2], factual_disputes=[], consistency_analysis={})
        analyses = analyzer.analyze_elements([issue], [rule], profile)
        self.assertIn('issue1_element_Duty', analyses)
        ea = analyses['issue1_element_Duty']
        self.assertIsInstance(ea, LRM.ElementAnalysis)
        # supporting > threshold, contradicting < threshold
        self.assertIn('e1', ea.supporting_evidence)
        self.assertIn('e2', ea.contradicting_evidence)

class TestBurdenAssessor(unittest.TestCase):
    def test_assess_burdens(self):
        assessor = LRM.BurdenAssessor()
        issue = LRM.LegalIssue(
            id='issue1', description='Negligence', claim_type=LRM.ClaimType.CIVIL,
            elements=['Duty'], burden_standard=LRM.BurdenStandard.PREPONDERANCE, burden_party=LRM.Party.PLAINTIFF
        )
        ea = LRM.ElementAnalysis(
            element_id='issue1_element_Duty', element_text='Duty', satisfied=True,
            confidence_score=0.8, supporting_evidence=['e1'], contradicting_evidence=[], reasoning_chain=[]
        )
        results = assessor.assess_burdens([issue], {'issue1_element_Duty': ea})
        self.assertIn('issue1', results)
        ba = results['issue1']
        self.assertIsInstance(ba, LRM.BurdenAnalysis)
        self.assertTrue(ba.met)

class TestReasoningEngine(unittest.TestCase):
    def test_generate_reasoning(self):
        engine = LRM.ReasoningEngine()
        issue = LRM.LegalIssue(
            id='issue1', description='Negligence', claim_type=LRM.ClaimType.CIVIL,
            elements=['Duty'], burden_standard=LRM.BurdenStandard.PREPONDERANCE, burden_party=LRM.Party.PLAINTIFF
        )
        rule = LRM.LegalRule(id='r1', rule_text='Duty of care', source_type='STATUTE', source_id='s1', jurisdiction='FEDERAL')
        ea = LRM.ElementAnalysis(
            element_id='issue1_element_Duty', element_text='Duty', satisfied=True,
            confidence_score=0.8, supporting_evidence=['e1'], contradicting_evidence=[], reasoning_chain=['r1']
        )
        ba = LRM.BurdenAnalysis(
            standard=LRM.BurdenStandard.PREPONDERANCE, party=LRM.Party.PLAINTIFF,
            met=True, confidence_score=0.8, explanation='ok'
        )
        chains, conclusions, overall_conf, explanation = engine.generate_reasoning(
            [issue], [rule], {'issue1_element_Duty': ea}, {'issue1': ba}
        )
        self.assertIn('issue1', chains)
        self.assertIn('issue1', conclusions)
        self.assertIsInstance(chains['issue1'], LRM.ReasoningChain)
        self.assertTrue(conclusions['issue1'])
        self.assertIsInstance(overall_conf, float)
        self.assertIsInstance(explanation, str)

class TestLegalReasoningModule(unittest.TestCase):
    def test_process_and_to_json(self):
        kb = DummyKB()
        module = LRM.LegalReasoningModule(kb)
        # use same dummy evidence profile
        ev = DummyEvidenceItem('e1', {'Duty':0.8})
        profile = LRM.EvidenceProfile(case_id='C1', evidence_items=[ev], factual_disputes=[], consistency_analysis={})
        context = {'case_id':'C1','case_type':'CIVIL','jurisdiction':'FEDERAL'}
        rp = module.process_case(profile, context)
        self.assertIsInstance(rp, LRM.ReasoningProfile)
        json_str = module.to_json(rp)
        data = json.loads(json_str)
        self.assertEqual(data['case_id'], 'C1')
        self.assertIn('legal_issues', data)

if __name__ == '__main__':
    unittest.main()
