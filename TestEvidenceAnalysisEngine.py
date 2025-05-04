import unittest
from unittest.mock import patch
import json
from uuid import uuid4
from EvidenceAnalysisEngine import EvidenceProcessor, EvidenceItem, LegalIssue, EvidenceProfile, EvidenceValidator, FactualAnalyzer, EvidenceWeighter

class TestEvidenceProcessor(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test."""
        self.processor = EvidenceProcessor()
        self.case_id = "test_case_001"
        self.jurisdiction = "federal"
        self.sample_evidence = [
            {
                "content": "Witness John Smith saw the defendant at the crime scene on July 15, 2023.",
                "source": "John Smith",
                "metadata": {"context": "Witness testimony", "source_role": "plaintiff"}
            },
            {
                "content": "Contract signed between ABC Corp and XYZ Inc on January 1, 2023.",
                "source": "Court Records",
                "metadata": {"authenticated": True}
            }
        ]
        self.case_context = {
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

    def test_evidence_processor_initialization(self):
        """Test that EvidenceProcessor initializes its components correctly."""
        self.assertIsInstance(self.processor.validator, EvidenceValidator)
        self.assertIsInstance(self.processor.factual_analyzer, FactualAnalyzer)
        self.assertIsInstance(self.processor.weighter, EvidenceWeighter)

    def test_categorize_evidence_testimony(self):
        """Test evidence categorization for testimony."""
        evidence = {
            "content": "Witness stated that the event occurred.",
            "metadata": {"context": "Witness testimony"}
        }
        category, confidence = self.processor.categorize_evidence(evidence)
        self.assertEqual(category, "TESTIMONY")
        self.assertGreaterEqual(confidence, 0.5)

    def test_categorize_evidence_document(self):
        """Test evidence categorization for document."""
        evidence = {
            "content": "Signed contract between parties.",
            "metadata": {"authenticated": True}
        }
        category, confidence = self.processor.categorize_evidence(evidence)
        self.assertEqual(category, "DOCUMENT")
        self.assertGreaterEqual(confidence, 0.5)

    def test_validate_evidence_missing_content(self):
        """Test evidence validation with missing content."""
        evidence = {
            "source": "John Smith",
            "metadata": {}
        }
        is_valid, issues = self.processor.validator.validate_evidence(evidence, self.jurisdiction)
        self.assertFalse(is_valid)
        self.assertIn("missing_content", issues)

    def test_validate_evidence_admissibility(self):
        """Test evidence admissibility check."""
        evidence = {
            "content": "Hearsay statement about the event.",
            "source": "Anonymous",
            "category": "HEARSAY",
            "metadata": {}
        }
        is_valid, issues = self.processor.validator.validate_evidence(evidence, self.jurisdiction)
        self.assertFalse(is_valid)
        self.assertIn("admissibility", issues)
        self.assertIn("hearsay", issues["admissibility"]["issues"])

    @patch('EvidenceAnalysisEngine.nlp')
    def test_identify_hearsay(self, mock_nlp):
        """Test hearsay identification."""
        # Define MockToken to simulate spaCy tokens
        class MockToken:
            def __init__(self, text, lemma):
                self.text = text
                self.lemma_ = lemma
        
        # Define MockSent to simulate spaCy sentences
        class MockSent:
            def __init__(self, text):
                self.text = text
                self.tokens = [MockToken("said", "say")]  # Simulate reporting verb
            def __iter__(self):
                return iter(self.tokens)
        
        # Define MockDoc to simulate spaCy Doc
        class MockDoc:
            def __init__(self):
                self.sents = [
                    MockSent("He said the defendant was there."),
                    MockSent("No quotes here.")
                ]
        
        mock_nlp.return_value = MockDoc()

        hearsay_instances = self.processor.validator.identify_hearsay("He said the defendant was there. No quotes here.")
        self.assertGreater(len(hearsay_instances), 0)
        self.assertEqual(hearsay_instances[0]["text"], "He said the defendant was there.")
        self.assertIn("reporting_verb", hearsay_instances[0])

    def test_process_evidence(self):
        """Test full evidence processing pipeline."""
        profile = self.processor.process_evidence(
            self.case_id,
            self.sample_evidence,
            self.jurisdiction,
            self.case_context
        )
        self.assertIsInstance(profile, EvidenceProfile)
        self.assertEqual(profile.case_id, self.case_id)
        self.assertEqual(len(profile.evidence_items), len(self.sample_evidence))
        self.assertIsNotNone(profile.summary)
        self.assertIsNotNone(profile.consistency_analysis)

    def test_weight_evidence(self):
        """Test evidence weighting."""
        evidence_item = EvidenceItem(
            id="test_001",
            content="Test evidence",
            category="DOCUMENT",
            source="Official Records",
            metadata={"authenticated": True}
        )
        weight, factors = self.processor.weighter.weight_evidence(evidence_item, self.case_context)
        self.assertGreaterEqual(weight, 0.0)
        self.assertLessEqual(weight, 1.0)
        self.assertIn("type_weight", factors)
        self.assertIn("reliability_adjustment", factors)
        self.assertIn("relevance_adjustment", factors)

    @patch('EvidenceAnalysisEngine.nlp')
    def test_factual_analysis(self, mock_nlp):
        """Test factual analysis."""
        # Define MockToken to simulate spaCy tokens
        class MockToken:
            def __init__(self, text, like_num=False):
                self.text = text
                self.like_num = like_num
        
        # Define MockSent to simulate spaCy sentences
        class MockSent:
            def __init__(self, text):
                self.text = text
                self.ents = []
                self.tokens = [MockToken("Fact", False), MockToken("one", True)]  # Simulate number
            def __iter__(self):
                return iter(self.tokens)
        
        # Define MockDoc to simulate spaCy Doc
        class MockDoc:
            def __init__(self):
                self.sents = [MockSent("Fact one.")]
                self.ents = []
                self.noun_chunks = []
        
        mock_nlp.return_value = MockDoc()
        mock_nlp.pipe.return_value = [MockDoc()]

        evidence_items = [
            EvidenceItem(
                id="fact_001",
                content="Fact one.",
                category="TESTIMONY",
                source="Witness",
                reliability_score=0.7
            )
        ]
        analysis = self.processor.factual_analyzer.analyze_facts(self.case_id, evidence_items)
        self.assertIn("factual_claims", analysis)
        self.assertIn("claim_groups", analysis)
        self.assertIn("contradictions", analysis)
        self.assertIn("fact_summary", analysis)

    def test_create_fact_summary(self):
        """Test fact summary creation."""
        evidence_items = [
            EvidenceItem(
                id="fact_001",
                content="Defendant was at the scene.",
                category="TESTIMONY",
                source="Witness",
                admissibility=True,
                reliability_score=0.7,
                relevance_scores={"issue_1": 0.8}
            )
        ]
        profile = EvidenceProfile(
            case_id=self.case_id,
            evidence_items=evidence_items,
            factual_disputes=[]
        )
        summary = self.processor.factual_analyzer.create_fact_summary(profile, self.case_context)
        self.assertEqual(summary["case_id"], self.case_id)
        self.assertGreaterEqual(len(summary["established_facts"]), 0)
        self.assertIn("legal_issue_relevance", summary)
        self.assertIn("summary_metrics", summary)

    def test_get_evidence_summary(self):
        """Test evidence summary generation."""
        evidence_items = [
            EvidenceItem(
                id="ev_001",
                content="Test evidence",
                category="DOCUMENT",
                source="Records",
                admissibility=True,
                reliability_score=0.8
            )
        ]
        summary = self.processor.get_evidence_summary(self.case_id, evidence_items)
        self.assertEqual(summary["case_id"], self.case_id)
        self.assertEqual(summary["total_items"], 1)
        self.assertIn("DOCUMENT", summary["category_distribution"])
        self.assertEqual(summary["admissibility"]["admissible"], 1)
        self.assertGreaterEqual(summary["average_reliability"], 0.0)

if __name__ == '__main__':
    unittest.main()