import unittest
import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, List

from LegalKnowledgeBaseModule import (
    LegalKnowledgeBase,
    LegalStatute,
    CaseLaw,
    LegalPrinciple
)


class TestLegalKnowledgeBase(unittest.TestCase):
    """Unit tests for the Legal Knowledge Base module."""

    def setUp(self) -> None:
        """Initialize test environment before each test."""
        # Use an in-memory database for testing
        self.test_db_path = "file:memdb1?mode=memory&cache=shared"
        self.kb = LegalKnowledgeBase(db_path=self.test_db_path)

        # Sample vector embedding for testing
        self.sample_vector = [0.1] * 768

        # Sample statute for testing
        self.sample_statute = LegalStatute(
            id="stat-001",
            title="Consumer Protection Act",
            text="This act protects consumers from unfair business practices...",
            jurisdiction="Federal",
            category="Consumer Law",
            effective_date="2020-01-01",
            last_amended="2023-05-15",
            sections={
                "1": "Purpose",
                "2": "Definitions",
                "3": "Prohibited Acts"
            },
            citations=["case-001", "case-002"],
            vector_embedding=self.sample_vector
        )

        # Sample case law for testing
        self.sample_case = CaseLaw(
            id="case-001",
            title="Smith v. Company Inc.",
            citation="123 F.3d 456 (2022)",
            court="Federal Court of Appeals",
            date_decided="2022-03-15",
            jurisdiction="Federal",
            summary="Case regarding consumer rights under the Consumer Protection Act",
            full_text="The court found that the defendant violated section 3 of the CPA...",
            judges=["Judge Smith", "Judge Brown"],
            parties={
                "plaintiff": "John Smith",
                "defendant": "Company Inc."
            },
            outcome="Judgment for plaintiff",
            legal_principles=["principle-001"],
            related_cases=["case-002"],
            vector_embedding=self.sample_vector
        )

        # Sample legal principle for testing
        self.sample_principle = LegalPrinciple(
            id="principle-001",
            name="Caveat Venditor",
            description="Let the seller beware, which places the burden on sellers...",
            category="Consumer Law",
            jurisdiction="Federal",
            source_cases=["case-001"],
            related_principles=["principle-002"],
            exceptions=["In cases of explicit disclaimers"],
            vector_embedding=self.sample_vector
        )

    def tearDown(self) -> None:
        """Clean up test environment after each test."""
        self.kb.conn.close()
        if self.test_db_path != ":memory:" and os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)

    def test_add_and_get_statute(self) -> None:
        """Test adding and retrieving a statute from the knowledge base."""
        # Add statute
        result = self.kb.add_statute(self.sample_statute)
        self.assertTrue(result, "Failed to add statute")

        # Retrieve and verify statute
        retrieved = self.kb.get_statute("stat-001")
        self.assertIsNotNone(retrieved, "Retrieved statute is None")
        self.assertEqual(retrieved.id, "stat-001", "Incorrect statute ID")
        self.assertEqual(retrieved.title, "Consumer Protection Act", "Incorrect statute title")
        self.assertEqual(retrieved.jurisdiction, "Federal", "Incorrect jurisdiction")
        self.assertEqual(retrieved.category, "Consumer Law", "Incorrect category")
        self.assertEqual(len(retrieved.sections), 3, "Incorrect number of sections")

        # Test retrieving non-existent statute
        nonexistent = self.kb.get_statute("nonexistent")
        self.assertIsNone(nonexistent, "Non-existent statute should be None")

    def test_add_and_get_case_law(self) -> None:
        """Test adding and retrieving a case law from the knowledge base."""
        # Add case law
        result = self.kb.add_case_law(self.sample_case)
        self.assertTrue(result, "Failed to add case law")

        # Retrieve and verify case law
        retrieved = self.kb.get_case_law("case-001")
        self.assertIsNotNone(retrieved, "Retrieved case law is None")
        self.assertEqual(retrieved.id, "case-001", "Incorrect case law ID")
        self.assertEqual(retrieved.title, "Smith v. Company Inc.", "Incorrect case law title")
        self.assertEqual(retrieved.court, "Federal Court of Appeals", "Incorrect court")
        self.assertEqual(len(retrieved.judges), 2, "Incorrect number of judges")
        self.assertEqual(retrieved.outcome, "Judgment for plaintiff", "Incorrect outcome")

        # Test retrieving non-existent case law
        nonexistent = self.kb.get_case_law("nonexistent")
        self.assertIsNone(nonexistent, "Non-existent case law should be None")

    def test_add_and_get_legal_principle(self) -> None:
        """Test adding and retrieving a legal principle from the knowledge base."""
        # Add legal principle
        result = self.kb.add_legal_principle(self.sample_principle)
        self.assertTrue(result, "Failed to add legal principle")

        # Retrieve and verify legal principle
        retrieved = self.kb.get_legal_principle("principle-001")
        self.assertIsNotNone(retrieved, "Retrieved principle is None")
        self.assertEqual(retrieved.id, "principle-001", "Incorrect principle ID")
        self.assertEqual(retrieved.name, "Caveat Venditor", "Incorrect principle name")
        self.assertEqual(retrieved.category, "Consumer Law", "Incorrect category")
        self.assertEqual(len(retrieved.source_cases), 1, "Incorrect number of source cases")
        self.assertEqual(len(retrieved.exceptions), 1, "Incorrect number of exceptions")

        # Test retrieving non-existent principle
        nonexistent = self.kb.get_legal_principle("nonexistent")
        self.assertIsNone(nonexistent, "Non-existent principle should be None")

    def test_search_statutes(self) -> None:
        """Test searching statutes with various filters."""
        # Add test statutes
        self.kb.add_statute(self.sample_statute)
        another_statute = LegalStatute(
            id="stat-002",
            title="State Consumer Law",
            text="This state law protects consumers...",
            jurisdiction="California",
            category="Consumer Law",
            effective_date="2021-01-01"
        )
        self.kb.add_statute(another_statute)

        # Search by text
        results = self.kb.search_statutes("protect")
        self.assertEqual(len(results), 2, "Expected 2 statutes for 'protect' search")

        # Search with jurisdiction filter
        results = self.kb.search_statutes("consumer", jurisdiction="Federal")
        self.assertEqual(len(results), 1, "Expected 1 statute for Federal jurisdiction")
        self.assertEqual(results[0].id, "stat-001", "Incorrect statute ID")

        # Search with category filter
        results = self.kb.search_statutes("law", category="Consumer Law")
        self.assertEqual(len(results), 1, "Expected 1 statute for Consumer Law category")

        # Search with no results
        results = self.kb.search_statutes("nonexistent")
        self.assertEqual(len(results), 0, "Expected 0 statutes for non-existent query")

    def test_search_case_law(self) -> None:
        """Test searching case law with various filters."""
        # Add test case law
        self.kb.add_case_law(self.sample_case)
        another_case = CaseLaw(
            id="case-002",
            title="Johnson v. Other Corp",
            citation="456 F.3d 789 (2023)",
            court="Supreme Court",
            date_decided="2023-01-15",
            jurisdiction="Federal",
            summary="Related case on consumer rights",
            full_text="The Supreme Court affirmed the lower court's decision..."
        )
        self.kb.add_case_law(another_case)

        # Search by text
        results = self.kb.search_case_law("consumer")
        self.assertEqual(len(results), 2, "Expected 2 cases for 'consumer' search")

        # Search with court_level filter
        results = self.kb.search_case_law("court", court_level="Supreme Court")
        self.assertEqual(len(results), 1, "Expected 1 case for Supreme Court filter")
        self.assertEqual(results[0].id, "case-002", "Incorrect case ID")

        # Search with jurisdiction filter
        results = self.kb.search_case_law("case", jurisdiction="Federal")
        self.assertEqual(len(results), 2, "Expected 2 cases for Federal jurisdiction")

        # Search with no results
        results = self.kb.search_case_law("nonexistent")
        self.assertEqual(len(results), 0, "Expected 0 cases for non-existent query")

    def test_search_legal_principles(self) -> None:
        """Test searching legal principles with various filters."""
        # Add test principles
        self.kb.add_legal_principle(self.sample_principle)
        another_principle = LegalPrinciple(
            id="principle-002",
            name="Res Ipsa Loquitur",
            description="The thing speaks for itself...",
            category="Tort Law",
            jurisdiction="Common Law"
        )
        self.kb.add_legal_principle(another_principle)

        # Search by text
        results = self.kb.search_legal_principles("thing")
        self.assertEqual(len(results), 1, "Expected 1 principle for 'thing' search")
        self.assertEqual(results[0].id, "principle-002", "Incorrect principle ID")

        # Search with category filter
        results = self.kb.search_legal_principles("seller", category="Consumer Law")
        self.assertEqual(len(results), 1, "Expected 1 principle for Consumer Law category")
        self.assertEqual(results[0].id, "principle-001", "Incorrect principle ID")

        # Search with jurisdiction filter
        results = self.kb.search_legal_principles("Res", jurisdiction="Common Law")
        self.assertEqual(len(results), 1, "Expected 1 principle for Common Law jurisdiction")
        self.assertEqual(results[0].id, "principle-002", "Incorrect principle ID")

        # Search with no results
        results = self.kb.search_legal_principles("nonexistent")
        self.assertEqual(len(results), 0, "Expected 0 principles for non-existent query")

    def test_semantic_search(self) -> None:
        """Test semantic search using vector embeddings."""
        # Add test data
        self.kb.add_statute(self.sample_statute)
        self.kb.add_case_law(self.sample_case)
        self.kb.add_legal_principle(self.sample_principle)

        # Create query vector
        query_vector = [0.1] * 768
        query_vector[0] = 0.2

        # Search statutes
        results = self.kb.semantic_search(query_vector, "statutes")
        self.assertEqual(len(results), 1, "Expected 1 statute in semantic search")
        self.assertEqual(results[0].id, "stat-001", "Incorrect statute ID")

        # Search cases
        results = self.kb.semantic_search(query_vector, "cases")
        self.assertEqual(len(results), 1, "Expected 1 case in semantic search")
        self.assertEqual(results[0].id, "case-001", "Incorrect case ID")

        # Search principles
        results = self.kb.semantic_search(query_vector, "principles")
        self.assertEqual(len(results), 1, "Expected 1 principle in semantic search")
        self.assertEqual(results[0].id, "principle-001", "Incorrect principle ID")

    def test_import_export_json(self) -> None:
        """Test importing and exporting data to/from JSON."""
        # Add test data
        self.kb.add_statute(self.sample_statute)
        self.kb.add_case_law(self.sample_case)
        self.kb.add_legal_principle(self.sample_principle)

        # Export to JSON
        test_json_path = "test_export.json"
        export_result = self.kb.export_to_json(test_json_path)
        self.assertTrue(export_result, "Failed to export to JSON")
        self.assertTrue(os.path.exists(test_json_path), "JSON file not created")

        # Import into new knowledge base
        new_kb = LegalKnowledgeBase(db_path=":memory:")
        import_result = new_kb.import_from_json(test_json_path)
        self.assertEqual(import_result, (1, 1, 1), "Incorrect import result count")

        # Verify imported data
        statute = new_kb.get_statute("stat-001")
        self.assertIsNotNone(statute, "Imported statute is None")
        self.assertEqual(statute.title, "Consumer Protection Act", "Incorrect statute title")

        case = new_kb.get_case_law("case-001")
        self.assertIsNotNone(case, "Imported case is None")
        self.assertEqual(case.title, "Smith v. Company Inc.", "Incorrect case title")

        principle = new_kb.get_legal_principle("principle-001")
        self.assertIsNotNone(principle, "Imported principle is None")
        self.assertEqual(principle.name, "Caveat Venditor", "Incorrect principle name")

        # Clean up
        if os.path.exists(test_json_path):
            os.remove(test_json_path)

    def test_statistics(self) -> None:
        """Test retrieving statistics from the knowledge base."""
        # Add initial test data
        self.kb.add_statute(self.sample_statute)
        self.kb.add_case_law(self.sample_case)
        self.kb.add_legal_principle(self.sample_principle)

        # Add additional data for varied jurisdictions and categories
        self.kb.add_statute(LegalStatute(
            id="stat-002",
            title="California Consumer Protection",
            text="State level consumer protection...",
            jurisdiction="California",
            category="Consumer Law",
            effective_date="2021-01-01"
        ))
        self.kb.add_case_law(CaseLaw(
            id="case-002",
            title="Another Case",
            citation="789 F.3d 123 (2023)",
            court="State Court",
            date_decided="2023-05-01",
            jurisdiction="California",
            summary="State case summary",
            full_text="Full text of the case..."
        ))
        self.kb.add_legal_principle(LegalPrinciple(
            id="principle-002",
            name="Due Process",
            description="Fundamental legal principle...",
            category="Constitutional Law",
            jurisdiction="Federal"
        ))

        # Retrieve and verify statistics
        stats = self.kb.get_statistics()
        self.assertEqual(stats['total_statutes'], 2, "Incorrect total statutes")
        self.assertEqual(stats['total_cases'], 2, "Incorrect total cases")
        self.assertEqual(stats['total_principles'], 2, "Incorrect total principles")
        self.assertEqual(
            stats['statutes_by_jurisdiction']['Federal'], 1,
            "Incorrect Federal statute count"
        )
        self.assertEqual(
            stats['statutes_by_jurisdiction']['California'], 1,
            "Incorrect California statute count"
        )
        self.assertEqual(
            stats['cases_by_jurisdiction']['Federal'], 1,
            "Incorrect Federal case count"
        )
        self.assertEqual(
            stats['cases_by_jurisdiction']['California'], 1,
            "Incorrect California case count"
        )
        self.assertEqual(
            stats['statutes_by_category']['Consumer Law'], 2,
            "Incorrect Consumer Law statute count"
        )
        self.assertEqual(
            stats['principles_by_category']['Consumer Law'], 1,
            "Incorrect Consumer Law principle count"
        )
        self.assertEqual(
            stats['principles_by_category']['Constitutional Law'], 1,
            "Incorrect Constitutional Law principle count"
        )


if __name__ == "__main__":
    unittest.main()