"""
Legal Knowledge Base for AI Judge System
Manages and provides access to legal knowledge, statutes, case laws, and principles.
"""

import os
import json
import logging
import sqlite3
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class LegalStatute:
    """Representation of a legal statute or regulation."""
    id: str
    title: str
    text: str
    jurisdiction: str
    category: str
    effective_date: str
    last_amended: Optional[str] = None
    sections: Optional[Dict[str, str]] = None
    citations: Optional[List[str]] = None
    vector_embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding vector embedding."""
        result = {
            "id": self.id,
            "title": self.title,
            "text": self.text,
            "jurisdiction": self.jurisdiction,
            "category": self.category,
            "effective_date": self.effective_date
        }
        
        if self.last_amended:
            result["last_amended"] = self.last_amended
        if self.sections:
            result["sections"] = self.sections
        if self.citations:
            result["citations"] = self.citations
            
        return result


@dataclass
class CaseLaw:
    """Representation of a case law precedent."""
    id: str
    title: str
    citation: str
    court: str
    date_decided: str
    jurisdiction: str
    summary: str
    full_text: str
    judges: Optional[List[str]] = None
    parties: Optional[Dict[str, str]] = None
    outcome: Optional[str] = None
    legal_principles: Optional[List[str]] = None
    related_cases: Optional[List[str]] = None
    vector_embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding vector embedding."""
        result = {
            "id": self.id,
            "title": self.title,
            "citation": self.citation,
            "court": self.court,
            "date_decided": self.date_decided,
            "jurisdiction": self.jurisdiction,
            "summary": self.summary,
            "full_text": self.full_text
        }
        
        if self.judges:
            result["judges"] = self.judges
        if self.parties:
            result["parties"] = self.parties
        if self.outcome:
            result["outcome"] = self.outcome
        if self.legal_principles:
            result["legal_principles"] = self.legal_principles
        if self.related_cases:
            result["related_cases"] = self.related_cases
            
        return result


@dataclass
class LegalPrinciple:
    """Representation of a legal principle or doctrine."""
    id: str
    name: str
    description: str
    category: str
    jurisdiction: Optional[str] = None
    source_cases: Optional[List[str]] = None
    related_principles: Optional[List[str]] = None
    exceptions: Optional[List[str]] = None
    vector_embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding vector embedding."""
        result = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category
        }
        
        if self.jurisdiction:
            result["jurisdiction"] = self.jurisdiction
        if self.source_cases:
            result["source_cases"] = self.source_cases
        if self.related_principles:
            result["related_principles"] = self.related_principles
        if self.exceptions:
            result["exceptions"] = self.exceptions
            
        return result


class VectorDBInterface:
    """
    Interface for vector database operations.
    This could be replaced with actual vector database implementations
    like Pinecone, Weaviate, Milvus, etc.
    """
    
    def __init__(self, vector_dim: int = 768):
        """
        Initialize vector database interface.
        
        Args:
            vector_dim: Dimension of vector embeddings
        """
        self.vector_dim = vector_dim
        self.statutes_db = {}  # id -> vector
        self.cases_db = {}     # id -> vector
        self.principles_db = {}  # id -> vector
        
    def add_vector(self, collection: str, id: str, vector: List[float]) -> bool:
        """
        Add a vector to the database.
        
        Args:
            collection: Collection name ("statutes", "cases", or "principles")
            id: Item ID
            vector: Vector embedding
            
        Returns:
            Success status
        """
        if len(vector) != self.vector_dim:
            logger.error(f"Vector dimension mismatch: expected {self.vector_dim}, got {len(vector)}")
            return False
        
        if collection == "statutes":
            self.statutes_db[id] = np.array(vector)
        elif collection == "cases":
            self.cases_db[id] = np.array(vector)
        elif collection == "principles":
            self.principles_db[id] = np.array(vector)
        else:
            logger.error(f"Unknown collection: {collection}")
            return False
            
        return True
    
    def search(self, collection: str, query_vector: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for similar vectors.
        
        Args:
            collection: Collection name
            query_vector: Query vector
            top_k: Number of results to return
            
        Returns:
            List of (id, similarity) tuples
        """
        query = np.array(query_vector)
        
        if collection == "statutes":
            db = self.statutes_db
        elif collection == "cases":
            db = self.cases_db
        elif collection == "principles":
            db = self.principles_db
        else:
            logger.error(f"Unknown collection: {collection}")
            return []
        
        # Simple cosine similarity search
        results = []
        for id, vector in db.items():
            # Compute cosine similarity
            dot_product = np.dot(query, vector)
            norm_query = np.linalg.norm(query)
            norm_vector = np.linalg.norm(vector)
            
            if norm_query == 0 or norm_vector == 0:
                similarity = 0
            else:
                similarity = dot_product / (norm_query * norm_vector)
                
            results.append((id, float(similarity)))
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]


class LegalKnowledgeBase:
    """
    Main class for the Legal Knowledge Base.
    Provides access to legal knowledge including statutes, case law, and principles.
    """
    
    def __init__(self, db_path: str = "legal_knowledge.db"):
        """
        Initialize the Legal Knowledge Base.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.vector_db = VectorDBInterface()
        
        # Create database if it doesn't exist
        self._initialize_db()
        
    def _initialize_db(self):
        """Initialize the SQLite database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create statutes table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS statutes (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            text TEXT NOT NULL,
            jurisdiction TEXT NOT NULL,
            category TEXT NOT NULL,
            effective_date TEXT NOT NULL,
            last_amended TEXT,
            sections TEXT,
            citations TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        ''')
        
        # Create case law table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS case_law (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            citation TEXT NOT NULL,
            court TEXT NOT NULL,
            date_decided TEXT NOT NULL,
            jurisdiction TEXT NOT NULL,
            summary TEXT NOT NULL,
            full_text TEXT NOT NULL,
            judges TEXT,
            parties TEXT,
            outcome TEXT,
            legal_principles TEXT,
            related_cases TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        ''')
        
        # Create legal principles table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS legal_principles (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT NOT NULL,
            category TEXT NOT NULL,
            jurisdiction TEXT,
            source_cases TEXT,
            related_principles TEXT,
            exceptions TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        ''')
        
        conn.commit()
        conn.close()
        
    def add_statute(self, statute: LegalStatute) -> bool:
        """
        Add a legal statute to the knowledge base.
        
        Args:
            statute: LegalStatute object
            
        Returns:
            Success status
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Convert sections, citations to JSON strings
            sections_json = json.dumps(statute.sections) if statute.sections else None
            citations_json = json.dumps(statute.citations) if statute.citations else None
            
            now = datetime.now().isoformat()
            
            # Insert into database
            cursor.execute('''
            INSERT OR REPLACE INTO statutes (
                id, title, text, jurisdiction, category, effective_date, 
                last_amended, sections, citations, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                statute.id, statute.title, statute.text, statute.jurisdiction,
                statute.category, statute.effective_date, statute.last_amended,
                sections_json, citations_json, now, now
            ))
            
            conn.commit()
            conn.close()
            
            # Add vector embedding if available
            if statute.vector_embedding:
                self.vector_db.add_vector("statutes", statute.id, statute.vector_embedding)
                
            return True
            
        except Exception as e:
            logger.error(f"Error adding statute: {str(e)}")
            return False
    
    def add_case_law(self, case: CaseLaw) -> bool:
        """
        Add a case law to the knowledge base.
        
        Args:
            case: CaseLaw object
            
        Returns:
            Success status
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Convert lists and dicts to JSON strings
            judges_json = json.dumps(case.judges) if case.judges else None
            parties_json = json.dumps(case.parties) if case.parties else None
            principles_json = json.dumps(case.legal_principles) if case.legal_principles else None
            related_json = json.dumps(case.related_cases) if case.related_cases else None
            
            now = datetime.now().isoformat()
            
            # Insert into database
            cursor.execute('''
            INSERT OR REPLACE INTO case_law (
                id, title, citation, court, date_decided, jurisdiction, summary, full_text,
                judges, parties, outcome, legal_principles, related_cases, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                case.id, case.title, case.citation, case.court, case.date_decided,
                case.jurisdiction, case.summary, case.full_text, judges_json, parties_json,
                case.outcome, principles_json, related_json, now, now
            ))
            
            conn.commit()
            conn.close()
            
            # Add vector embedding if available
            if case.vector_embedding:
                self.vector_db.add_vector("cases", case.id, case.vector_embedding)
                
            return True
            
        except Exception as e:
            logger.error(f"Error adding case law: {str(e)}")
            return False
    
    def add_legal_principle(self, principle: LegalPrinciple) -> bool:
        """
        Add a legal principle to the knowledge base.
        
        Args:
            principle: LegalPrinciple object
            
        Returns:
            Success status
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Convert lists to JSON strings
            source_cases_json = json.dumps(principle.source_cases) if principle.source_cases else None
            related_json = json.dumps(principle.related_principles) if principle.related_principles else None
            exceptions_json = json.dumps(principle.exceptions) if principle.exceptions else None
            
            now = datetime.now().isoformat()
            
            # Insert into database
            cursor.execute('''
            INSERT OR REPLACE INTO legal_principles (
                id, name, description, category, jurisdiction, source_cases,
                related_principles, exceptions, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                principle.id, principle.name, principle.description, principle.category,
                principle.jurisdiction, source_cases_json, related_json, exceptions_json,
                now, now
            ))
            
            conn.commit()
            conn.close()
            
            # Add vector embedding if available
            if principle.vector_embedding:
                self.vector_db.add_vector("principles", principle.id, principle.vector_embedding)
                
            return True
            
        except Exception as e:
            logger.error(f"Error adding legal principle: {str(e)}")
            return False
        
    def get_statute(self, statute_id: str) -> Optional[LegalStatute]:
        """
        Retrieve a statute by ID.
        
        Args:
            statute_id: ID of the statute
            
        Returns:
            LegalStatute object or None if not found
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM statutes WHERE id = ?", (statute_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
                
            # Parse JSON fields
            sections = json.loads(row['sections']) if row['sections'] else None
            citations = json.loads(row['citations']) if row['citations'] else None
            
            statute = LegalStatute(
                id=row['id'],
                title=row['title'],
                text=row['text'],
                jurisdiction=row['jurisdiction'],
                category=row['category'],
                effective_date=row['effective_date'],
                last_amended=row['last_amended'],
                sections=sections,
                citations=citations
            )
            
            conn.close()
            return statute
            
        except Exception as e:
            logger.error(f"Error retrieving statute: {str(e)}")
            return None

    def get_case_law(self, case_id: str) -> Optional[CaseLaw]:
        """
        Retrieve a case law by ID.
        
        Args:
            case_id: ID of the case law
            
        Returns:
            CaseLaw object or None if not found
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM case_law WHERE id = ?", (case_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
                
            # Parse JSON fields
            judges = json.loads(row['judges']) if row['judges'] else None
            parties = json.loads(row['parties']) if row['parties'] else None
            legal_principles = json.loads(row['legal_principles']) if row['legal_principles'] else None
            related_cases = json.loads(row['related_cases']) if row['related_cases'] else None
            
            case = CaseLaw(
                id=row['id'],
                title=row['title'],
                citation=row['citation'],
                court=row['court'],
                date_decided=row['date_decided'],
                jurisdiction=row['jurisdiction'],
                summary=row['summary'],
                full_text=row['full_text'],
                judges=judges,
                parties=parties,
                outcome=row['outcome'],
                legal_principles=legal_principles,
                related_cases=related_cases
            )
            
            conn.close()
            return case
            
        except Exception as e:
            logger.error(f"Error retrieving case law: {str(e)}")
            return None

    def get_legal_principle(self, principle_id: str) -> Optional[LegalPrinciple]:
        """
        Retrieve a legal principle by ID.
        
        Args:
            principle_id: ID of the legal principle
            
        Returns:
            LegalPrinciple object or None if not found
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM legal_principles WHERE id = ?", (principle_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
                
            # Parse JSON fields
            source_cases = json.loads(row['source_cases']) if row['source_cases'] else None
            related_principles = json.loads(row['related_principles']) if row['related_principles'] else None
            exceptions = json.loads(row['exceptions']) if row['exceptions'] else None
            
            principle = LegalPrinciple(
                id=row['id'],
                name=row['name'],
                description=row['description'],
                category=row['category'],
                jurisdiction=row['jurisdiction'],
                source_cases=source_cases,
                related_principles=related_principles,
                exceptions=exceptions
            )
            
            conn.close()
            return principle
            
        except Exception as e:
            logger.error(f"Error retrieving legal principle: {str(e)}")
            return None

    def search_statutes(self, query: str, jurisdiction: Optional[str] = None, 
                    category: Optional[str] = None, limit: int = 10) -> List[LegalStatute]:
        """
        Search for statutes by text query and optional filters.
        
        Args:
            query: Text query to search for
            jurisdiction: Optional jurisdiction filter
            category: Optional category filter
            limit: Maximum number of results to return
            
        Returns:
            List of matching LegalStatute objects
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query_params = [f"%{query}%", f"%{query}%"]
            sql = """
            SELECT * FROM statutes 
            WHERE (title LIKE ? OR text LIKE ?) 
            """
            
            if jurisdiction:
                sql += "AND jurisdiction = ? "
                query_params.append(jurisdiction)
                
            if category:
                sql += "AND category = ? "
                query_params.append(category)
                
            sql += "LIMIT ?"
            query_params.append(limit)
            
            cursor.execute(sql, tuple(query_params))
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                # Parse JSON fields
                sections = json.loads(row['sections']) if row['sections'] else None
                citations = json.loads(row['citations']) if row['citations'] else None
                
                statute = LegalStatute(
                    id=row['id'],
                    title=row['title'],
                    text=row['text'],
                    jurisdiction=row['jurisdiction'],
                    category=row['category'],
                    effective_date=row['effective_date'],
                    last_amended=row['last_amended'],
                    sections=sections,
                    citations=citations
                )
                results.append(statute)
            
            conn.close()
            return results
            
        except Exception as e:
            logger.error(f"Error searching statutes: {str(e)}")
            return []

    def search_case_law(self, query: str, jurisdiction: Optional[str] = None, 
                    court: Optional[str] = None, limit: int = 10) -> List[CaseLaw]:
        """
        Search for case law by text query and optional filters.
        
        Args:
            query: Text query to search for
            jurisdiction: Optional jurisdiction filter
            court: Optional court filter
            limit: Maximum number of results to return
            
        Returns:
            List of matching CaseLaw objects
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query_params = [f"%{query}%", f"%{query}%", f"%{query}%"]
            sql = """
            SELECT * FROM case_law 
            WHERE (title LIKE ? OR summary LIKE ? OR full_text LIKE ?) 
            """
            
            if jurisdiction:
                sql += "AND jurisdiction = ? "
                query_params.append(jurisdiction)
                
            if court:
                sql += "AND court = ? "
                query_params.append(court)
                
            sql += "LIMIT ?"
            query_params.append(limit)
            
            cursor.execute(sql, tuple(query_params))
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                # Parse JSON fields
                judges = json.loads(row['judges']) if row['judges'] else None
                parties = json.loads(row['parties']) if row['parties'] else None
                legal_principles = json.loads(row['legal_principles']) if row['legal_principles'] else None
                related_cases = json.loads(row['related_cases']) if row['related_cases'] else None
                
                case = CaseLaw(
                    id=row['id'],
                    title=row['title'],
                    citation=row['citation'],
                    court=row['court'],
                    date_decided=row['date_decided'],
                    jurisdiction=row['jurisdiction'],
                    summary=row['summary'],
                    full_text=row['full_text'],
                    judges=judges,
                    parties=parties,
                    outcome=row['outcome'],
                    legal_principles=legal_principles,
                    related_cases=related_cases
                )
                results.append(case)
            
            conn.close()
            return results
            
        except Exception as e:
            logger.error(f"Error searching case law: {str(e)}")
            return []

    def search_legal_principles(self, query: str, category: Optional[str] = None, 
                            jurisdiction: Optional[str] = None, limit: int = 10) -> List[LegalPrinciple]:
        """
        Search for legal principles by text query and optional filters.
        
        Args:
            query: Text query to search for
            category: Optional category filter
            jurisdiction: Optional jurisdiction filter
            limit: Maximum number of results to return
            
        Returns:
            List of matching LegalPrinciple objects
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query_params = [f"%{query}%", f"%{query}%"]
            sql = """
            SELECT * FROM legal_principles 
            WHERE (name LIKE ? OR description LIKE ?) 
            """
            
            if category:
                sql += "AND category = ? "
                query_params.append(category)
                
            if jurisdiction:
                sql += "AND jurisdiction = ? "
                query_params.append(jurisdiction)
                
            sql += "LIMIT ?"
            query_params.append(limit)
            
            cursor.execute(sql, tuple(query_params))
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                # Parse JSON fields
                source_cases = json.loads(row['source_cases']) if row['source_cases'] else None
                related_principles = json.loads(row['related_principles']) if row['related_principles'] else None
                exceptions = json.loads(row['exceptions']) if row['exceptions'] else None
                
                principle = LegalPrinciple(
                    id=row['id'],
                    name=row['name'],
                    description=row['description'],
                    category=row['category'],
                    jurisdiction=row['jurisdiction'],
                    source_cases=source_cases,
                    related_principles=related_principles,
                    exceptions=exceptions
                )
                results.append(principle)
            
            conn.close()
            return results
            
        except Exception as e:
            logger.error(f"Error searching legal principles: {str(e)}")
            return []

    def semantic_search(self, query_vector: List[float], collection: str, top_k: int = 5) -> List[Union[LegalStatute, CaseLaw, LegalPrinciple]]:
        """
        Perform semantic search using vector embeddings.
        
        Args:
            query_vector: Vector embedding of the query
            collection: Collection to search ("statutes", "cases", or "principles")
            top_k: Number of results to return
            
        Returns:
            List of matching objects
        """
        # Get similar vectors
        results = self.vector_db.search(collection, query_vector, top_k)
        
        # Retrieve full objects
        items = []
        for id, similarity in results:
            if collection == "statutes":
                item = self.get_statute(id)
            elif collection == "cases":
                item = self.get_case_law(id)
            elif collection == "principles":
                item = self.get_legal_principle(id)
            else:
                continue
                
            if item:
                items.append(item)
        
        return items

    def import_from_json(self, file_path: str) -> Tuple[int, int, int]:
        """
        Import legal knowledge from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Tuple of (statutes_count, cases_count, principles_count)
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return (0, 0, 0)
                
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            statutes_count = 0
            cases_count = 0
            principles_count = 0
            
            # Import statutes
            for statute_data in data.get('statutes', []):
                try:
                    statute = LegalStatute(
                        id=statute_data['id'],
                        title=statute_data['title'],
                        text=statute_data['text'],
                        jurisdiction=statute_data['jurisdiction'],
                        category=statute_data['category'],
                        effective_date=statute_data['effective_date'],
                        last_amended=statute_data.get('last_amended'),
                        sections=statute_data.get('sections'),
                        citations=statute_data.get('citations'),
                        vector_embedding=statute_data.get('vector_embedding')
                    )
                    
                    if self.add_statute(statute):
                        statutes_count += 1
                except Exception as e:
                    logger.error(f"Error importing statute: {str(e)}")
                    
            # Import case law
            for case_data in data.get('case_law', []):
                try:
                    case = CaseLaw(
                        id=case_data['id'],
                        title=case_data['title'],
                        citation=case_data['citation'],
                        court=case_data['court'],
                        date_decided=case_data['date_decided'],
                        jurisdiction=case_data['jurisdiction'],
                        summary=case_data['summary'],
                        full_text=case_data['full_text'],
                        judges=case_data.get('judges'),
                        parties=case_data.get('parties'),
                        outcome=case_data.get('outcome'),
                        legal_principles=case_data.get('legal_principles'),
                        related_cases=case_data.get('related_cases'),
                        vector_embedding=case_data.get('vector_embedding')
                    )
                    
                    if self.add_case_law(case):
                        cases_count += 1
                except Exception as e:
                    logger.error(f"Error importing case law: {str(e)}")
                    
            # Import legal principles
            for principle_data in data.get('legal_principles', []):
                try:
                    principle = LegalPrinciple(
                        id=principle_data['id'],
                        name=principle_data['name'],
                        description=principle_data['description'],
                        category=principle_data['category'],
                        jurisdiction=principle_data.get('jurisdiction'),
                        source_cases=principle_data.get('source_cases'),
                        related_principles=principle_data.get('related_principles'),
                        exceptions=principle_data.get('exceptions'),
                        vector_embedding=principle_data.get('vector_embedding')
                    )
                    
                    if self.add_legal_principle(principle):
                        principles_count += 1
                except Exception as e:
                    logger.error(f"Error importing legal principle: {str(e)}")
                    
            return (statutes_count, cases_count, principles_count)
            
        except Exception as e:
            logger.error(f"Error importing from JSON: {str(e)}")
            return (0, 0, 0)

    def export_to_json(self, file_path: str) -> bool:
        """
        Export legal knowledge to a JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Success status
        """
        try:
            # Get all data
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get statutes
            cursor.execute("SELECT * FROM statutes")
            statute_rows = cursor.fetchall()
            
            statutes = []
            for row in statute_rows:
                statute_dict = dict(row)
                
                # Parse JSON fields
                if statute_dict['sections']:
                    statute_dict['sections'] = json.loads(statute_dict['sections'])
                if statute_dict['citations']:
                    statute_dict['citations'] = json.loads(statute_dict['citations'])
                    
                # Remove timestamp fields
                statute_dict.pop('created_at', None)
                statute_dict.pop('updated_at', None)
                
                statutes.append(statute_dict)
                
            # Get case law
            cursor.execute("SELECT * FROM case_law")
            case_rows = cursor.fetchall()
            
            cases = []
            for row in case_rows:
                case_dict = dict(row)
                
                # Parse JSON fields
                if case_dict['judges']:
                    case_dict['judges'] = json.loads(case_dict['judges'])
                if case_dict['parties']:
                    case_dict['parties'] = json.loads(case_dict['parties'])
                if case_dict['legal_principles']:
                    case_dict['legal_principles'] = json.loads(case_dict['legal_principles'])
                if case_dict['related_cases']:
                    case_dict['related_cases'] = json.loads(case_dict['related_cases'])
                    
                # Remove timestamp fields
                case_dict.pop('created_at', None)
                case_dict.pop('updated_at', None)
                
                cases.append(case_dict)
                
            # Get legal principles
            cursor.execute("SELECT * FROM legal_principles")
            principle_rows = cursor.fetchall()
            
            principles = []
            for row in principle_rows:
                principle_dict = dict(row)
                
                # Parse JSON fields
                if principle_dict['source_cases']:
                    principle_dict['source_cases'] = json.loads(principle_dict['source_cases'])
                if principle_dict['related_principles']:
                    principle_dict['related_principles'] = json.loads(principle_dict['related_principles'])
                if principle_dict['exceptions']:
                    principle_dict['exceptions'] = json.loads(principle_dict['exceptions'])
                    
                # Remove timestamp fields
                principle_dict.pop('created_at', None)
                principle_dict.pop('updated_at', None)
                
                principles.append(principle_dict)
                
            conn.close()
            
            # Create output data
            output_data = {
                'statutes': statutes,
                'case_law': cases,
                'legal_principles': principles
            }
            
            # Write to file
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(output_data, f, indent=2)
                
            return True
            
        except Exception as e:
            logger.error(f"Error exporting to JSON: {str(e)}")
            return False

    def get_statistics(self) -> Dict[str, int]:
    """
    Get statistics about the knowledge base.
    
    Returns:
        Dictionary with count statistics
    """
    try:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Count statutes
        cursor.execute("SELECT COUNT(*) FROM statutes")
        statutes_count = cursor.fetchone()[0]
        
        # Count case law
        cursor.execute("SELECT COUNT(*) FROM case_law")
        cases_count = cursor.fetchone()[0]
        
        # Count legal principles
        cursor.execute("SELECT COUNT(*) FROM legal_principles")
        principles_count = cursor.fetchone()[0]
        
        # Count by jurisdiction for statutes
        cursor.execute("SELECT jurisdiction, COUNT(*) FROM statutes GROUP BY jurisdiction")
        statutes_by_jurisdiction = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Count by jurisdiction for case law
        cursor.execute("SELECT jurisdiction, COUNT(*) FROM case_law GROUP BY jurisdiction")
        cases_by_jurisdiction = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Count by category for statutes
        cursor.execute("SELECT category, COUNT(*) FROM statutes GROUP BY category")
        statutes_by_category = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Count by category for legal principles
        cursor.execute("SELECT category, COUNT(*) FROM legal_principles GROUP BY category")
        principles_by_category = {row[0]: row[1] for row in cursor.fetchall()}
        
        conn.close()
        
        return {
            'total_statutes': statutes_count,
            'total_cases': cases_count,
            'total_principles': principles_count,
            'statutes_by_jurisdiction': statutes_by_jurisdiction,
            'cases_by_jurisdiction': cases_by_jurisdiction,
            'statutes_by_category': statutes_by_category,
            'principles_by_category': principles_by_category
        }
        
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        return {
            'total_statutes': 0,
            'total_cases': 0,
            'total_principles': 0
        }