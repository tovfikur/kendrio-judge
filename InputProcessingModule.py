"""
Input Processing Module for AI Judge System
Converts user-provided case details into a standardized format.
"""

import re
import json
import logging
import spacy
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict

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
class CaseInput:
    """Standardized structure for case input."""
    facts: str
    evidence: Optional[List[str]] = None
    parties: Optional[Dict[str, str]] = None
    jurisdiction: Optional[str] = None
    legal_question: Optional[str] = None
    case_type: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

class InputValidator:
    """Validates the completeness and structure of case inputs."""
    
    @staticmethod
    def is_valid_case(case: CaseInput) -> Tuple[bool, str]:
        """
        Check if a case input is valid.
        
        Args:
            case: The case input to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Facts are mandatory
        if not case.facts or len(case.facts.strip()) < 20:
            return False, "Case facts are missing or too brief"
        
        # Check minimum word count for facts (at least 10 words)
        if len(case.facts.split()) < 10:
            return False, "Case facts should be more detailed (at least 10 words)"
        
        # All checks passed
        return True, ""


class InputProcessor:
    """Processes raw case inputs into standardized format."""
    
    def __init__(self):
        """Initialize the input processor."""
        self.validator = InputValidator()
    
    def process(self, raw_input: str) -> Optional[CaseInput]:
        """
        Process raw text input into a structured case.
        
        Args:
            raw_input: Raw text describing the case
            
        Returns:
            Structured CaseInput object or None if processing fails
        """
        try:
            # First, try to parse as JSON if input looks like JSON
            if raw_input.strip().startswith('{') and raw_input.strip().endswith('}'):
                return self._process_json_input(raw_input)
            
            # Otherwise process as free text
            return self._process_text_input(raw_input)
            
        except Exception as e:
            logger.error(f"Error processing input: {str(e)}")
            return None
    
    def _process_json_input(self, json_input: str) -> Optional[CaseInput]:
        """Process input that appears to be in JSON format."""
        try:
            data = json.loads(json_input)
            
            # Extract required fields
            if "facts" not in data:
                logger.error("JSON input missing required 'facts' field")
                return None
            
            # Create case input object
            case = CaseInput(
                facts=data.get("facts", ""),
                evidence=data.get("evidence", []),
                parties=data.get("parties", {}),
                jurisdiction=data.get("jurisdiction"),
                legal_question=data.get("legal_question"),
                case_type=data.get("case_type")
            )
            
            # Validate the case
            valid, error_msg = self.validator.is_valid_case(case)
            if not valid:
                logger.error(f"Invalid case input: {error_msg}")
                return None
                
            return case
            
        except json.JSONDecodeError:
            logger.warning("Input appeared to be JSON but couldn't be parsed")
            return self._process_text_input(json_input)
    
    def _process_text_input(self, text_input: str) -> Optional[CaseInput]:
        """Process free-form text input into a structured case."""
        # Basic facts extraction - assume entire text is the facts if no clear structure
        facts = text_input.strip()
        evidence = []
        parties = {}
        jurisdiction = None
        legal_question = None
        case_type = None
        
        # Try to extract structured information using NLP if available
        if nlp:
            doc = nlp(text_input)
            
            # Extract potential evidence statements (sentences with factual claims)
            evidence_candidates = []
            for sent in doc.sents:
                # Simple heuristic: sentences with dates, numbers, or specific entities
                # are likely to be evidence
                if any(token.like_num for token in sent) or any(ent.label_ in ["DATE", "TIME"] for ent in sent.ents):
                    evidence_candidates.append(sent.text.strip())
            
            if evidence_candidates:
                evidence = evidence_candidates[:5]  # Limit to top 5 for now
            
            # Extract parties (named entities that might be people or organizations)
            party_candidates = {}
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE"]:
                    party_candidates[ent.text] = ent.label_
            
            # Filter to most frequently mentioned parties
            if party_candidates:
                # Get top 2 most common entities
                top_parties = sorted(party_candidates.items(), 
                                    key=lambda x: text_input.count(x[0]), 
                                    reverse=True)[:2]
                parties = {name: label for name, label in top_parties}
            
            # Try to identify jurisdiction
            jurisdiction_keywords = ["federal", "state", "district", "supreme court", 
                                    "circuit", "county", "municipal"]
            for keyword in jurisdiction_keywords:
                if keyword in text_input.lower():
                    # Find the sentence with this keyword
                    for sent in doc.sents:
                        if keyword in sent.text.lower():
                            jurisdiction = sent.text.strip()
                            break
                    if jurisdiction:
                        break
            
            # Try to identify the legal question
            question_indicators = ["whether", "should", "can", "must", "?", "legal question"]
            for sent in doc.sents:
                sent_text = sent.text.lower()
                if any(indicator in sent_text for indicator in question_indicators):
                    legal_question = sent.text.strip()
                    break
            
            # Try to determine case type
            case_types = {
                "criminal": ["criminal", "prosecution", "defendant", "guilty", "crime"],
                "civil": ["civil", "damages", "liability", "plaintiff", "defendant", "sue"],
                "family": ["divorce", "custody", "family", "child support"],
                "contract": ["contract", "agreement", "breach", "terms"],
                "property": ["property", "land", "deed", "easement", "real estate"],
                "tort": ["negligence", "injury", "damages", "tort"]
            }
            
            case_type_scores = {case_type: 0 for case_type in case_types}
            for case_type, keywords in case_types.items():
                for keyword in keywords:
                    case_type_scores[case_type] += text_input.lower().count(keyword)
            
            # Assign the case type with the highest score if any matches found
            max_score = max(case_type_scores.values())
            if max_score > 0:
                case_type = max(case_type_scores.items(), key=lambda x: x[1])[0]
        
        # Create case input object
        case = CaseInput(
            facts=facts,
            evidence=evidence if evidence else None,
            parties=parties if parties else None,
            jurisdiction=jurisdiction,
            legal_question=legal_question,
            case_type=case_type
        )
        
        # Validate the case
        valid, error_msg = self.validator.is_valid_case(case)
        if not valid:
            logger.error(f"Invalid case input: {error_msg}")
            return None
            
        return case

class InputEnricher:
    """Enriches case inputs with additional context and information."""
    
    def __init__(self):
        """Initialize the input enricher."""
        pass
    
    def enrich(self, case: CaseInput) -> CaseInput:
        """
        Enrich the case with additional information.
        
        Args:
            case: The case input to enrich
            
        Returns:
            Enriched case input
        """
        # Placeholder for enrichment logic
        # In a real system, this might:
        # 1. Identify and extract additional entities
        # 2. Perform sentiment analysis on arguments
        # 3. Link to relevant legal concepts
        # 4. Add contextual information
        
        return case


class InputNormalizer:
    """Normalizes case inputs to ensure consistency."""
    
    def normalize(self, case: CaseInput) -> CaseInput:
        """
        Normalize the case input for consistency.
        
        Args:
            case: The case input to normalize
            
        Returns:
            Normalized case input
        """
        # Normalize facts - remove excessive whitespace and standardize formatting
        if case.facts:
            case.facts = re.sub(r'\s+', ' ', case.facts).strip()
        
        # Normalize evidence list
        if case.evidence:
            case.evidence = [re.sub(r'\s+', ' ', e).strip() for e in case.evidence]
            # Remove duplicates while preserving order
            seen = set()
            case.evidence = [x for x in case.evidence if not (x in seen or seen.add(x))]
        
        # Normalize party names (e.g., consistent capitalization)
        if case.parties:
            normalized_parties = {}
            for name, role in case.parties.items():
                # Properly capitalize names (Title Case)
                normalized_name = ' '.join(word.capitalize() for word in name.split())
                normalized_parties[normalized_name] = role
            case.parties = normalized_parties
        
        # Normalize jurisdiction if present
        if case.jurisdiction:
            case.jurisdiction = re.sub(r'\s+', ' ', case.jurisdiction).strip()
        
        # Normalize legal question if present
        if case.legal_question:
            # Ensure legal question ends with a question mark
            case.legal_question = re.sub(r'\s+', ' ', case.legal_question).strip()
            if not case.legal_question.endswith('?'):
                case.legal_question += '?'
        
        # Normalize case type to standard vocabulary if present
        if case.case_type:
            # Map synonyms to standard case types
            type_mapping = {
                "criminal case": "criminal",
                "civil case": "civil",
                "family law": "family",
                "contract dispute": "contract",
                "property dispute": "property",
                "tort case": "tort"
            }
            
            case.case_type = case.case_type.lower()
            for synonym, standard in type_mapping.items():
                if case.case_type == synonym or case.case_type.startswith(synonym):
                    case.case_type = standard
                    break
        
        return case

# Main pipeline for processing inputs
class InputPipeline:
    """Full pipeline for processing and validating case inputs."""
    
    def __init__(self):
        """Initialize the input pipeline with its components."""
        self.processor = InputProcessor()
        self.validator = InputValidator()
        self.enricher = InputEnricher()
        self.normalizer = InputNormalizer()
    
    def process(self, raw_input: str) -> Optional[CaseInput]:
        """
        Process raw input through the complete pipeline.
        
        Args:
            raw_input: Raw text or JSON describing the case
            
        Returns:
            Processed and enriched CaseInput object or None if processing fails
        """
        # Initial processing
        case = self.processor.process(raw_input)
        if not case:
            return None
        
        # Normalize
        case = self.normalizer.normalize(case)
        
        # Enrich
        case = self.enricher.enrich(case)
        
        # Final validation
        valid, error_msg = self.validator.is_valid_case(case)
        if not valid:
            logger.error(f"Input failed final validation: {error_msg}")
            return None
        
        return case

if __name__ == "__main__":
    # Example usage
    pipeline = InputPipeline()
    
    # Test with a simple text input
    sample_input = """
    John Smith and ABC Corporation are involved in a contract dispute.
    On January 15, 2023, Smith signed a contract to provide consulting services to ABC Corp.
    After completing the work, ABC refused to pay the agreed amount of $50,000.
    Smith is now suing for breach of contract in the New York State court.
    The question is whether ABC Corp is liable for the full contract amount.
    """
    
    result = pipeline.process(sample_input)
    if result:
        print("Processed Case:")
        print(result.to_json())
    else:
        print("Failed to process input")