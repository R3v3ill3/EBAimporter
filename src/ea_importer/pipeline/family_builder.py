"""
Family Builder for EA Importer - Gold text selection and family management.
"""

import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

from ..core.config import get_settings
from ..core.logging import get_logger, log_function_call
from ..models import ClusterCandidate, ClauseSegment
from ..utils import Fingerprinter

logger = get_logger(__name__)


class FamilyBuilder:
    """Family management system with gold text selection."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.settings = get_settings()
        self.config = config or {}
        self.fingerprinter = Fingerprinter()
    
    @log_function_call
    def build_families_from_clusters(
        self, 
        cluster_candidates: List[ClusterCandidate],
        document_clauses: Dict[str, List[ClauseSegment]]
    ) -> Dict[str, Dict[str, Any]]:
        """Build agreement families from cluster candidates."""
        
        families = {}
        
        for candidate in cluster_candidates:
            if candidate.confidence_level in ['high', 'medium']:
                family_data = self._build_single_family(candidate, document_clauses)
                families[candidate.cluster_id] = family_data
        
        logger.info(f"Built {len(families)} families from clusters")
        return families
    
    def _build_single_family(
        self, 
        candidate: ClusterCandidate, 
        document_clauses: Dict[str, List[ClauseSegment]]
    ) -> Dict[str, Any]:
        """Build a single family from cluster candidate."""
        
        # Select gold document (most complete/recent)
        gold_doc_id = self._select_gold_document(candidate.document_ids, document_clauses)
        
        # Get all clauses for family members
        all_clauses = {}
        for doc_id in candidate.document_ids:
            if doc_id in document_clauses:
                all_clauses[doc_id] = document_clauses[doc_id]
        
        # Build merged clause structure
        family_clauses = self._merge_family_clauses(gold_doc_id, all_clauses)
        
        return {
            'family_id': candidate.cluster_id,
            'title': candidate.suggested_title,
            'gold_document_id': gold_doc_id,
            'member_document_ids': candidate.document_ids,
            'confidence_level': candidate.confidence_level,
            'family_clauses': family_clauses,
            'created_at': datetime.now().isoformat()
        }
    
    def _select_gold_document(
        self, 
        document_ids: List[str], 
        document_clauses: Dict[str, List[ClauseSegment]]
    ) -> str:
        """Select the best document as gold standard."""
        
        best_doc = document_ids[0]
        best_score = 0
        
        for doc_id in document_ids:
            if doc_id not in document_clauses:
                continue
                
            clauses = document_clauses[doc_id]
            
            # Score based on completeness and structure
            score = len(clauses)  # More clauses = more complete
            score += sum(1 for c in clauses if c.heading)  # Bonus for headings
            score += sum(len(c.path) for c in clauses) * 0.1  # Bonus for structure
            
            if score > best_score:
                best_score = score
                best_doc = doc_id
        
        return best_doc
    
    def _merge_family_clauses(
        self, 
        gold_doc_id: str, 
        all_clauses: Dict[str, List[ClauseSegment]]
    ) -> List[Dict[str, Any]]:
        """Merge clauses from family members using gold document as base."""
        
        if gold_doc_id not in all_clauses:
            return []
        
        gold_clauses = all_clauses[gold_doc_id]
        
        # Start with gold document structure
        family_clauses = []
        for clause in gold_clauses:
            family_clause = {
                'clause_id': clause.clause_id,
                'heading': clause.heading,
                'text': clause.text,
                'path': clause.path,
                'source_document_id': gold_doc_id,
                'hash_sha256': clause.hash_sha256,
                'token_count': clause.token_count,
                'variants': []
            }
            
            # Find variations in other documents
            for doc_id, doc_clauses in all_clauses.items():
                if doc_id == gold_doc_id:
                    continue
                    
                # Find matching clause by ID or similarity
                matching_clause = self._find_matching_clause(clause, doc_clauses)
                if matching_clause and matching_clause.text != clause.text:
                    family_clause['variants'].append({
                        'source_document_id': doc_id,
                        'text': matching_clause.text,
                        'hash_sha256': matching_clause.hash_sha256
                    })
            
            family_clauses.append(family_clause)
        
        return family_clauses
    
    def _find_matching_clause(
        self, 
        target_clause: ClauseSegment, 
        candidate_clauses: List[ClauseSegment]
    ) -> Optional[ClauseSegment]:
        """Find matching clause in candidate list."""
        
        # Try exact clause ID match first
        for clause in candidate_clauses:
            if clause.clause_id == target_clause.clause_id:
                return clause
        
        # Try path-based matching
        for clause in candidate_clauses:
            if clause.path == target_clause.path:
                return clause
        
        # Try text similarity matching
        best_match = None
        best_similarity = 0.7  # Minimum threshold
        
        for clause in candidate_clauses:
            # Simple word overlap similarity
            target_words = set(target_clause.text.lower().split())
            candidate_words = set(clause.text.lower().split())
            
            if len(target_words) > 0:
                similarity = len(target_words & candidate_words) / len(target_words | candidate_words)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = clause
        
        return best_match
    
    def save_family(self, family_data: Dict[str, Any], output_dir: Path) -> Path:
        """Save family data to file."""
        family_id = family_data['family_id']
        family_dir = output_dir / family_id
        family_dir.mkdir(parents=True, exist_ok=True)
        
        # Save family definition
        family_file = family_dir / "family.json"
        with open(family_file, 'w') as f:
            json.dump(family_data, f, indent=2)
        
        # Save clauses as JSONL
        clauses_file = family_dir / "family_clauses.jsonl"
        with open(clauses_file, 'w') as f:
            for clause in family_data['family_clauses']:
                f.write(json.dumps(clause) + '\n')
        
        logger.info(f"Saved family {family_id} to {family_dir}")
        return family_file


__all__ = ['FamilyBuilder']