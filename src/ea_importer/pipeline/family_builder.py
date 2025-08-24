"""
Family Builder - Implements family clustering and gold text selection logic.
"""

import json
import uuid
from datetime import datetime, date
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from collections import defaultdict

import pandas as pd
from sqlalchemy.orm import Session

from ..core.config import get_settings
from ..core.logging import LoggerMixin
from ..database import get_database_manager
from ..models import (
    AgreementFamily, FamilyClause, FamilyRate, FamilyRule, 
    IngestRun, ClusterRun
)
from ..pipeline.clustering import ClusteringResult, ClusterCandidate
from ..utils.text_cleaner import create_text_cleaner
from ..utils.text_segmenter import ClauseSegment
from ..utils.fingerprinter import create_clause_fingerprinter


@dataclass
class GoldTextCandidate:
    """Represents a candidate for gold text selection."""
    ea_id: str
    score: float
    recency_score: float
    completeness_score: float
    quality_score: float
    num_clauses: int
    text_length: int
    creation_date: Optional[datetime] = None


@dataclass
class FamilyBuildResult:
    """Result of family building process."""
    family_id: str
    title: str
    ea_ids: List[str]
    gold_ea_id: str
    num_clauses: int
    confidence_score: float
    jurisdiction: str
    effective_from: Optional[date] = None
    effective_to: Optional[date] = None


class FamilyBuilder(LoggerMixin):
    """Builds EA families from clustering results with gold text selection."""
    
    def __init__(self):
        """Initialize the family builder."""
        self.settings = get_settings()
        self.db_manager = get_database_manager()
        self.text_cleaner = create_text_cleaner()
        self.clause_fingerprinter = create_clause_fingerprinter()
    
    def load_clause_files(self, ea_ids: List[str]) -> Dict[str, List[Dict]]:
        """
        Load clause files for a list of EA IDs.
        
        Args:
            ea_ids: List of EA identifiers
            
        Returns:
            Dictionary mapping EA ID to list of clauses
        """
        clause_data = {}
        
        for ea_id in ea_ids:
            clauses_file = self.settings.clauses_dir / f"{ea_id}.jsonl"
            
            if not clauses_file.exists():
                self.logger.warning(f"Clause file not found for {ea_id}: {clauses_file}")
                clause_data[ea_id] = []
                continue
            
            clauses = []
            try:
                with open(clauses_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        clause = json.loads(line.strip())
                        clauses.append(clause)
                
                clause_data[ea_id] = clauses
                self.logger.debug(f"Loaded {len(clauses)} clauses for {ea_id}")
                
            except Exception as e:
                self.logger.error(f"Failed to load clauses for {ea_id}: {e}")
                clause_data[ea_id] = []
        
        return clause_data
    
    def calculate_completeness_score(self, clauses: List[Dict]) -> float:
        """
        Calculate completeness score based on clause coverage.
        
        Args:
            clauses: List of clause dictionaries
            
        Returns:
            Completeness score (0.0 to 1.0)
        """
        if not clauses:
            return 0.0
        
        # Check for key EA sections
        required_sections = {
            'definition': ['definition', 'interpret', 'meaning'],
            'classification': ['classification', 'grade', 'level'],
            'rates': ['rate', 'wage', 'salary', 'pay'],
            'hours': ['hours', 'working time', 'ordinary hours'],
            'leave': ['leave', 'holiday', 'vacation'],
            'penalty': ['penalty', 'overtime', 'shift'],
            'allowance': ['allowance', 'expense', 'reimbursement'],
            'procedure': ['procedure', 'process', 'application'],
        }
        
        sections_found = set()
        
        for clause in clauses:
            clause_text = (clause.get('heading', '') + ' ' + clause.get('text', '')).lower()
            
            for section_type, keywords in required_sections.items():
                if any(keyword in clause_text for keyword in keywords):
                    sections_found.add(section_type)
        
        # Score based on section coverage
        coverage_score = len(sections_found) / len(required_sections)
        
        # Bonus for having many clauses (indicates comprehensive coverage)
        clause_count_score = min(1.0, len(clauses) / 50)  # Normalize to 50 clauses
        
        # Weighted combination
        return 0.7 * coverage_score + 0.3 * clause_count_score
    
    def calculate_quality_score(self, clauses: List[Dict]) -> float:
        """
        Calculate quality score based on clause content quality.
        
        Args:
            clauses: List of clause dictionaries
            
        Returns:
            Quality score (0.0 to 1.0)
        """
        if not clauses:
            return 0.0
        
        quality_indicators = []
        
        for clause in clauses:
            text = clause.get('text', '')
            
            # Length indicators (not too short, not too long)
            length = len(text)
            if 50 <= length <= 2000:
                length_score = 1.0
            elif length < 20:
                length_score = 0.0
            else:
                length_score = max(0.0, 1.0 - abs(length - 500) / 1500)
            
            # Structure indicators
            has_numbering = bool(clause.get('clause_id'))
            has_heading = bool(clause.get('heading'))
            has_hierarchy = len(clause.get('path', [])) > 0
            
            structure_score = (has_numbering + has_heading + has_hierarchy) / 3
            
            # Content quality indicators
            has_definitions = any(word in text.lower() for word in ['means', 'defined as', 'refers to'])
            has_amounts = bool(__import__('re').search(r'\$\d+|\d+\.\d{2}', text))
            has_dates = bool(__import__('re').search(r'\d{1,2}[/-]\d{1,2}[/-]\d{4}|\d{4}-\d{2}-\d{2}', text))
            
            content_score = (has_definitions + has_amounts + has_dates) / 3
            
            # Combined clause quality
            clause_quality = 0.4 * length_score + 0.3 * structure_score + 0.3 * content_score
            quality_indicators.append(clause_quality)
        
        return sum(quality_indicators) / len(quality_indicators)
    
    def calculate_recency_score(self, ea_id: str) -> float:
        """
        Calculate recency score based on file modification time or extracted dates.
        
        Args:
            ea_id: EA identifier
            
        Returns:
            Recency score (0.0 to 1.0)
        """
        try:
            # Try to get file modification time
            text_file = self.settings.text_dir / f"{ea_id}.txt"
            if text_file.exists():
                mtime = datetime.fromtimestamp(text_file.stat().st_mtime)
                days_old = (datetime.now() - mtime).days
                
                # Score based on age (newer is better)
                if days_old <= 30:
                    return 1.0
                elif days_old <= 365:
                    return 1.0 - (days_old - 30) / 335
                else:
                    return max(0.1, 1.0 - days_old / 3650)  # 10 year decay
            
        except Exception as e:
            self.logger.debug(f"Could not get recency for {ea_id}: {e}")
        
        return 0.5  # Default middle score
    
    def select_gold_candidate(self, cluster: ClusterCandidate) -> GoldTextCandidate:
        """
        Select the best gold text candidate from a cluster.
        
        Args:
            cluster: Cluster of similar EAs
            
        Returns:
            Selected gold text candidate
        """
        self.logger.info(f"Selecting gold candidate from {len(cluster.ea_ids)} EAs")
        
        # Load clause data for all EAs in cluster
        clause_data = self.load_clause_files(cluster.ea_ids)
        
        candidates = []
        
        for ea_id in cluster.ea_ids:
            clauses = clause_data.get(ea_id, [])
            
            # Calculate scores
            completeness_score = self.calculate_completeness_score(clauses)
            quality_score = self.calculate_quality_score(clauses)
            recency_score = self.calculate_recency_score(ea_id)
            
            # Combined score (weighted)
            overall_score = (
                0.4 * completeness_score +
                0.3 * quality_score +
                0.3 * recency_score
            )
            
            candidate = GoldTextCandidate(
                ea_id=ea_id,
                score=overall_score,
                recency_score=recency_score,
                completeness_score=completeness_score,
                quality_score=quality_score,
                num_clauses=len(clauses),
                text_length=sum(len(c.get('text', '')) for c in clauses)
            )
            
            candidates.append(candidate)
            
            self.logger.debug(f"{ea_id}: score={overall_score:.3f} "
                            f"(completeness={completeness_score:.3f}, "
                            f"quality={quality_score:.3f}, "
                            f"recency={recency_score:.3f})")
        
        # Sort by score and select best
        candidates.sort(key=lambda x: x.score, reverse=True)
        best_candidate = candidates[0]
        
        self.logger.info(f"Selected gold candidate: {best_candidate.ea_id} "
                        f"(score={best_candidate.score:.3f})")
        
        return best_candidate
    
    def merge_clause_content(self, 
                           cluster: ClusterCandidate, 
                           gold_ea_id: str) -> List[Dict]:
        """
        Merge clause content using gold EA as base with selective enhancement.
        
        Args:
            cluster: Cluster of EAs
            gold_ea_id: Selected gold EA ID
            
        Returns:
            List of merged clause dictionaries
        """
        self.logger.info(f"Merging clause content for family with gold EA: {gold_ea_id}")
        
        # Load all clause data
        clause_data = self.load_clause_files(cluster.ea_ids)
        gold_clauses = clause_data.get(gold_ea_id, [])
        
        if not gold_clauses:
            self.logger.error(f"No clauses found for gold EA: {gold_ea_id}")
            return []
        
        # Use gold clauses as base
        merged_clauses = []
        
        for gold_clause in gold_clauses:
            merged_clause = gold_clause.copy()
            
            # Clean the text
            cleaned_text = self.text_cleaner.clean_clause_text(merged_clause.get('text', ''))
            merged_clause['text'] = cleaned_text
            
            # Add family metadata
            merged_clause['family_gold_source'] = gold_ea_id
            merged_clause['merge_confidence'] = 1.0  # Gold clauses have highest confidence
            
            # Look for similar clauses in other EAs to potentially enhance
            similar_clauses = self._find_similar_clauses_in_cluster(
                gold_clause, clause_data, gold_ea_id
            )
            
            if similar_clauses:
                # Enhance with additional information if available
                enhanced_text = self._enhance_clause_text(gold_clause, similar_clauses)
                if enhanced_text != merged_clause['text']:
                    merged_clause['text'] = enhanced_text
                    merged_clause['enhanced_from'] = [sc['ea_id'] for sc in similar_clauses]
            
            merged_clauses.append(merged_clause)
        
        self.logger.info(f"Merged {len(merged_clauses)} clauses for family")
        return merged_clauses
    
    def _find_similar_clauses_in_cluster(self,
                                       target_clause: Dict,
                                       clause_data: Dict[str, List[Dict]],
                                       exclude_ea_id: str) -> List[Dict]:
        """Find similar clauses across cluster members."""
        target_text = target_clause.get('text', '')
        target_clause_id = target_clause.get('clause_id', '')
        
        similar_clauses = []
        
        for ea_id, clauses in clause_data.items():
            if ea_id == exclude_ea_id:
                continue
            
            for clause in clauses:
                # Check clause ID similarity first
                if target_clause_id and clause.get('clause_id') == target_clause_id:
                    similarity = self.clause_fingerprinter.compare_clauses(
                        target_text, clause.get('text', '')
                    )
                    
                    if similarity >= 0.7:  # High similarity threshold
                        similar_clauses.append({
                            'ea_id': ea_id,
                            'clause': clause,
                            'similarity': similarity
                        })
        
        # Sort by similarity
        similar_clauses.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_clauses[:3]  # Top 3 most similar
    
    def _enhance_clause_text(self, 
                           gold_clause: Dict, 
                           similar_clauses: List[Dict]) -> str:
        """
        Enhance gold clause text with information from similar clauses.
        
        Args:
            gold_clause: Gold clause to enhance
            similar_clauses: Similar clauses from other EAs
            
        Returns:
            Enhanced clause text
        """
        gold_text = gold_clause.get('text', '')
        
        # For now, just return the gold text
        # In a more sophisticated implementation, we could:
        # 1. Extract missing definitions
        # 2. Add clarifying language
        # 3. Merge complementary information
        # 4. Standardize formatting
        
        # Simple enhancement: if gold text is significantly shorter,
        # consider using a longer similar clause
        if len(gold_text) < 100:
            for similar in similar_clauses:
                similar_text = similar['clause'].get('text', '')
                if (len(similar_text) > len(gold_text) * 1.5 and 
                    similar['similarity'] > 0.9):
                    
                    self.logger.debug(f"Enhanced short clause with longer similar text")
                    return similar_text
        
        return gold_text
    
    def extract_family_metadata(self, 
                              cluster: ClusterCandidate, 
                              gold_clauses: List[Dict]) -> Dict[str, Any]:
        """
        Extract family-level metadata from clauses.
        
        Args:
            cluster: EA cluster
            gold_clauses: Gold clause data
            
        Returns:
            Family metadata dictionary
        """
        # Extract title from first few clauses
        title = "Enterprise Agreement Family"
        
        for clause in gold_clauses[:5]:
            heading = clause.get('heading', '')
            text = clause.get('text', '')
            
            # Look for agreement title patterns
            if any(word in heading.lower() for word in ['agreement', 'title', 'name']):
                if len(heading) > 10:
                    title = heading
                    break
            
            # Look in text for title patterns
            import re
            title_match = re.search(r'(?:this|the)\s+(.+?)\s+(?:agreement|enterprise)', 
                                  text, re.IGNORECASE)
            if title_match:
                potential_title = title_match.group(1).strip()
                if 10 <= len(potential_title) <= 100:
                    title = potential_title + " Enterprise Agreement"
                    break
        
        # Extract jurisdiction (default to settings)
        jurisdiction = self.settings.jurisdiction
        
        # Try to extract from text
        jurisdiction_patterns = [
            r'(?:New South Wales|NSW)',
            r'(?:Victoria|VIC)', 
            r'(?:Queensland|QLD)',
            r'(?:Western Australia|WA)',
            r'(?:South Australia|SA)',
            r'(?:Tasmania|TAS)',
            r'(?:Northern Territory|NT)',
            r'(?:Australian Capital Territory|ACT)',
            r'(?:Commonwealth|Federal|National)'
        ]
        
        for clause in gold_clauses[:10]:
            text = clause.get('text', '')
            for pattern in jurisdiction_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    jurisdiction = re.search(pattern, text, re.IGNORECASE).group(0)
                    break
        
        # Extract effective dates
        effective_from = None
        effective_to = None
        
        date_pattern = r'(\d{1,2}[/-]\d{1,2}[/-]\d{4}|\d{4}-\d{2}-\d{2})'
        
        for clause in gold_clauses[:10]:
            text = clause.get('text', '')
            
            if 'commencement' in text.lower() or 'effective' in text.lower():
                dates = re.findall(date_pattern, text)
                if dates:
                    try:
                        # Try to parse first date as effective_from
                        date_str = dates[0].replace('/', '-')
                        if len(date_str.split('-')[0]) == 2:
                            # Day/month/year format
                            parts = date_str.split('-')
                            date_str = f"{parts[2]}-{parts[1]}-{parts[0]}"
                        
                        effective_from = datetime.strptime(date_str, '%Y-%m-%d').date()
                        
                        if len(dates) > 1:
                            # Second date might be expiry
                            date_str = dates[1].replace('/', '-')
                            if len(date_str.split('-')[0]) == 2:
                                parts = date_str.split('-')
                                date_str = f"{parts[2]}-{parts[1]}-{parts[0]}"
                            effective_to = datetime.strptime(date_str, '%Y-%m-%d').date()
                        
                        break
                    except ValueError:
                        continue
        
        return {
            'title': title,
            'jurisdiction': jurisdiction,
            'effective_from': effective_from,
            'effective_to': effective_to,
            'description': f"Family of {len(cluster.ea_ids)} similar Enterprise Agreements",
            'industry_sector': None,  # Could be extracted with more sophisticated NLP
            'coverage': None,
            'source_documents': cluster.ea_ids
        }
    
    def build_family_from_cluster(self, 
                                cluster: ClusterCandidate,
                                ingest_run_id: Optional[str] = None) -> FamilyBuildResult:
        """
        Build a complete EA family from a cluster.
        
        Args:
            cluster: Cluster of similar EAs
            ingest_run_id: Optional ingest run ID for tracking
            
        Returns:
            Family build result
        """
        self.logger.info(f"Building family from cluster of {len(cluster.ea_ids)} EAs")
        
        # Step 1: Select gold candidate
        gold_candidate = self.select_gold_candidate(cluster)
        
        # Step 2: Merge clause content
        merged_clauses = self.merge_clause_content(cluster, gold_candidate.ea_id)
        
        # Step 3: Extract family metadata
        metadata = self.extract_family_metadata(cluster, merged_clauses)
        
        # Step 4: Create family in database
        family_id = str(uuid.uuid4())
        
        try:
            with self.db_manager.session_scope() as session:
                # Create family record
                family = AgreementFamily(
                    id=uuid.UUID(family_id),
                    title=metadata['title'],
                    jurisdiction=metadata['jurisdiction'],
                    version=self.settings.target_version,
                    effective_from=metadata['effective_from'],
                    effective_to=metadata['effective_to'],
                    checksum="",  # Will be computed from clauses
                    description=metadata['description'],
                    industry_sector=metadata['industry_sector'],
                    coverage=metadata['coverage'],
                    source_documents=metadata['source_documents'],
                    ingest_run_id=uuid.UUID(ingest_run_id) if ingest_run_id else None,
                    gold_document_id=gold_candidate.ea_id
                )
                session.add(family)
                session.flush()  # Get the ID
                
                # Create clause records
                checksum_data = []
                for clause_dict in merged_clauses:
                    clause = FamilyClause(
                        family_id=family.id,
                        clause_id=clause_dict.get('clause_id', ''),
                        heading=clause_dict.get('heading'),
                        text=clause_dict.get('text', ''),
                        path=clause_dict.get('path', []),
                        hash_sha256=clause_dict.get('hash_sha256', ''),
                        tokens=clause_dict.get('tokens'),
                        page_spans=clause_dict.get('page_spans'),
                        clause_type=clause_dict.get('clause_type')
                    )
                    session.add(clause)
                    checksum_data.append(clause_dict.get('text', ''))
                
                # Compute and update family checksum
                import hashlib
                family_content = '\n'.join(checksum_data)
                family.checksum = hashlib.sha256(family_content.encode('utf-8')).hexdigest()
                
                session.commit()
                
                self.logger.info(f"Created family {family_id} with {len(merged_clauses)} clauses")
        
        except Exception as e:
            self.logger.error(f"Failed to create family in database: {e}")
            raise
        
        # Step 5: Save family files
        self._save_family_files(family_id, merged_clauses, metadata)
        
        return FamilyBuildResult(
            family_id=family_id,
            title=metadata['title'],
            ea_ids=cluster.ea_ids,
            gold_ea_id=gold_candidate.ea_id,
            num_clauses=len(merged_clauses),
            confidence_score=cluster.confidence_score,
            jurisdiction=metadata['jurisdiction'],
            effective_from=metadata['effective_from'],
            effective_to=metadata['effective_to']
        )
    
    def _save_family_files(self, 
                         family_id: str, 
                         merged_clauses: List[Dict], 
                         metadata: Dict[str, Any]):
        """
        Save family files to disk.
        
        Args:
            family_id: Family identifier
            merged_clauses: Merged clause data
            metadata: Family metadata
        """
        family_dir = self.settings.families_dir / family_id / "gold"
        family_dir.mkdir(parents=True, exist_ok=True)
        
        # Save family clauses as JSONL
        clauses_file = family_dir / "family_clauses.jsonl"
        with open(clauses_file, 'w', encoding='utf-8') as f:
            for clause in merged_clauses:
                clause_copy = clause.copy()
                clause_copy['family_id'] = family_id
                f.write(json.dumps(clause_copy, ensure_ascii=False) + '\n')
        
        # Save family metadata
        metadata_file = family_dir / "family_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            # Convert dates to strings for JSON serialization
            serializable_metadata = metadata.copy()
            for key, value in serializable_metadata.items():
                if isinstance(value, date):
                    serializable_metadata[key] = value.isoformat()
            
            json.dump(serializable_metadata, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved family files to {family_dir}")
    
    def build_families_from_clustering_result(self, 
                                            clustering_result: ClusteringResult,
                                            ingest_run_id: Optional[str] = None) -> List[FamilyBuildResult]:
        """
        Build families from clustering result.
        
        Args:
            clustering_result: Result from clustering analysis
            ingest_run_id: Optional ingest run ID for tracking
            
        Returns:
            List of family build results
        """
        self.logger.info(f"Building families from {len(clustering_result.clusters)} clusters")
        
        family_results = []
        
        for i, cluster in enumerate(clustering_result.clusters):
            try:
                self.logger.info(f"Processing cluster {i+1}/{len(clustering_result.clusters)}")
                
                result = self.build_family_from_cluster(cluster, ingest_run_id)
                family_results.append(result)
                
                self.logger.info(f"Successfully built family {result.family_id}")
                
            except Exception as e:
                self.logger.error(f"Failed to build family from cluster {i+1}: {e}")
                continue
        
        self.logger.info(f"Built {len(family_results)} families successfully")
        
        # Save summary report
        self._save_family_build_report(family_results, clustering_result)
        
        return family_results
    
    def _save_family_build_report(self, 
                                family_results: List[FamilyBuildResult],
                                clustering_result: ClusteringResult):
        """Save family building report."""
        report_dir = self.settings.reports_dir / "families"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Summary CSV
        summary_data = []
        for result in family_results:
            summary_data.append({
                'family_id': result.family_id,
                'title': result.title,
                'ea_count': len(result.ea_ids),
                'gold_ea_id': result.gold_ea_id,
                'num_clauses': result.num_clauses,
                'confidence_score': result.confidence_score,
                'jurisdiction': result.jurisdiction,
                'effective_from': result.effective_from,
                'effective_to': result.effective_to,
                'ea_ids': ','.join(result.ea_ids)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = report_dir / f"family_build_{clustering_result.run_id}.csv"
        summary_df.to_csv(summary_file, index=False)
        
        self.logger.info(f"Family build report saved to {summary_file}")


def create_family_builder() -> FamilyBuilder:
    """Factory function to create a family builder."""
    return FamilyBuilder()