"""
Instance Manager - Handles EA instances, parameters, and overlay generation.
"""

import json
import uuid
from datetime import datetime, date
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict

import pandas as pd
from sqlalchemy.orm import Session

from ..core.config import get_settings
from ..core.logging import LoggerMixin
from ..database import get_database_manager
from ..models import (
    AgreementInstance, InstanceParam, InstanceOverlay, 
    AgreementFamily, FamilyClause, OverlayType, AgreementStatus
)
from ..utils.text_cleaner import create_text_cleaner
from ..utils.fingerprinter import create_clause_fingerprinter


@dataclass
class InstanceData:
    """Data for creating an EA instance."""
    instance_id: str
    family_key: str
    employer_name: str
    employer_abn: Optional[str] = None
    fwc_id: Optional[str] = None
    commencement: Optional[date] = None
    nominal_expiry: Optional[date] = None
    pay_steps_json: Optional[str] = None
    title: Optional[str] = None
    coverage_description: Optional[str] = None
    status: str = AgreementStatus.ACTIVE.value
    additional_params: Optional[Dict[str, Any]] = None


@dataclass
class OverlayData:
    """Data for creating an overlay."""
    clause_id: str
    overlay_type: str
    payload: Dict[str, Any]
    effective_from: Optional[date] = None
    effective_to: Optional[date] = None
    description: Optional[str] = None


@dataclass
class DiffResult:
    """Result of comparing instance against family gold text."""
    instance_id: str
    family_id: str
    differences: List[Dict[str, Any]]
    similarity_score: float
    overlays_needed: List[OverlayData]


class InstanceManager(LoggerMixin):
    """Manages EA instances, parameters, and overlays."""
    
    def __init__(self):
        """Initialize the instance manager."""
        self.settings = get_settings()
        self.db_manager = get_database_manager()
        self.text_cleaner = create_text_cleaner()
        self.clause_fingerprinter = create_clause_fingerprinter()
    
    def load_instances_from_csv(self, csv_file: Path) -> List[InstanceData]:
        """
        Load instance data from CSV file.
        
        Args:
            csv_file: Path to CSV file with instance data
            
        Returns:
            List of instance data objects
        """
        self.logger.info(f"Loading instances from {csv_file}")
        
        try:
            df = pd.read_csv(csv_file)
            
            # Validate required columns
            required_columns = ['instance_id', 'family_key', 'employer_name']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            instances = []
            
            for _, row in df.iterrows():
                # Parse dates
                commencement = None
                if pd.notna(row.get('commencement')):
                    try:
                        commencement = pd.to_datetime(row['commencement']).date()
                    except Exception as e:
                        self.logger.warning(f"Invalid commencement date for {row['instance_id']}: {e}")
                
                nominal_expiry = None
                if pd.notna(row.get('nominal_expiry')):
                    try:
                        nominal_expiry = pd.to_datetime(row['nominal_expiry']).date()
                    except Exception as e:
                        self.logger.warning(f"Invalid expiry date for {row['instance_id']}: {e}")
                
                # Parse additional parameters
                additional_params = {}
                for col in df.columns:
                    if col not in required_columns + ['commencement', 'nominal_expiry', 
                                                    'pay_steps_json', 'title', 'coverage_description',
                                                    'employer_abn', 'fwc_id', 'status']:
                        if pd.notna(row[col]):
                            additional_params[col] = row[col]
                
                instance = InstanceData(
                    instance_id=str(row['instance_id']),
                    family_key=str(row['family_key']),
                    employer_name=str(row['employer_name']),
                    employer_abn=str(row['employer_abn']) if pd.notna(row.get('employer_abn')) else None,
                    fwc_id=str(row['fwc_id']) if pd.notna(row.get('fwc_id')) else None,
                    commencement=commencement,
                    nominal_expiry=nominal_expiry,
                    pay_steps_json=str(row['pay_steps_json']) if pd.notna(row.get('pay_steps_json')) else None,
                    title=str(row['title']) if pd.notna(row.get('title')) else None,
                    coverage_description=str(row['coverage_description']) if pd.notna(row.get('coverage_description')) else None,
                    status=str(row.get('status', AgreementStatus.ACTIVE.value)),
                    additional_params=additional_params if additional_params else None
                )
                
                instances.append(instance)
            
            self.logger.info(f"Loaded {len(instances)} instances from CSV")
            return instances
            
        except Exception as e:
            self.logger.error(f"Failed to load instances from CSV: {e}")
            raise
    
    def resolve_family_key(self, family_key: str) -> Optional[uuid.UUID]:
        """
        Resolve family key to family ID.
        
        Args:
            family_key: Family key (could be title pattern, ID, etc.)
            
        Returns:
            Family UUID or None if not found
        """
        try:
            with self.db_manager.session_scope() as session:
                # Try exact match on title
                family = session.query(AgreementFamily).filter(
                    AgreementFamily.title.ilike(f'%{family_key}%')
                ).first()
                
                if family:
                    return family.id
                
                # Try matching on source documents
                family = session.query(AgreementFamily).filter(
                    AgreementFamily.source_documents.contains([family_key])
                ).first()
                
                if family:
                    return family.id
                
                # Try UUID parsing
                try:
                    family_uuid = uuid.UUID(family_key)
                    family = session.query(AgreementFamily).filter(
                        AgreementFamily.id == family_uuid
                    ).first()
                    
                    if family:
                        return family.id
                        
                except ValueError:
                    pass
                
                self.logger.warning(f"Could not resolve family key: {family_key}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error resolving family key {family_key}: {e}")
            return None
    
    def create_instance(self, instance_data: InstanceData) -> bool:
        """
        Create an EA instance in the database.
        
        Args:
            instance_data: Instance data to create
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Resolve family key
            family_id = self.resolve_family_key(instance_data.family_key)
            if not family_id:
                self.logger.error(f"Could not resolve family for instance {instance_data.instance_id}")
                return False
            
            with self.db_manager.session_scope() as session:
                # Check if instance already exists
                existing = session.query(AgreementInstance).filter(
                    AgreementInstance.id == uuid.UUID(instance_data.instance_id)
                ).first()
                
                if existing:
                    self.logger.warning(f"Instance {instance_data.instance_id} already exists")
                    return True
                
                # Create instance
                instance = AgreementInstance(
                    id=uuid.UUID(instance_data.instance_id),
                    family_id=family_id,
                    employer_id=instance_data.employer_abn or instance_data.employer_name,
                    fwc_id=instance_data.fwc_id,
                    commencement=instance_data.commencement,
                    nominal_expiry=instance_data.nominal_expiry,
                    status=instance_data.status,
                    title=instance_data.title,
                    employer_name=instance_data.employer_name,
                    coverage_description=instance_data.coverage_description
                )
                
                session.add(instance)
                session.flush()  # Get the ID
                
                # Create parameters
                if instance_data.pay_steps_json:
                    param = InstanceParam(
                        instance_id=instance.id,
                        key='pay_steps',
                        value=json.loads(instance_data.pay_steps_json)
                    )
                    session.add(param)
                
                if instance_data.employer_abn:
                    param = InstanceParam(
                        instance_id=instance.id,
                        key='employer_abn',
                        value=instance_data.employer_abn
                    )
                    session.add(param)
                
                # Add additional parameters
                if instance_data.additional_params:
                    for key, value in instance_data.additional_params.items():
                        param = InstanceParam(
                            instance_id=instance.id,
                            key=key,
                            value=value
                        )
                        session.add(param)
                
                session.commit()
                
                self.logger.info(f"Created instance {instance_data.instance_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to create instance {instance_data.instance_id}: {e}")
            return False
    
    def batch_create_instances(self, instances: List[InstanceData]) -> Tuple[int, int]:
        """
        Create multiple instances in batch.
        
        Args:
            instances: List of instance data
            
        Returns:
            Tuple of (successful_count, failed_count)
        """
        successful = 0
        failed = 0
        
        for instance_data in instances:
            if self.create_instance(instance_data):
                successful += 1
            else:
                failed += 1
        
        self.logger.info(f"Batch create completed: {successful} successful, {failed} failed")
        return successful, failed
    
    def load_family_clauses(self, family_id: uuid.UUID) -> List[Dict[str, Any]]:
        """
        Load family clauses from database.
        
        Args:
            family_id: Family UUID
            
        Returns:
            List of clause dictionaries
        """
        try:
            with self.db_manager.session_scope() as session:
                clauses = session.query(FamilyClause).filter(
                    FamilyClause.family_id == family_id
                ).all()
                
                clause_list = []
                for clause in clauses:
                    clause_dict = {
                        'id': str(clause.id),
                        'clause_id': clause.clause_id,
                        'heading': clause.heading,
                        'text': clause.text,
                        'path': clause.path,
                        'hash_sha256': clause.hash_sha256,
                        'tokens': clause.tokens,
                        'page_spans': clause.page_spans,
                        'clause_type': clause.clause_type
                    }
                    clause_list.append(clause_dict)
                
                return clause_list
                
        except Exception as e:
            self.logger.error(f"Failed to load family clauses: {e}")
            return []
    
    def load_instance_clauses(self, instance_ea_id: str) -> List[Dict[str, Any]]:
        """
        Load clauses for a specific EA instance.
        
        Args:
            instance_ea_id: EA identifier for the instance
            
        Returns:
            List of clause dictionaries
        """
        clauses_file = self.settings.clauses_dir / f"{instance_ea_id}.jsonl"
        
        if not clauses_file.exists():
            self.logger.warning(f"Clauses file not found: {clauses_file}")
            return []
        
        clauses = []
        try:
            with open(clauses_file, 'r', encoding='utf-8') as f:
                for line in f:
                    clause = json.loads(line.strip())
                    clauses.append(clause)
            
            return clauses
            
        except Exception as e:
            self.logger.error(f"Failed to load instance clauses: {e}")
            return []
    
    def compare_instance_to_family(self, 
                                 instance_id: str,
                                 instance_ea_id: str) -> Optional[DiffResult]:
        """
        Compare an instance against its family gold text.
        
        Args:
            instance_id: Instance identifier
            instance_ea_id: EA identifier for the instance
            
        Returns:
            Diff result or None if comparison failed
        """
        try:
            # Get instance from database
            with self.db_manager.session_scope() as session:
                instance = session.query(AgreementInstance).filter(
                    AgreementInstance.id == uuid.UUID(instance_id)
                ).first()
                
                if not instance:
                    self.logger.error(f"Instance not found: {instance_id}")
                    return None
                
                family_id = instance.family_id
            
            # Load family and instance clauses
            family_clauses = self.load_family_clauses(family_id)
            instance_clauses = self.load_instance_clauses(instance_ea_id)
            
            if not family_clauses or not instance_clauses:
                self.logger.error(f"Could not load clauses for comparison")
                return None
            
            # Create clause lookup by clause_id
            family_lookup = {clause['clause_id']: clause for clause in family_clauses}
            instance_lookup = {clause['clause_id']: clause for clause in instance_clauses}
            
            differences = []
            overlays_needed = []
            similarity_scores = []
            
            # Compare clauses
            all_clause_ids = set(family_lookup.keys()) | set(instance_lookup.keys())
            
            for clause_id in all_clause_ids:
                family_clause = family_lookup.get(clause_id)
                instance_clause = instance_lookup.get(clause_id)
                
                if not family_clause and instance_clause:
                    # Instance has extra clause
                    differences.append({
                        'type': 'added',
                        'clause_id': clause_id,
                        'instance_text': instance_clause['text'],
                        'family_text': None,
                        'description': f"Clause {clause_id} exists in instance but not in family"
                    })
                    
                    overlays_needed.append(OverlayData(
                        clause_id=clause_id,
                        overlay_type=OverlayType.ADD_CLAUSE.value,
                        payload={'text': instance_clause['text']},
                        description=f"Add clause {clause_id}"
                    ))
                
                elif family_clause and not instance_clause:
                    # Instance missing clause
                    differences.append({
                        'type': 'removed',
                        'clause_id': clause_id,
                        'instance_text': None,
                        'family_text': family_clause['text'],
                        'description': f"Clause {clause_id} exists in family but not in instance"
                    })
                    
                    overlays_needed.append(OverlayData(
                        clause_id=clause_id,
                        overlay_type=OverlayType.REMOVE_CLAUSE.value,
                        payload={},
                        description=f"Remove clause {clause_id}"
                    ))
                
                elif family_clause and instance_clause:
                    # Compare text similarity
                    similarity = self.clause_fingerprinter.compare_clauses(
                        family_clause['text'], instance_clause['text']
                    )
                    
                    similarity_scores.append(similarity)
                    
                    if similarity < 0.9:  # Significant difference
                        differences.append({
                            'type': 'modified',
                            'clause_id': clause_id,
                            'instance_text': instance_clause['text'],
                            'family_text': family_clause['text'],
                            'similarity': similarity,
                            'description': f"Clause {clause_id} differs significantly (similarity: {similarity:.2f})"
                        })
                        
                        overlays_needed.append(OverlayData(
                            clause_id=clause_id,
                            overlay_type=OverlayType.REPLACE_TEXT.value,
                            payload={'text': instance_clause['text']},
                            description=f"Replace text for clause {clause_id}"
                        ))
            
            # Calculate overall similarity
            overall_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
            
            result = DiffResult(
                instance_id=instance_id,
                family_id=str(family_id),
                differences=differences,
                similarity_score=overall_similarity,
                overlays_needed=overlays_needed
            )
            
            self.logger.info(f"Comparison complete: {len(differences)} differences, "
                           f"similarity: {overall_similarity:.2f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to compare instance to family: {e}")
            return None
    
    def create_overlays_from_diff(self, 
                                diff_result: DiffResult,
                                auto_apply: bool = False) -> List[str]:
        """
        Create overlay records from diff result.
        
        Args:
            diff_result: Result from instance comparison
            auto_apply: Whether to automatically apply overlays
            
        Returns:
            List of created overlay IDs
        """
        overlay_ids = []
        
        try:
            with self.db_manager.session_scope() as session:
                for overlay_data in diff_result.overlays_needed:
                    overlay = InstanceOverlay(
                        instance_id=uuid.UUID(diff_result.instance_id),
                        clause_id=overlay_data.clause_id,
                        overlay_type=overlay_data.overlay_type,
                        payload_jsonb=overlay_data.payload,
                        effective_from=overlay_data.effective_from,
                        effective_to=overlay_data.effective_to,
                        description=overlay_data.description
                    )
                    
                    session.add(overlay)
                    session.flush()
                    
                    overlay_ids.append(str(overlay.id))
                
                session.commit()
                
                self.logger.info(f"Created {len(overlay_ids)} overlays for instance {diff_result.instance_id}")
        
        except Exception as e:
            self.logger.error(f"Failed to create overlays: {e}")
        
        return overlay_ids
    
    def save_diff_report(self, 
                        diff_result: DiffResult,
                        output_file: Path):
        """
        Save diff report to file.
        
        Args:
            diff_result: Diff result to save
            output_file: Output file path
        """
        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            report_data = {
                'instance_id': diff_result.instance_id,
                'family_id': diff_result.family_id,
                'comparison_date': datetime.now().isoformat(),
                'similarity_score': diff_result.similarity_score,
                'differences_count': len(diff_result.differences),
                'overlays_needed_count': len(diff_result.overlays_needed),
                'differences': diff_result.differences,
                'overlays_needed': [asdict(overlay) for overlay in diff_result.overlays_needed]
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"Diff report saved to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save diff report: {e}")
    
    def generate_html_diff_report(self, 
                                diff_result: DiffResult,
                                output_file: Path):
        """
        Generate HTML diff report for human review.
        
        Args:
            diff_result: Diff result
            output_file: Output HTML file path
        """
        try:
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>EA Instance Diff Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; }}
                    .difference {{ margin: 10px 0; padding: 10px; border-radius: 5px; }}
                    .added {{ background-color: #d4edda; border-left: 4px solid #28a745; }}
                    .removed {{ background-color: #f8d7da; border-left: 4px solid #dc3545; }}
                    .modified {{ background-color: #fff3cd; border-left: 4px solid #ffc107; }}
                    .clause-id {{ font-weight: bold; color: #495057; }}
                    .text {{ margin: 5px 0; padding: 5px; background-color: #f8f9fa; }}
                    .similarity {{ font-size: 0.9em; color: #6c757d; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>EA Instance Diff Report</h1>
                    <p><strong>Instance ID:</strong> {diff_result.instance_id}</p>
                    <p><strong>Family ID:</strong> {diff_result.family_id}</p>
                    <p><strong>Overall Similarity:</strong> {diff_result.similarity_score:.2%}</p>
                    <p><strong>Differences:</strong> {len(diff_result.differences)}</p>
                    <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
            """
            
            for diff in diff_result.differences:
                diff_type = diff['type']
                clause_id = diff['clause_id']
                
                html_content += f"""
                <div class="difference {diff_type}">
                    <div class="clause-id">Clause {clause_id} ({diff_type})</div>
                    <p>{diff['description']}</p>
                """
                
                if diff.get('family_text'):
                    html_content += f"""
                    <div><strong>Family Text:</strong></div>
                    <div class="text">{diff['family_text'][:500]}{'...' if len(diff['family_text']) > 500 else ''}</div>
                    """
                
                if diff.get('instance_text'):
                    html_content += f"""
                    <div><strong>Instance Text:</strong></div>
                    <div class="text">{diff['instance_text'][:500]}{'...' if len(diff['instance_text']) > 500 else ''}</div>
                    """
                
                if diff.get('similarity'):
                    html_content += f"""
                    <div class="similarity">Similarity: {diff['similarity']:.2%}</div>
                    """
                
                html_content += "</div>"
            
            html_content += """
            </body>
            </html>
            """
            
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"HTML diff report saved to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate HTML diff report: {e}")
    
    def process_instance_batch(self, 
                             csv_file: Path,
                             generate_overlays: bool = True) -> Dict[str, Any]:
        """
        Process a batch of instances from CSV.
        
        Args:
            csv_file: CSV file with instance data
            generate_overlays: Whether to generate overlays automatically
            
        Returns:
            Processing summary
        """
        self.logger.info(f"Processing instance batch from {csv_file}")
        
        # Load instances
        instances = self.load_instances_from_csv(csv_file)
        
        # Create instances
        successful, failed = self.batch_create_instances(instances)
        
        # Generate diff reports and overlays
        diff_results = []
        overlays_created = 0
        
        if generate_overlays:
            for instance_data in instances:
                # Try to find corresponding EA file
                instance_ea_id = f"EA_{instance_data.instance_id}"
                clauses_file = self.settings.clauses_dir / f"{instance_ea_id}.jsonl"
                
                if clauses_file.exists():
                    diff_result = self.compare_instance_to_family(
                        instance_data.instance_id, 
                        instance_ea_id
                    )
                    
                    if diff_result:
                        diff_results.append(diff_result)
                        
                        # Save reports
                        report_dir = self.settings.reports_dir / "instances" / instance_data.instance_id
                        
                        self.save_diff_report(
                            diff_result, 
                            report_dir / "diff_report.json"
                        )
                        
                        self.generate_html_diff_report(
                            diff_result,
                            report_dir / "diff_report.html"
                        )
                        
                        # Create overlays
                        overlay_ids = self.create_overlays_from_diff(diff_result)
                        overlays_created += len(overlay_ids)
        
        summary = {
            'instances_loaded': len(instances),
            'instances_created': successful,
            'instances_failed': failed,
            'diff_reports_generated': len(diff_results),
            'overlays_created': overlays_created,
            'processing_date': datetime.now().isoformat()
        }
        
        self.logger.info(f"Batch processing complete: {summary}")
        
        return summary


def create_instance_manager() -> InstanceManager:
    """Factory function to create an instance manager."""
    return InstanceManager()