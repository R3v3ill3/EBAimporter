"""
Instance Manager for EA Importer - Handle EA instances and overlays.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from ..core.logging import get_logger, log_function_call
from ..models import OverlayType

logger = get_logger(__name__)


class InstanceManager:
    """Manage EA instances with parameter packs and overlays."""
    
    def __init__(self):
        pass
    
    @log_function_call
    def create_instance(
        self,
        instance_id: str,
        family_id: str,
        employer_name: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a new EA instance."""
        
        instance_data = {
            'instance_id': instance_id,
            'family_id': family_id,
            'employer_name': employer_name,
            'parameters': parameters,
            'created_at': datetime.now().isoformat(),
            'overlays': []
        }
        
        logger.info(f"Created instance {instance_id} for family {family_id}")
        return instance_data
    
    def generate_overlay(
        self,
        instance_data: Dict[str, Any],
        family_clauses: List[Dict[str, Any]],
        instance_clauses: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate overlays for instance differences."""
        
        overlays = []
        
        # Create lookup for family clauses
        family_lookup = {clause['clause_id']: clause for clause in family_clauses}
        
        for instance_clause in instance_clauses:
            clause_id = instance_clause['clause_id']
            
            if clause_id in family_lookup:
                family_clause = family_lookup[clause_id]
                
                # Check for differences
                if instance_clause['text'] != family_clause['text']:
                    overlay = {
                        'clause_id': clause_id,
                        'overlay_type': OverlayType.REPLACE_TEXT.value,
                        'payload': {
                            'original_text': family_clause['text'],
                            'replacement_text': instance_clause['text']
                        },
                        'effective_from': datetime.now().isoformat(),
                        'created_at': datetime.now().isoformat()
                    }
                    overlays.append(overlay)
            else:
                # New clause not in family
                overlay = {
                    'clause_id': clause_id,
                    'overlay_type': OverlayType.ADD_CLAUSE.value,
                    'payload': {
                        'text': instance_clause['text'],
                        'heading': instance_clause.get('heading', '')
                    },
                    'effective_from': datetime.now().isoformat(),
                    'created_at': datetime.now().isoformat()
                }
                overlays.append(overlay)
        
        logger.info(f"Generated {len(overlays)} overlays for instance")
        return overlays
    
    def import_instances_from_csv(self, csv_path: Path) -> List[Dict[str, Any]]:
        """Import instances from CSV file."""
        
        import pandas as pd
        
        df = pd.read_csv(csv_path)
        instances = []
        
        required_columns = ['instance_id', 'family_id', 'employer_name']
        
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")
        
        for _, row in df.iterrows():
            parameters = {}
            
            # Extract standard parameters
            for col in df.columns:
                if col not in required_columns:
                    parameters[col] = row[col]
            
            instance = self.create_instance(
                instance_id=row['instance_id'],
                family_id=row['family_id'],
                employer_name=row['employer_name'],
                parameters=parameters
            )
            
            instances.append(instance)
        
        logger.info(f"Imported {len(instances)} instances from CSV")
        return instances
    
    def save_instance(self, instance_data: Dict[str, Any], output_dir: Path) -> Path:
        """Save instance data to file."""
        
        instance_id = instance_data['instance_id']
        instance_dir = output_dir / instance_id
        instance_dir.mkdir(parents=True, exist_ok=True)
        
        instance_file = instance_dir / "instance.json"
        with open(instance_file, 'w') as f:
            json.dump(instance_data, f, indent=2)
        
        # Save overlays separately
        if instance_data.get('overlays'):
            overlay_file = instance_dir / "overlay.json"
            with open(overlay_file, 'w') as f:
                json.dump(instance_data['overlays'], f, indent=2)
        
        logger.info(f"Saved instance {instance_id} to {instance_dir}")
        return instance_file


__all__ = ['InstanceManager']