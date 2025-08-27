"""
Version control utilities for EA Importer.

Creates simple, immutable manifests and exports of families/instances to a
versioned directory, with checksum support.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Tuple, List
from datetime import datetime

from ..core.logging import get_logger, log_function_call
from ..models import VersionManifest


logger = get_logger(__name__)


def _sha256_of_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


class VersionController:
    """Basic version control: manifest generation and export."""

    @log_function_call
    def create_version(
        self,
        version_name: str,
        families_data: Dict[str, Any],
        instances_data: Dict[str, Any],
        notes: str | None = None,
    ) -> VersionManifest:
        checksums: Dict[str, str] = {}
        # Hash canonical JSON of each family/instance
        for fid, fdata in families_data.items():
            checksums[f"family:{fid}"] = _sha256_of_bytes(json.dumps(fdata, sort_keys=True).encode("utf-8"))
        for iid, idata in instances_data.items():
            checksums[f"instance:{iid}"] = _sha256_of_bytes(json.dumps(idata, sort_keys=True).encode("utf-8"))

        manifest = VersionManifest(
            version=version_name,
            created_at=datetime.utcnow(),
            locked_at=None,
            commit_sha=None,
            families_count=len(families_data),
            instances_count=len(instances_data),
            checksums=checksums,
            notes=notes,
        )
        return manifest

    @log_function_call
    def save_version(
        self,
        manifest: VersionManifest,
        families_data: Dict[str, Any],
        instances_data: Dict[str, Any],
        versions_dir: Path,
    ) -> Path:
        target = versions_dir / manifest.version
        target.mkdir(parents=True, exist_ok=True)

        # Save manifest
        manifest_path = target / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(
                {
                    "version": manifest.version,
                    "created_at": manifest.created_at.isoformat(),
                    "locked_at": manifest.locked_at.isoformat() if manifest.locked_at else None,
                    "commit_sha": manifest.commit_sha,
                    "families_count": manifest.families_count,
                    "instances_count": manifest.instances_count,
                    "checksums": manifest.checksums,
                    "notes": manifest.notes,
                },
                f,
                indent=2,
            )

        # Save families
        fam_dir = target / "families"
        fam_dir.mkdir(exist_ok=True)
        for fid, fdata in families_data.items():
            with open(fam_dir / f"{fid}.json", "w") as f:
                json.dump(fdata, f, indent=2)

        # Save instances
        inst_dir = target / "instances"
        inst_dir.mkdir(exist_ok=True)
        for iid, idata in instances_data.items():
            with open(inst_dir / f"{iid}.json", "w") as f:
                json.dump(idata, f, indent=2)

        logger.info(f"Saved version {manifest.version} to {target}")
        return target

    @log_function_call
    def list_versions(self, versions_dir: Path) -> List[Dict[str, Any]]:
        if not versions_dir.exists():
            return []
        versions = []
        for child in sorted(versions_dir.iterdir()):
            if child.is_dir() and (child / "manifest.json").exists():
                try:
                    with open(child / "manifest.json", "r") as f:
                        versions.append(json.load(f))
                except Exception:
                    continue
        return versions

"""
Version Control for EA Importer - Corpus versioning and locking.
"""

import hashlib
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from ..core.logging import get_logger, log_function_call
from ..models import VersionManifest

logger = get_logger(__name__)


class VersionController:
    """Version control system for EA corpus management."""
    
    def __init__(self):
        pass
    
    @log_function_call
    def create_version(
        self,
        version_name: str,
        families_data: Dict[str, Any],
        instances_data: Dict[str, Any],
        notes: Optional[str] = None
    ) -> VersionManifest:
        """Create a new corpus version."""
        
        # Calculate checksums
        checksums = {}
        
        # Checksum for families
        families_json = json.dumps(families_data, sort_keys=True)
        checksums['families'] = hashlib.sha256(families_json.encode()).hexdigest()
        
        # Checksum for instances
        instances_json = json.dumps(instances_data, sort_keys=True)
        checksums['instances'] = hashlib.sha256(instances_json.encode()).hexdigest()
        
        # Overall corpus checksum
        corpus_data = {'families': families_data, 'instances': instances_data}
        corpus_json = json.dumps(corpus_data, sort_keys=True)
        checksums['corpus'] = hashlib.sha256(corpus_json.encode()).hexdigest()
        
        manifest = VersionManifest(
            version=version_name,
            created_at=datetime.now(),
            locked_at=None,
            commit_sha=None,
            families_count=len(families_data),
            instances_count=len(instances_data),
            checksums=checksums,
            notes=notes
        )
        
        logger.info(f"Created version {version_name} with {manifest.families_count} families and {manifest.instances_count} instances")
        return manifest
    
    def lock_version(self, manifest: VersionManifest, commit_sha: Optional[str] = None) -> VersionManifest:
        """Lock a version to prevent further changes."""
        
        manifest.locked_at = datetime.now()
        manifest.commit_sha = commit_sha
        
        logger.info(f"Locked version {manifest.version} at {manifest.locked_at}")
        return manifest
    
    def save_version(
        self,
        manifest: VersionManifest,
        families_data: Dict[str, Any],
        instances_data: Dict[str, Any],
        output_dir: Path
    ) -> Path:
        """Save version data to filesystem."""
        
        version_dir = output_dir / manifest.version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Save manifest
        manifest_file = version_dir / "manifest.json"
        manifest_data = {
            'version': manifest.version,
            'created_at': manifest.created_at.isoformat(),
            'locked_at': manifest.locked_at.isoformat() if manifest.locked_at else None,
            'commit_sha': manifest.commit_sha,
            'families_count': manifest.families_count,
            'instances_count': manifest.instances_count,
            'checksums': manifest.checksums,
            'notes': manifest.notes
        }
        
        with open(manifest_file, 'w') as f:
            json.dump(manifest_data, f, indent=2)
        
        # Save families data
        families_file = version_dir / "families.json"
        with open(families_file, 'w') as f:
            json.dump(families_data, f, indent=2)
        
        # Save instances data
        instances_file = version_dir / "instances.json"
        with open(instances_file, 'w') as f:
            json.dump(instances_data, f, indent=2)
        
        # Create index file for quick access
        index_file = version_dir / "index.json"
        index_data = {
            'version': manifest.version,
            'type': 'ea_corpus',
            'created_at': manifest.created_at.isoformat(),
            'locked': manifest.locked_at is not None,
            'families': list(families_data.keys()),
            'instances': list(instances_data.keys()),
            'checksums': manifest.checksums
        }
        
        with open(index_file, 'w') as f:
            json.dump(index_data, f, indent=2)
        
        logger.info(f"Saved version {manifest.version} to {version_dir}")
        return version_dir
    
    def load_version(self, version_path: Path) -> Tuple[VersionManifest, Dict[str, Any], Dict[str, Any]]:
        """Load version data from filesystem."""
        
        # Load manifest
        manifest_file = version_path / "manifest.json"
        with open(manifest_file, 'r') as f:
            manifest_data = json.load(f)
        
        manifest = VersionManifest(
            version=manifest_data['version'],
            created_at=datetime.fromisoformat(manifest_data['created_at']),
            locked_at=datetime.fromisoformat(manifest_data['locked_at']) if manifest_data['locked_at'] else None,
            commit_sha=manifest_data.get('commit_sha'),
            families_count=manifest_data['families_count'],
            instances_count=manifest_data['instances_count'],
            checksums=manifest_data['checksums'],
            notes=manifest_data.get('notes')
        )
        
        # Load families data
        families_file = version_path / "families.json"
        with open(families_file, 'r') as f:
            families_data = json.load(f)
        
        # Load instances data
        instances_file = version_path / "instances.json"
        with open(instances_file, 'r') as f:
            instances_data = json.load(f)
        
        logger.info(f"Loaded version {manifest.version}")
        return manifest, families_data, instances_data
    
    def list_versions(self, versions_dir: Path) -> List[Dict[str, Any]]:
        """List all available versions."""
        
        versions = []
        
        for version_dir in versions_dir.iterdir():
            if version_dir.is_dir() and (version_dir / "manifest.json").exists():
                try:
                    index_file = version_dir / "index.json"
                    if index_file.exists():
                        with open(index_file, 'r') as f:
                            index_data = json.load(f)
                            versions.append(index_data)
                    else:
                        # Fallback to manifest
                        manifest_file = version_dir / "manifest.json"
                        with open(manifest_file, 'r') as f:
                            manifest_data = json.load(f)
                            versions.append({
                                'version': manifest_data['version'],
                                'created_at': manifest_data['created_at'],
                                'locked': manifest_data['locked_at'] is not None,
                                'families_count': manifest_data['families_count'],
                                'instances_count': manifest_data['instances_count']
                            })
                except Exception as e:
                    logger.warning(f"Failed to read version {version_dir.name}: {e}")
        
        # Sort by creation date
        versions.sort(key=lambda x: x['created_at'], reverse=True)
        
        return versions
    
    def verify_version_integrity(self, version_path: Path) -> Dict[str, bool]:
        """Verify the integrity of a version."""
        
        results = {
            'manifest_exists': False,
            'families_exists': False,
            'instances_exists': False,
            'checksums_valid': False
        }
        
        try:
            # Check file existence
            manifest_file = version_path / "manifest.json"
            families_file = version_path / "families.json"
            instances_file = version_path / "instances.json"
            
            results['manifest_exists'] = manifest_file.exists()
            results['families_exists'] = families_file.exists()
            results['instances_exists'] = instances_file.exists()
            
            if all([results['manifest_exists'], results['families_exists'], results['instances_exists']]):
                # Verify checksums
                manifest, families_data, instances_data = self.load_version(version_path)
                
                # Recalculate checksums
                families_json = json.dumps(families_data, sort_keys=True)
                families_checksum = hashlib.sha256(families_json.encode()).hexdigest()
                
                instances_json = json.dumps(instances_data, sort_keys=True)
                instances_checksum = hashlib.sha256(instances_json.encode()).hexdigest()
                
                # Verify against stored checksums
                results['checksums_valid'] = (
                    manifest.checksums.get('families') == families_checksum and
                    manifest.checksums.get('instances') == instances_checksum
                )
        
        except Exception as e:
            logger.error(f"Error verifying version integrity: {e}")
        
        return results
    
    def generate_changelog(self, old_version_path: Path, new_version_path: Path) -> Dict[str, Any]:
        """Generate changelog between two versions."""
        
        try:
            # Load both versions
            old_manifest, old_families, old_instances = self.load_version(old_version_path)
            new_manifest, new_families, new_instances = self.load_version(new_version_path)
            
            changelog = {
                'from_version': old_manifest.version,
                'to_version': new_manifest.version,
                'changes': {
                    'families': {
                        'added': [],
                        'removed': [],
                        'modified': []
                    },
                    'instances': {
                        'added': [],
                        'removed': [],
                        'modified': []
                    }
                }
            }
            
            # Compare families
            old_family_keys = set(old_families.keys())
            new_family_keys = set(new_families.keys())
            
            changelog['changes']['families']['added'] = list(new_family_keys - old_family_keys)
            changelog['changes']['families']['removed'] = list(old_family_keys - new_family_keys)
            
            # Check for modifications
            for family_id in old_family_keys & new_family_keys:
                old_family_json = json.dumps(old_families[family_id], sort_keys=True)
                new_family_json = json.dumps(new_families[family_id], sort_keys=True)
                
                if old_family_json != new_family_json:
                    changelog['changes']['families']['modified'].append(family_id)
            
            # Compare instances
            old_instance_keys = set(old_instances.keys())
            new_instance_keys = set(new_instances.keys())
            
            changelog['changes']['instances']['added'] = list(new_instance_keys - old_instance_keys)
            changelog['changes']['instances']['removed'] = list(old_instance_keys - new_instance_keys)
            
            # Check for modifications
            for instance_id in old_instance_keys & new_instance_keys:
                old_instance_json = json.dumps(old_instances[instance_id], sort_keys=True)
                new_instance_json = json.dumps(new_instances[instance_id], sort_keys=True)
                
                if old_instance_json != new_instance_json:
                    changelog['changes']['instances']['modified'].append(instance_id)
            
            return changelog
            
        except Exception as e:
            logger.error(f"Error generating changelog: {e}")
            return {}


__all__ = ['VersionController']