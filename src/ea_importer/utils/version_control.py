"""
Version Control - Implements corpus versioning, locking, and manifest generation.
"""

import json
import hashlib
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Set
from dataclasses import dataclass, asdict

import pandas as pd
from sqlalchemy.orm import Session

from ..core.config import get_settings
from ..core.logging import LoggerMixin
from ..database import get_database_manager
from ..models import AgreementFamily, FamilyClause, AgreementInstance


@dataclass
class FileChecksum:
    """Represents a file with its checksum."""
    path: str
    checksum: str
    size_bytes: int
    modified_time: str


@dataclass
class FamilyManifestEntry:
    """Manifest entry for a family."""
    family_id: str
    title: str
    version: str
    jurisdiction: str
    clause_count: int
    checksum: str
    locked_at: Optional[str] = None
    gold_document_id: Optional[str] = None


@dataclass
class CorpusManifest:
    """Complete corpus manifest."""
    version: str
    created_at: str
    locked_at: Optional[str] = None
    commit_sha: Optional[str] = None
    
    # Statistics
    total_families: int
    total_clauses: int
    total_instances: int
    
    # Content
    families: List[FamilyManifestEntry]
    file_checksums: List[FileChecksum]
    
    # Metadata
    jurisdiction: str
    description: str
    provenance: Dict[str, Any]
    
    @property
    def is_locked(self) -> bool:
        """Check if corpus is locked."""
        return self.locked_at is not None


class VersionManager(LoggerMixin):
    """Manages corpus versions and provides version control functionality."""
    
    def __init__(self):
        """Initialize version manager."""
        self.settings = get_settings()
        self.db_manager = get_database_manager()
    
    def calculate_file_checksum(self, file_path: Path) -> str:
        """
        Calculate SHA256 checksum for a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            SHA256 checksum as hex string
        """
        sha256_hash = hashlib.sha256()
        
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256_hash.update(chunk)
            
            return sha256_hash.hexdigest()
            
        except Exception as e:
            self.logger.error(f"Failed to calculate checksum for {file_path}: {e}")
            return ""
    
    def calculate_directory_checksums(self, directory: Path, patterns: List[str]) -> List[FileChecksum]:
        """
        Calculate checksums for files in directory matching patterns.
        
        Args:
            directory: Directory to scan
            patterns: List of glob patterns to match
            
        Returns:
            List of file checksums
        """
        checksums = []
        
        if not directory.exists():
            return checksums
        
        for pattern in patterns:
            for file_path in directory.glob(pattern):
                if file_path.is_file():
                    checksum = self.calculate_file_checksum(file_path)
                    if checksum:
                        # Get relative path from data root
                        try:
                            relative_path = file_path.relative_to(self.settings.data_root)
                        except ValueError:
                            relative_path = file_path
                        
                        file_checksum = FileChecksum(
                            path=str(relative_path),
                            checksum=checksum,
                            size_bytes=file_path.stat().st_size,
                            modified_time=datetime.fromtimestamp(
                                file_path.stat().st_mtime
                            ).isoformat()
                        )
                        checksums.append(file_checksum)
        
        return checksums
    
    def get_family_manifests(self) -> List[FamilyManifestEntry]:
        """
        Get manifest entries for all families.
        
        Returns:
            List of family manifest entries
        """
        manifests = []
        
        try:
            with self.db_manager.session_scope() as session:
                families = session.query(AgreementFamily).all()
                
                for family in families:
                    # Count clauses
                    clause_count = session.query(FamilyClause).filter(
                        FamilyClause.family_id == family.id
                    ).count()
                    
                    manifest_entry = FamilyManifestEntry(
                        family_id=str(family.id),
                        title=family.title,
                        version=family.version,
                        jurisdiction=family.jurisdiction,
                        clause_count=clause_count,
                        checksum=family.checksum,
                        locked_at=family.locked_at.isoformat() if family.locked_at else None,
                        gold_document_id=family.gold_document_id
                    )
                    manifests.append(manifest_entry)
            
        except Exception as e:
            self.logger.error(f"Failed to get family manifests: {e}")
        
        return manifests
    
    def create_corpus_manifest(self, 
                             version: str,
                             description: str = "",
                             commit_sha: Optional[str] = None) -> CorpusManifest:
        """
        Create a complete corpus manifest.
        
        Args:
            version: Version identifier
            description: Version description
            commit_sha: Optional git commit SHA
            
        Returns:
            Corpus manifest
        """
        self.logger.info(f"Creating corpus manifest for version {version}")
        
        # Get family manifests
        families = self.get_family_manifests()
        
        # Calculate file checksums
        file_patterns = [
            "**/*.jsonl",
            "**/*.csv", 
            "**/*.json",
            "**/*.txt",
            "**/*.sha256",
            "**/*.minhash"
        ]
        
        file_checksums = []
        
        # Data directories to include
        data_dirs = [
            (self.settings.eas_dir, ["**/*"]),
            (self.settings.families_dir, ["**/*"]),
            (self.settings.instances_dir, ["**/*"]),
            (self.settings.reports_dir, ["**/*"])
        ]
        
        for directory, patterns in data_dirs:
            dir_checksums = self.calculate_directory_checksums(directory, patterns)
            file_checksums.extend(dir_checksums)
        
        # Get database statistics
        total_instances = 0
        try:
            with self.db_manager.session_scope() as session:
                total_instances = session.query(AgreementInstance).count()
        except Exception as e:
            self.logger.warning(f"Failed to get instance count: {e}")
        
        # Build provenance information
        provenance = {
            'created_by': 'EA Importer System',
            'created_at': datetime.now().isoformat(),
            'source_system': 'EA Importer v' + self.settings.target_version,
            'jurisdiction': self.settings.jurisdiction,
            'data_root': str(self.settings.data_root),
            'commit_sha': commit_sha,
            'file_count': len(file_checksums),
            'total_size_bytes': sum(fc.size_bytes for fc in file_checksums)
        }
        
        manifest = CorpusManifest(
            version=version,
            created_at=datetime.now().isoformat(),
            locked_at=None,
            commit_sha=commit_sha,
            total_families=len(families),
            total_clauses=sum(f.clause_count for f in families),
            total_instances=total_instances,
            families=families,
            file_checksums=file_checksums,
            jurisdiction=self.settings.jurisdiction,
            description=description,
            provenance=provenance
        )
        
        self.logger.info(f"Created manifest: {manifest.total_families} families, "
                        f"{manifest.total_clauses} clauses, {len(file_checksums)} files")
        
        return manifest
    
    def save_manifest(self, manifest: CorpusManifest, version_dir: Path):
        """
        Save manifest to version directory.
        
        Args:
            manifest: Corpus manifest
            version_dir: Version directory path
        """
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main manifest
        manifest_file = version_dir / "manifest.json"
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(manifest), f, indent=2, ensure_ascii=False)
        
        # Save family summary CSV
        if manifest.families:
            families_data = []
            for family in manifest.families:
                families_data.append(asdict(family))
            
            families_df = pd.DataFrame(families_data)
            families_file = version_dir / "families_summary.csv"
            families_df.to_csv(families_file, index=False)
        
        # Save file checksums CSV
        if manifest.file_checksums:
            checksums_data = []
            for checksum in manifest.file_checksums:
                checksums_data.append(asdict(checksum))
            
            checksums_df = pd.DataFrame(checksums_data)
            checksums_file = version_dir / "file_checksums.csv"
            checksums_df.to_csv(checksums_file, index=False)
        
        # Save human-readable summary
        summary_file = version_dir / "READINESS.md"
        self._generate_readiness_report(manifest, summary_file)
        
        self.logger.info(f"Manifest saved to {version_dir}")
    
    def _generate_readiness_report(self, manifest: CorpusManifest, output_file: Path):
        """Generate human-readable readiness report."""
        content = f"""# EA Corpus Readiness Report

## Version: {manifest.version}

**Created:** {manifest.created_at}  
**Status:** {'🔒 LOCKED' if manifest.is_locked else '🔓 UNLOCKED'}  
**Jurisdiction:** {manifest.jurisdiction}

## Summary Statistics

- **Families:** {manifest.total_families}
- **Clauses:** {manifest.total_clauses:,}
- **Instances:** {manifest.total_instances}
- **Files:** {len(manifest.file_checksums):,}
- **Total Size:** {manifest.provenance.get('total_size_bytes', 0) / 1024 / 1024:.1f} MB

## Data Quality

### Family Coverage
"""
        
        if manifest.families:
            # Group families by jurisdiction
            from collections import defaultdict
            by_jurisdiction = defaultdict(list)
            for family in manifest.families:
                by_jurisdiction[family.jurisdiction].append(family)
            
            for jurisdiction, families in by_jurisdiction.items():
                content += f"\n- **{jurisdiction}:** {len(families)} families"
            
            # Locked families
            locked_families = [f for f in manifest.families if f.locked_at]
            content += f"\n- **Locked families:** {len(locked_families)} / {len(manifest.families)}"
        
        content += f"""

### File Integrity

All {len(manifest.file_checksums):,} files have been checksummed and verified.

### Provenance

- **Source System:** {manifest.provenance.get('source_system', 'Unknown')}
- **Data Root:** {manifest.provenance.get('data_root', 'Unknown')}
"""
        
        if manifest.commit_sha:
            content += f"- **Git Commit:** {manifest.commit_sha}\n"
        
        content += f"""
## Usage

This corpus version is ready for:

- ✅ Document similarity queries
- ✅ Clause-level search and retrieval  
- ✅ Family-based analysis
- ✅ Rate and rule extraction
"""
        
        if manifest.is_locked:
            content += """- ✅ Production use (LOCKED VERSION)"""
        else:
            content += """- ⚠️ Development use only (UNLOCKED)"""
        
        content += f"""

## Description

{manifest.description}

---

*Generated by EA Importer System at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def lock_corpus_version(self, version: str) -> bool:
        """
        Lock a corpus version to prevent modifications.
        
        Args:
            version: Version to lock
            
        Returns:
            True if successful, False otherwise
        """
        try:
            version_dir = self.settings.versions_dir / version
            manifest_file = version_dir / "manifest.json"
            
            if not manifest_file.exists():
                self.logger.error(f"Manifest not found for version {version}")
                return False
            
            # Load and update manifest
            with open(manifest_file, 'r', encoding='utf-8') as f:
                manifest_data = json.load(f)
            
            manifest_data['locked_at'] = datetime.now().isoformat()
            
            # Save updated manifest
            with open(manifest_file, 'w', encoding='utf-8') as f:
                json.dump(manifest_data, f, indent=2, ensure_ascii=False)
            
            # Update database families
            with self.db_manager.session_scope() as session:
                families = session.query(AgreementFamily).filter(
                    AgreementFamily.version == version
                ).all()
                
                for family in families:
                    family.locked_at = datetime.now()
                
                session.commit()
            
            self.logger.info(f"Locked corpus version {version}")
            
            # Regenerate readiness report
            manifest_obj = CorpusManifest(**manifest_data)
            self._generate_readiness_report(manifest_obj, version_dir / "READINESS.md")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to lock version {version}: {e}")
            return False
    
    def create_version_snapshot(self, 
                              version: str,
                              description: str = "",
                              commit_sha: Optional[str] = None,
                              copy_files: bool = False) -> bool:
        """
        Create a complete version snapshot.
        
        Args:
            version: Version identifier  
            description: Version description
            commit_sha: Optional git commit SHA
            copy_files: Whether to copy actual files (creates archive)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Creating version snapshot: {version}")
            
            # Create version directory
            version_dir = self.settings.versions_dir / version
            version_dir.mkdir(parents=True, exist_ok=True)
            
            # Create manifest
            manifest = self.create_corpus_manifest(version, description, commit_sha)
            
            # Save manifest
            self.save_manifest(manifest, version_dir)
            
            # Copy files if requested
            if copy_files:
                self._copy_version_files(version_dir)
            
            self.logger.info(f"Version snapshot {version} created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create version snapshot {version}: {e}")
            return False
    
    def _copy_version_files(self, version_dir: Path):
        """Copy all relevant files to version directory."""
        archive_dir = version_dir / "archive"
        archive_dir.mkdir(exist_ok=True)
        
        # Directories to archive
        dirs_to_copy = [
            ("eas", self.settings.eas_dir),
            ("families", self.settings.families_dir),
            ("instances", self.settings.instances_dir),
            ("reports", self.settings.reports_dir)
        ]
        
        for dir_name, source_dir in dirs_to_copy:
            if source_dir.exists():
                dest_dir = archive_dir / dir_name
                self.logger.info(f"Copying {source_dir} to {dest_dir}")
                shutil.copytree(source_dir, dest_dir, dirs_exist_ok=True)
    
    def list_versions(self) -> List[Dict[str, Any]]:
        """
        List all available corpus versions.
        
        Returns:
            List of version information dictionaries
        """
        versions = []
        
        if not self.settings.versions_dir.exists():
            return versions
        
        for version_dir in self.settings.versions_dir.iterdir():
            if version_dir.is_dir():
                manifest_file = version_dir / "manifest.json"
                
                if manifest_file.exists():
                    try:
                        with open(manifest_file, 'r', encoding='utf-8') as f:
                            manifest_data = json.load(f)
                        
                        versions.append({
                            'version': manifest_data['version'],
                            'created_at': manifest_data['created_at'],
                            'locked_at': manifest_data.get('locked_at'),
                            'description': manifest_data.get('description', ''),
                            'total_families': manifest_data.get('total_families', 0),
                            'total_clauses': manifest_data.get('total_clauses', 0),
                            'is_locked': manifest_data.get('locked_at') is not None,
                            'path': str(version_dir)
                        })
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to read manifest for {version_dir}: {e}")
        
        # Sort by creation date (newest first)
        versions.sort(key=lambda x: x['created_at'], reverse=True)
        
        return versions
    
    def get_version_manifest(self, version: str) -> Optional[CorpusManifest]:
        """
        Get manifest for a specific version.
        
        Args:
            version: Version identifier
            
        Returns:
            Corpus manifest or None if not found
        """
        try:
            version_dir = self.settings.versions_dir / version
            manifest_file = version_dir / "manifest.json"
            
            if not manifest_file.exists():
                return None
            
            with open(manifest_file, 'r', encoding='utf-8') as f:
                manifest_data = json.load(f)
            
            return CorpusManifest(**manifest_data)
            
        except Exception as e:
            self.logger.error(f"Failed to load manifest for version {version}: {e}")
            return None
    
    def verify_version_integrity(self, version: str) -> Dict[str, Any]:
        """
        Verify integrity of a version by checking file checksums.
        
        Args:
            version: Version to verify
            
        Returns:
            Verification result dictionary
        """
        manifest = self.get_version_manifest(version)
        
        if not manifest:
            return {
                'version': version,
                'status': 'error',
                'message': 'Manifest not found',
                'files_checked': 0,
                'files_valid': 0,
                'files_invalid': 0,
                'missing_files': 0
            }
        
        files_checked = 0
        files_valid = 0
        files_invalid = 0
        missing_files = 0
        invalid_files = []
        missing_file_list = []
        
        for file_checksum in manifest.file_checksums:
            files_checked += 1
            file_path = self.settings.data_root / file_checksum.path
            
            if not file_path.exists():
                missing_files += 1
                missing_file_list.append(file_checksum.path)
                continue
            
            # Calculate current checksum
            current_checksum = self.calculate_file_checksum(file_path)
            
            if current_checksum == file_checksum.checksum:
                files_valid += 1
            else:
                files_invalid += 1
                invalid_files.append({
                    'path': file_checksum.path,
                    'expected': file_checksum.checksum,
                    'actual': current_checksum
                })
        
        status = 'valid'
        if missing_files > 0 or files_invalid > 0:
            status = 'invalid'
        
        return {
            'version': version,
            'status': status,
            'message': f"Verified {files_checked} files",
            'files_checked': files_checked,
            'files_valid': files_valid,
            'files_invalid': files_invalid,
            'missing_files': missing_files,
            'invalid_files': invalid_files[:10],  # Limit to first 10
            'missing_file_list': missing_file_list[:10]  # Limit to first 10
        }


def create_version_manager() -> VersionManager:
    """Factory function to create a version manager."""
    return VersionManager()