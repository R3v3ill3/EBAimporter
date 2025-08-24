"""
Main ingestion pipeline for processing Enterprise Agreement PDFs.
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import uuid

from tqdm import tqdm
import pandas as pd

from ..core.config import get_settings
from ..core.logging import LoggerMixin
from ..database import get_database_manager
from ..models import IngestRun, DocumentFingerprint as DBDocumentFingerprint
from ..utils.pdf_processor import create_pdf_processor, PDFDocument
from ..utils.text_cleaner import create_text_cleaner, CleaningStats
from ..utils.text_segmenter import create_text_segmenter, ClauseSegment
from ..utils.fingerprinter import create_text_fingerprinter, DocumentFingerprint


@dataclass
class ProcessingStats:
    """Statistics for a single document processing."""
    ea_id: str
    file_path: str
    file_size_bytes: int
    processing_time_seconds: float
    success: bool
    error_message: Optional[str] = None
    
    # PDF processing
    total_pages: int = 0
    ocr_used: bool = False
    
    # Text processing
    original_text_length: int = 0
    cleaned_text_length: int = 0
    cleaning_stats: Optional[CleaningStats] = None
    
    # Segmentation
    num_clauses: int = 0
    segmentation_method: str = "pattern"
    
    # Fingerprinting
    sha256_hash: Optional[str] = None
    minhash_computed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        if self.cleaning_stats:
            result['cleaning_stats'] = asdict(self.cleaning_stats)
        return result


@dataclass
class IngestRunStats:
    """Statistics for an entire ingestion run."""
    run_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    
    files_found: int = 0
    files_processed: int = 0
    files_succeeded: int = 0
    files_failed: int = 0
    
    total_pages: int = 0
    total_clauses: int = 0
    total_text_chars: int = 0
    
    processing_stats: List[ProcessingStats] = None
    
    def __post_init__(self):
        if self.processing_stats is None:
            self.processing_stats = []
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.files_processed == 0:
            return 0.0
        return self.files_succeeded / self.files_processed
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate run duration."""
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class EAIngestPipeline(LoggerMixin):
    """Main pipeline for ingesting Enterprise Agreement PDFs."""
    
    def __init__(self):
        """Initialize the ingest pipeline."""
        self.settings = get_settings()
        self.db_manager = get_database_manager()
        
        # Create processors
        self.pdf_processor = create_pdf_processor()
        self.text_cleaner = create_text_cleaner()
        self.text_segmenter = create_text_segmenter()
        self.fingerprinter = create_text_fingerprinter()
        
        self.logger.info("EA Ingest Pipeline initialized")
    
    def generate_ea_id(self, file_path: Path) -> str:
        """
        Generate a unique EA ID from file path.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Unique EA identifier
        """
        # Use filename without extension as base
        base_name = file_path.stem
        
        # Clean up common patterns
        base_name = base_name.replace(" ", "_")
        base_name = base_name.replace("-", "_")
        
        # Add hash suffix to ensure uniqueness
        file_hash = hashlib.md5(str(file_path).encode()).hexdigest()[:8]
        
        ea_id = f"EA_{base_name}_{file_hash}"
        
        return ea_id
    
    def process_single_pdf(self,
                          file_path: Path,
                          ea_id: Optional[str] = None,
                          force_ocr: bool = False) -> ProcessingStats:
        """
        Process a single PDF file through the complete pipeline.
        
        Args:
            file_path: Path to PDF file
            ea_id: Optional EA identifier (generated if not provided)
            force_ocr: Force OCR even if text layer exists
            
        Returns:
            Processing statistics
        """
        start_time = datetime.now()
        
        if ea_id is None:
            ea_id = self.generate_ea_id(file_path)
        
        stats = ProcessingStats(
            ea_id=ea_id,
            file_path=str(file_path),
            file_size_bytes=file_path.stat().st_size,
            processing_time_seconds=0.0,
            success=False
        )
        
        try:
            self.logger.info(f"Processing {ea_id}: {file_path.name}")
            
            # Step 1: Extract text from PDF
            pdf_document = self.pdf_processor.process_pdf(file_path, force_ocr=force_ocr)
            
            stats.total_pages = pdf_document.total_pages
            stats.ocr_used = force_ocr or not self.pdf_processor.has_text_layer(file_path)
            stats.original_text_length = len(pdf_document.full_text)
            
            if not pdf_document.full_text.strip():
                raise ValueError("No text extracted from PDF")
            
            # Step 2: Clean text
            cleaned_text, cleaning_stats = self.text_cleaner.clean_text(pdf_document.full_text)
            
            stats.cleaned_text_length = len(cleaned_text)
            stats.cleaning_stats = cleaning_stats
            
            # Step 3: Segment into clauses
            try:
                clauses = self.text_segmenter.segment_text(cleaned_text)
                stats.segmentation_method = "pattern"
            except Exception as e:
                self.logger.warning(f"Pattern segmentation failed for {ea_id}: {e}")
                clauses = self.text_segmenter.segment_by_ml_fallback(cleaned_text)
                stats.segmentation_method = "fallback"
            
            stats.num_clauses = len(clauses)
            
            # Validate minimum clause count
            if len(clauses) < self.settings.min_clause_count:
                self.logger.warning(f"{ea_id} has only {len(clauses)} clauses "
                                  f"(minimum: {self.settings.min_clause_count})")
            
            # Step 4: Generate fingerprints
            fingerprint = self.fingerprinter.fingerprint_document(
                ea_id=ea_id,
                full_text=cleaned_text,
                num_clauses=len(clauses)
            )
            
            stats.sha256_hash = fingerprint.sha256_hash
            stats.minhash_computed = True
            
            # Step 5: Save outputs
            self._save_processing_outputs(
                ea_id=ea_id,
                pdf_document=pdf_document,
                cleaned_text=cleaned_text,
                clauses=clauses,
                fingerprint=fingerprint
            )
            
            stats.success = True
            self.logger.info(f"Successfully processed {ea_id}: "
                           f"{stats.total_pages} pages, {stats.num_clauses} clauses")
            
        except Exception as e:
            stats.error_message = str(e)
            self.logger.error(f"Failed to process {ea_id}: {e}")
        
        finally:
            stats.processing_time_seconds = (datetime.now() - start_time).total_seconds()
        
        return stats
    
    def _save_processing_outputs(self,
                               ea_id: str,
                               pdf_document: PDFDocument,
                               cleaned_text: str,
                               clauses: List[ClauseSegment],
                               fingerprint: DocumentFingerprint):
        """
        Save processing outputs to files and database.
        
        Args:
            ea_id: EA identifier
            pdf_document: Processed PDF document
            cleaned_text: Cleaned text
            clauses: Segmented clauses
            fingerprint: Document fingerprint
        """
        # Ensure output directories exist
        self.settings.ensure_directories()
        
        # Save cleaned text
        text_file = self.settings.text_dir / f"{ea_id}.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        
        # Save clauses as JSONL
        clauses_file = self.settings.clauses_dir / f"{ea_id}.jsonl"
        with open(clauses_file, 'w', encoding='utf-8') as f:
            for clause in clauses:
                clause_dict = clause.to_dict()
                clause_dict['ea_id'] = ea_id
                
                # Add metadata
                clause_dict['tokens'] = len(clause.text.split())
                clause_dict['hash_sha256'] = hashlib.sha256(
                    clause.text.encode('utf-8')
                ).hexdigest()
                
                f.write(json.dumps(clause_dict, ensure_ascii=False) + '\n')
        
        # Save SHA256 fingerprint
        sha256_file = self.settings.fingerprints_dir / f"{ea_id}.sha256"
        with open(sha256_file, 'w') as f:
            f.write(fingerprint.sha256_hash)
        
        # Save MinHash fingerprint
        minhash_file = self.settings.fingerprints_dir / f"{ea_id}.minhash"
        fingerprint_dict = fingerprint.to_dict()
        with open(minhash_file, 'w') as f:
            json.dump(fingerprint_dict, f)
        
        # Save to database
        try:
            with self.db_manager.session_scope() as session:
                db_fingerprint = DBDocumentFingerprint(
                    ea_id=ea_id,
                    sha256_hash=fingerprint.sha256_hash,
                    minhash_signature=fingerprint.minhash_signature.digest(),
                    file_path=str(pdf_document.file_path),
                    file_size=pdf_document.file_path.stat().st_size,
                    num_pages=pdf_document.total_pages,
                    num_clauses=fingerprint.num_clauses
                )
                session.add(db_fingerprint)
                
        except Exception as e:
            self.logger.warning(f"Failed to save fingerprint to database: {e}")
    
    def batch_ingest(self,
                    input_dir: Path,
                    file_pattern: str = "*.pdf",
                    force_ocr: bool = False,
                    max_files: Optional[int] = None) -> IngestRunStats:
        """
        Batch ingest all PDFs in a directory.
        
        Args:
            input_dir: Directory containing PDF files
            file_pattern: File pattern to match
            force_ocr: Force OCR for all files
            max_files: Maximum number of files to process
            
        Returns:
            Ingestion run statistics
        """
        run_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        self.logger.info(f"Starting batch ingest run {run_id}")
        self.logger.info(f"Input directory: {input_dir}")
        self.logger.info(f"File pattern: {file_pattern}")
        
        # Find PDF files
        pdf_files = list(input_dir.glob(file_pattern))
        
        if max_files:
            pdf_files = pdf_files[:max_files]
        
        run_stats = IngestRunStats(
            run_id=run_id,
            started_at=start_time,
            files_found=len(pdf_files)
        )
        
        if not pdf_files:
            self.logger.warning(f"No PDF files found in {input_dir} matching {file_pattern}")
            run_stats.completed_at = datetime.now()
            return run_stats
        
        # Create database ingest run record
        db_run = None
        try:
            with self.db_manager.session_scope() as session:
                db_run = IngestRun(
                    id=uuid.UUID(run_id),
                    config_snapshot={
                        'input_dir': str(input_dir),
                        'file_pattern': file_pattern,
                        'force_ocr': force_ocr,
                        'max_files': max_files,
                        'settings': {
                            'min_clause_count': self.settings.min_clause_count,
                            'ocr_language': self.settings.ocr_language,
                            'ocr_dpi': self.settings.ocr_dpi,
                        }
                    }
                )
                session.add(db_run)
                
        except Exception as e:
            self.logger.warning(f"Failed to create database ingest run: {e}")
        
        # Process files with progress bar
        self.logger.info(f"Processing {len(pdf_files)} PDF files...")
        
        progress_bar = tqdm(pdf_files, desc="Processing PDFs")
        
        for pdf_file in progress_bar:
            try:
                progress_bar.set_description(f"Processing {pdf_file.name}")
                
                # Process the file
                stats = self.process_single_pdf(pdf_file, force_ocr=force_ocr)
                run_stats.processing_stats.append(stats)
                
                # Update counters
                run_stats.files_processed += 1
                if stats.success:
                    run_stats.files_succeeded += 1
                    run_stats.total_pages += stats.total_pages
                    run_stats.total_clauses += stats.num_clauses
                    run_stats.total_text_chars += stats.cleaned_text_length
                else:
                    run_stats.files_failed += 1
                
                # Update progress
                progress_bar.set_postfix({
                    'Success': f"{run_stats.files_succeeded}/{run_stats.files_processed}",
                    'Rate': f"{run_stats.success_rate:.1%}"
                })
                
            except KeyboardInterrupt:
                self.logger.info("Batch ingest interrupted by user")
                break
            except Exception as e:
                self.logger.error(f"Unexpected error processing {pdf_file}: {e}")
                run_stats.files_failed += 1
        
        run_stats.completed_at = datetime.now()
        
        # Update database ingest run
        if db_run:
            try:
                with self.db_manager.session_scope() as session:
                    db_run = session.get(IngestRun, uuid.UUID(run_id))
                    if db_run:
                        db_run.completed_at = run_stats.completed_at
                        db_run.files_processed = run_stats.files_processed
                        db_run.files_succeeded = run_stats.files_succeeded
                        db_run.files_failed = run_stats.files_failed
                        db_run.total_clauses = run_stats.total_clauses
                        
            except Exception as e:
                self.logger.warning(f"Failed to update database ingest run: {e}")
        
        # Save run statistics
        self._save_run_stats(run_stats)
        
        self.logger.info(f"Batch ingest completed: {run_stats.files_succeeded}/{run_stats.files_processed} "
                        f"files processed successfully ({run_stats.success_rate:.1%})")
        
        return run_stats
    
    def _save_run_stats(self, run_stats: IngestRunStats):
        """
        Save run statistics to file.
        
        Args:
            run_stats: Run statistics to save
        """
        try:
            # Create reports directory
            reports_dir = self.settings.reports_dir / "ingest_runs"
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            # Save detailed stats
            stats_file = reports_dir / f"{run_stats.run_id}_stats.json"
            
            stats_dict = asdict(run_stats)
            stats_dict['started_at'] = run_stats.started_at.isoformat()
            if run_stats.completed_at:
                stats_dict['completed_at'] = run_stats.completed_at.isoformat()
            
            with open(stats_file, 'w') as f:
                json.dump(stats_dict, f, indent=2)
            
            # Save summary CSV
            if run_stats.processing_stats:
                summary_data = []
                for stats in run_stats.processing_stats:
                    summary_data.append({
                        'ea_id': stats.ea_id,
                        'file_name': Path(stats.file_path).name,
                        'success': stats.success,
                        'processing_time_seconds': stats.processing_time_seconds,
                        'total_pages': stats.total_pages,
                        'num_clauses': stats.num_clauses,
                        'text_length': stats.cleaned_text_length,
                        'ocr_used': stats.ocr_used,
                        'error_message': stats.error_message or ""
                    })
                
                summary_df = pd.DataFrame(summary_data)
                summary_file = reports_dir / f"{run_stats.run_id}_summary.csv"
                summary_df.to_csv(summary_file, index=False)
            
            self.logger.info(f"Run statistics saved to {reports_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to save run statistics: {e}")
    
    def validate_outputs(self, ea_ids: List[str]) -> Dict[str, bool]:
        """
        Validate that all expected outputs exist for given EA IDs.
        
        Args:
            ea_ids: List of EA identifiers to validate
            
        Returns:
            Dictionary mapping EA ID to validation status
        """
        results = {}
        
        for ea_id in ea_ids:
            required_files = [
                self.settings.text_dir / f"{ea_id}.txt",
                self.settings.clauses_dir / f"{ea_id}.jsonl",
                self.settings.fingerprints_dir / f"{ea_id}.sha256",
                self.settings.fingerprints_dir / f"{ea_id}.minhash",
            ]
            
            all_exist = all(file_path.exists() for file_path in required_files)
            results[ea_id] = all_exist
            
            if not all_exist:
                missing = [str(f) for f in required_files if not f.exists()]
                self.logger.warning(f"{ea_id} missing files: {missing}")
        
        return results


def create_ingest_pipeline() -> EAIngestPipeline:
    """Factory function to create an ingest pipeline."""
    return EAIngestPipeline()