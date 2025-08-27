"""
CSV Batch Import Utility for EA Importer

Supports batch importing of Enterprise Agreements from:
- CSV files with document URLs and metadata
- FWC (Fair Work Commission) document search results
- Remote PDF URLs with automatic downloading
- Bulk processing with error handling and resume capability

CSV Format:
- url: Direct URL to PDF document
- title: Document title/name
- employer: Employer organization name
- union: Union name (optional)
- industry: Industry classification
- effective_date: Agreement effective date (YYYY-MM-DD)
- expiry_date: Agreement expiry date (YYYY-MM-DD) 
- status: Agreement status (active, expired, terminated)
- fwc_code: FWC agreement code (if applicable)
- metadata: Additional JSON metadata (optional)
"""

import csv
import json
import asyncio
import aiohttp
import aiofiles
from pathlib import Path
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime, date
import logging
from urllib.parse import urlparse, urljoin
import hashlib
import tempfile
import time

from ..core.config import get_settings
from ..core.logging import get_logger, log_function_call

# Optional DB imports (DB-optional mode)
try:
    from ..database import get_db_session  # type: ignore
    DB_SESSION_AVAILABLE = True
except Exception:
    get_db_session = None  # type: ignore
    DB_SESSION_AVAILABLE = False

try:
    from ..models import Document as DocumentDB  # type: ignore
    from ..models import BatchImportJob, BatchImportResult  # type: ignore
    DB_MODELS_AVAILABLE = True
except Exception:
    DocumentDB = None  # type: ignore
    BatchImportJob = None  # type: ignore
    BatchImportResult = None  # type: ignore
    DB_MODELS_AVAILABLE = False

from ..utils.pdf_processor import PDFProcessor
from ..utils.text_cleaner import TextCleaner
from ..utils.text_segmenter import TextSegmenter
from ..utils.fingerprinter import Fingerprinter

logger = get_logger(__name__)


class CSVBatchImporter:
    """
    Batch importer for Enterprise Agreements from CSV files with URLs.
    """
    
    def __init__(self, session_timeout: int = 30, max_concurrent: int = 5):
        """
        Initialize batch importer.
        
        Args:
            session_timeout: HTTP session timeout in seconds
            max_concurrent: Maximum concurrent downloads
        """
        self.settings = get_settings()
        self.session_timeout = session_timeout
        self.max_concurrent = max_concurrent
        self.pdf_processor = PDFProcessor()
        self.text_cleaner = TextCleaner()
        self.text_segmenter = TextSegmenter()
        self.fingerprinter = Fingerprinter()
        self._db_enabled = DB_SESSION_AVAILABLE and DB_MODELS_AVAILABLE
        
        # Create download directory
        self.download_dir = Path(self.settings.paths.upload_dir) / "batch_downloads"
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
    @log_function_call
    async def import_from_csv(
        self,
        csv_file_path: str,
        job_name: str = None,
        auto_process: bool = True,
        resume_job_id: Optional[int] = None
    ) -> Any:
        """
        Import Enterprise Agreements from CSV file.
        
        Args:
            csv_file_path: Path to CSV file with document URLs
            job_name: Optional name for the import job
            auto_process: Whether to automatically process downloaded PDFs
            resume_job_id: Resume an existing job (optional)
            
        Returns:
            BatchImportJob with results
        """
        
        if not job_name:
            job_name = f"CSV_Import_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create or resume batch import job (DB-optional)
        if self._db_enabled and get_db_session and BatchImportJob:
            with get_db_session() as session:  # type: ignore[arg-type]
                if resume_job_id:
                    job = session.query(BatchImportJob).filter(  # type: ignore[call-arg]
                        BatchImportJob.id == resume_job_id  # type: ignore[attr-defined]
                    ).first()
                    if not job:
                        raise ValueError(f"Job {resume_job_id} not found")
                    logger.info(f"Resuming batch import job: {job.job_name}")
                else:
                    job = BatchImportJob(  # type: ignore[call-arg]
                        job_name=job_name,
                        source_type="csv",
                        source_path=csv_file_path,
                        status="running",
                        total_items=0,
                        processed_items=0,
                        successful_items=0,
                        failed_items=0,
                        settings={
                            "auto_process": auto_process,
                            "session_timeout": self.session_timeout,
                            "max_concurrent": self.max_concurrent
                        }
                    )
                    session.add(job)
                    session.flush()  # Get job ID
                    logger.info(f"Created batch import job: {job.job_name} (ID: {job.id})")
        else:
            # Simple in-memory job object for CLI display compatibility
            class _SimpleJob:
                def __init__(self, name: str, source_path: str):
                    self.id = 0
                    self.job_name = name
                    self.source_type = "csv"
                    self.source_path = source_path
                    self.status = "running"
                    self.total_items = 0
                    self.processed_items = 0
                    self.successful_items = 0
                    self.failed_items = 0
                    self.created_at = datetime.utcnow()
                    self.completed_at = None
                    self.error_message = None
            job = _SimpleJob(job_name, csv_file_path)
        
        try:
            # Parse CSV file
            csv_records = await self._parse_csv_file(csv_file_path)
            
            # Update total items count
            if self._db_enabled and get_db_session and BatchImportJob:
                with get_db_session() as session:  # type: ignore[arg-type]
                    db_job = session.query(BatchImportJob).filter(BatchImportJob.id == job.id).first()  # type: ignore[call-arg]
                    if db_job:
                        db_job.total_items = len(csv_records)
                        session.commit()
            else:
                job.total_items = len(csv_records)
            
            logger.info(f"Found {len(csv_records)} records in CSV file")
            
            # Process records with concurrency control
            semaphore = asyncio.Semaphore(self.max_concurrent)
            
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.session_timeout)
            ) as http_session:
                
                tasks = [
                    self._process_csv_record(semaphore, http_session, getattr(job, 'id', 0), record, auto_process)
                    for record in csv_records
                ]
                
                # Process with progress tracking
                completed_tasks = 0
                for task in asyncio.as_completed(tasks):
                    try:
                        result = await task
                        completed_tasks += 1
                        
                        # Update progress every 10 items
                        if completed_tasks % 10 == 0:
                            await self._update_job_progress(job.id)
                            
                    except Exception as e:
                        logger.error(f"Task failed: {e}")
                        completed_tasks += 1
            
            # Final progress update
            if self._db_enabled:
                await self._update_job_progress(getattr(job, 'id', 0))
            
            # Mark job as completed
            if self._db_enabled and get_db_session and BatchImportJob:
                with get_db_session() as session:  # type: ignore[arg-type]
                    db_job = session.query(BatchImportJob).filter(BatchImportJob.id == job.id).first()  # type: ignore[call-arg]
                    if db_job:
                        db_job.status = "completed"
                        db_job.completed_at = datetime.utcnow()
                        session.commit()
            else:
                job.status = "completed"
                job.completed_at = datetime.utcnow()
                
            logger.info(f"Batch import job completed: {job.successful_items}/{job.total_items} successful")
            return job
            
        except Exception as e:
            logger.error(f"Batch import job failed: {e}")
            
            # Mark job as failed
            if self._db_enabled and get_db_session and BatchImportJob:
                with get_db_session() as session:  # type: ignore[arg-type]
                    db_job = session.query(BatchImportJob).filter(BatchImportJob.id == job.id).first()  # type: ignore[call-arg]
                    if db_job:
                        db_job.status = "failed"
                        db_job.error_message = str(e)
                        db_job.completed_at = datetime.utcnow()
                        session.commit()
            else:
                job.status = "failed"
                job.error_message = str(e)
            
            raise
    
    async def _parse_csv_file(self, csv_file_path: str) -> List[Dict[str, Any]]:
        """Parse CSV file and validate records."""
        
        records = []
        
        async with aiofiles.open(csv_file_path, mode='r', encoding='utf-8') as file:
            content = await file.read()
            
        # Parse CSV content
        csv_reader = csv.DictReader(content.splitlines())
        
        # We accept multiple header variants and map to our canonical fields
        required_fields = ['url', 'title']
        
        for row_num, row in enumerate(csv_reader, start=2):  # Start at 2 for header
            try:
                # Case-insensitive key access
                lower_row = {k.strip(): v for k, v in row.items()}
                key_map = {k.lower(): k for k in lower_row.keys()}

                def get_any(keys: List[str]) -> str:
                    for k in keys:
                        if k.lower() in key_map and lower_row.get(key_map[k.lower()]):
                            return lower_row[key_map[k.lower()]].strip()
                    return ""

                # Map fields
                url_val = get_any(["url", "PDF_URL", "pdf_url"]) or get_any(["link", "href"])
                title_val = get_any(["title", "Title", "DocumentTitle", "AgreementTitle"]) or ""
                employer_val = get_any(["employer", "PartyName"]) or ""
                fwc_code_val = get_any(["fwc_code", "PublicationID", "AgmntMNC"]) or ""
                status_val = get_any(["status"]) or "unknown"
                industry_val = get_any(["industry"]) or ""
                union_val = get_any(["union"]) or ""

                # Validate required
                if not url_val or not self._is_valid_url(url_val):
                    logger.warning(f"Row {row_num}: Invalid or missing URL")
                    continue
                if not title_val:
                    logger.warning(f"Row {row_num}: Missing title")
                    continue

                record = {
                    'row_number': row_num,
                    'url': url_val,
                    'title': title_val,
                    'employer': employer_val,
                    'union': union_val,
                    'industry': industry_val,
                    'status': status_val,
                    'fwc_code': fwc_code_val,
                }

                # Dates
                eff_dt = get_any(["effective_date"]) or ""
                exp_dt = get_any(["expiry_date", "NominalExpiryDate"]) or ""
                def parse_date(val: str) -> Optional[date]:
                    if not val:
                        return None
                    try:
                        return datetime.strptime(val, '%Y-%m-%d').date()
                    except Exception:
                        try:
                            # ISO datetime with Z
                            return datetime.fromisoformat(val.replace('Z', '+00:00')).date()
                        except Exception:
                            logger.warning(f"Row {row_num}: Invalid date format: {val}")
                            return None
                record['effective_date'] = parse_date(eff_dt)
                record['expiry_date'] = parse_date(exp_dt)

                # Extra metadata
                metadata: Dict[str, Any] = {}
                for key in [
                    'Summary', 'AgreementMembers', 'ABN', 'MatterName',
                    'DocumentTitle', 'AgreementTitle'
                ]:
                    val = get_any([key])
                    if val:
                        metadata[key] = val
                # If provided explicit metadata json field, merge
                metadata_json = get_any(["metadata"])
                if metadata_json:
                    try:
                        metadata.update(json.loads(metadata_json))
                    except json.JSONDecodeError:
                        logger.warning(f"Row {row_num}: Invalid JSON metadata: {metadata_json}")
                record['metadata'] = metadata

                records.append(record)

            except Exception as e:
                logger.error(f"Row {row_num}: Error parsing record: {e}")
                continue
        
        logger.info(f"Parsed {len(records)} valid records from CSV file")
        return records
    
    async def _process_csv_record(
        self,
        semaphore: asyncio.Semaphore,
        http_session: aiohttp.ClientSession,
        job_id: int,
        record: Dict[str, Any],
        auto_process: bool
    ) -> Any:
        """Process a single CSV record."""
        
        async with semaphore:
            start_time = time.time()
            
            try:
                # Generate unique filename
                url_hash = hashlib.md5(record['url'].encode()).hexdigest()[:8]
                safe_title = "".join(c for c in record['title'] if c.isalnum() or c in (' ', '-', '_')).strip()[:50]
                filename = f"{safe_title}_{url_hash}.pdf"
                file_path = self.download_dir / filename
                
                # Download PDF
                logger.info(f"Downloading: {record['url']}")
                await self._download_pdf(http_session, record['url'], file_path)
                
                # Validate PDF
                if not await self._validate_pdf(file_path):
                    raise ValueError("Downloaded file is not a valid PDF")
                
                document_id = None
                if self._db_enabled and get_db_session and DocumentDB:
                    with get_db_session() as session:  # type: ignore[arg-type]
                        document = DocumentDB(
                            file_path=str(file_path),
                            original_filename=filename,
                            file_size_bytes=file_path.stat().st_size,
                            status=record['status'],
                            fwc_id=record['fwc_code'],
                            title=record['title'],
                            jurisdiction=None,
                        )  # type: ignore[call-arg]
                        session.add(document)
                        session.flush()
                        document_id = document.id
                
                # Auto-process if enabled
                if auto_process:
                    try:
                        # Full processing pipeline using utilities
                        document = self.pdf_processor.process_pdf(file_path)
                        cleaned_doc = self.text_cleaner.clean_document(document)
                        clauses = self.text_segmenter.segment_document(cleaned_doc)
                        fingerprint = self.fingerprinter.fingerprint_document(document)

                        # Save artifacts
                        ea_id = document.metadata.get('ea_id')
                        text_dir = Path(self.settings.paths.text_dir)
                        text_dir.mkdir(parents=True, exist_ok=True)
                        with open(text_dir / f"{ea_id}.txt", 'w', encoding='utf-8') as f:
                            f.write(document.full_text)

                        clauses_dir = Path(self.settings.paths.clauses_dir)
                        clauses_dir.mkdir(parents=True, exist_ok=True)
                        with open(clauses_dir / f"{ea_id}.jsonl", 'w', encoding='utf-8') as f:
                            for c in clauses:
                                f.write(json.dumps({
                                    'ea_id': c.ea_id,
                                    'clause_id': c.clause_id,
                                    'heading': c.heading,
                                    'text': c.text,
                                    'path': c.path,
                                    'hash_sha256': c.hash_sha256,
                                    'token_count': c.token_count
                                }) + '\n')

                        fp_dir = Path(self.settings.paths.fingerprints_dir)
                        self.fingerprinter.save_fingerprint(fingerprint, fp_dir)

                        processing_status = 'completed'
                        error_message = None
                    except Exception as e:
                        logger.error(f"Processing failed for {filename}: {e}")
                        processing_status = 'processing_failed'
                        error_message = str(e)
                        if self._db_enabled and get_db_session and DocumentDB and document_id is not None:
                            with get_db_session() as session:  # type: ignore[arg-type]
                                doc = session.query(DocumentDB).filter(DocumentDB.id == document_id).first()
                                if doc:
                                    setattr(doc, 'status', 'processing_failed')
                                    session.commit()
                else:
                    processing_status = 'downloaded'
                    error_message = None
                
                # Create result record or simple dict
                if self._db_enabled and get_db_session and BatchImportResult:
                    result = BatchImportResult(  # type: ignore[call-arg]
                        job_id=job_id,
                        row_number=record['row_number'],
                        source_url=record['url'],
                        status='success',
                        # Store limited info; schema differs from ORM here
                    )
                    with get_db_session() as session:  # type: ignore[arg-type]
                        session.add(result)
                        session.commit()
                    logger.info(f"Successfully processed: {record['title']}")
                    return result
                else:
                    logger.info(f"Successfully processed: {record['title']}")
                    return {
                        'row_number': record['row_number'],
                        'source_url': record['url'],
                        'status': 'success',
                        'processing_time_seconds': time.time() - start_time,
                        'file_path': str(file_path),
                        'file_size': file_path.stat().st_size
                    }
                
            except Exception as e:
                logger.error(f"Failed to process record {record['row_number']}: {e}")
                
                # Create error result
                if self._db_enabled and get_db_session and BatchImportResult:
                    result = BatchImportResult(  # type: ignore[call-arg]
                        job_id=job_id,
                        row_number=record['row_number'],
                        source_url=record['url'],
                        status='failed',
                    )
                    with get_db_session() as session:  # type: ignore[arg-type]
                        session.add(result)
                        session.commit()
                    return result
                else:
                    return {
                        'row_number': record['row_number'],
                        'source_url': record['url'],
                        'status': 'failed',
                        'error_message': str(e),
                        'processing_time_seconds': time.time() - start_time
                    }
    
    async def _download_pdf(
        self,
        session: aiohttp.ClientSession,
        url: str,
        file_path: Path
    ) -> None:
        """Download PDF from URL."""
        
        async with session.get(url) as response:
            if response.status != 200:
                raise ValueError(f"HTTP {response.status}: Failed to download {url}")
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' not in content_type and not url.lower().endswith('.pdf'):
                logger.warning(f"Unexpected content type: {content_type} for {url}")
            
            # Download file
            async with aiofiles.open(file_path, 'wb') as f:
                async for chunk in response.content.iter_chunked(8192):
                    await f.write(chunk)
    
    async def _validate_pdf(self, file_path: Path) -> bool:
        """Validate that downloaded file is a PDF."""
        
        if not file_path.exists():
            return False
        
        # Check file size
        if file_path.stat().st_size < 100:  # Too small to be a valid PDF
            return False
        
        # Check PDF header
        try:
            async with aiofiles.open(file_path, 'rb') as f:
                header = await f.read(5)
                return header.startswith(b'%PDF-')
        except Exception:
            return False
    
    async def _process_document(self, document_id: int, file_path: Path) -> None:
        """Process downloaded document through the ingestion pipeline."""
        
        # Use the ingest pipeline for processing
        await asyncio.get_event_loop().run_in_executor(
            None,
            self._sync_process_document,
            document_id,
            file_path
        )
    
    def _sync_process_document(self, document_id: int, file_path: Path) -> None:
        """Synchronous document processing for executor."""
        
        # Placeholder that marks document as completed in DB if available
        if get_db_session and DocumentDB:
            try:
                with get_db_session() as session:  # type: ignore[arg-type]
                    document = session.query(DocumentDB).filter(DocumentDB.id == document_id).first()
                    if document:
                        setattr(document, 'status', 'completed')
                        session.commit()
            except Exception:
                # Swallow DB errors to avoid crashing background processing
                return
    
    async def _update_job_progress(self, job_id: int) -> None:
        """Update job progress statistics."""
        
        if not (self._db_enabled and get_db_session and BatchImportJob and BatchImportResult):
            return
        with get_db_session() as session:  # type: ignore[arg-type]
            job = session.query(BatchImportJob).filter(BatchImportJob.id == job_id).first()  # type: ignore[call-arg]
            if not job:
                return
            results = session.query(BatchImportResult).filter(BatchImportResult.job_id == job_id).all()  # type: ignore[call-arg]
            job.processed_items = len(results)
            job.successful_items = len([r for r in results if getattr(r, 'status', '') == 'success'])
            job.failed_items = len([r for r in results if getattr(r, 'status', '') == 'failed'])
            session.commit()
    
    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format."""
        
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    @log_function_call
    def get_job_status(self, job_id: int) -> Optional[BatchImportJob]:
        """Get status of a batch import job."""
        
        with get_db_session() as session:
            return session.query(BatchImportJob).filter(
                BatchImportJob.id == job_id
            ).first()
    
    @log_function_call
    def list_jobs(self, limit: int = 50) -> List[BatchImportJob]:
        """List recent batch import jobs."""
        
        with get_db_session() as session:
            return session.query(BatchImportJob).order_by(
                BatchImportJob.created_at.desc()
            ).limit(limit).all()
    
    @log_function_call
    def cancel_job(self, job_id: int) -> bool:
        """Cancel a running batch import job."""
        
        with get_db_session() as session:
            job = session.query(BatchImportJob).filter(BatchImportJob.id == job_id).first()
            if job and job.status == 'running':
                job.status = 'cancelled'
                job.completed_at = datetime.utcnow()
                session.commit()
                return True
        
        return False


# Utility functions for common CSV batch import scenarios

async def import_fwc_search_results(csv_file_path: str, job_name: str = None) -> Any:
    """
    Import Enterprise Agreements from FWC search results CSV.
    
    Args:
        csv_file_path: Path to CSV file from FWC website export
        job_name: Optional job name
        
    Returns:
        BatchImportJob
    """
    
    importer = CSVBatchImporter()
    return await importer.import_from_csv(
        csv_file_path=csv_file_path,
        job_name=job_name or "FWC_Search_Import",
        auto_process=True
    )


async def import_url_list(urls: List[str], job_name: str = None) -> Any:
    """
    Import Enterprise Agreements from a list of URLs.
    
    Args:
        urls: List of PDF URLs
        job_name: Optional job name
        
    Returns:
        BatchImportJob
    """
    
    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        writer = csv.writer(f)
        writer.writerow(['url', 'title'])
        
        for i, url in enumerate(urls):
            title = f"Document_{i+1}"
            writer.writerow([url, title])
        
        temp_csv_path = f.name
    
    try:
        importer = CSVBatchImporter()
        return await importer.import_from_csv(
            csv_file_path=temp_csv_path,
            job_name=job_name or "URL_List_Import",
            auto_process=True
        )
    finally:
        # Clean up temporary file
        Path(temp_csv_path).unlink(missing_ok=True)


__all__ = [
    'CSVBatchImporter',
    'import_fwc_search_results',
    'import_url_list'
]