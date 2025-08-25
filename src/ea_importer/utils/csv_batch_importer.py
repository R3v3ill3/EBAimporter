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
from ..database import get_db_session
from ..models import DocumentDB, BatchImportJob, BatchImportResult
from ..utils.pdf_processor import PDFProcessor
from ..pipeline.ingest_pipeline import IngestPipeline

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
        self.ingest_pipeline = IngestPipeline()
        
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
    ) -> BatchImportJob:
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
        
        # Create or resume batch import job
        with get_db_session() as session:
            if resume_job_id:
                job = session.query(BatchImportJob).filter(
                    BatchImportJob.id == resume_job_id
                ).first()
                if not job:
                    raise ValueError(f"Job {resume_job_id} not found")
                logger.info(f"Resuming batch import job: {job.job_name}")
            else:
                job = BatchImportJob(
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
        
        try:
            # Parse CSV file
            csv_records = await self._parse_csv_file(csv_file_path)
            
            # Update total items count
            with get_db_session() as session:
                job = session.query(BatchImportJob).filter(BatchImportJob.id == job.id).first()
                job.total_items = len(csv_records)
                session.commit()
            
            logger.info(f"Found {len(csv_records)} records in CSV file")
            
            # Process records with concurrency control
            semaphore = asyncio.Semaphore(self.max_concurrent)
            
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.session_timeout)
            ) as http_session:
                
                tasks = [
                    self._process_csv_record(semaphore, http_session, job.id, record, auto_process)
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
            await self._update_job_progress(job.id)
            
            # Mark job as completed
            with get_db_session() as session:
                job = session.query(BatchImportJob).filter(BatchImportJob.id == job.id).first()
                job.status = "completed"
                job.completed_at = datetime.utcnow()
                session.commit()
                
            logger.info(f"Batch import job completed: {job.successful_items}/{job.total_items} successful")
            return job
            
        except Exception as e:
            logger.error(f"Batch import job failed: {e}")
            
            # Mark job as failed
            with get_db_session() as session:
                job = session.query(BatchImportJob).filter(BatchImportJob.id == job.id).first()
                job.status = "failed"
                job.error_message = str(e)
                job.completed_at = datetime.utcnow()
                session.commit()
            
            raise
    
    async def _parse_csv_file(self, csv_file_path: str) -> List[Dict[str, Any]]:
        """Parse CSV file and validate records."""
        
        records = []
        
        async with aiofiles.open(csv_file_path, mode='r', encoding='utf-8') as file:
            content = await file.read()
            
        # Parse CSV content
        csv_reader = csv.DictReader(content.splitlines())
        
        required_fields = ['url', 'title']
        optional_fields = [
            'employer', 'union', 'industry', 'effective_date', 'expiry_date',
            'status', 'fwc_code', 'metadata'
        ]
        
        for row_num, row in enumerate(csv_reader, start=2):  # Start at 2 for header
            try:
                # Validate required fields
                missing_fields = [field for field in required_fields if not row.get(field)]
                if missing_fields:
                    logger.warning(f"Row {row_num}: Missing required fields: {missing_fields}")
                    continue
                
                # Validate URL
                if not self._is_valid_url(row['url']):
                    logger.warning(f"Row {row_num}: Invalid URL: {row['url']}")
                    continue
                
                # Parse dates
                record = {
                    'row_number': row_num,
                    'url': row['url'].strip(),
                    'title': row['title'].strip(),
                    'employer': row.get('employer', '').strip(),
                    'union': row.get('union', '').strip(),
                    'industry': row.get('industry', '').strip(),
                    'status': row.get('status', 'unknown').strip(),
                    'fwc_code': row.get('fwc_code', '').strip()
                }
                
                # Parse dates
                for date_field in ['effective_date', 'expiry_date']:
                    date_str = row.get(date_field, '').strip()
                    if date_str:
                        try:
                            record[date_field] = datetime.strptime(date_str, '%Y-%m-%d').date()
                        except ValueError:
                            logger.warning(f"Row {row_num}: Invalid date format for {date_field}: {date_str}")
                            record[date_field] = None
                    else:
                        record[date_field] = None
                
                # Parse metadata JSON
                metadata_str = row.get('metadata', '').strip()
                if metadata_str:
                    try:
                        record['metadata'] = json.loads(metadata_str)
                    except json.JSONDecodeError:
                        logger.warning(f"Row {row_num}: Invalid JSON metadata: {metadata_str}")
                        record['metadata'] = {}
                else:
                    record['metadata'] = {}
                
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
    ) -> BatchImportResult:
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
                
                # Create document record
                with get_db_session() as session:
                    document = DocumentDB(
                        file_path=str(file_path),
                        file_name=filename,
                        file_size=file_path.stat().st_size,
                        source_url=record['url'],
                        title=record['title'],
                        employer=record['employer'],
                        union=record['union'],
                        industry=record['industry'],
                        effective_date=record['effective_date'],
                        expiry_date=record['expiry_date'],
                        status=record['status'],
                        fwc_code=record['fwc_code'],
                        metadata=record['metadata'],
                        processing_status='downloaded'
                    )
                    session.add(document)
                    session.flush()
                    document_id = document.id
                
                # Auto-process if enabled
                if auto_process:
                    try:
                        await self._process_document(document_id, file_path)
                        processing_status = 'completed'
                        error_message = None
                    except Exception as e:
                        logger.error(f"Processing failed for {filename}: {e}")
                        processing_status = 'processing_failed'
                        error_message = str(e)
                        
                        # Update document status
                        with get_db_session() as session:
                            doc = session.query(DocumentDB).filter(DocumentDB.id == document_id).first()
                            if doc:
                                doc.processing_status = processing_status
                                doc.error_message = error_message
                                session.commit()
                else:
                    processing_status = 'downloaded'
                    error_message = None
                
                # Create result record
                result = BatchImportResult(
                    job_id=job_id,
                    row_number=record['row_number'],
                    source_url=record['url'],
                    document_id=document_id,
                    status='success',
                    processing_time_seconds=time.time() - start_time,
                    file_path=str(file_path),
                    file_size=file_path.stat().st_size
                )
                
                with get_db_session() as session:
                    session.add(result)
                    session.commit()
                
                logger.info(f"Successfully processed: {record['title']}")
                return result
                
            except Exception as e:
                logger.error(f"Failed to process record {record['row_number']}: {e}")
                
                # Create error result
                result = BatchImportResult(
                    job_id=job_id,
                    row_number=record['row_number'],
                    source_url=record['url'],
                    status='failed',
                    error_message=str(e),
                    processing_time_seconds=time.time() - start_time
                )
                
                with get_db_session() as session:
                    session.add(result)
                    session.commit()
                
                return result
    
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
        
        # This would integrate with the main ingest pipeline
        # For now, we'll update the status
        with get_db_session() as session:
            document = session.query(DocumentDB).filter(DocumentDB.id == document_id).first()
            if document:
                document.processing_status = 'processed'
                session.commit()
    
    async def _update_job_progress(self, job_id: int) -> None:
        """Update job progress statistics."""
        
        with get_db_session() as session:
            job = session.query(BatchImportJob).filter(BatchImportJob.id == job_id).first()
            if not job:
                return
            
            # Count results
            results = session.query(BatchImportResult).filter(
                BatchImportResult.job_id == job_id
            ).all()
            
            job.processed_items = len(results)
            job.successful_items = len([r for r in results if r.status == 'success'])
            job.failed_items = len([r for r in results if r.status == 'failed'])
            
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

async def import_fwc_search_results(csv_file_path: str, job_name: str = None) -> BatchImportJob:
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


async def import_url_list(urls: List[str], job_name: str = None) -> BatchImportJob:
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