"""
CSV batch import utility for downloading PDFs from URLs.
"""

import asyncio
import csv
import time
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from urllib.parse import urlparse, urljoin
import re

import httpx
import pandas as pd
from tqdm import tqdm

from ..core.config import get_settings
from ..core.logging import LoggerMixin


@dataclass
class DownloadResult:
    """Result of a single PDF download."""
    url: str
    local_path: Optional[Path] = None
    success: bool = False
    error_message: Optional[str] = None
    file_size_bytes: int = 0
    download_time_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'url': self.url,
            'local_path': str(self.local_path) if self.local_path else None,
            'success': self.success,
            'error_message': self.error_message,
            'file_size_bytes': self.file_size_bytes,
            'download_time_seconds': self.download_time_seconds,
        }


@dataclass
class BatchImportResult:
    """Result of a batch import operation."""
    total_urls: int
    downloaded: int
    failed: int
    skipped: int
    download_results: List[DownloadResult]
    execution_time_seconds: float
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_urls == 0:
            return 0.0
        return self.downloaded / self.total_urls


class CSVBatchImporter(LoggerMixin):
    """Handles batch import of PDFs from CSV files with URLs."""
    
    def __init__(self):
        """Initialize the CSV batch importer."""
        self.settings = get_settings()
        self.client = None
    
    def _create_http_client(self) -> httpx.AsyncClient:
        """Create HTTP client with appropriate settings."""
        headers = {
            'User-Agent': 'EA-Importer/1.0 (Enterprise Agreement Processor)',
            'Accept': 'application/pdf,application/octet-stream,*/*',
        }
        
        timeout = httpx.Timeout(30.0, connect=10.0)
        
        return httpx.AsyncClient(
            headers=headers,
            timeout=timeout,
            follow_redirects=True,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
    
    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename for safe filesystem storage.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        # Remove or replace invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Remove excessive whitespace
        filename = re.sub(r'\s+', '_', filename)
        
        # Limit length
        if len(filename) > 200:
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            filename = name[:195] + '.' + ext if ext else name[:200]
        
        # Ensure it ends with .pdf
        if not filename.lower().endswith('.pdf'):
            filename += '.pdf'
        
        return filename
    
    def extract_filename_from_url(self, url: str) -> str:
        """
        Extract filename from URL.
        
        Args:
            url: PDF URL
            
        Returns:
            Extracted filename
        """
        parsed = urlparse(url)
        
        # Try to get filename from path
        path_parts = parsed.path.split('/')
        filename = path_parts[-1] if path_parts else ''
        
        # If no filename or doesn't look like PDF, generate one
        if not filename or not filename.lower().endswith('.pdf'):
            # Try to extract document ID from URL
            doc_id_match = re.search(r'(?:document|doc|pdf)[/_-]?(\w+)', url, re.IGNORECASE)
            if doc_id_match:
                filename = f"EA_{doc_id_match.group(1)}.pdf"
            else:
                # Generate from domain and path
                domain = parsed.netloc.replace('.', '_')
                path_clean = re.sub(r'[^a-zA-Z0-9]', '_', parsed.path)
                filename = f"EA_{domain}_{path_clean}.pdf"
        
        return self.sanitize_filename(filename)
    
    def extract_fwc_document_id(self, url: str) -> Optional[str]:
        """
        Extract FWC document ID from FWC URLs.
        
        Args:
            url: FWC URL
            
        Returns:
            Document ID if found
        """
        # Common FWC patterns
        patterns = [
            r'fwc\.gov\.au.*?/([A-Z]{2}\d{6})',  # MA123456 format
            r'document[/_-]([A-Z]{2}\d{6})',
            r'agreement[/_-]([A-Z]{2}\d{6})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        return None
    
    async def download_pdf(self, url: str, output_dir: Path) -> DownloadResult:
        """
        Download a single PDF from URL.
        
        Args:
            url: PDF URL
            output_dir: Output directory
            
        Returns:
            Download result
        """
        start_time = time.time()
        result = DownloadResult(url=url)
        
        try:
            # Generate filename
            filename = self.extract_filename_from_url(url)
            
            # Add FWC ID if available
            fwc_id = self.extract_fwc_document_id(url)
            if fwc_id:
                name_part, ext = filename.rsplit('.', 1)
                filename = f"{fwc_id}_{name_part}.{ext}"
            
            output_path = output_dir / filename
            
            # Skip if file already exists
            if output_path.exists():
                result.local_path = output_path
                result.success = True
                result.file_size_bytes = output_path.stat().st_size
                result.error_message = "File already exists (skipped)"
                self.logger.debug(f"Skipping existing file: {filename}")
                return result
            
            # Download the file
            self.logger.debug(f"Downloading: {url} -> {filename}")
            
            async with self.client.stream('GET', url) as response:
                response.raise_for_status()
                
                # Check content type
                content_type = response.headers.get('content-type', '').lower()
                if 'pdf' not in content_type and 'octet-stream' not in content_type:
                    raise ValueError(f"Not a PDF file: content-type={content_type}")
                
                # Write file
                with open(output_path, 'wb') as f:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        f.write(chunk)
                
                result.local_path = output_path
                result.success = True
                result.file_size_bytes = output_path.stat().st_size
                
                self.logger.debug(f"Downloaded {result.file_size_bytes} bytes to {filename}")
                
        except httpx.HTTPStatusError as e:
            result.error_message = f"HTTP {e.response.status_code}: {e.response.reason_phrase}"
            self.logger.warning(f"HTTP error downloading {url}: {result.error_message}")
            
        except httpx.RequestError as e:
            result.error_message = f"Request error: {str(e)}"
            self.logger.warning(f"Request error downloading {url}: {result.error_message}")
            
        except Exception as e:
            result.error_message = f"Unexpected error: {str(e)}"
            self.logger.error(f"Unexpected error downloading {url}: {result.error_message}")
        
        finally:
            result.download_time_seconds = time.time() - start_time
        
        return result
    
    async def batch_download(
        self,
        urls: List[str],
        output_dir: Path,
        max_concurrent: int = 3,
        delay_seconds: float = 1.0
    ) -> List[DownloadResult]:
        """
        Download multiple PDFs concurrently with rate limiting.
        
        Args:
            urls: List of PDF URLs
            output_dir: Output directory
            max_concurrent: Maximum concurrent downloads
            delay_seconds: Delay between requests
            
        Returns:
            List of download results
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def download_with_limit(url: str) -> DownloadResult:
            async with semaphore:
                result = await self.download_pdf(url, output_dir)
                # Rate limiting
                if delay_seconds > 0:
                    await asyncio.sleep(delay_seconds)
                return result
        
        # Create HTTP client
        self.client = self._create_http_client()
        
        try:
            # Download all URLs
            results = await asyncio.gather(
                *[download_with_limit(url) for url in urls],
                return_exceptions=True
            )
            
            # Convert exceptions to failed results
            final_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    final_results.append(DownloadResult(
                        url=urls[i],
                        success=False,
                        error_message=str(result)
                    ))
                else:
                    final_results.append(result)
            
            return final_results
            
        finally:
            await self.client.aclose()
    
    def load_urls_from_csv(self, csv_file: Path, url_column: str = 'url') -> List[Dict[str, str]]:
        """
        Load URLs from CSV file.
        
        Args:
            csv_file: Path to CSV file
            url_column: Name of column containing URLs
            
        Returns:
            List of row dictionaries
        """
        try:
            df = pd.read_csv(csv_file)
            
            if url_column not in df.columns:
                available_cols = ', '.join(df.columns)
                raise ValueError(f"URL column '{url_column}' not found. Available columns: {available_cols}")
            
            # Filter out empty URLs
            df = df.dropna(subset=[url_column])
            df = df[df[url_column].str.strip() != '']
            
            self.logger.info(f"Loaded {len(df)} URLs from {csv_file}")
            
            return df.to_dict('records')
            
        except Exception as e:
            self.logger.error(f"Failed to load CSV file {csv_file}: {e}")
            raise
    
    async def import_from_csv(
        self,
        csv_file: Path,
        output_dir: Path,
        url_column: str = 'url',
        max_concurrent: int = 3,
        delay_seconds: float = 1.0,
        max_files: Optional[int] = None
    ) -> BatchImportResult:
        """
        Import PDFs from CSV file containing URLs.
        
        Args:
            csv_file: Path to CSV file
            output_dir: Output directory for PDFs
            url_column: Name of column containing URLs
            max_concurrent: Maximum concurrent downloads
            delay_seconds: Delay between requests
            max_files: Maximum number of files to download
            
        Returns:
            Batch import result
        """
        start_time = time.time()
        
        self.logger.info(f"Starting batch import from {csv_file}")
        
        # Load URLs from CSV
        rows = self.load_urls_from_csv(csv_file, url_column)
        
        if max_files:
            rows = rows[:max_files]
        
        urls = [row[url_column] for row in rows]
        
        self.logger.info(f"Downloading {len(urls)} PDFs to {output_dir}")
        
        # Download PDFs
        download_results = await self.batch_download(
            urls=urls,
            output_dir=output_dir,
            max_concurrent=max_concurrent,
            delay_seconds=delay_seconds
        )
        
        # Calculate statistics
        downloaded = sum(1 for r in download_results if r.success and r.error_message != "File already exists (skipped)")
        failed = sum(1 for r in download_results if not r.success)
        skipped = sum(1 for r in download_results if r.success and r.error_message == "File already exists (skipped)")
        
        result = BatchImportResult(
            total_urls=len(urls),
            downloaded=downloaded,
            failed=failed,
            skipped=skipped,
            download_results=download_results,
            execution_time_seconds=time.time() - start_time
        )
        
        self.logger.info(f"Batch import completed: {downloaded} downloaded, {failed} failed, {skipped} skipped")
        
        return result
    
    def save_import_report(self, result: BatchImportResult, output_file: Path):
        """
        Save batch import report to file.
        
        Args:
            result: Batch import result
            output_file: Output file path
        """
        try:
            # Convert results to DataFrame
            data = []
            for download_result in result.download_results:
                data.append(download_result.to_dict())
            
            df = pd.DataFrame(data)
            
            # Add summary at the top
            summary_data = {
                'url': 'SUMMARY',
                'local_path': f"{result.downloaded} downloaded, {result.failed} failed, {result.skipped} skipped",
                'success': result.success_rate,
                'error_message': f"Total: {result.total_urls}",
                'file_size_bytes': sum(r.file_size_bytes for r in result.download_results),
                'download_time_seconds': result.execution_time_seconds,
            }
            
            summary_df = pd.DataFrame([summary_data])
            final_df = pd.concat([summary_df, df], ignore_index=True)
            
            # Save to CSV
            final_df.to_csv(output_file, index=False)
            
            self.logger.info(f"Import report saved to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save import report: {e}")


def create_csv_importer() -> CSVBatchImporter:
    """Factory function to create a CSV batch importer."""
    return CSVBatchImporter()


# CLI command integration
async def cli_csv_import(
    csv_file: Path,
    output_dir: Optional[Path] = None,
    url_column: str = 'url',
    max_concurrent: int = 3,
    delay_seconds: float = 1.0,
    max_files: Optional[int] = None
):
    """
    CLI command for CSV batch import.
    
    Args:
        csv_file: Path to CSV file
        output_dir: Output directory (defaults to raw EAs directory)
        url_column: Name of column containing URLs
        max_concurrent: Maximum concurrent downloads
        delay_seconds: Delay between requests (seconds)
        max_files: Maximum number of files to download
    """
    settings = get_settings()
    
    if output_dir is None:
        output_dir = settings.raw_eas_dir
    
    importer = create_csv_importer()
    
    result = await importer.import_from_csv(
        csv_file=csv_file,
        output_dir=output_dir,
        url_column=url_column,
        max_concurrent=max_concurrent,
        delay_seconds=delay_seconds,
        max_files=max_files
    )
    
    # Save report
    report_file = settings.reports_dir / "csv_imports" / f"import_{csv_file.stem}.csv"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    importer.save_import_report(result, report_file)
    
    return result