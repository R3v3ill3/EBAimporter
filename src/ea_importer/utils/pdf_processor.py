"""
PDF Processing utility for EA Importer.

Handles PDF text extraction with multiple fallback strategies:
1. Direct text extraction from PDF text layer
2. OCR processing for scanned documents
3. Quality validation and error handling

Supports multiple PDF libraries and OCR engines for maximum compatibility.
"""

import os
import io
import logging
import hashlib
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
import time

# PDF processing libraries
try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

# OCR and image processing
try:
    import pytesseract
    from PIL import Image
    import cv2
    import numpy as np
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

try:
    from pdf2image import convert_from_path
    HAS_PDF2IMAGE = True
except ImportError:
    HAS_PDF2IMAGE = False

from ..core.config import get_settings
from ..core.logging import get_logger, log_function_call
from ..models import PDFDocument, PDFPage

logger = get_logger(__name__)


class PDFProcessingError(Exception):
    """Custom exception for PDF processing errors"""
    pass


class OCRNotAvailableError(PDFProcessingError):
    """Raised when OCR is requested but not available"""
    pass


class PDFProcessor:
    """
    Advanced PDF processor with multiple extraction strategies and OCR fallback.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize PDF processor.
        
        Args:
            config: Optional configuration override
        """
        self.settings = get_settings()
        self.config = config or {}
        
        # OCR configuration
        self.ocr_dpi = self.config.get('ocr_dpi', self.settings.ocr.dpi)
        self.ocr_language = self.config.get('ocr_language', self.settings.ocr.language)
        self.min_text_ratio = self.config.get('min_text_ratio', self.settings.ocr.min_text_ratio)
        self.ocr_timeout = self.config.get('ocr_timeout', self.settings.ocr.timeout)
        
        # Processing limits
        self.max_file_size = self.config.get('max_file_size_mb', self.settings.processing.max_file_size_mb)
        
        # Verify dependencies
        self._check_dependencies()
    
    def _check_dependencies(self) -> None:
        """Check and log available dependencies"""
        available = []
        missing = []
        
        if HAS_PYPDF2:
            available.append("PyPDF2")
        else:
            missing.append("PyPDF2")
            
        if HAS_PDFPLUMBER:
            available.append("pdfplumber")
        else:
            missing.append("pdfplumber")
            
        if HAS_PYMUPDF:
            available.append("PyMuPDF")
        else:
            missing.append("PyMuPDF")
            
        if HAS_OCR:
            available.append("OCR (pytesseract)")
        else:
            missing.append("OCR (pytesseract)")
            
        if HAS_PDF2IMAGE:
            available.append("pdf2image")
        else:
            missing.append("pdf2image")
        
        logger.info(f"PDF processing dependencies - Available: {available}")
        if missing:
            logger.warning(f"Missing optional dependencies: {missing}")
    
    @log_function_call
    def process_pdf(
        self, 
        file_path: Union[str, Path], 
        force_ocr: bool = False,
        ea_id: Optional[str] = None
    ) -> PDFDocument:
        """
        Process a PDF file with automatic fallback strategies.
        
        Args:
            file_path: Path to PDF file
            force_ocr: Force OCR even if text layer exists
            ea_id: Optional EA identifier
            
        Returns:
            PDFDocument with extracted content
            
        Raises:
            PDFProcessingError: If processing fails
        """
        file_path = Path(file_path)
        
        # Validate file
        self._validate_file(file_path)
        
        # Generate EA ID if not provided
        if not ea_id:
            ea_id = self._generate_ea_id(file_path)
        
        logger.info(f"Processing PDF: {file_path} (EA ID: {ea_id})")
        start_time = time.time()
        
        try:
            # Attempt text extraction first (unless forced OCR)
            pages = []
            metadata = {}
            ocr_used = False
            
            if not force_ocr and self.has_text_layer(file_path):
                logger.debug("PDF has text layer, attempting direct extraction")
                try:
                    pages, metadata = self._extract_text_layer(file_path)
                    logger.info(f"Successfully extracted text from {len(pages)} pages")
                except Exception as e:
                    logger.warning(f"Text layer extraction failed: {e}")
                    pages = []
            
            # Fall back to OCR if needed
            if not pages or force_ocr:
                if not HAS_OCR:
                    raise OCRNotAvailableError("OCR requested but pytesseract not available")
                
                logger.info("Falling back to OCR processing")
                pages = self._extract_with_ocr(file_path)
                ocr_used = True
                logger.info(f"OCR completed for {len(pages)} pages")
            
            # Validate extraction quality
            self._validate_extraction(pages, file_path)
            
            # Create PDFDocument
            document = PDFDocument(
                file_path=file_path,
                pages=pages,
                metadata={
                    **metadata,
                    'ea_id': ea_id,
                    'file_size_bytes': file_path.stat().st_size,
                    'processed_at': datetime.now().isoformat(),
                    'ocr_used': ocr_used,
                    'processing_time_seconds': time.time() - start_time,
                }
            )
            
            logger.info(f"PDF processing completed in {time.time() - start_time:.2f}s")
            return document
            
        except Exception as e:
            logger.error(f"PDF processing failed for {file_path}: {e}")
            raise PDFProcessingError(f"Failed to process PDF {file_path}: {e}")
    
    def _validate_file(self, file_path: Path) -> None:
        """Validate PDF file before processing"""
        if not file_path.exists():
            raise PDFProcessingError(f"File not found: {file_path}")
        
        if not file_path.is_file():
            raise PDFProcessingError(f"Not a file: {file_path}")
        
        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.max_file_size:
            raise PDFProcessingError(
                f"File too large: {file_size_mb:.1f}MB > {self.max_file_size}MB"
            )
        
        # Check file extension
        if file_path.suffix.lower() != '.pdf':
            logger.warning(f"File extension is not .pdf: {file_path}")
    
    def _generate_ea_id(self, file_path: Path) -> str:
        """Generate unique EA ID from file path and content"""
        # Use filename and file hash for ID generation
        file_stem = file_path.stem
        
        # Calculate file hash for uniqueness
        file_hash = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                file_hash.update(chunk)
        
        hash_short = file_hash.hexdigest()[:8]
        return f"EA-{file_stem}-{hash_short}".upper()
    
    def has_text_layer(self, file_path: Path, min_text_ratio: Optional[float] = None) -> bool:
        """
        Check if PDF has a meaningful text layer.
        
        Args:
            file_path: Path to PDF file
            min_text_ratio: Minimum ratio of text characters to consider meaningful
            
        Returns:
            True if PDF has sufficient text layer
        """
        if min_text_ratio is None:
            min_text_ratio = self.min_text_ratio
        
        try:
            # Try with pdfplumber first (most reliable)
            if HAS_PDFPLUMBER:
                with pdfplumber.open(file_path) as pdf:
                    total_chars = 0
                    text_chars = 0
                    
                    # Sample first few pages for efficiency
                    pages_to_check = min(3, len(pdf.pages))
                    
                    for i in range(pages_to_check):
                        page = pdf.pages[i]
                        page_text = page.extract_text() or ""
                        
                        total_chars += len(page_text)
                        text_chars += sum(1 for c in page_text if c.isalnum())
                    
                    if total_chars == 0:
                        return False
                    
                    text_ratio = text_chars / total_chars
                    logger.debug(f"Text layer analysis: {text_ratio:.3f} ratio ({text_chars}/{total_chars})")
                    return text_ratio >= min_text_ratio
            
            # Fallback to PyPDF2
            elif HAS_PYPDF2:
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    
                    if len(reader.pages) == 0:
                        return False
                    
                    # Check first page
                    page = reader.pages[0]
                    text = page.extract_text()
                    
                    return len(text.strip()) > 50  # Arbitrary threshold
            
            else:
                logger.warning("No PDF library available for text layer detection")
                return False
                
        except Exception as e:
            logger.warning(f"Error checking text layer for {file_path}: {e}")
            return False
    
    def _extract_text_layer(self, file_path: Path) -> Tuple[List[PDFPage], Dict[str, Any]]:
        """Extract text from PDF using text layer"""
        pages = []
        metadata = {}
        
        # Try pdfplumber first (best for tables and layout)
        if HAS_PDFPLUMBER:
            pages, metadata = self._extract_with_pdfplumber(file_path)
        elif HAS_PYMUPDF:
            pages, metadata = self._extract_with_pymupdf(file_path)
        elif HAS_PYPDF2:
            pages, metadata = self._extract_with_pypdf2(file_path)
        else:
            raise PDFProcessingError("No PDF processing library available")
        
        return pages, metadata
    
    def _extract_with_pdfplumber(self, file_path: Path) -> Tuple[List[PDFPage], Dict[str, Any]]:
        """Extract text using pdfplumber (preserves layout and tables)"""
        pages = []
        metadata = {}
        
        with pdfplumber.open(file_path) as pdf:
            metadata = {
                'total_pages': len(pdf.pages),
                'metadata': pdf.metadata or {},
                'extraction_method': 'pdfplumber'
            }
            
            for page_num, page in enumerate(pdf.pages):
                # Extract text
                text = page.extract_text() or ""
                
                # Extract tables
                tables = []
                try:
                    page_tables = page.extract_tables()
                    if page_tables:
                        for table in page_tables:
                            tables.append({
                                'rows': table,
                                'bbox': getattr(table, 'bbox', None)
                            })
                except Exception as e:
                    logger.debug(f"Table extraction failed for page {page_num + 1}: {e}")
                
                # Get page dimensions
                bbox = (0, 0, page.width, page.height) if hasattr(page, 'width') else None
                
                # Check for images
                has_images = len(page.images) > 0 if hasattr(page, 'images') else False
                
                pdf_page = PDFPage(
                    page_number=page_num + 1,
                    text=text,
                    bbox=bbox,
                    has_images=has_images,
                    tables=tables
                )
                
                pages.append(pdf_page)
        
        return pages, metadata
    
    def _extract_with_pymupdf(self, file_path: Path) -> Tuple[List[PDFPage], Dict[str, Any]]:
        """Extract text using PyMuPDF"""
        pages = []
        metadata = {}
        
        doc = fitz.open(file_path)
        
        try:
            metadata = {
                'total_pages': len(doc),
                'metadata': doc.metadata,
                'extraction_method': 'pymupdf'
            }
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text
                text = page.get_text()
                
                # Get page dimensions
                rect = page.rect
                bbox = (rect.x0, rect.y0, rect.x1, rect.y1)
                
                # Check for images
                image_list = page.get_images()
                has_images = len(image_list) > 0
                
                # Extract tables (basic)
                tables = []
                try:
                    tabs = page.find_tables()
                    for tab in tabs:
                        table_data = tab.extract()
                        tables.append({
                            'rows': table_data,
                            'bbox': tab.bbox
                        })
                except Exception as e:
                    logger.debug(f"Table extraction failed for page {page_num + 1}: {e}")
                
                pdf_page = PDFPage(
                    page_number=page_num + 1,
                    text=text,
                    bbox=bbox,
                    has_images=has_images,
                    tables=tables
                )
                
                pages.append(pdf_page)
        
        finally:
            doc.close()
        
        return pages, metadata
    
    def _extract_with_pypdf2(self, file_path: Path) -> Tuple[List[PDFPage], Dict[str, Any]]:
        """Extract text using PyPDF2 (basic extraction)"""
        pages = []
        metadata = {}
        
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            
            metadata = {
                'total_pages': len(reader.pages),
                'metadata': reader.metadata or {},
                'extraction_method': 'pypdf2'
            }
            
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                
                # Get basic page info
                media_box = page.mediabox if hasattr(page, 'mediabox') else None
                bbox = None
                if media_box:
                    bbox = (float(media_box[0]), float(media_box[1]), 
                           float(media_box[2]), float(media_box[3]))
                
                pdf_page = PDFPage(
                    page_number=page_num + 1,
                    text=text,
                    bbox=bbox,
                    has_images=False,  # PyPDF2 doesn't easily detect images
                    tables=[]  # No table extraction in PyPDF2
                )
                
                pages.append(pdf_page)
        
        return pages, metadata
    
    def _extract_with_ocr(self, file_path: Path) -> List[PDFPage]:
        """Extract text using OCR"""
        if not HAS_OCR or not HAS_PDF2IMAGE:
            raise OCRNotAvailableError("OCR dependencies not available")
        
        logger.debug(f"Starting OCR processing with DPI={self.ocr_dpi}")
        
        try:
            # Convert PDF to images
            images = convert_from_path(
                file_path,
                dpi=self.ocr_dpi,
                fmt='PNG'
            )
            
            pages = []
            
            for page_num, image in enumerate(images):
                logger.debug(f"OCR processing page {page_num + 1}/{len(images)}")
                
                # Preprocess image for better OCR
                image_array = np.array(image)
                processed_image = self._preprocess_image_for_ocr(image_array)
                
                # Run OCR
                custom_config = f'--oem 3 --psm 6 -l {self.ocr_language}'
                
                try:
                    text = pytesseract.image_to_string(
                        processed_image,
                        config=custom_config,
                        timeout=self.ocr_timeout
                    )
                except Exception as e:
                    logger.warning(f"OCR failed for page {page_num + 1}: {e}")
                    text = ""
                
                # Create PDFPage
                pdf_page = PDFPage(
                    page_number=page_num + 1,
                    text=text,
                    bbox=None,  # No bbox info from OCR
                    has_images=True,  # Scanned document
                    tables=[]  # Table extraction would need additional processing
                )
                
                pages.append(pdf_page)
            
            return pages
            
        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            raise PDFProcessingError(f"OCR processing failed: {e}")
    
    def _preprocess_image_for_ocr(self, image_array: np.ndarray) -> Image.Image:
        """
        Preprocess image for better OCR results.
        
        Args:
            image_array: Input image as numpy array
            
        Returns:
            Preprocessed PIL Image
        """
        # Convert to grayscale
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Convert back to PIL Image
        return Image.fromarray(thresh)
    
    def _validate_extraction(self, pages: List[PDFPage], file_path: Path) -> None:
        """Validate extraction quality"""
        if not pages:
            raise PDFProcessingError("No pages extracted from PDF")
        
        # Check for minimum text content
        total_text_length = sum(len(page.text) for page in pages)
        
        if total_text_length < 100:  # Minimum threshold
            logger.warning(f"Very little text extracted ({total_text_length} chars)")
        
        # Check for reasonable clause count (heuristic)
        total_text = " ".join(page.text for page in pages)
        clause_indicators = total_text.count('.') + total_text.count(';')
        
        min_expected_clauses = self.settings.minimum_clause_count
        if clause_indicators < min_expected_clauses:
            logger.warning(
                f"Low clause count detected: {clause_indicators} < {min_expected_clauses}"
            )
        
        logger.info(f"Extraction validation passed: {len(pages)} pages, {total_text_length} chars")
    
    def get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """
        Get basic information about a PDF file without full processing.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Dictionary with file information
        """
        info = {
            'file_path': str(file_path),
            'file_size_bytes': file_path.stat().st_size,
            'file_size_mb': file_path.stat().st_size / (1024 * 1024),
            'modified_time': datetime.fromtimestamp(file_path.stat().st_mtime),
            'has_text_layer': False,
            'page_count': 0,
            'processing_feasible': True,
            'errors': []
        }
        
        try:
            # Check if file is too large
            if info['file_size_mb'] > self.max_file_size:
                info['processing_feasible'] = False
                info['errors'].append(f"File too large: {info['file_size_mb']:.1f}MB")
            
            # Get page count and text layer info
            if HAS_PDFPLUMBER:
                try:
                    with pdfplumber.open(file_path) as pdf:
                        info['page_count'] = len(pdf.pages)
                        info['has_text_layer'] = self.has_text_layer(file_path)
                except Exception as e:
                    info['errors'].append(f"PDF analysis failed: {e}")
            
        except Exception as e:
            info['processing_feasible'] = False
            info['errors'].append(f"File access failed: {e}")
        
        return info


# Convenience functions for batch processing
def process_pdf_batch(
    pdf_files: List[Path],
    output_dir: Path,
    force_ocr: bool = False,
    max_workers: Optional[int] = None
) -> Dict[str, Any]:
    """
    Process multiple PDFs in batch.
    
    Args:
        pdf_files: List of PDF file paths
        output_dir: Output directory for results
        force_ocr: Force OCR for all files
        max_workers: Maximum number of parallel workers
        
    Returns:
        Batch processing results
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    settings = get_settings()
    max_workers = max_workers or settings.processing.max_workers
    
    processor = PDFProcessor()
    results = {
        'successful': [],
        'failed': [],
        'total_files': len(pdf_files),
        'start_time': datetime.now()
    }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    def process_single_pdf(pdf_path: Path) -> Dict[str, Any]:
        """Process a single PDF and return result info"""
        try:
            document = processor.process_pdf(pdf_path, force_ocr=force_ocr)
            
            # Save extracted text
            text_file = output_dir / "text" / f"{document.metadata['ea_id']}.txt"
            text_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(document.full_text)
            
            return {
                'status': 'success',
                'file_path': str(pdf_path),
                'ea_id': document.metadata['ea_id'],
                'pages': len(document.pages),
                'text_length': len(document.full_text),
                'ocr_used': document.metadata.get('ocr_used', False),
                'processing_time': document.metadata.get('processing_time_seconds', 0)
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'file_path': str(pdf_path),
                'error': str(e)
            }
    
    # Process in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {executor.submit(process_single_pdf, pdf_path): pdf_path 
                         for pdf_path in pdf_files}
        
        for future in as_completed(future_to_path):
            result = future.result()
            
            if result['status'] == 'success':
                results['successful'].append(result)
                logger.info(f"Successfully processed: {result['file_path']}")
            else:
                results['failed'].append(result)
                logger.error(f"Failed to process: {result['file_path']} - {result['error']}")
    
    results['end_time'] = datetime.now()
    results['duration'] = (results['end_time'] - results['start_time']).total_seconds()
    
    logger.info(f"Batch processing completed: {len(results['successful'])}/{len(pdf_files)} successful")
    
    return results


# Export main classes
__all__ = [
    'PDFProcessor',
    'PDFProcessingError',
    'OCRNotAvailableError',
    'process_pdf_batch',
]