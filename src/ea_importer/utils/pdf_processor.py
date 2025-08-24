"""
PDF processing utilities for text extraction and OCR.
"""

import io
import os
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass

import PyPDF2
import pdfplumber
from pdf2image import convert_from_path, convert_from_bytes
import pytesseract
from PIL import Image
import cv2
import numpy as np

from ..core.config import get_settings
from ..core.logging import LoggerMixin


@dataclass
class PDFPage:
    """Represents a single page from a PDF."""
    page_number: int
    text: str
    bbox: Optional[Tuple[float, float, float, float]] = None  # (x0, y0, x1, y1)
    has_images: bool = False
    tables: List[Dict] = None
    
    def __post_init__(self):
        if self.tables is None:
            self.tables = []


@dataclass
class PDFDocument:
    """Represents a complete PDF document."""
    file_path: Path
    pages: List[PDFPage]
    metadata: Dict
    total_pages: int
    
    @property
    def full_text(self) -> str:
        """Get all text concatenated."""
        return "\n\n".join(page.text for page in self.pages if page.text.strip())
    
    @property
    def page_texts(self) -> List[str]:
        """Get list of page texts."""
        return [page.text for page in self.pages]


class PDFProcessor(LoggerMixin):
    """Handles PDF text extraction and OCR operations."""
    
    def __init__(self):
        self.settings = get_settings()
        self._setup_tesseract()
    
    def _setup_tesseract(self):
        """Configure Tesseract OCR."""
        if hasattr(pytesseract, 'pytesseract'):
            if os.path.exists(self.settings.tesseract_cmd):
                pytesseract.pytesseract.tesseract_cmd = self.settings.tesseract_cmd
            else:
                self.logger.warning(f"Tesseract not found at {self.settings.tesseract_cmd}")
    
    def extract_text_pypdf2(self, file_path: Path) -> List[str]:
        """
        Extract text using PyPDF2 (fast but limited).
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of page texts
        """
        texts = []
        
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(reader.pages):
                    try:
                        text = page.extract_text()
                        texts.append(text or "")
                    except Exception as e:
                        self.logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                        texts.append("")
                        
        except Exception as e:
            self.logger.error(f"Failed to read PDF with PyPDF2: {e}")
            raise
        
        return texts
    
    def extract_text_pdfplumber(self, file_path: Path) -> Tuple[List[PDFPage], Dict]:
        """
        Extract text and metadata using pdfplumber (more accurate).
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Tuple of (list of PDFPage objects, metadata dict)
        """
        pages = []
        metadata = {}
        
        try:
            with pdfplumber.open(file_path) as pdf:
                # Extract metadata
                metadata = {
                    'title': getattr(pdf.metadata, 'get', lambda x, y: y)('Title', ''),
                    'author': getattr(pdf.metadata, 'get', lambda x, y: y)('Author', ''),
                    'subject': getattr(pdf.metadata, 'get', lambda x, y: y)('Subject', ''),
                    'creator': getattr(pdf.metadata, 'get', lambda x, y: y)('Creator', ''),
                    'producer': getattr(pdf.metadata, 'get', lambda x, y: y)('Producer', ''),
                    'creation_date': getattr(pdf.metadata, 'get', lambda x, y: y)('CreationDate', ''),
                    'modification_date': getattr(pdf.metadata, 'get', lambda x, y: y)('ModDate', ''),
                }
                
                for page_num, page in enumerate(pdf.pages):
                    try:
                        # Extract text
                        text = page.extract_text() or ""
                        
                        # Extract tables
                        tables = []
                        try:
                            page_tables = page.extract_tables()
                            if page_tables:
                                for table in page_tables:
                                    if table:  # Skip empty tables
                                        tables.append({
                                            'data': table,
                                            'bbox': None  # pdfplumber doesn't provide bbox by default
                                        })
                        except Exception as e:
                            self.logger.debug(f"No tables found on page {page_num + 1}: {e}")
                        
                        # Check for images
                        has_images = len(page.images) > 0 if hasattr(page, 'images') else False
                        
                        pdf_page = PDFPage(
                            page_number=page_num + 1,
                            text=text,
                            bbox=None,
                            has_images=has_images,
                            tables=tables
                        )
                        pages.append(pdf_page)
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to extract from page {page_num + 1}: {e}")
                        # Create empty page
                        pdf_page = PDFPage(
                            page_number=page_num + 1,
                            text="",
                            tables=[]
                        )
                        pages.append(pdf_page)
                        
        except Exception as e:
            self.logger.error(f"Failed to read PDF with pdfplumber: {e}")
            raise
        
        return pages, metadata
    
    def preprocess_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for better OCR results.
        
        Args:
            image: PIL Image
            
        Returns:
            Preprocessed PIL Image
        """
        # Convert PIL to OpenCV format
        img_array = np.array(image)
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply preprocessing
        # 1. Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (1, 1), 0)
        
        # 2. Adaptive thresholding for better contrast
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # 3. Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Convert back to PIL
        return Image.fromarray(cleaned)
    
    def ocr_image(self, image: Image.Image, preprocess: bool = True) -> str:
        """
        Perform OCR on an image.
        
        Args:
            image: PIL Image
            preprocess: Whether to preprocess the image
            
        Returns:
            Extracted text
        """
        try:
            if preprocess:
                image = self.preprocess_image_for_ocr(image)
            
            # Configure Tesseract
            custom_config = r'--oem 3 --psm 6 -l ' + self.settings.ocr_language
            
            text = pytesseract.image_to_string(image, config=custom_config)
            return text.strip()
            
        except Exception as e:
            self.logger.error(f"OCR failed: {e}")
            return ""
    
    def extract_text_with_ocr(self, file_path: Path, dpi: int = None) -> List[str]:
        """
        Extract text using OCR on PDF pages converted to images.
        
        Args:
            file_path: Path to PDF file
            dpi: DPI for image conversion (default from settings)
            
        Returns:
            List of page texts
        """
        dpi = dpi or self.settings.ocr_dpi
        texts = []
        
        try:
            self.logger.info(f"Converting PDF to images at {dpi} DPI...")
            
            # Convert PDF to images
            images = convert_from_path(str(file_path), dpi=dpi)
            
            self.logger.info(f"Performing OCR on {len(images)} pages...")
            
            for page_num, image in enumerate(images):
                try:
                    text = self.ocr_image(image)
                    texts.append(text)
                    self.logger.debug(f"OCR completed for page {page_num + 1}")
                except Exception as e:
                    self.logger.warning(f"OCR failed for page {page_num + 1}: {e}")
                    texts.append("")
                    
        except Exception as e:
            self.logger.error(f"Failed to convert PDF to images: {e}")
            raise
        
        return texts
    
    def has_text_layer(self, file_path: Path, min_text_ratio: float = 0.1) -> bool:
        """
        Check if PDF has sufficient extractable text.
        
        Args:
            file_path: Path to PDF file
            min_text_ratio: Minimum ratio of pages with text to consider as having text layer
            
        Returns:
            True if PDF has text layer
        """
        try:
            texts = self.extract_text_pypdf2(file_path)
            
            # Count pages with meaningful text (more than just whitespace/numbers)
            pages_with_text = 0
            for text in texts:
                # Remove whitespace and common non-content characters
                clean_text = ''.join(c for c in text if c.isalpha())
                if len(clean_text) > 50:  # At least 50 alphabetic characters
                    pages_with_text += 1
            
            text_ratio = pages_with_text / len(texts) if texts else 0
            has_text = text_ratio >= min_text_ratio
            
            self.logger.debug(f"Text layer check: {pages_with_text}/{len(texts)} pages "
                            f"have text ({text_ratio:.2%}), threshold: {min_text_ratio:.2%}")
            
            return has_text
            
        except Exception as e:
            self.logger.warning(f"Failed to check text layer: {e}")
            return False
    
    def process_pdf(self, file_path: Path, force_ocr: bool = False) -> PDFDocument:
        """
        Process a PDF file, automatically choosing between text extraction and OCR.
        
        Args:
            file_path: Path to PDF file
            force_ocr: Force OCR even if text layer exists
            
        Returns:
            PDFDocument object
        """
        self.logger.info(f"Processing PDF: {file_path}")
        
        # Check if file exists
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        try:
            # First try pdfplumber for text and metadata
            pages, metadata = self.extract_text_pdfplumber(file_path)
            
            # Check if we need OCR
            needs_ocr = force_ocr or not self.has_text_layer(file_path)
            
            if needs_ocr:
                self.logger.info("Performing OCR due to insufficient text layer")
                ocr_texts = self.extract_text_with_ocr(file_path)
                
                # Update pages with OCR text where original text is poor
                for i, (page, ocr_text) in enumerate(zip(pages, ocr_texts)):
                    original_clean = ''.join(c for c in page.text if c.isalpha())
                    ocr_clean = ''.join(c for c in ocr_text if c.isalpha())
                    
                    # Use OCR text if it's significantly longer or original is very short
                    if len(ocr_clean) > len(original_clean) * 1.5 or len(original_clean) < 50:
                        pages[i].text = ocr_text
                        self.logger.debug(f"Page {i+1}: Used OCR text ({len(ocr_clean)} vs {len(original_clean)} chars)")
            
            # Create document
            document = PDFDocument(
                file_path=file_path,
                pages=pages,
                metadata=metadata,
                total_pages=len(pages)
            )
            
            self.logger.info(f"Successfully processed PDF: {len(pages)} pages, "
                           f"{len(document.full_text)} total characters")
            
            return document
            
        except Exception as e:
            self.logger.error(f"Failed to process PDF {file_path}: {e}")
            raise


def create_pdf_processor() -> PDFProcessor:
    """Factory function to create a PDF processor."""
    return PDFProcessor()