"""
Text Cleaning and Normalization utility for EA Importer.

Handles comprehensive text cleaning for extracted PDF content:
- Header/footer removal
- Hyphenation repair
- Whitespace normalization
- Quote mark standardization
- Page break preservation
- Special character handling
- Legal document formatting preservation

Designed specifically for Australian Enterprise Agreement documents.
"""

import re
import logging
import unicodedata
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path

from ..core.logging import get_logger, log_function_call
from ..models import PDFDocument, PDFPage

logger = get_logger(__name__)


class TextCleaningError(Exception):
    """Custom exception for text cleaning errors"""
    pass


class TextCleaner:
    """
    Advanced text cleaning and normalization for legal documents.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize text cleaner.
        
        Args:
            config: Optional configuration override
        """
        self.config = config or {}
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
        
        # Common headers/footers in EA documents
        self.common_headers = self._load_common_headers()
        self.common_footers = self._load_common_footers()
    
    def _compile_patterns(self):
        """Compile commonly used regex patterns"""
        
        # Hyphenation patterns
        self.hyphen_patterns = {
            # End-of-line hyphenation (word split across lines)
            'line_break': re.compile(r'(\w+)-\s*\n\s*(\w+)', re.MULTILINE),
            
            # Common hyphenated words that shouldn't be joined
            'keep_hyphen': re.compile(r'\b(part-time|full-time|co-worker|ex-employee|multi-\w+|non-\w+|pre-\w+|post-\w+|self-\w+|anti-\w+)\b', re.IGNORECASE)
        }
        
        # Whitespace patterns
        self.whitespace_patterns = {
            # Multiple spaces
            'multiple_spaces': re.compile(r' {2,}'),
            
            # Multiple newlines (preserve up to 2)
            'multiple_newlines': re.compile(r'\n{3,}'),
            
            # Tab characters
            'tabs': re.compile(r'\t'),
            
            # Trailing whitespace
            'trailing': re.compile(r'[ \t]+$', re.MULTILINE),
            
            # Leading whitespace (preserve some for indentation)
            'excessive_leading': re.compile(r'^\s{5,}', re.MULTILINE)
        }
        
        # Quote patterns
        self.quote_patterns = {
            # Smart quotes to regular quotes
            'smart_single': re.compile(r'[\u2018\u2019]'),  # ' '
            'smart_double': re.compile(r'[\u201c\u201d]'),  # " "
            
            # Other quote-like characters
            'backtick': re.compile(r'`'),
            'prime': re.compile(r'[\u2032\u2033]')  # ′ ″
        }
        
        # Page number patterns
        self.page_patterns = {
            'page_number': re.compile(r'^\s*(?:page\s+)?\d+\s*(?:of\s+\d+)?\s*$', re.IGNORECASE | re.MULTILINE),
            'page_footer': re.compile(r'^\s*-?\s*\d+\s*-?\s*$', re.MULTILINE)
        }
        
        # Header/footer patterns
        self.header_footer_patterns = {
            'document_title': re.compile(r'^.{0,5}(ENTERPRISE AGREEMENT|CERTIFIED AGREEMENT|AWARD).{0,50}$', re.IGNORECASE | re.MULTILINE),
            'date_line': re.compile(r'^\s*(?:Date|Updated|Version|Effective).*\d{4}\s*$', re.IGNORECASE | re.MULTILINE),
            'copyright': re.compile(r'©|\(c\)|copyright|proprietary|confidential', re.IGNORECASE),
            'url_line': re.compile(r'^\s*(?:https?://|www\.)\S+\s*$', re.MULTILINE),
            'email_line': re.compile(r'^\s*\S+@\S+\.\S+\s*$', re.MULTILINE)
        }
        
        # Legal reference patterns
        self.legal_patterns = {
            'section_ref': re.compile(r'\b(?:section|clause|part|schedule|appendix)\s+\d+(?:\.\d+)*\b', re.IGNORECASE),
            'act_reference': re.compile(r'\b\w+\s+Act\s+\d{4}\b'),
            'regulation_ref': re.compile(r'\b\w+\s+Regulation\s+\d{4}\b')
        }
        
        # Special character patterns
        self.special_char_patterns = {
            'bullet_points': re.compile(r'^[\s]*[•·▪▫◦‣⁃]\s*', re.MULTILINE),
            'em_dash': re.compile(r'—'),
            'en_dash': re.compile(r'–'),
            'ellipsis': re.compile(r'…'),
            'non_breaking_space': re.compile(r'\u00a0')  # Non-breaking space
        }
    
    def _load_common_headers(self) -> Set[str]:
        """Load common header patterns found in EA documents"""
        return {
            'fair work commission',
            'fair work australia', 
            'australian industrial relations commission',
            'enterprise agreement',
            'certified agreement',
            'workplace agreement',
            'page',
            'document'
        }
    
    def _load_common_footers(self) -> Set[str]:
        """Load common footer patterns found in EA documents"""
        return {
            'page',
            'confidential',
            'proprietary',
            'copyright',
            '©',
            'draft',
            'version'
        }
    
    @log_function_call
    def clean_document(self, document: PDFDocument) -> PDFDocument:
        """
        Clean all pages in a PDF document.
        
        Args:
            document: PDFDocument to clean
            
        Returns:
            PDFDocument with cleaned text
        """
        logger.info(f"Cleaning document with {len(document.pages)} pages")
        
        cleaned_pages = []
        
        for page_num, page in enumerate(document.pages):
            try:
                cleaned_page = self.clean_page(page, document_context={
                    'total_pages': len(document.pages),
                    'page_number': page_num + 1,
                    'file_path': document.file_path
                })
                cleaned_pages.append(cleaned_page)
                
            except Exception as e:
                logger.warning(f"Failed to clean page {page_num + 1}: {e}")
                # Keep original page if cleaning fails
                cleaned_pages.append(page)
        
        # Create new document with cleaned pages
        cleaned_document = PDFDocument(
            file_path=document.file_path,
            pages=cleaned_pages,
            metadata={
                **document.metadata,
                'text_cleaned': True,
                'cleaning_stats': self._calculate_cleaning_stats(document.pages, cleaned_pages)
            }
        )
        
        logger.info("Document cleaning completed")
        return cleaned_document
    
    def clean_page(self, page: PDFPage, document_context: Optional[Dict[str, Any]] = None) -> PDFPage:
        """
        Clean text on a single page.
        
        Args:
            page: PDFPage to clean
            document_context: Optional context about the document
            
        Returns:
            PDFPage with cleaned text
        """
        if not page.text.strip():
            return page  # Nothing to clean
        
        original_text = page.text
        text = original_text
        
        # Step 1: Normalize Unicode characters
        text = self._normalize_unicode(text)
        
        # Step 2: Remove headers and footers
        text = self._remove_headers_footers(text, document_context)
        
        # Step 3: Fix hyphenation
        text = self._fix_hyphenation(text)
        
        # Step 4: Normalize quotes
        text = self._normalize_quotes(text)
        
        # Step 5: Clean whitespace
        text = self._normalize_whitespace(text)
        
        # Step 6: Handle special characters
        text = self._normalize_special_characters(text)
        
        # Step 7: Preserve legal structure
        text = self._preserve_legal_structure(text)
        
        # Step 8: Final cleanup
        text = self._final_cleanup(text)
        
        # Create cleaned page
        cleaned_page = PDFPage(
            page_number=page.page_number,
            text=text,
            bbox=page.bbox,
            has_images=page.has_images,
            tables=page.tables  # Keep table data unchanged
        )
        
        return cleaned_page
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters to standard forms"""
        # Normalize to NFKC form (canonical decomposition, then canonical composition)
        text = unicodedata.normalize('NFKC', text)
        
        # Remove or replace problematic Unicode characters
        replacements = {
            '\ufeff': '',  # Byte order mark
            '\u200b': '',  # Zero width space
            '\u200c': '',  # Zero width non-joiner
            '\u200d': '',  # Zero width joiner
            '\u2060': '',  # Word joiner
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def _remove_headers_footers(self, text: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Remove common headers and footers"""
        lines = text.split('\n')
        cleaned_lines = []
        
        for i, line in enumerate(lines):
            line_clean = line.strip().lower()
            
            # Skip empty lines
            if not line_clean:
                cleaned_lines.append(line)
                continue
            
            # Check for page numbers
            if self.page_patterns['page_number'].match(line) or self.page_patterns['page_footer'].match(line):
                continue
            
            # Check for common headers (first few lines)
            if i < 3 and any(header in line_clean for header in self.common_headers):
                continue
            
            # Check for common footers (last few lines)
            if i >= len(lines) - 3 and any(footer in line_clean for footer in self.common_footers):
                continue
            
            # Check for document metadata lines
            if (self.header_footer_patterns['date_line'].match(line) or
                self.header_footer_patterns['url_line'].match(line) or
                self.header_footer_patterns['email_line'].match(line)):
                continue
            
            # Keep the line
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _fix_hyphenation(self, text: str) -> str:
        """Fix hyphenation issues from PDF extraction"""
        
        # First, protect legitimate hyphenated words
        protected_words = []
        for match in self.hyphen_patterns['keep_hyphen'].finditer(text):
            protected_words.append(match.group())
        
        # Fix line-break hyphenation
        def fix_line_hyphen(match):
            word1, word2 = match.groups()
            combined = word1 + word2
            
            # Don't combine if it would create a protected word incorrectly
            if any(protected in combined.lower() for protected in ['part-time', 'full-time']):
                return match.group(0)  # Keep original
            
            return combined
        
        text = self.hyphen_patterns['line_break'].sub(fix_line_hyphen, text)
        
        return text
    
    def _normalize_quotes(self, text: str) -> str:
        """Normalize quote characters to standard ASCII"""
        
        # Convert smart quotes
        text = self.quote_patterns['smart_single'].sub("'", text)
        text = self.quote_patterns['smart_double'].sub('"', text)
        
        # Convert other quote-like characters
        text = self.quote_patterns['backtick'].sub("'", text)
        text = self.quote_patterns['prime'].sub("'", text)
        
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace while preserving document structure"""
        
        # Convert tabs to spaces
        text = self.whitespace_patterns['tabs'].sub('    ', text)
        
        # Remove trailing whitespace
        text = self.whitespace_patterns['trailing'].sub('', text)
        
        # Normalize multiple spaces (but preserve some indentation)
        text = self.whitespace_patterns['multiple_spaces'].sub(' ', text)
        
        # Limit excessive leading whitespace (preserve some for structure)
        text = self.whitespace_patterns['excessive_leading'].sub('    ', text)
        
        # Limit multiple newlines (preserve paragraph breaks)
        text = self.whitespace_patterns['multiple_newlines'].sub('\n\n', text)
        
        return text
    
    def _normalize_special_characters(self, text: str) -> str:
        """Normalize special characters"""
        
        # Replace non-breaking spaces with regular spaces
        text = self.special_char_patterns['non_breaking_space'].sub(' ', text)
        
        # Normalize dashes
        text = self.special_char_patterns['em_dash'].sub(' - ', text)
        text = self.special_char_patterns['en_dash'].sub(' - ', text)
        
        # Replace ellipsis
        text = self.special_char_patterns['ellipsis'].sub('...', text)
        
        # Standardize bullet points
        text = self.special_char_patterns['bullet_points'].sub('• ', text)
        
        return text
    
    def _preserve_legal_structure(self, text: str) -> str:
        """Preserve important legal document structure"""
        
        # Ensure proper spacing around section references
        def fix_section_ref(match):
            return ' ' + match.group().strip() + ' '
        
        text = self.legal_patterns['section_ref'].sub(fix_section_ref, text)
        
        # Preserve legal references
        text = re.sub(r'\b(Fair Work Act 2009|Industrial Relations Act|Workplace Relations Act)\b', 
                     lambda m: ' ' + m.group() + ' ', text)
        
        return text
    
    def _final_cleanup(self, text: str) -> str:
        """Final cleanup and validation"""
        
        # Remove excessive whitespace
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Ensure text ends with newline
        text = text.rstrip() + '\n'
        
        # Remove any remaining problematic characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        
        return text
    
    def _calculate_cleaning_stats(self, original_pages: List[PDFPage], cleaned_pages: List[PDFPage]) -> Dict[str, Any]:
        """Calculate statistics about the cleaning process"""
        
        original_text = '\n'.join(page.text for page in original_pages)
        cleaned_text = '\n'.join(page.text for page in cleaned_pages)
        
        stats = {
            'original_length': len(original_text),
            'cleaned_length': len(cleaned_text),
            'length_reduction': len(original_text) - len(cleaned_text),
            'length_reduction_percent': ((len(original_text) - len(cleaned_text)) / len(original_text) * 100) if original_text else 0,
            
            'original_lines': original_text.count('\n'),
            'cleaned_lines': cleaned_text.count('\n'),
            
            'original_words': len(original_text.split()),
            'cleaned_words': len(cleaned_text.split()),
        }
        
        return stats
    
    def clean_text(self, text: str, preserve_structure: bool = True) -> str:
        """
        Clean raw text without page context.
        
        Args:
            text: Raw text to clean
            preserve_structure: Whether to preserve document structure
            
        Returns:
            Cleaned text
        """
        if not text.strip():
            return text
        
        # Apply basic cleaning steps
        text = self._normalize_unicode(text)
        text = self._fix_hyphenation(text)
        text = self._normalize_quotes(text)
        text = self._normalize_whitespace(text)
        text = self._normalize_special_characters(text)
        
        if preserve_structure:
            text = self._preserve_legal_structure(text)
        
        text = self._final_cleanup(text)
        
        return text
    
    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """
        Extract metadata from cleaned text.
        
        Args:
            text: Cleaned text
            
        Returns:
            Extracted metadata
        """
        metadata = {
            'length': len(text),
            'word_count': len(text.split()),
            'line_count': text.count('\n'),
            'paragraph_count': text.count('\n\n'),
            'has_legal_references': bool(self.legal_patterns['section_ref'].search(text)),
            'has_act_references': bool(self.legal_patterns['act_reference'].search(text)),
        }
        
        return metadata


# Utility functions for batch text cleaning
def clean_text_batch(texts: List[str], **kwargs) -> List[str]:
    """
    Clean multiple text strings in batch.
    
    Args:
        texts: List of text strings to clean
        **kwargs: Arguments passed to TextCleaner
        
    Returns:
        List of cleaned text strings
    """
    cleaner = TextCleaner(**kwargs)
    return [cleaner.clean_text(text) for text in texts]


def clean_document_batch(documents: List[PDFDocument], **kwargs) -> List[PDFDocument]:
    """
    Clean multiple PDF documents in batch.
    
    Args:
        documents: List of PDFDocuments to clean
        **kwargs: Arguments passed to TextCleaner
        
    Returns:
        List of cleaned PDFDocuments
    """
    cleaner = TextCleaner(**kwargs)
    return [cleaner.clean_document(doc) for doc in documents]


# Export main classes
__all__ = [
    'TextCleaner',
    'TextCleaningError',
    'clean_text_batch',
    'clean_document_batch',
]