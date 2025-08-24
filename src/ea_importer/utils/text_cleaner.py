"""
Text cleaning and normalization utilities for EA documents.
"""

import re
import unicodedata
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from ..core.logging import LoggerMixin


@dataclass
class CleaningStats:
    """Statistics from text cleaning operations."""
    original_length: int
    cleaned_length: int
    lines_removed: int = 0
    hyphenations_fixed: int = 0
    whitespace_normalized: int = 0
    headers_footers_removed: int = 0
    
    @property
    def reduction_ratio(self) -> float:
        """Calculate text reduction ratio."""
        if self.original_length == 0:
            return 0.0
        return (self.original_length - self.cleaned_length) / self.original_length


class TextCleaner(LoggerMixin):
    """Handles text cleaning and normalization for EA documents."""
    
    # Common header/footer patterns in EA documents
    HEADER_FOOTER_PATTERNS = [
        r'^Page \d+ of \d+$',
        r'^\d+$',  # Just page numbers
        r'^Fair Work Commission.*$',
        r'^FWC.*$',
        r'^www\.fwc\.gov\.au.*$',
        r'^Published \d{1,2} \w+ \d{4}$',
        r'^MA\d{6}.*$',  # FWC matter numbers
        r'^.*Enterprise Agreement.*$',
        r'^DRAFT.*$',
        r'^CONFIDENTIAL.*$',
        r'^FOR APPROVAL.*$',
    ]
    
    # Patterns for hyphenation repair
    HYPHENATION_PATTERNS = [
        (r'(\w+)-\s*\n\s*(\w+)', r'\1\2'),  # word- \n word -> wordword
        (r'(\w+)-\s+(\w+)', r'\1\2'),       # word- word -> wordword (same line)
    ]
    
    # Quote marks to normalize
    QUOTE_NORMALIZATION = {
        '"': '"',  # Left double quotation mark
        '"': '"',  # Right double quotation mark
        ''': "'",  # Left single quotation mark
        ''': "'",  # Right single quotation mark
        '`': "'",  # Grave accent
        '´': "'",  # Acute accent
    }
    
    def __init__(self):
        """Initialize the text cleaner."""
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficiency."""
        self.header_footer_regex = [
            re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            for pattern in self.HEADER_FOOTER_PATTERNS
        ]
        
        self.hyphenation_regex = [
            (re.compile(pattern), replacement)
            for pattern, replacement in self.HYPHENATION_PATTERNS
        ]
    
    def normalize_unicode(self, text: str) -> str:
        """
        Normalize Unicode characters to standard forms.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Normalize to NFC form (canonical decomposition followed by canonical composition)
        normalized = unicodedata.normalize('NFC', text)
        
        # Replace non-standard quotes
        for old_char, new_char in self.QUOTE_NORMALIZATION.items():
            normalized = normalized.replace(old_char, new_char)
        
        return normalized
    
    def fix_hyphenation(self, text: str) -> Tuple[str, int]:
        """
        Fix broken hyphenation from PDF extraction.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (fixed text, number of fixes made)
        """
        fixes = 0
        result = text
        
        for pattern, replacement in self.hyphenation_regex:
            new_result, count = pattern.subn(replacement, result)
            result = new_result
            fixes += count
        
        return result, fixes
    
    def remove_headers_footers(self, lines: List[str]) -> Tuple[List[str], int]:
        """
        Remove headers and footers from text lines.
        
        Args:
            lines: List of text lines
            
        Returns:
            Tuple of (cleaned lines, number of lines removed)
        """
        cleaned_lines = []
        removed_count = 0
        
        for line in lines:
            line_stripped = line.strip()
            
            # Skip empty lines
            if not line_stripped:
                cleaned_lines.append(line)
                continue
            
            # Check against header/footer patterns
            is_header_footer = False
            for pattern in self.header_footer_regex:
                if pattern.match(line_stripped):
                    is_header_footer = True
                    break
            
            # Additional heuristics for headers/footers
            if not is_header_footer:
                # Very short lines that are just numbers or single words
                if len(line_stripped) <= 3 and (line_stripped.isdigit() or 
                                               len(line_stripped.split()) == 1):
                    is_header_footer = True
                
                # Lines that are mostly uppercase and short
                elif (len(line_stripped) < 50 and 
                      len([c for c in line_stripped if c.isupper()]) > len(line_stripped) * 0.7):
                    is_header_footer = True
            
            if is_header_footer:
                removed_count += 1
                self.logger.debug(f"Removed header/footer: '{line_stripped}'")
            else:
                cleaned_lines.append(line)
        
        return cleaned_lines, removed_count
    
    def normalize_whitespace(self, text: str) -> Tuple[str, int]:
        """
        Normalize whitespace in text.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (normalized text, number of normalizations)
        """
        original_lines = len(text.split('\n'))
        
        # Replace multiple spaces with single spaces
        text = re.sub(r' +', ' ', text)
        
        # Replace multiple newlines with at most two newlines
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove trailing whitespace from lines
        lines = text.split('\n')
        lines = [line.rstrip() for line in lines]
        text = '\n'.join(lines)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        new_lines = len(text.split('\n'))
        normalizations = max(0, original_lines - new_lines)
        
        return text, normalizations
    
    def preserve_page_breaks(self, text: str, page_marker: str = "\n\n[PAGE_BREAK]\n\n") -> str:
        """
        Insert page break markers to preserve page boundaries.
        
        Args:
            text: Input text
            page_marker: Marker to insert at page breaks
            
        Returns:
            Text with page break markers
        """
        # This is a simple heuristic - in practice, you'd pass page boundaries
        # from the PDF processor
        
        # Look for patterns that suggest page breaks
        page_break_patterns = [
            r'\n\s*Page \d+\s*\n',
            r'\n\s*\d+\s*\n\s*\n',  # Page numbers
            r'\n\s*[A-Z\s]{10,}\s*\n',  # All caps headers
        ]
        
        result = text
        for pattern in page_break_patterns:
            result = re.sub(pattern, page_marker, result, flags=re.IGNORECASE)
        
        return result
    
    def clean_text(
        self,
        text: str,
        remove_headers_footers: bool = True,
        fix_hyphenation: bool = True,
        normalize_unicode: bool = True,
        normalize_whitespace: bool = True,
        preserve_page_breaks: bool = True
    ) -> Tuple[str, CleaningStats]:
        """
        Perform comprehensive text cleaning.
        
        Args:
            text: Input text
            remove_headers_footers: Whether to remove headers/footers
            fix_hyphenation: Whether to fix hyphenation
            normalize_unicode: Whether to normalize Unicode
            normalize_whitespace: Whether to normalize whitespace
            preserve_page_breaks: Whether to preserve page break markers
            
        Returns:
            Tuple of (cleaned text, cleaning statistics)
        """
        original_length = len(text)
        result = text
        
        stats = CleaningStats(original_length=original_length, cleaned_length=0)
        
        # Unicode normalization
        if normalize_unicode:
            result = self.normalize_unicode(result)
        
        # Fix hyphenation
        if fix_hyphenation:
            result, stats.hyphenations_fixed = self.fix_hyphenation(result)
        
        # Remove headers and footers
        if remove_headers_footers:
            lines = result.split('\n')
            cleaned_lines, stats.headers_footers_removed = self.remove_headers_footers(lines)
            result = '\n'.join(cleaned_lines)
            stats.lines_removed = stats.headers_footers_removed
        
        # Preserve page breaks before normalizing whitespace
        if preserve_page_breaks:
            result = self.preserve_page_breaks(result)
        
        # Normalize whitespace
        if normalize_whitespace:
            result, stats.whitespace_normalized = self.normalize_whitespace(result)
        
        stats.cleaned_length = len(result)
        
        self.logger.debug(f"Text cleaning completed: {original_length} -> {stats.cleaned_length} chars "
                         f"({stats.reduction_ratio:.1%} reduction)")
        
        return result, stats
    
    def clean_clause_text(self, text: str) -> str:
        """
        Clean text specifically for clause content.
        
        Args:
            text: Clause text
            
        Returns:
            Cleaned clause text
        """
        # More conservative cleaning for clause text
        result, _ = self.clean_text(
            text,
            remove_headers_footers=False,  # Don't remove headers in clause context
            fix_hyphenation=True,
            normalize_unicode=True,
            normalize_whitespace=True,
            preserve_page_breaks=False  # Don't preserve page breaks in clauses
        )
        
        # Additional clause-specific cleaning
        # Remove excessive blank lines within clauses
        result = re.sub(r'\n\s*\n\s*\n+', '\n\n', result)
        
        # Ensure clause ends with proper punctuation or newline
        result = result.strip()
        if result and not result[-1] in '.!?':
            # Don't add punctuation if it ends with a colon (might be introducing a list)
            if not result.endswith(':'):
                result += '.'
        
        return result
    
    def extract_potential_headings(self, text: str) -> List[Dict[str, any]]:
        """
        Extract potential section headings from text.
        
        Args:
            text: Input text
            
        Returns:
            List of potential headings with metadata
        """
        headings = []
        lines = text.split('\n')
        
        heading_patterns = [
            # Numbered sections: "1. Introduction", "2.1 Definitions"
            r'^(\d+(?:\.\d+)*\.?)\s+(.+)$',
            # Letter sections: "A. General", "B.1 Specific"
            r'^([A-Z](?:\.\d+)*\.?)\s+(.+)$',
            # Parenthetical: "(a) First item", "(i) Sub item"
            r'^\(([a-z]+|[ivx]+)\)\s+(.+)$',
            # All caps short lines (potential headings)
            r'^([A-Z\s]{3,30})$',
        ]
        
        for line_num, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped or len(line_stripped) < 3:
                continue
            
            for pattern in heading_patterns:
                match = re.match(pattern, line_stripped)
                if match:
                    if len(match.groups()) == 2:
                        number, title = match.groups()
                    else:
                        number = ""
                        title = match.group(1)
                    
                    headings.append({
                        'line_number': line_num,
                        'number': number.strip(),
                        'title': title.strip(),
                        'full_text': line_stripped,
                        'pattern': pattern
                    })
                    break
        
        return headings


def create_text_cleaner() -> TextCleaner:
    """Factory function to create a text cleaner."""
    return TextCleaner()