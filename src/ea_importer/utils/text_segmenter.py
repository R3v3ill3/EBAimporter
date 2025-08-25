"""
Text Segmentation utility for EA Importer.

Handles intelligent clause segmentation with hierarchical numbering detection:
- Pattern-based segmentation using regex for common numbering schemes
- Machine learning fallback for complex documents
- Hierarchical structure preservation
- Legal document-specific patterns
- Quality validation and error handling

Designed specifically for Australian Enterprise Agreement documents.
"""

import re
import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False

from ..core.config import get_settings
from ..core.logging import get_logger, log_function_call
from ..models import PDFDocument, ClauseSegment

logger = get_logger(__name__)


class SegmentationError(Exception):
    """Custom exception for segmentation errors"""
    pass


@dataclass
class NumberingPattern:
    """Represents a hierarchical numbering pattern"""
    pattern: str
    level: int
    regex: re.Pattern
    description: str
    priority: int = 0


class TextSegmenter:
    """
    Advanced text segmentation for legal documents with hierarchical structure detection.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize text segmenter.
        
        Args:
            config: Optional configuration override
        """
        self.settings = get_settings()
        self.config = config or {}
        
        # Segmentation parameters
        self.min_clause_length = self.config.get('min_clause_length', self.settings.processing.min_clause_length)
        self.max_clause_length = self.config.get('max_clause_length', self.settings.processing.max_clause_length)
        
        # Initialize numbering patterns
        self.numbering_patterns = self._initialize_numbering_patterns()
        
        # Initialize NLP model if available
        self.nlp_model = None
        if HAS_SPACY:
            try:
                self.nlp_model = spacy.load("en_core_web_sm")
                logger.info("Loaded spaCy model for ML-based segmentation")
            except OSError:
                logger.warning("spaCy model not found, using pattern-based segmentation only")
    
    def _initialize_numbering_patterns(self) -> List[NumberingPattern]:
        """Initialize hierarchical numbering patterns for EA documents"""
        
        patterns = [
            # Level 1: Main sections (1., 2., 3., etc.)
            NumberingPattern(
                pattern=r'^\s*(\d+)\.\s*([^\n]+?)(?:\n|$)',
                level=1,
                regex=re.compile(r'^\s*(\d+)\.\s*([^\n]+?)(?:\n|$)', re.MULTILINE),
                description="Main sections (1., 2., 3.)",
                priority=1
            ),
            
            # Level 2: Subsections (1.1, 1.2, 2.1, etc.)
            NumberingPattern(
                pattern=r'^\s*(\d+)\.(\d+)\s*([^\n]+?)(?:\n|$)',
                level=2,
                regex=re.compile(r'^\s*(\d+)\.(\d+)\s*([^\n]+?)(?:\n|$)', re.MULTILINE),
                description="Subsections (1.1, 1.2, 2.1)",
                priority=2
            ),
            
            # Level 3: Sub-subsections (1.1.1, 2.3.4, etc.)
            NumberingPattern(
                pattern=r'^\s*(\d+)\.(\d+)\.(\d+)\s*([^\n]+?)(?:\n|$)',
                level=3,
                regex=re.compile(r'^\s*(\d+)\.(\d+)\.(\d+)\s*([^\n]+?)(?:\n|$)', re.MULTILINE),
                description="Sub-subsections (1.1.1, 2.3.4)",
                priority=3
            ),
            
            # Level 4: Lettered clauses (a), b), c), etc.)
            NumberingPattern(
                pattern=r'^\s*\(([a-z])\)\s*([^\n]+?)(?:\n|$)',
                level=4,
                regex=re.compile(r'^\s*\(([a-z])\)\s*([^\n]+?)(?:\n|$)', re.MULTILINE),
                description="Lettered clauses (a), b), c)",
                priority=4
            ),
            
            # Level 4 Alternative: a., b., c.
            NumberingPattern(
                pattern=r'^\s*([a-z])\.\s*([^\n]+?)(?:\n|$)',
                level=4,
                regex=re.compile(r'^\s*([a-z])\.\s*([^\n]+?)(?:\n|$)', re.MULTILINE),
                description="Lettered sections (a., b., c.)",
                priority=4
            ),
            
            # Level 5: Roman numerals (i), ii), iii), etc.)
            NumberingPattern(
                pattern=r'^\s*\(([ivx]+)\)\s*([^\n]+?)(?:\n|$)',
                level=5,
                regex=re.compile(r'^\s*\(([ivx]+)\)\s*([^\n]+?)(?:\n|$)', re.MULTILINE),
                description="Roman numerals (i), ii), iii)",
                priority=5
            ),
            
            # Level 5 Alternative: i., ii., iii.
            NumberingPattern(
                pattern=r'^\s*([ivx]+)\.\s*([^\n]+?)(?:\n|$)',
                level=5,
                regex=re.compile(r'^\s*([ivx]+)\.\s*([^\n]+?)(?:\n|$)', re.MULTILINE),
                description="Roman numerals (i., ii., iii.)",
                priority=5
            ),
            
            # Special patterns for EA documents
            
            # Schedule/Appendix patterns
            NumberingPattern(
                pattern=r'^\s*(SCHEDULE|APPENDIX)\s+([A-Z0-9]+)[\s\-]*([^\n]+?)(?:\n|$)',
                level=1,
                regex=re.compile(r'^\s*(SCHEDULE|APPENDIX)\s+([A-Z0-9]+)[\s\-]*([^\n]+?)(?:\n|$)', re.MULTILINE | re.IGNORECASE),
                description="Schedules and Appendices",
                priority=1
            ),
            
            # Part patterns
            NumberingPattern(
                pattern=r'^\s*PART\s+([A-Z0-9]+)[\s\-]*([^\n]+?)(?:\n|$)',
                level=1,
                regex=re.compile(r'^\s*PART\s+([A-Z0-9]+)[\s\-]*([^\n]+?)(?:\n|$)', re.MULTILINE | re.IGNORECASE),
                description="Parts",
                priority=1
            ),
            
            # Clause patterns (common in legal docs)
            NumberingPattern(
                pattern=r'^\s*CLAUSE\s+(\d+(?:\.\d+)*)\s*[\-\:]?\s*([^\n]+?)(?:\n|$)',
                level=2,
                regex=re.compile(r'^\s*CLAUSE\s+(\d+(?:\.\d+)*)\s*[\-\:]?\s*([^\n]+?)(?:\n|$)', re.MULTILINE | re.IGNORECASE),
                description="Explicit clause references",
                priority=2
            )
        ]
        
        # Sort by priority for matching order
        patterns.sort(key=lambda p: p.priority)
        
        return patterns
    
    @log_function_call
    def segment_document(self, document: PDFDocument, ea_id: Optional[str] = None) -> List[ClauseSegment]:
        """
        Segment a PDF document into clauses.
        
        Args:
            document: PDFDocument to segment
            ea_id: Optional EA identifier override
            
        Returns:
            List of ClauseSegment objects
        """
        if ea_id is None:
            ea_id = document.metadata.get('ea_id', 'UNKNOWN')
        
        logger.info(f"Segmenting document {ea_id} with {len(document.pages)} pages")
        
        # Combine all page text
        full_text = document.full_text
        
        # Try pattern-based segmentation first
        clauses = self._segment_with_patterns(full_text, ea_id, document)
        
        # Fall back to ML-based segmentation if patterns fail
        if not clauses and self.nlp_model:
            logger.info("Pattern-based segmentation failed, trying ML approach")
            clauses = self._segment_with_ml(full_text, ea_id, document)
        
        # Final fallback: simple paragraph-based segmentation
        if not clauses:
            logger.warning("Both pattern and ML segmentation failed, using paragraph fallback")
            clauses = self._segment_by_paragraphs(full_text, ea_id, document)
        
        # Validate and clean segments
        clauses = self._validate_and_clean_segments(clauses)
        
        logger.info(f"Segmentation completed: {len(clauses)} clauses extracted")
        return clauses
    
    def _segment_with_patterns(self, text: str, ea_id: str, document: PDFDocument) -> List[ClauseSegment]:
        """Segment text using hierarchical numbering patterns"""
        
        segments = []
        hierarchical_stack = []  # Track current hierarchical position
        
        # Find all pattern matches with their positions
        matches = []
        for pattern in self.numbering_patterns:
            for match in pattern.regex.finditer(text):
                matches.append({
                    'match': match,
                    'pattern': pattern,
                    'start': match.start(),
                    'end': match.end(),
                    'level': pattern.level,
                    'text': match.group()
                })
        
        # Sort matches by position in text
        matches.sort(key=lambda m: m['start'])
        
        if not matches:
            logger.warning("No numbering patterns found in text")
            return []
        
        # Process matches to create segments
        for i, match_info in enumerate(matches):
            match = match_info['match']
            pattern = match_info['pattern']
            level = match_info['level']
            
            # Determine clause ID based on pattern and groups
            clause_id = self._extract_clause_id(match, pattern)
            
            # Extract heading from match
            heading = self._extract_heading(match, pattern)
            
            # Determine text content (from this match to next match or end)
            start_pos = match.end()
            if i + 1 < len(matches):
                end_pos = matches[i + 1]['start']
            else:
                end_pos = len(text)
            
            clause_text = text[start_pos:end_pos].strip()
            
            # Skip if text is too short or too long
            if len(clause_text) < self.min_clause_length:
                logger.debug(f"Skipping short clause {clause_id}: {len(clause_text)} chars")
                continue
            
            if len(clause_text) > self.max_clause_length:
                logger.warning(f"Clause {clause_id} is very long: {len(clause_text)} chars")
            
            # Update hierarchical stack
            hierarchical_stack = self._update_hierarchy(hierarchical_stack, level, clause_id)
            
            # Calculate page spans
            page_spans = self._calculate_page_spans(start_pos, end_pos, text, document)
            
            # Create clause segment
            segment = ClauseSegment(
                ea_id=ea_id,
                clause_id=clause_id,
                path=hierarchical_stack.copy(),
                heading=heading,
                text=clause_text,
                hash_sha256=hashlib.sha256(clause_text.encode()).hexdigest(),
                token_count=len(clause_text.split()),
                page_spans=page_spans
            )
            
            segments.append(segment)
        
        return segments
    
    def _extract_clause_id(self, match: re.Match, pattern: NumberingPattern) -> str:
        """Extract clause ID from regex match"""
        
        if pattern.description.startswith("Main sections"):
            return match.group(1)
        
        elif pattern.description.startswith("Subsections"):
            return f"{match.group(1)}.{match.group(2)}"
        
        elif pattern.description.startswith("Sub-subsections"):
            return f"{match.group(1)}.{match.group(2)}.{match.group(3)}"
        
        elif "lettered" in pattern.description.lower():
            return match.group(1)
        
        elif "roman" in pattern.description.lower():
            return match.group(1)
        
        elif pattern.description.startswith("Schedules"):
            return f"{match.group(1)}_{match.group(2)}"
        
        elif pattern.description.startswith("Parts"):
            return f"PART_{match.group(1)}"
        
        elif pattern.description.startswith("Explicit clause"):
            return f"CLAUSE_{match.group(1)}"
        
        else:
            # Fallback: use first captured group
            return match.group(1) if match.groups() else "UNKNOWN"
    
    def _extract_heading(self, match: re.Match, pattern: NumberingPattern) -> str:
        """Extract heading text from regex match"""
        
        # Usually the last captured group contains the heading
        groups = match.groups()
        if len(groups) >= 2:
            return groups[-1].strip()
        
        return ""
    
    def _update_hierarchy(self, stack: List[str], level: int, clause_id: str) -> List[str]:
        """Update hierarchical stack based on current level"""
        
        # Trim stack to current level
        while len(stack) >= level:
            stack.pop()
        
        # Add current clause to stack
        stack.append(clause_id)
        
        return stack
    
    def _calculate_page_spans(self, start_pos: int, end_pos: int, full_text: str, document: PDFDocument) -> List[Tuple[int, int]]:
        """Calculate which pages contain this text segment"""
        
        page_spans = []
        current_pos = 0
        
        for page_num, page in enumerate(document.pages):
            page_start = current_pos
            page_end = current_pos + len(page.text)
            
            # Check if segment overlaps with this page
            if (start_pos < page_end and end_pos > page_start):
                page_spans.append((page_num + 1, page_num + 1))  # Same page start/end
            
            current_pos = page_end + 1  # +1 for page break
        
        # Merge consecutive pages
        if page_spans:
            merged_spans = []
            current_start = page_spans[0][0]
            current_end = page_spans[0][1]
            
            for start, end in page_spans[1:]:
                if start == current_end + 1:
                    current_end = end
                else:
                    merged_spans.append((current_start, current_end))
                    current_start = start
                    current_end = end
            
            merged_spans.append((current_start, current_end))
            return merged_spans
        
        return page_spans
    
    def _segment_with_ml(self, text: str, ea_id: str, document: PDFDocument) -> List[ClauseSegment]:
        """Segment text using ML-based approach with spaCy"""
        
        if not self.nlp_model:
            return []
        
        logger.info("Using ML-based segmentation")
        
        # Process text with spaCy
        doc = self.nlp_model(text)
        
        segments = []
        current_segment = []
        clause_counter = 1
        
        # Simple sentence-based segmentation with ML features
        for sent in doc.sents:
            sent_text = sent.text.strip()
            
            # Check if sentence looks like a heading or new section
            is_heading = self._is_likely_heading(sent_text, sent)
            
            if is_heading and current_segment:
                # Save current segment
                segment_text = ' '.join(current_segment)
                if len(segment_text) >= self.min_clause_length:
                    segment = ClauseSegment(
                        ea_id=ea_id,
                        clause_id=f"ML_{clause_counter}",
                        path=[f"ML_{clause_counter}"],
                        heading="",
                        text=segment_text,
                        hash_sha256=hashlib.sha256(segment_text.encode()).hexdigest(),
                        token_count=len(segment_text.split()),
                        page_spans=[]  # Would need more complex calculation
                    )
                    segments.append(segment)
                    clause_counter += 1
                
                # Start new segment
                current_segment = [sent_text]
            else:
                current_segment.append(sent_text)
        
        # Handle final segment
        if current_segment:
            segment_text = ' '.join(current_segment)
            if len(segment_text) >= self.min_clause_length:
                segment = ClauseSegment(
                    ea_id=ea_id,
                    clause_id=f"ML_{clause_counter}",
                    path=[f"ML_{clause_counter}"],
                    heading="",
                    text=segment_text,
                    hash_sha256=hashlib.sha256(segment_text.encode()).hexdigest(),
                    token_count=len(segment_text.split()),
                    page_spans=[]
                )
                segments.append(segment)
        
        logger.info(f"ML segmentation produced {len(segments)} segments")
        return segments
    
    def _is_likely_heading(self, text: str, sent) -> bool:
        """Determine if a sentence is likely a heading using ML features"""
        
        # Rule-based heuristics
        if len(text) > 100:  # Too long for heading
            return False
        
        if text.endswith('.') and len(text.split()) > 10:  # Looks like sentence
            return False
        
        # Check for common heading patterns
        heading_patterns = [
            r'^\d+\.',  # Starts with number
            r'^[A-Z][A-Z\s]+$',  # All caps
            r'^(PART|SECTION|CLAUSE|SCHEDULE)',  # Common legal headings
        ]
        
        for pattern in heading_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True
        
        # Use spaCy features if available
        if hasattr(sent, 'ents'):
            # Check if contains legal entities
            legal_labels = ['LAW', 'ORG', 'PERSON']  # Common in headings
            for ent in sent.ents:
                if ent.label_ in legal_labels:
                    return True
        
        return False
    
    def _segment_by_paragraphs(self, text: str, ea_id: str, document: PDFDocument) -> List[ClauseSegment]:
        """Fallback segmentation using simple paragraph breaks"""
        
        logger.info("Using paragraph-based fallback segmentation")
        
        paragraphs = text.split('\n\n')
        segments = []
        
        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            
            if len(paragraph) >= self.min_clause_length:
                segment = ClauseSegment(
                    ea_id=ea_id,
                    clause_id=f"PARA_{i+1}",
                    path=[f"PARA_{i+1}"],
                    heading="",
                    text=paragraph,
                    hash_sha256=hashlib.sha256(paragraph.encode()).hexdigest(),
                    token_count=len(paragraph.split()),
                    page_spans=[]
                )
                segments.append(segment)
        
        logger.info(f"Paragraph segmentation produced {len(segments)} segments")
        return segments
    
    def _validate_and_clean_segments(self, segments: List[ClauseSegment]) -> List[ClauseSegment]:
        """Validate and clean clause segments"""
        
        cleaned_segments = []
        
        for segment in segments:
            # Skip empty or very short segments
            if len(segment.text.strip()) < self.min_clause_length:
                continue
            
            # Truncate very long segments
            if len(segment.text) > self.max_clause_length:
                segment.text = segment.text[:self.max_clause_length] + "..."
                logger.warning(f"Truncated long segment {segment.clause_id}")
            
            # Ensure hash is calculated
            if not segment.hash_sha256:
                segment.hash_sha256 = hashlib.sha256(segment.text.encode()).hexdigest()
            
            # Ensure token count is calculated
            if not segment.token_count:
                segment.token_count = len(segment.text.split())
            
            cleaned_segments.append(segment)
        
        return cleaned_segments
    
    def segment_text(self, text: str, ea_id: str = "UNKNOWN") -> List[ClauseSegment]:
        """
        Segment raw text without page context.
        
        Args:
            text: Raw text to segment
            ea_id: EA identifier
            
        Returns:
            List of ClauseSegment objects
        """
        # Create minimal document structure
        from ..models import PDFPage, PDFDocument
        
        page = PDFPage(page_number=1, text=text)
        document = PDFDocument(file_path=Path("unknown"), pages=[page])
        
        return self.segment_document(document, ea_id)
    
    def get_segmentation_stats(self, segments: List[ClauseSegment]) -> Dict[str, Any]:
        """
        Calculate statistics about segmentation results.
        
        Args:
            segments: List of clause segments
            
        Returns:
            Dictionary with segmentation statistics
        """
        if not segments:
            return {'total_segments': 0}
        
        text_lengths = [len(seg.text) for seg in segments]
        token_counts = [seg.token_count or 0 for seg in segments]
        
        stats = {
            'total_segments': len(segments),
            'avg_text_length': sum(text_lengths) / len(text_lengths),
            'min_text_length': min(text_lengths),
            'max_text_length': max(text_lengths),
            'avg_token_count': sum(token_counts) / len(token_counts) if token_counts else 0,
            'total_tokens': sum(token_counts),
            'hierarchy_levels': len(set(len(seg.path) for seg in segments)),
            'unique_clause_ids': len(set(seg.clause_id for seg in segments)),
        }
        
        return stats


# Utility functions for batch processing
def segment_documents_batch(documents: List[PDFDocument], **kwargs) -> Dict[str, List[ClauseSegment]]:
    """
    Segment multiple documents in batch.
    
    Args:
        documents: List of PDFDocuments to segment
        **kwargs: Arguments passed to TextSegmenter
        
    Returns:
        Dictionary mapping EA IDs to clause segments
    """
    segmenter = TextSegmenter(**kwargs)
    results = {}
    
    for document in documents:
        ea_id = document.metadata.get('ea_id', 'UNKNOWN')
        try:
            segments = segmenter.segment_document(document, ea_id)
            results[ea_id] = segments
        except Exception as e:
            logger.error(f"Failed to segment document {ea_id}: {e}")
            results[ea_id] = []
    
    return results


# Export main classes
__all__ = [
    'TextSegmenter',
    'SegmentationError',
    'NumberingPattern',
    'segment_documents_batch',
]