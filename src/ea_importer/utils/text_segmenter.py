"""
Text segmentation utilities for clause detection and hierarchical parsing.
"""

import re
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from ..core.logging import LoggerMixin


class ClauseType(Enum):
    """Types of clauses in EA documents."""
    DEFINITION = "definition"
    CLASSIFICATION = "classification"
    RATE = "rate"
    PROCEDURE = "procedure"
    CONDITION = "condition"
    PENALTY = "penalty"
    ALLOWANCE = "allowance"
    LEAVE = "leave"
    DISPUTE = "dispute"
    GENERAL = "general"


@dataclass
class ClauseSegment:
    """Represents a single clause segment."""
    clause_id: str
    heading: Optional[str]
    text: str
    path: List[str]
    level: int
    start_line: int
    end_line: int
    page_spans: List[Tuple[int, int]] = field(default_factory=list)
    clause_type: Optional[ClauseType] = None
    has_subsections: bool = False
    parent_clause_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'clause_id': self.clause_id,
            'heading': self.heading,
            'text': self.text,
            'path': self.path,
            'level': self.level,
            'start_line': self.start_line,
            'end_line': self.end_line,
            'page_spans': self.page_spans,
            'clause_type': self.clause_type.value if self.clause_type else None,
            'has_subsections': self.has_subsections,
            'parent_clause_id': self.parent_clause_id,
        }


class TextSegmenter(LoggerMixin):
    """Segments text into hierarchical clauses using pattern recognition."""
    
    # Hierarchical numbering patterns (in order of priority)
    NUMBERING_PATTERNS = [
        # Main sections: "1.", "2.", "10."
        (r'^(\d+)\.\s+(.+)', 1, 'numeric_main'),
        # Subsections: "1.1", "2.3", "10.15"
        (r'^(\d+\.\d+)\s+(.+)', 2, 'numeric_sub'),
        # Sub-subsections: "1.1.1", "2.3.4"
        (r'^(\d+\.\d+\.\d+)\s+(.+)', 3, 'numeric_subsub'),
        # Four-level: "1.1.1.1"
        (r'^(\d+\.\d+\.\d+\.\d+)\s+(.+)', 4, 'numeric_four'),
        
        # Letter sections: "A.", "B.", "Z."
        (r'^([A-Z])\.\s+(.+)', 1, 'letter_main'),
        # Letter subsections: "A.1", "B.3"
        (r'^([A-Z]\.\d+)\s+(.+)', 2, 'letter_numeric'),
        
        # Parenthetical: "(a)", "(i)", "(1)"
        (r'^\(([a-z])\)\s+(.+)', 3, 'paren_letter'),
        (r'^\(([ivx]+)\)\s+(.+)', 3, 'paren_roman'),
        (r'^\((\d+)\)\s+(.+)', 3, 'paren_numeric'),
        
        # Indented items: "a.", "i.", "1."
        (r'^\s{2,}([a-z])\.\s+(.+)', 4, 'indent_letter'),
        (r'^\s{2,}([ivx]+)\.\s+(.+)', 4, 'indent_roman'),
        (r'^\s{2,}(\d+)\.\s+(.+)', 4, 'indent_numeric'),
    ]
    
    # Clause type detection patterns
    TYPE_PATTERNS = {
        ClauseType.DEFINITION: [
            r'definition|means|shall mean|interpreted|construed',
            r'^definitions?$',
        ],
        ClauseType.CLASSIFICATION: [
            r'classification|grade|level|category',
            r'employee.*classification',
        ],
        ClauseType.RATE: [
            r'rate of pay|hourly rate|weekly rate|salary|wage',
            r'minimum.*rate|base.*rate',
            r'\$\d+\.\d{2}|\d+\.\d{2}\s*per\s*hour',
        ],
        ClauseType.PENALTY: [
            r'overtime|penalty|shift|weekend|public holiday',
            r'time and a half|double time|loading',
        ],
        ClauseType.ALLOWANCE: [
            r'allowance|reimbursement|expense|travel|meal',
            r'tool allowance|first aid|supervisory',
        ],
        ClauseType.LEAVE: [
            r'annual leave|sick leave|personal leave|carer',
            r'parental leave|compassionate|bereavement',
        ],
        ClauseType.PROCEDURE: [
            r'procedure|process|shall|must|required',
            r'notification|application|approval',
        ],
        ClauseType.DISPUTE: [
            r'dispute|grievance|complaint|resolution',
            r'mediation|arbitration|fair work',
        ],
    }
    
    def __init__(self):
        """Initialize the text segmenter."""
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficiency."""
        self.numbering_regex = [
            (re.compile(pattern, re.IGNORECASE), level, pattern_type)
            for pattern, level, pattern_type in self.NUMBERING_PATTERNS
        ]
        
        self.type_regex = {}
        for clause_type, patterns in self.TYPE_PATTERNS.items():
            self.type_regex[clause_type] = [
                re.compile(pattern, re.IGNORECASE)
                for pattern in patterns
            ]
    
    def detect_clause_type(self, heading: str, text: str) -> Optional[ClauseType]:
        """
        Detect the type of a clause based on heading and content.
        
        Args:
            heading: Clause heading
            text: Clause text content
            
        Returns:
            Detected clause type or None
        """
        combined_text = f"{heading or ''} {text}".lower()
        
        for clause_type, patterns in self.type_regex.items():
            for pattern in patterns:
                if pattern.search(combined_text):
                    return clause_type
        
        return ClauseType.GENERAL
    
    def parse_numbering(self, line: str) -> Optional[Tuple[str, str, int, str]]:
        """
        Parse numbering from a line.
        
        Args:
            line: Text line to parse
            
        Returns:
            Tuple of (number, heading, level, pattern_type) or None
        """
        line_stripped = line.strip()
        
        for pattern, level, pattern_type in self.numbering_regex:
            match = pattern.match(line_stripped)
            if match:
                number = match.group(1)
                heading = match.group(2).strip() if len(match.groups()) > 1 else ""
                return number, heading, level, pattern_type
        
        return None
    
    def build_clause_id(self, number: str, pattern_type: str) -> str:
        """
        Build a standardized clause ID from numbering.
        
        Args:
            number: Raw number from text
            pattern_type: Type of numbering pattern
            
        Returns:
            Standardized clause ID
        """
        # Convert different numbering schemes to consistent format
        if 'roman' in pattern_type:
            # Convert roman numerals to arabic
            roman_map = {'i': '1', 'ii': '2', 'iii': '3', 'iv': '4', 'v': '5', 
                        'vi': '6', 'vii': '7', 'viii': '8', 'ix': '9', 'x': '10'}
            number = roman_map.get(number.lower(), number)
        
        return number
    
    def build_hierarchical_path(self, segments: List[ClauseSegment]) -> List[str]:
        """
        Build hierarchical path for a clause based on previous segments.
        
        Args:
            segments: List of previous clause segments
            
        Returns:
            Hierarchical path
        """
        if not segments:
            return []
        
        current_segment = segments[-1]
        path = []
        
        # Build path by traversing up the hierarchy
        for i in range(len(segments) - 1, -1, -1):
            segment = segments[i]
            if segment.level < current_segment.level:
                path.insert(0, segment.heading or segment.clause_id)
                current_segment = segment
            elif segment.level == current_segment.level and i == len(segments) - 1:
                # Same level as current, include if it's the current one
                continue
        
        return path
    
    def segment_by_headings(
        self,
        text: str,
        min_clause_length: int = 50,
        max_clause_length: int = 10000
    ) -> List[ClauseSegment]:
        """
        Segment text into clauses based on hierarchical headings.
        
        Args:
            text: Input text to segment
            min_clause_length: Minimum clause length in characters
            max_clause_length: Maximum clause length in characters
            
        Returns:
            List of clause segments
        """
        lines = text.split('\n')
        segments = []
        current_clause = None
        current_text_lines = []
        
        self.logger.info(f"Segmenting text with {len(lines)} lines")
        
        for line_num, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Skip empty lines
            if not line_stripped:
                if current_text_lines:
                    current_text_lines.append('')
                continue
            
            # Try to parse as a heading
            numbering_info = self.parse_numbering(line)
            
            if numbering_info:
                number, heading, level, pattern_type = numbering_info
                
                # Save previous clause if exists
                if current_clause and current_text_lines:
                    clause_text = '\n'.join(current_text_lines).strip()
                    if len(clause_text) >= min_clause_length:
                        current_clause.text = clause_text
                        current_clause.end_line = line_num - 1
                        
                        # Build hierarchical path
                        current_clause.path = self.build_hierarchical_path(segments)
                        if current_clause.heading:
                            current_clause.path.append(current_clause.heading)
                        
                        # Detect clause type
                        current_clause.clause_type = self.detect_clause_type(
                            current_clause.heading, clause_text
                        )
                        
                        segments.append(current_clause)
                        self.logger.debug(f"Added clause {current_clause.clause_id}: {current_clause.heading}")
                
                # Start new clause
                clause_id = self.build_clause_id(number, pattern_type)
                current_clause = ClauseSegment(
                    clause_id=clause_id,
                    heading=heading,
                    text="",
                    path=[],
                    level=level,
                    start_line=line_num,
                    end_line=line_num,
                )
                current_text_lines = []
                
            else:
                # Add to current clause content
                if current_text_lines or line_stripped:  # Don't start with empty lines
                    current_text_lines.append(line)
        
        # Save final clause
        if current_clause and current_text_lines:
            clause_text = '\n'.join(current_text_lines).strip()
            if len(clause_text) >= min_clause_length:
                current_clause.text = clause_text
                current_clause.end_line = len(lines) - 1
                current_clause.path = self.build_hierarchical_path(segments)
                if current_clause.heading:
                    current_clause.path.append(current_clause.heading)
                current_clause.clause_type = self.detect_clause_type(
                    current_clause.heading, clause_text
                )
                segments.append(current_clause)
        
        # Post-process: set parent relationships and subsection flags
        self._set_parent_relationships(segments)
        
        self.logger.info(f"Segmentation complete: {len(segments)} clauses extracted")
        
        return segments
    
    def _set_parent_relationships(self, segments: List[ClauseSegment]):
        """Set parent-child relationships between clauses."""
        for i, segment in enumerate(segments):
            # Find parent (previous clause with lower level)
            for j in range(i - 1, -1, -1):
                if segments[j].level < segment.level:
                    segment.parent_clause_id = segments[j].clause_id
                    segments[j].has_subsections = True
                    break
    
    def segment_by_ml_fallback(self, text: str) -> List[ClauseSegment]:
        """
        Fallback segmentation using simple heuristics when pattern matching fails.
        
        Args:
            text: Input text
            
        Returns:
            List of clause segments
        """
        self.logger.warning("Using ML fallback segmentation")
        
        # Simple paragraph-based segmentation
        paragraphs = re.split(r'\n\s*\n', text)
        segments = []
        
        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if len(paragraph) < 50:  # Skip very short paragraphs
                continue
            
            # Try to extract a heading from first line
            lines = paragraph.split('\n')
            first_line = lines[0].strip()
            
            # Simple heuristic: if first line is short and rest is longer, treat as heading
            if len(lines) > 1 and len(first_line) < 100:
                heading = first_line
                text_content = '\n'.join(lines[1:]).strip()
            else:
                heading = f"Section {i + 1}"
                text_content = paragraph
            
            if text_content:
                segment = ClauseSegment(
                    clause_id=f"fallback_{i + 1}",
                    heading=heading,
                    text=text_content,
                    path=[],
                    level=1,
                    start_line=0,  # Line numbers not available in fallback
                    end_line=0,
                    clause_type=ClauseType.GENERAL
                )
                segments.append(segment)
        
        return segments
    
    def segment_text(
        self,
        text: str,
        use_ml_fallback: bool = True,
        min_clause_count: int = None
    ) -> List[ClauseSegment]:
        """
        Main text segmentation method.
        
        Args:
            text: Input text to segment
            use_ml_fallback: Whether to use ML fallback if pattern matching fails
            min_clause_count: Minimum expected clause count
            
        Returns:
            List of clause segments
        """
        # Try pattern-based segmentation first
        segments = self.segment_by_headings(text)
        
        # Check if segmentation was successful
        min_expected = min_clause_count or self.settings.min_clause_count if hasattr(self, 'settings') else 10
        
        if len(segments) < min_expected and use_ml_fallback:
            self.logger.warning(f"Pattern segmentation yielded only {len(segments)} clauses, "
                              f"expected at least {min_expected}. Trying fallback.")
            segments = self.segment_by_ml_fallback(text)
        
        return segments


def create_text_segmenter() -> TextSegmenter:
    """Factory function to create a text segmenter."""
    return TextSegmenter()