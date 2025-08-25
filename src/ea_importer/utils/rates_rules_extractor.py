"""
Rates and Rules Extractor for EA Importer.
Extracts structured data from Enterprise Agreement clauses.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import pandas as pd

from ..core.logging import get_logger, log_function_call
from ..models import ClauseSegment, RateExtraction, RuleExtraction

logger = get_logger(__name__)


class RatesRulesExtractor:
    """Extract rates and rules from Enterprise Agreement clauses."""
    
    def __init__(self):
        self._init_patterns()
    
    def _init_patterns(self):
        """Initialize regex patterns for extraction."""
        
        # Rate patterns
        self.rate_patterns = {
            'hourly': re.compile(r'\$(\d+(?:\.\d{2})?)\s*(?:per\s+)?hour', re.IGNORECASE),
            'weekly': re.compile(r'\$(\d+(?:\.\d{2})?)\s*(?:per\s+)?week', re.IGNORECASE),
            'annual': re.compile(r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:per\s+)?(?:year|annum)', re.IGNORECASE),
        }
        
        # Classification patterns
        self.classification_patterns = {
            'level': re.compile(r'level\s+(\d+)', re.IGNORECASE),
            'grade': re.compile(r'grade\s+(\d+)', re.IGNORECASE),
            'classification': re.compile(r'classification\s+(\d+)', re.IGNORECASE),
        }
        
        # Rule patterns
        self.rule_patterns = {
            'overtime': re.compile(r'overtime.*?(\d+(?:\.\d+)?)\s*times?\s*(?:ordinary|normal|base)', re.IGNORECASE),
            'penalty_weekend': re.compile(r'(?:saturday|sunday).*?(\d+(?:\.\d+)?)\s*times?\s*(?:ordinary|normal|base)', re.IGNORECASE),
            'penalty_night': re.compile(r'night.*?(\d+(?:\.\d+)?)\s*times?\s*(?:ordinary|normal|base)', re.IGNORECASE),
            'allowance': re.compile(r'allowance.*?\$(\d+(?:\.\d{2})?)', re.IGNORECASE),
        }
    
    @log_function_call
    def extract_rates_from_clauses(self, clauses: List[ClauseSegment]) -> List[RateExtraction]:
        """Extract wage rates from clauses."""
        
        rates = []
        
        for clause in clauses:
            # Look for rate-related keywords
            if any(keyword in clause.text.lower() for keyword in ['wage', 'rate', 'salary', 'pay']):
                clause_rates = self._extract_rates_from_text(clause)
                rates.extend(clause_rates)
        
        logger.info(f"Extracted {len(rates)} rates from {len(clauses)} clauses")
        return rates
    
    def _extract_rates_from_text(self, clause: ClauseSegment) -> List[RateExtraction]:
        """Extract rates from a single clause."""
        
        rates = []
        text = clause.text
        
        # Find classifications
        classifications = self._extract_classifications(text)
        
        # Find rates
        for rate_type, pattern in self.rate_patterns.items():
            matches = pattern.finditer(text)
            
            for match in matches:
                rate_value = float(match.group(1).replace(',', ''))
                
                # Try to associate with classification
                classification = self._find_nearest_classification(match.start(), text, classifications)
                
                rate = RateExtraction(
                    classification=classification or "General",
                    level=None,
                    base_rate=rate_value,
                    unit=rate_type,
                    source_clause_id=clause.clause_id,
                    effective_from=None,
                    effective_to=None
                )
                
                rates.append(rate)
        
        return rates
    
    def _extract_classifications(self, text: str) -> List[Tuple[int, str]]:
        """Extract classifications with their positions in text."""
        
        classifications = []
        
        for class_type, pattern in self.classification_patterns.items():
            matches = pattern.finditer(text)
            
            for match in matches:
                pos = match.start()
                classification = f"{class_type.title()} {match.group(1)}"
                classifications.append((pos, classification))
        
        return classifications
    
    def _find_nearest_classification(self, rate_pos: int, text: str, classifications: List[Tuple[int, str]]) -> Optional[str]:
        """Find the nearest classification to a rate."""
        
        if not classifications:
            return None
        
        # Find classification closest to rate position
        best_classification = None
        min_distance = float('inf')
        
        for class_pos, classification in classifications:
            distance = abs(rate_pos - class_pos)
            
            # Prefer classifications that come before the rate
            if class_pos <= rate_pos:
                distance *= 0.5
            
            if distance < min_distance:
                min_distance = distance
                best_classification = classification
        
        return best_classification
    
    @log_function_call
    def extract_rules_from_clauses(self, clauses: List[ClauseSegment]) -> List[RuleExtraction]:
        """Extract employment rules from clauses."""
        
        rules = []
        
        for clause in clauses:
            # Look for rule-related keywords
            if any(keyword in clause.text.lower() for keyword in 
                   ['overtime', 'penalty', 'allowance', 'shift', 'weekend', 'holiday']):
                clause_rules = self._extract_rules_from_text(clause)
                rules.extend(clause_rules)
        
        logger.info(f"Extracted {len(rules)} rules from {len(clauses)} clauses")
        return rules
    
    def _extract_rules_from_text(self, clause: ClauseSegment) -> List[RuleExtraction]:
        """Extract rules from a single clause."""
        
        rules = []
        text = clause.text
        
        for rule_type, pattern in self.rule_patterns.items():
            matches = pattern.finditer(text)
            
            for match in matches:
                if rule_type == 'allowance':
                    rule_data = {
                        'amount': float(match.group(1)),
                        'type': 'fixed_amount',
                        'description': self._extract_rule_context(match, text)
                    }
                else:
                    multiplier = float(match.group(1))
                    rule_data = {
                        'multiplier': multiplier,
                        'base': 'ordinary_rate',
                        'description': self._extract_rule_context(match, text)
                    }
                
                rule = RuleExtraction(
                    key=f"{rule_type}_{len(rules) + 1}",
                    rule_type=rule_type,
                    rule_data=rule_data,
                    source_clause_id=clause.clause_id,
                    effective_from=None,
                    effective_to=None
                )
                
                rules.append(rule)
        
        return rules
    
    def _extract_rule_context(self, match: re.Match, text: str, context_chars: int = 100) -> str:
        """Extract context around a rule match."""
        
        start = max(0, match.start() - context_chars)
        end = min(len(text), match.end() + context_chars)
        
        return text[start:end].strip()
    
    def extract_tables_from_clauses(self, clauses: List[ClauseSegment]) -> List[Dict[str, Any]]:
        """Extract table-like structures from clauses."""
        
        tables = []
        
        for clause in clauses:
            if self._looks_like_table(clause.text):
                table_data = self._parse_table_text(clause.text)
                if table_data:
                    tables.append({
                        'source_clause_id': clause.clause_id,
                        'table_type': self._classify_table(clause.text),
                        'headers': table_data.get('headers', []),
                        'rows': table_data.get('rows', []),
                        'raw_text': clause.text
                    })
        
        logger.info(f"Extracted {len(tables)} tables from clauses")
        return tables
    
    def _looks_like_table(self, text: str) -> bool:
        """Check if text contains table-like structure."""
        
        # Simple heuristics for table detection
        lines = text.split('\n')
        
        # Look for multiple lines with similar structure
        structured_lines = 0
        for line in lines:
            # Count lines with multiple dollar amounts or numbers
            dollar_count = len(re.findall(r'\$\d+', line))
            number_count = len(re.findall(r'\b\d+\.\d+\b', line))
            
            if dollar_count >= 2 or number_count >= 2:
                structured_lines += 1
        
        return structured_lines >= 2
    
    def _parse_table_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse table structure from text."""
        
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        if len(lines) < 2:
            return None
        
        # Try to identify header row
        potential_headers = []
        data_rows = []
        
        for i, line in enumerate(lines):
            if self._looks_like_header(line):
                potential_headers = self._split_table_row(line)
                data_rows = lines[i+1:]
                break
        
        if not potential_headers:
            # No clear header, treat first line as header
            potential_headers = self._split_table_row(lines[0])
            data_rows = lines[1:]
        
        # Parse data rows
        parsed_rows = []
        for row_text in data_rows:
            row_data = self._split_table_row(row_text)
            if row_data:
                parsed_rows.append(row_data)
        
        return {
            'headers': potential_headers,
            'rows': parsed_rows
        }
    
    def _looks_like_header(self, line: str) -> bool:
        """Check if line looks like a table header."""
        
        # Headers often contain descriptive words without numbers
        has_descriptive_words = any(word in line.lower() for word in 
                                   ['classification', 'level', 'rate', 'hourly', 'weekly'])
        has_few_numbers = len(re.findall(r'\$?\d+', line)) <= 1
        
        return has_descriptive_words and has_few_numbers
    
    def _split_table_row(self, row_text: str) -> List[str]:
        """Split table row into columns."""
        
        # Try different separators
        separators = ['\t', '  ', ' | ', ',']
        
        best_split = [row_text]
        max_parts = 1
        
        for sep in separators:
            parts = [part.strip() for part in row_text.split(sep) if part.strip()]
            if len(parts) > max_parts:
                max_parts = len(parts)
                best_split = parts
        
        return best_split
    
    def _classify_table(self, table_text: str) -> str:
        """Classify the type of table."""
        
        text_lower = table_text.lower()
        
        if 'overtime' in text_lower:
            return 'overtime_rates'
        elif 'penalty' in text_lower:
            return 'penalty_rates'
        elif 'allowance' in text_lower:
            return 'allowances'
        elif any(word in text_lower for word in ['wage', 'salary', 'rate']):
            return 'wage_rates'
        else:
            return 'general'
    
    def export_rates_to_csv(self, rates: List[RateExtraction], output_path: str):
        """Export rates to CSV format."""
        
        data = []
        for rate in rates:
            data.append({
                'classification': rate.classification,
                'level': rate.level,
                'base_rate': rate.base_rate,
                'unit': rate.unit,
                'source_clause_id': rate.source_clause_id,
                'effective_from': rate.effective_from,
                'effective_to': rate.effective_to
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(rates)} rates to {output_path}")
    
    def export_rules_to_json(self, rules: List[RuleExtraction], output_path: str):
        """Export rules to JSON format."""
        
        import json
        
        rules_data = []
        for rule in rules:
            rules_data.append({
                'key': rule.key,
                'rule_type': rule.rule_type,
                'rule_data': rule.rule_data,
                'source_clause_id': rule.source_clause_id,
                'effective_from': rule.effective_from.isoformat() if rule.effective_from else None,
                'effective_to': rule.effective_to.isoformat() if rule.effective_to else None
            })
        
        with open(output_path, 'w') as f:
            json.dump(rules_data, f, indent=2)
        
        logger.info(f"Exported {len(rules)} rules to {output_path}")


__all__ = ['RatesRulesExtractor']