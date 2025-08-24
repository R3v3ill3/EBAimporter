"""
Rates and Rules Extraction Engine for parsing tables and normalizing business rules.
"""

import re
import json
import uuid
from datetime import datetime, date
from decimal import Decimal, InvalidOperation
from typing import List, Dict, Optional, Tuple, Any, Set, Union
from dataclasses import dataclass, asdict
from collections import defaultdict

import pandas as pd

from ..core.config import get_settings
from ..core.logging import LoggerMixin


@dataclass
class RateEntry:
    """Represents a single rate entry."""
    classification: str
    level: Optional[str]
    base_rate: Decimal
    unit: str  # 'hour', 'week', 'year'
    effective_from: Optional[date] = None
    effective_to: Optional[date] = None
    description: Optional[str] = None
    conditions: Optional[Dict[str, Any]] = None


@dataclass
class RuleEntry:
    """Represents a business rule."""
    key: str
    rule_type: str  # 'penalty', 'allowance', 'condition', 'procedure'
    parameters: Dict[str, Any]
    description: str
    conditions: Optional[Dict[str, Any]] = None
    priority: int = 0


@dataclass
class ExtractionResult:
    """Result of rates and rules extraction."""
    family_id: str
    rates: List[RateEntry]
    rules: List[RuleEntry]
    extraction_confidence: float
    errors: List[str]
    warnings: List[str]


class RatesRulesExtractor(LoggerMixin):
    """Extracts rates and rules from EA clause data."""
    
    # Currency patterns
    CURRENCY_PATTERNS = [
        r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)',  # $25.50, $1,234.56
        r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*dollars?',  # 25.50 dollars
        r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*per\s+hour',  # 25.50 per hour
    ]
    
    # Rate table patterns
    RATE_TABLE_HEADERS = [
        'classification', 'level', 'grade', 'rate', 'hourly', 'weekly', 'annual',
        'minimum', 'base', 'salary', 'wage', 'pay', 'amount'
    ]
    
    # Penalty/allowance patterns
    PENALTY_PATTERNS = {
        'overtime_weekday': [
            r'overtime.*?(\d+(?:\.\d+)?)\s*times?\s*ordinary',
            r'time\s*and\s*a\s*half',
            r'1\.5\s*times?\s*ordinary',
            r'150%.*?ordinary',
        ],
        'overtime_weekend': [
            r'weekend.*?overtime.*?(\d+(?:\.\d+)?)\s*times?',
            r'saturday.*?(\d+(?:\.\d+)?)\s*times?',
            r'sunday.*?(\d+(?:\.\d+)?)\s*times?',
            r'double\s*time',
            r'2\.0?\s*times?\s*ordinary',
        ],
        'shift_penalty': [
            r'shift.*?penalty.*?(\d+(?:\.\d+)?)',
            r'night.*?shift.*?(\d+(?:\.\d+)?)',
            r'evening.*?shift.*?(\d+(?:\.\d+)?)',
        ],
        'public_holiday': [
            r'public\s*holiday.*?(\d+(?:\.\d+)?)\s*times?',
            r'holiday.*?penalty.*?(\d+(?:\.\d+)?)',
            r'2\.5\s*times?\s*ordinary',
        ]
    }
    
    # Allowance patterns
    ALLOWANCE_PATTERNS = {
        'tool_allowance': [
            r'tool\s*allowance.*?\$?(\d+(?:\.\d{2})?)',
            r'tools?.*?\$?(\d+(?:\.\d{2})?)',
        ],
        'travel_allowance': [
            r'travel\s*allowance.*?\$?(\d+(?:\.\d{2})?)',
            r'mileage.*?\$?(\d+(?:\.\d{2})?)',
            r'kilometre.*?\$?(\d+(?:\.\d{2})?)',
        ],
        'meal_allowance': [
            r'meal\s*allowance.*?\$?(\d+(?:\.\d{2})?)',
            r'food\s*allowance.*?\$?(\d+(?:\.\d{2})?)',
        ],
        'first_aid_allowance': [
            r'first\s*aid.*?\$?(\d+(?:\.\d{2})?)',
        ],
        'supervisory_allowance': [
            r'supervisor.*?allowance.*?\$?(\d+(?:\.\d{2})?)',
            r'leading\s*hand.*?\$?(\d+(?:\.\d{2})?)',
        ]
    }
    
    def __init__(self):
        """Initialize the rates and rules extractor."""
        self.settings = get_settings()
    
    def extract_currency_amounts(self, text: str) -> List[Tuple[Decimal, str]]:
        """
        Extract currency amounts from text.
        
        Args:
            text: Input text
            
        Returns:
            List of (amount, context) tuples
        """
        amounts = []
        
        for pattern in self.CURRENCY_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                try:
                    amount_str = match.group(1).replace(',', '')
                    amount = Decimal(amount_str)
                    
                    # Get context around the match
                    start = max(0, match.start() - 20)
                    end = min(len(text), match.end() + 20)
                    context = text[start:end].strip()
                    
                    amounts.append((amount, context))
                    
                except (InvalidOperation, IndexError):
                    continue
        
        return amounts
    
    def detect_rate_tables_in_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect and extract rate tables from text.
        
        Args:
            text: Input text
            
        Returns:
            List of detected table dictionaries
        """
        tables = []
        
        # Split text into lines for table detection
        lines = text.split('\n')
        
        # Look for table-like structures
        table_start = None
        table_lines = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                if table_start is not None and len(table_lines) >= 2:
                    # End of potential table
                    table = self._parse_table_lines(table_lines, table_start)
                    if table:
                        tables.append(table)
                
                table_start = None
                table_lines = []
                continue
            
            # Check if line looks like a table header
            if self._is_table_header_line(line):
                table_start = i
                table_lines = [line]
            elif table_start is not None:
                # Check if line looks like table data
                if self._is_table_data_line(line):
                    table_lines.append(line)
                else:
                    # End of table
                    if len(table_lines) >= 2:
                        table = self._parse_table_lines(table_lines, table_start)
                        if table:
                            tables.append(table)
                    
                    table_start = None
                    table_lines = []
        
        # Handle table at end of text
        if table_start is not None and len(table_lines) >= 2:
            table = self._parse_table_lines(table_lines, table_start)
            if table:
                tables.append(table)
        
        return tables
    
    def _is_table_header_line(self, line: str) -> bool:
        """Check if line looks like a table header."""
        line_lower = line.lower()
        
        # Must contain at least 2 header keywords
        header_count = sum(1 for header in self.RATE_TABLE_HEADERS 
                          if header in line_lower)
        
        return header_count >= 2
    
    def _is_table_data_line(self, line: str) -> bool:
        """Check if line looks like table data."""
        # Must contain at least one currency amount
        currency_matches = 0
        for pattern in self.CURRENCY_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                currency_matches += 1
        
        # Must have multiple columns (spaces or tabs)
        has_columns = len(re.split(r'\s{2,}|\t', line)) >= 2
        
        return currency_matches > 0 and has_columns
    
    def _parse_table_lines(self, 
                          lines: List[str], 
                          start_line: int) -> Optional[Dict[str, Any]]:
        """
        Parse table lines into structured data.
        
        Args:
            lines: Table lines
            start_line: Starting line number
            
        Returns:
            Parsed table dictionary or None
        """
        if len(lines) < 2:
            return None
        
        try:
            # Use pandas to parse the table
            # Join lines and split by multiple spaces or tabs
            table_text = '\n'.join(lines)
            
            # Try different separators
            separators = [r'\s{2,}', r'\t', r'\|']
            
            for sep in separators:
                try:
                    # Split each line by separator
                    rows = []
                    for line in lines:
                        columns = re.split(sep, line.strip())
                        if len(columns) >= 2:
                            rows.append(columns)
                    
                    if len(rows) >= 2:
                        # Create DataFrame
                        df = pd.DataFrame(rows[1:], columns=rows[0])
                        
                        # Clean column names
                        df.columns = [col.strip().lower() for col in df.columns]
                        
                        return {
                            'type': 'rate_table',
                            'start_line': start_line,
                            'dataframe': df,
                            'raw_lines': lines,
                            'confidence': self._calculate_table_confidence(df)
                        }
                
                except Exception:
                    continue
            
        except Exception as e:
            self.logger.debug(f"Failed to parse table: {e}")
        
        return None
    
    def _calculate_table_confidence(self, df: pd.DataFrame) -> float:
        """Calculate confidence score for detected table."""
        confidence = 0.0
        
        # Check for rate-related columns
        rate_columns = 0
        for col in df.columns:
            if any(keyword in col for keyword in 
                  ['rate', 'wage', 'salary', 'pay', 'amount', 'hourly', 'weekly']):
                rate_columns += 1
        
        confidence += min(0.4, rate_columns * 0.2)
        
        # Check for classification columns
        class_columns = 0
        for col in df.columns:
            if any(keyword in col for keyword in 
                  ['classification', 'level', 'grade', 'position', 'role']):
                class_columns += 1
        
        confidence += min(0.3, class_columns * 0.3)
        
        # Check for currency values in data
        currency_cells = 0
        total_cells = 0
        
        for col in df.columns:
            for val in df[col]:
                total_cells += 1
                if isinstance(val, str):
                    for pattern in self.CURRENCY_PATTERNS:
                        if re.search(pattern, val, re.IGNORECASE):
                            currency_cells += 1
                            break
        
        if total_cells > 0:
            confidence += (currency_cells / total_cells) * 0.3
        
        return min(1.0, confidence)
    
    def extract_rates_from_tables(self, tables: List[Dict[str, Any]]) -> List[RateEntry]:
        """
        Extract rate entries from detected tables.
        
        Args:
            tables: List of detected tables
            
        Returns:
            List of rate entries
        """
        rates = []
        
        for table in tables:
            if table.get('confidence', 0) < 0.3:
                continue
            
            df = table['dataframe']
            
            # Identify columns
            classification_col = self._find_column(df, ['classification', 'position', 'role', 'job'])
            level_col = self._find_column(df, ['level', 'grade', 'step'])
            rate_col = self._find_column(df, ['rate', 'hourly', 'wage', 'salary', 'pay', 'amount'])
            
            if not classification_col or not rate_col:
                self.logger.debug(f"Could not identify required columns in table")
                continue
            
            # Extract rates
            for _, row in df.iterrows():
                try:
                    classification = str(row[classification_col]).strip()
                    level = str(row[level_col]).strip() if level_col else None
                    rate_text = str(row[rate_col]).strip()
                    
                    # Extract currency amount
                    amounts = self.extract_currency_amounts(rate_text)
                    if not amounts:
                        continue
                    
                    base_rate = amounts[0][0]  # Take first amount
                    
                    # Determine unit
                    unit = 'hour'  # Default
                    if any(word in rate_text.lower() for word in ['weekly', 'week']):
                        unit = 'week'
                    elif any(word in rate_text.lower() for word in ['annual', 'year', 'yearly']):
                        unit = 'year'
                    
                    rate_entry = RateEntry(
                        classification=classification,
                        level=level,
                        base_rate=base_rate,
                        unit=unit,
                        description=f"From table at line {table['start_line']}"
                    )
                    
                    rates.append(rate_entry)
                    
                except Exception as e:
                    self.logger.debug(f"Failed to extract rate from row: {e}")
                    continue
        
        return rates
    
    def _find_column(self, df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
        """Find column matching keywords."""
        for col in df.columns:
            for keyword in keywords:
                if keyword in col.lower():
                    return col
        return None
    
    def extract_penalty_rules(self, clauses: List[Dict]) -> List[RuleEntry]:
        """
        Extract penalty rules from clauses.
        
        Args:
            clauses: List of clause dictionaries
            
        Returns:
            List of penalty rule entries
        """
        rules = []
        
        for clause in clauses:
            text = clause.get('text', '')
            clause_id = clause.get('clause_id', '')
            
            for rule_key, patterns in self.PENALTY_PATTERNS.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    
                    for match in matches:
                        try:
                            # Extract multiplier
                            multiplier = None
                            if match.groups():
                                multiplier = float(match.group(1))
                            elif 'time and a half' in match.group(0).lower():
                                multiplier = 1.5
                            elif 'double time' in match.group(0).lower():
                                multiplier = 2.0
                            elif '150%' in match.group(0):
                                multiplier = 1.5
                            
                            if multiplier:
                                rule = RuleEntry(
                                    key=f"{rule_key}_{clause_id}",
                                    rule_type='penalty',
                                    parameters={
                                        'multiplier': multiplier,
                                        'applies_to': 'ordinary_rate',
                                        'pattern_matched': pattern
                                    },
                                    description=f"{rule_key.replace('_', ' ').title()}: {multiplier}x ordinary rate",
                                    conditions={
                                        'source_clause': clause_id,
                                        'matched_text': match.group(0)
                                    }
                                )
                                rules.append(rule)
                                break  # Only one match per pattern per clause
                        
                        except (ValueError, IndexError) as e:
                            self.logger.debug(f"Failed to extract penalty rule: {e}")
                            continue
        
        return rules
    
    def extract_allowance_rules(self, clauses: List[Dict]) -> List[RuleEntry]:
        """
        Extract allowance rules from clauses.
        
        Args:
            clauses: List of clause dictionaries
            
        Returns:
            List of allowance rule entries
        """
        rules = []
        
        for clause in clauses:
            text = clause.get('text', '')
            clause_id = clause.get('clause_id', '')
            
            for rule_key, patterns in self.ALLOWANCE_PATTERNS.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    
                    for match in matches:
                        try:
                            # Extract amount
                            amount = None
                            if match.groups():
                                amount_str = match.group(1)
                                amount = float(amount_str)
                            
                            if amount:
                                # Determine unit
                                unit = 'per_day'  # Default
                                if any(word in text.lower() for word in ['per week', 'weekly']):
                                    unit = 'per_week'
                                elif any(word in text.lower() for word in ['per hour', 'hourly']):
                                    unit = 'per_hour'
                                elif any(word in text.lower() for word in ['per kilometre', 'per km']):
                                    unit = 'per_km'
                                
                                rule = RuleEntry(
                                    key=f"{rule_key}_{clause_id}",
                                    rule_type='allowance',
                                    parameters={
                                        'amount': amount,
                                        'unit': unit,
                                        'currency': 'AUD',
                                        'pattern_matched': pattern
                                    },
                                    description=f"{rule_key.replace('_', ' ').title()}: ${amount} {unit}",
                                    conditions={
                                        'source_clause': clause_id,
                                        'matched_text': match.group(0)
                                    }
                                )
                                rules.append(rule)
                                break  # Only one match per pattern per clause
                        
                        except (ValueError, IndexError) as e:
                            self.logger.debug(f"Failed to extract allowance rule: {e}")
                            continue
        
        return rules
    
    def extract_procedural_rules(self, clauses: List[Dict]) -> List[RuleEntry]:
        """
        Extract procedural rules from clauses.
        
        Args:
            clauses: List of clause dictionaries
            
        Returns:
            List of procedural rule entries
        """
        rules = []
        
        # Patterns for procedural rules
        procedure_patterns = {
            'notice_period': [
                r'(\d+)\s*(?:days?|weeks?|months?)\s*notice',
                r'notice.*?(\d+)\s*(?:days?|weeks?|months?)',
            ],
            'probation_period': [
                r'probation.*?(\d+)\s*(?:days?|weeks?|months?)',
                r'(\d+)\s*(?:days?|weeks?|months?)\s*probation',
            ],
            'annual_leave_entitlement': [
                r'(\d+)\s*(?:days?|weeks?)\s*annual\s*leave',
                r'annual\s*leave.*?(\d+)\s*(?:days?|weeks?)',
            ],
            'sick_leave_entitlement': [
                r'(\d+)\s*(?:days?|weeks?)\s*sick\s*leave',
                r'sick\s*leave.*?(\d+)\s*(?:days?|weeks?)',
            ]
        }
        
        for clause in clauses:
            text = clause.get('text', '')
            clause_id = clause.get('clause_id', '')
            
            for rule_key, patterns in procedure_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    
                    for match in matches:
                        try:
                            # Extract number
                            number = int(match.group(1))
                            
                            # Determine unit
                            unit = 'days'
                            if 'week' in match.group(0).lower():
                                unit = 'weeks'
                            elif 'month' in match.group(0).lower():
                                unit = 'months'
                            
                            rule = RuleEntry(
                                key=f"{rule_key}_{clause_id}",
                                rule_type='procedure',
                                parameters={
                                    'value': number,
                                    'unit': unit,
                                    'pattern_matched': pattern
                                },
                                description=f"{rule_key.replace('_', ' ').title()}: {number} {unit}",
                                conditions={
                                    'source_clause': clause_id,
                                    'matched_text': match.group(0)
                                }
                            )
                            rules.append(rule)
                            break  # Only one match per pattern per clause
                        
                        except (ValueError, IndexError) as e:
                            self.logger.debug(f"Failed to extract procedural rule: {e}")
                            continue
        
        return rules
    
    def extract_family_rates_and_rules(self, 
                                     family_id: str, 
                                     family_clauses: List[Dict]) -> ExtractionResult:
        """
        Extract rates and rules from family clauses.
        
        Args:
            family_id: Family identifier
            family_clauses: List of family clause dictionaries
            
        Returns:
            Extraction result
        """
        self.logger.info(f"Extracting rates and rules for family {family_id}")
        
        errors = []
        warnings = []
        
        # Extract rates from tables
        all_text = '\n'.join(clause.get('text', '') for clause in family_clauses)
        tables = self.detect_rate_tables_in_text(all_text)
        
        self.logger.info(f"Detected {len(tables)} potential rate tables")
        
        rates = self.extract_rates_from_tables(tables)
        
        if not rates:
            # Try extracting from individual clauses
            for clause in family_clauses:
                if any(keyword in clause.get('text', '').lower() 
                      for keyword in ['rate', 'wage', 'salary', 'pay']):
                    
                    amounts = self.extract_currency_amounts(clause.get('text', ''))
                    if amounts:
                        # Create basic rate entry
                        rate = RateEntry(
                            classification=clause.get('heading', 'Unspecified'),
                            level=None,
                            base_rate=amounts[0][0],
                            unit='hour',
                            description=f"Extracted from clause {clause.get('clause_id', '')}"
                        )
                        rates.append(rate)
        
        # Extract rules
        penalty_rules = self.extract_penalty_rules(family_clauses)
        allowance_rules = self.extract_allowance_rules(family_clauses)
        procedural_rules = self.extract_procedural_rules(family_clauses)
        
        all_rules = penalty_rules + allowance_rules + procedural_rules
        
        # Calculate confidence
        confidence = self._calculate_extraction_confidence(rates, all_rules, family_clauses)
        
        if not rates:
            warnings.append("No rates extracted from family clauses")
        
        if not all_rules:
            warnings.append("No rules extracted from family clauses")
        
        self.logger.info(f"Extracted {len(rates)} rates and {len(all_rules)} rules "
                        f"with confidence {confidence:.2f}")
        
        return ExtractionResult(
            family_id=family_id,
            rates=rates,
            rules=all_rules,
            extraction_confidence=confidence,
            errors=errors,
            warnings=warnings
        )
    
    def _calculate_extraction_confidence(self, 
                                       rates: List[RateEntry], 
                                       rules: List[RuleEntry],
                                       clauses: List[Dict]) -> float:
        """Calculate confidence score for extraction."""
        confidence = 0.0
        
        # Rate extraction confidence
        if rates:
            confidence += 0.4
            
            # Bonus for multiple rates
            if len(rates) > 1:
                confidence += 0.1
            
            # Bonus for different rate types
            units = set(rate.unit for rate in rates)
            confidence += len(units) * 0.05
        
        # Rule extraction confidence
        if rules:
            confidence += 0.3
            
            # Bonus for different rule types
            rule_types = set(rule.rule_type for rule in rules)
            confidence += len(rule_types) * 0.05
        
        # Coverage confidence
        rate_clauses = sum(1 for clause in clauses 
                          if any(keyword in clause.get('text', '').lower() 
                                for keyword in ['rate', 'wage', 'salary']))
        
        if rate_clauses > 0:
            coverage = len(rates) / rate_clauses
            confidence += min(0.2, coverage * 0.2)
        
        return min(1.0, confidence)
    
    def save_family_rates_and_rules(self, 
                                   extraction_result: ExtractionResult,
                                   family_dir: str):
        """
        Save extracted rates and rules to family directory.
        
        Args:
            extraction_result: Extraction result
            family_dir: Family directory path
        """
        family_path = Path(family_dir)
        family_path.mkdir(parents=True, exist_ok=True)
        
        # Save rates as CSV
        if extraction_result.rates:
            rates_data = []
            for rate in extraction_result.rates:
                rates_data.append({
                    'classification': rate.classification,
                    'level': rate.level,
                    'base_rate': float(rate.base_rate),
                    'unit': rate.unit,
                    'effective_from': rate.effective_from.isoformat() if rate.effective_from else None,
                    'effective_to': rate.effective_to.isoformat() if rate.effective_to else None,
                    'description': rate.description,
                    'conditions': json.dumps(rate.conditions) if rate.conditions else None
                })
            
            rates_df = pd.DataFrame(rates_data)
            rates_file = family_path / "family_rates.csv"
            rates_df.to_csv(rates_file, index=False)
            
            self.logger.info(f"Saved {len(rates_data)} rates to {rates_file}")
        
        # Save rules as JSON
        if extraction_result.rules:
            rules_data = {}
            for rule in extraction_result.rules:
                rules_data[rule.key] = {
                    'rule_type': rule.rule_type,
                    'parameters': rule.parameters,
                    'description': rule.description,
                    'conditions': rule.conditions,
                    'priority': rule.priority
                }
            
            rules_file = family_path / "family_rules.json"
            with open(rules_file, 'w', encoding='utf-8') as f:
                json.dump(rules_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved {len(rules_data)} rules to {rules_file}")
        
        # Save extraction metadata
        metadata = {
            'family_id': extraction_result.family_id,
            'extraction_date': datetime.now().isoformat(),
            'extraction_confidence': extraction_result.extraction_confidence,
            'rates_count': len(extraction_result.rates),
            'rules_count': len(extraction_result.rules),
            'errors': extraction_result.errors,
            'warnings': extraction_result.warnings
        }
        
        metadata_file = family_path / "extraction_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)


def create_rates_rules_extractor() -> RatesRulesExtractor:
    """Factory function to create a rates and rules extractor."""
    return RatesRulesExtractor()