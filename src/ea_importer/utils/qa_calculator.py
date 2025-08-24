"""
QA Calculator - Implements smoke testing with synthetic worker scenarios.
"""

import json
import uuid
from datetime import datetime, date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import random

import pandas as pd

from ..core.config import get_settings
from ..core.logging import LoggerMixin
from ..utils.rates_rules_extractor import RateEntry, RuleEntry


class WorkerType(Enum):
    """Types of workers for testing scenarios."""
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CASUAL = "casual"
    APPRENTICE = "apprentice"


class ShiftType(Enum):
    """Types of shifts."""
    ORDINARY = "ordinary"
    NIGHT = "night"
    WEEKEND = "weekend"
    PUBLIC_HOLIDAY = "public_holiday"


@dataclass
class WorkerScenario:
    """Represents a synthetic worker scenario for testing."""
    worker_id: str
    worker_type: WorkerType
    classification: str
    level: Optional[str]
    base_hourly_rate: Decimal
    
    # Work pattern
    ordinary_hours: float
    overtime_hours: float = 0.0
    night_hours: float = 0.0
    weekend_hours: float = 0.0
    public_holiday_hours: float = 0.0
    
    # Allowances
    tool_allowance: bool = False
    travel_distance_km: float = 0.0
    meal_allowance: bool = False
    first_aid_allowance: bool = False
    supervisory_allowance: bool = False
    
    # Scenario metadata
    scenario_name: str = ""
    week_ending: Optional[date] = None


@dataclass
class PayCalculationResult:
    """Result of pay calculation for a worker scenario."""
    worker_id: str
    scenario_name: str
    
    # Hours breakdown
    ordinary_hours: float
    overtime_hours: float
    penalty_hours: Dict[str, float]
    
    # Rate breakdown
    base_rate: Decimal
    overtime_rate: Decimal
    penalty_rates: Dict[str, Decimal]
    
    # Pay breakdown
    ordinary_pay: Decimal
    overtime_pay: Decimal
    penalty_pay: Dict[str, Decimal]
    allowances: Dict[str, Decimal]
    
    # Totals
    gross_pay: Decimal
    total_hours: float
    
    # Calculation metadata
    rules_applied: List[str]
    warnings: List[str]
    errors: List[str]
    calculation_date: datetime
    
    @property
    def total_penalty_pay(self) -> Decimal:
        """Calculate total penalty pay."""
        return sum(self.penalty_pay.values())
    
    @property
    def total_allowances(self) -> Decimal:
        """Calculate total allowances."""
        return sum(self.allowances.values())


@dataclass
class QATestResult:
    """Result of QA testing for a family."""
    family_id: str
    test_date: datetime
    
    scenarios_tested: int
    scenarios_passed: int
    scenarios_failed: int
    scenarios_with_warnings: int
    
    calculation_results: List[PayCalculationResult]
    anomalies: List[Dict[str, Any]]
    summary_statistics: Dict[str, Any]
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.scenarios_tested == 0:
            return 0.0
        return self.scenarios_passed / self.scenarios_tested
    
    @property
    def has_critical_anomalies(self) -> bool:
        """Check if there are critical anomalies."""
        return any(anomaly.get('severity') == 'critical' for anomaly in self.anomalies)


class PayCalculator(LoggerMixin):
    """Calculates pay based on rates, rules, and worker scenarios."""
    
    def __init__(self, rates: List[RateEntry], rules: List[RuleEntry]):
        """
        Initialize pay calculator.
        
        Args:
            rates: List of rate entries
            rules: List of rule entries
        """
        self.rates = rates
        self.rules = rules
        
        # Build lookup tables
        self.rates_lookup = self._build_rates_lookup(rates)
        self.rules_lookup = self._build_rules_lookup(rules)
    
    def _build_rates_lookup(self, rates: List[RateEntry]) -> Dict[str, RateEntry]:
        """Build rates lookup table."""
        lookup = {}
        for rate in rates:
            key = f"{rate.classification}_{rate.level or 'base'}"
            lookup[key] = rate
        return lookup
    
    def _build_rules_lookup(self, rules: List[RuleEntry]) -> Dict[str, RuleEntry]:
        """Build rules lookup table."""
        lookup = {}
        for rule in rules:
            lookup[rule.key] = rule
        return lookup
    
    def get_base_rate(self, classification: str, level: Optional[str] = None) -> Optional[Decimal]:
        """
        Get base hourly rate for classification and level.
        
        Args:
            classification: Worker classification
            level: Worker level (optional)
            
        Returns:
            Base hourly rate or None if not found
        """
        key = f"{classification}_{level or 'base'}"
        rate_entry = self.rates_lookup.get(key)
        
        if rate_entry:
            # Convert to hourly if needed
            if rate_entry.unit == 'hour':
                return rate_entry.base_rate
            elif rate_entry.unit == 'week':
                return rate_entry.base_rate / Decimal('38')  # Assume 38 hour week
            elif rate_entry.unit == 'year':
                return rate_entry.base_rate / Decimal('1976')  # 52 weeks * 38 hours
        
        return None
    
    def get_penalty_multiplier(self, penalty_type: str) -> Decimal:
        """
        Get penalty multiplier for a given penalty type.
        
        Args:
            penalty_type: Type of penalty
            
        Returns:
            Penalty multiplier
        """
        # Try to find specific rule
        for key, rule in self.rules_lookup.items():
            if penalty_type in key and rule.rule_type == 'penalty':
                return Decimal(str(rule.parameters.get('multiplier', 1.0)))
        
        # Default multipliers
        defaults = {
            'overtime_weekday': Decimal('1.5'),
            'overtime_weekend': Decimal('2.0'),
            'night_shift': Decimal('1.15'),
            'public_holiday': Decimal('2.5'),
        }
        
        return defaults.get(penalty_type, Decimal('1.0'))
    
    def get_allowance_amount(self, allowance_type: str) -> Decimal:
        """
        Get allowance amount for a given allowance type.
        
        Args:
            allowance_type: Type of allowance
            
        Returns:
            Allowance amount
        """
        for key, rule in self.rules_lookup.items():
            if allowance_type in key and rule.rule_type == 'allowance':
                return Decimal(str(rule.parameters.get('amount', 0.0)))
        
        return Decimal('0.0')
    
    def calculate_pay(self, scenario: WorkerScenario) -> PayCalculationResult:
        """
        Calculate pay for a worker scenario.
        
        Args:
            scenario: Worker scenario
            
        Returns:
            Pay calculation result
        """
        warnings = []
        errors = []
        rules_applied = []
        
        # Get base rate
        base_rate = self.get_base_rate(scenario.classification, scenario.level)
        
        if not base_rate:
            errors.append(f"No base rate found for {scenario.classification} {scenario.level}")
            base_rate = Decimal('25.00')  # Fallback rate
            warnings.append(f"Using fallback rate: ${base_rate}/hour")
        
        # Calculate ordinary pay
        ordinary_pay = Decimal(str(scenario.ordinary_hours)) * base_rate
        
        # Calculate overtime pay
        overtime_multiplier = self.get_penalty_multiplier('overtime_weekday')
        overtime_rate = base_rate * overtime_multiplier
        overtime_pay = Decimal(str(scenario.overtime_hours)) * overtime_rate
        
        if scenario.overtime_hours > 0:
            rules_applied.append(f"overtime_weekday_{overtime_multiplier}")
        
        # Calculate penalty pay
        penalty_hours = {}
        penalty_rates = {}
        penalty_pay = {}
        
        if scenario.night_hours > 0:
            penalty_hours['night_shift'] = scenario.night_hours
            penalty_rates['night_shift'] = base_rate * self.get_penalty_multiplier('night_shift')
            penalty_pay['night_shift'] = Decimal(str(scenario.night_hours)) * penalty_rates['night_shift']
            rules_applied.append('night_shift_penalty')
        
        if scenario.weekend_hours > 0:
            penalty_hours['weekend'] = scenario.weekend_hours
            penalty_rates['weekend'] = base_rate * self.get_penalty_multiplier('overtime_weekend')
            penalty_pay['weekend'] = Decimal(str(scenario.weekend_hours)) * penalty_rates['weekend']
            rules_applied.append('weekend_penalty')
        
        if scenario.public_holiday_hours > 0:
            penalty_hours['public_holiday'] = scenario.public_holiday_hours
            penalty_rates['public_holiday'] = base_rate * self.get_penalty_multiplier('public_holiday')
            penalty_pay['public_holiday'] = Decimal(str(scenario.public_holiday_hours)) * penalty_rates['public_holiday']
            rules_applied.append('public_holiday_penalty')
        
        # Calculate allowances
        allowances = {}
        
        if scenario.tool_allowance:
            allowances['tool'] = self.get_allowance_amount('tool_allowance')
            if allowances['tool'] > 0:
                rules_applied.append('tool_allowance')
        
        if scenario.travel_distance_km > 0:
            travel_rate = self.get_allowance_amount('travel_allowance')
            allowances['travel'] = Decimal(str(scenario.travel_distance_km)) * travel_rate
            if travel_rate > 0:
                rules_applied.append('travel_allowance')
        
        if scenario.meal_allowance:
            allowances['meal'] = self.get_allowance_amount('meal_allowance')
            if allowances['meal'] > 0:
                rules_applied.append('meal_allowance')
        
        if scenario.first_aid_allowance:
            allowances['first_aid'] = self.get_allowance_amount('first_aid_allowance')
            if allowances['first_aid'] > 0:
                rules_applied.append('first_aid_allowance')
        
        if scenario.supervisory_allowance:
            allowances['supervisory'] = self.get_allowance_amount('supervisory_allowance')
            if allowances['supervisory'] > 0:
                rules_applied.append('supervisory_allowance')
        
        # Calculate totals
        total_penalty_pay = sum(penalty_pay.values())
        total_allowances = sum(allowances.values())
        gross_pay = ordinary_pay + overtime_pay + total_penalty_pay + total_allowances
        total_hours = (scenario.ordinary_hours + scenario.overtime_hours + 
                      scenario.night_hours + scenario.weekend_hours + 
                      scenario.public_holiday_hours)
        
        # Round monetary amounts
        gross_pay = gross_pay.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        
        return PayCalculationResult(
            worker_id=scenario.worker_id,
            scenario_name=scenario.scenario_name,
            ordinary_hours=scenario.ordinary_hours,
            overtime_hours=scenario.overtime_hours,
            penalty_hours=penalty_hours,
            base_rate=base_rate,
            overtime_rate=overtime_rate,
            penalty_rates=penalty_rates,
            ordinary_pay=ordinary_pay,
            overtime_pay=overtime_pay,
            penalty_pay=penalty_pay,
            allowances=allowances,
            gross_pay=gross_pay,
            total_hours=total_hours,
            rules_applied=rules_applied,
            warnings=warnings,
            errors=errors,
            calculation_date=datetime.now()
        )


class SyntheticScenarioGenerator(LoggerMixin):
    """Generates synthetic worker scenarios for testing."""
    
    def __init__(self, rates: List[RateEntry], rules: List[RuleEntry]):
        """
        Initialize scenario generator.
        
        Args:
            rates: List of rate entries
            rules: List of rule entries
        """
        self.rates = rates
        self.rules = rules
        self.settings = get_settings()
    
    def get_available_classifications(self) -> List[Tuple[str, Optional[str]]]:
        """Get available worker classifications from rates."""
        classifications = []
        for rate in self.rates:
            classifications.append((rate.classification, rate.level))
        return list(set(classifications))
    
    def generate_scenarios(self, num_scenarios: int = 20) -> List[WorkerScenario]:
        """
        Generate synthetic worker scenarios.
        
        Args:
            num_scenarios: Number of scenarios to generate
            
        Returns:
            List of worker scenarios
        """
        scenarios = []
        classifications = self.get_available_classifications()
        
        if not classifications:
            self.logger.warning("No classifications available for scenario generation")
            return scenarios
        
        for i in range(num_scenarios):
            # Random classification
            classification, level = random.choice(classifications)
            
            # Get base rate
            calculator = PayCalculator(self.rates, self.rules)
            base_rate = calculator.get_base_rate(classification, level)
            
            if not base_rate:
                continue
            
            # Random worker type
            worker_type = random.choice(list(WorkerType))
            
            # Generate work hours based on worker type
            if worker_type == WorkerType.FULL_TIME:
                ordinary_hours = random.uniform(35, 40)
                overtime_hours = random.uniform(0, 10)
            elif worker_type == WorkerType.PART_TIME:
                ordinary_hours = random.uniform(15, 30)
                overtime_hours = random.uniform(0, 5)
            elif worker_type == WorkerType.CASUAL:
                ordinary_hours = random.uniform(10, 35)
                overtime_hours = random.uniform(0, 8)
            else:  # APPRENTICE
                ordinary_hours = random.uniform(30, 38)
                overtime_hours = random.uniform(0, 5)
            
            # Random penalty hours
            night_hours = random.uniform(0, ordinary_hours * 0.3) if random.random() < 0.3 else 0
            weekend_hours = random.uniform(0, 16) if random.random() < 0.4 else 0
            public_holiday_hours = random.uniform(0, 8) if random.random() < 0.1 else 0
            
            # Random allowances
            tool_allowance = random.random() < 0.5
            travel_distance_km = random.uniform(0, 100) if random.random() < 0.3 else 0
            meal_allowance = random.random() < 0.2
            first_aid_allowance = random.random() < 0.1
            supervisory_allowance = random.random() < 0.15
            
            scenario = WorkerScenario(
                worker_id=f"W{i+1:03d}",
                worker_type=worker_type,
                classification=classification,
                level=level,
                base_hourly_rate=base_rate,
                ordinary_hours=round(ordinary_hours, 2),
                overtime_hours=round(overtime_hours, 2),
                night_hours=round(night_hours, 2),
                weekend_hours=round(weekend_hours, 2),
                public_holiday_hours=round(public_holiday_hours, 2),
                tool_allowance=tool_allowance,
                travel_distance_km=round(travel_distance_km, 1),
                meal_allowance=meal_allowance,
                first_aid_allowance=first_aid_allowance,
                supervisory_allowance=supervisory_allowance,
                scenario_name=f"{worker_type.value}_{classification}_{i+1}",
                week_ending=date.today()
            )
            
            scenarios.append(scenario)
        
        self.logger.info(f"Generated {len(scenarios)} synthetic scenarios")
        return scenarios


class QATestRunner(LoggerMixin):
    """Runs QA tests on EA families."""
    
    def __init__(self):
        """Initialize QA test runner."""
        self.settings = get_settings()
    
    def detect_anomalies(self, results: List[PayCalculationResult]) -> List[Dict[str, Any]]:
        """
        Detect anomalies in calculation results.
        
        Args:
            results: List of calculation results
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        for result in results:
            # Check for negative pay
            if result.gross_pay < 0:
                anomalies.append({
                    'type': 'negative_pay',
                    'severity': 'critical',
                    'worker_id': result.worker_id,
                    'value': float(result.gross_pay),
                    'description': f"Worker {result.worker_id} has negative gross pay: ${result.gross_pay}"
                })
            
            # Check for zero pay with positive hours
            if result.gross_pay == 0 and result.total_hours > 0:
                anomalies.append({
                    'type': 'zero_pay_with_hours',
                    'severity': 'critical',
                    'worker_id': result.worker_id,
                    'hours': result.total_hours,
                    'description': f"Worker {result.worker_id} has zero pay despite {result.total_hours} hours"
                })
            
            # Check for extremely high hourly rate
            if result.total_hours > 0:
                effective_hourly_rate = result.gross_pay / Decimal(str(result.total_hours))
                if effective_hourly_rate > 200:  # $200/hour threshold
                    anomalies.append({
                        'type': 'excessive_hourly_rate',
                        'severity': 'warning',
                        'worker_id': result.worker_id,
                        'rate': float(effective_hourly_rate),
                        'description': f"Worker {result.worker_id} has excessive effective hourly rate: ${effective_hourly_rate:.2f}"
                    })
            
            # Check for calculation errors
            if result.errors:
                anomalies.append({
                    'type': 'calculation_error',
                    'severity': 'critical',
                    'worker_id': result.worker_id,
                    'errors': result.errors,
                    'description': f"Worker {result.worker_id} has calculation errors: {'; '.join(result.errors)}"
                })
            
            # Check for missing rules
            if result.total_hours > 40 and 'overtime' not in ' '.join(result.rules_applied):
                anomalies.append({
                    'type': 'missing_overtime_rule',
                    'severity': 'warning',
                    'worker_id': result.worker_id,
                    'hours': result.total_hours,
                    'description': f"Worker {result.worker_id} worked {result.total_hours} hours but no overtime rule applied"
                })
        
        return anomalies
    
    def calculate_summary_statistics(self, results: List[PayCalculationResult]) -> Dict[str, Any]:
        """
        Calculate summary statistics for results.
        
        Args:
            results: List of calculation results
            
        Returns:
            Summary statistics dictionary
        """
        if not results:
            return {}
        
        gross_pays = [float(r.gross_pay) for r in results]
        total_hours = [r.total_hours for r in results]
        
        stats = {
            'total_scenarios': len(results),
            'gross_pay': {
                'min': min(gross_pays),
                'max': max(gross_pays),
                'mean': sum(gross_pays) / len(gross_pays),
                'median': sorted(gross_pays)[len(gross_pays)//2]
            },
            'hours': {
                'min': min(total_hours),
                'max': max(total_hours),
                'mean': sum(total_hours) / len(total_hours),
                'median': sorted(total_hours)[len(total_hours)//2]
            },
            'rules_usage': {},
            'error_rate': len([r for r in results if r.errors]) / len(results),
            'warning_rate': len([r for r in results if r.warnings]) / len(results)
        }
        
        # Count rule usage
        all_rules = []
        for result in results:
            all_rules.extend(result.rules_applied)
        
        from collections import Counter
        rule_counts = Counter(all_rules)
        stats['rules_usage'] = dict(rule_counts)
        
        return stats
    
    def run_smoke_test(self, 
                      family_id: str,
                      rates: List[RateEntry],
                      rules: List[RuleEntry],
                      num_scenarios: int = 20) -> QATestResult:
        """
        Run smoke test for a family.
        
        Args:
            family_id: Family identifier
            rates: Family rates
            rules: Family rules
            num_scenarios: Number of test scenarios
            
        Returns:
            QA test result
        """
        self.logger.info(f"Running smoke test for family {family_id} with {num_scenarios} scenarios")
        
        # Generate scenarios
        generator = SyntheticScenarioGenerator(rates, rules)
        scenarios = generator.generate_scenarios(num_scenarios)
        
        if not scenarios:
            self.logger.error("No scenarios generated for testing")
            return QATestResult(
                family_id=family_id,
                test_date=datetime.now(),
                scenarios_tested=0,
                scenarios_passed=0,
                scenarios_failed=0,
                scenarios_with_warnings=0,
                calculation_results=[],
                anomalies=[],
                summary_statistics={}
            )
        
        # Calculate pay for each scenario
        calculator = PayCalculator(rates, rules)
        results = []
        
        for scenario in scenarios:
            try:
                result = calculator.calculate_pay(scenario)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to calculate pay for scenario {scenario.worker_id}: {e}")
                # Create error result
                error_result = PayCalculationResult(
                    worker_id=scenario.worker_id,
                    scenario_name=scenario.scenario_name,
                    ordinary_hours=scenario.ordinary_hours,
                    overtime_hours=scenario.overtime_hours,
                    penalty_hours={},
                    base_rate=Decimal('0'),
                    overtime_rate=Decimal('0'),
                    penalty_rates={},
                    ordinary_pay=Decimal('0'),
                    overtime_pay=Decimal('0'),
                    penalty_pay={},
                    allowances={},
                    gross_pay=Decimal('0'),
                    total_hours=0,
                    rules_applied=[],
                    warnings=[],
                    errors=[str(e)],
                    calculation_date=datetime.now()
                )
                results.append(error_result)
        
        # Analyze results
        anomalies = self.detect_anomalies(results)
        summary_stats = self.calculate_summary_statistics(results)
        
        # Count scenarios by status
        scenarios_passed = len([r for r in results if not r.errors])
        scenarios_failed = len([r for r in results if r.errors])
        scenarios_with_warnings = len([r for r in results if r.warnings])
        
        test_result = QATestResult(
            family_id=family_id,
            test_date=datetime.now(),
            scenarios_tested=len(scenarios),
            scenarios_passed=scenarios_passed,
            scenarios_failed=scenarios_failed,
            scenarios_with_warnings=scenarios_with_warnings,
            calculation_results=results,
            anomalies=anomalies,
            summary_statistics=summary_stats
        )
        
        self.logger.info(f"Smoke test completed: {test_result.success_rate:.1%} success rate, "
                        f"{len(anomalies)} anomalies detected")
        
        return test_result
    
    def save_qa_report(self, test_result: QATestResult, output_dir: Path):
        """
        Save QA test report.
        
        Args:
            test_result: QA test result
            output_dir: Output directory
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary CSV
        summary_data = []
        for result in test_result.calculation_results:
            summary_data.append({
                'worker_id': result.worker_id,
                'scenario_name': result.scenario_name,
                'total_hours': result.total_hours,
                'gross_pay': float(result.gross_pay),
                'base_rate': float(result.base_rate),
                'rules_applied': len(result.rules_applied),
                'warnings': len(result.warnings),
                'errors': len(result.errors),
                'status': 'PASS' if not result.errors else 'FAIL'
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = output_dir / "smoketest_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        
        # Save anomalies CSV
        if test_result.anomalies:
            anomalies_df = pd.DataFrame(test_result.anomalies)
            anomalies_file = output_dir / "discrepancies.csv"
            anomalies_df.to_csv(anomalies_file, index=False)
        
        # Save detailed JSON report
        report_data = {
            'family_id': test_result.family_id,
            'test_date': test_result.test_date.isoformat(),
            'summary': {
                'scenarios_tested': test_result.scenarios_tested,
                'scenarios_passed': test_result.scenarios_passed,
                'scenarios_failed': test_result.scenarios_failed,
                'scenarios_with_warnings': test_result.scenarios_with_warnings,
                'success_rate': test_result.success_rate,
                'has_critical_anomalies': test_result.has_critical_anomalies
            },
            'statistics': test_result.summary_statistics,
            'anomalies': test_result.anomalies,
            'detailed_results': [asdict(result) for result in test_result.calculation_results]
        }
        
        # Convert Decimal to float for JSON serialization
        def decimal_to_float(obj):
            if isinstance(obj, Decimal):
                return float(obj)
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, date):
                return obj.isoformat()
            return obj
        
        detailed_file = output_dir / "detailed_report.json"
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, default=decimal_to_float, ensure_ascii=False)
        
        self.logger.info(f"QA report saved to {output_dir}")


def create_qa_test_runner() -> QATestRunner:
    """Factory function to create a QA test runner."""
    return QATestRunner()