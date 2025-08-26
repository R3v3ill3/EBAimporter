"""
QA Calculator for EA Importer.

Provides synthetic worker generation and a simple rules-based smoke test
runner to validate rates/rules consistency at a basic level.
"""

from __future__ import annotations

import random
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

from ..core.logging import get_logger, log_function_call
from ..models import QATestResult


logger = get_logger(__name__)


@dataclass
class WorkerScenario:
    worker_id: str
    classification: str
    hours: float
    day: str


class QACalculator:
    """Minimal QA utility: synthetic workers and smoke tests."""

    @log_function_call
    def generate_synthetic_workers(self, family_id: str, count: int = 20) -> List[Dict[str, Any]]:
        workers: List[Dict[str, Any]] = []
        for i in range(count):
            workers.append(
                {
                    "worker_id": f"W{i+1:03d}",
                    "classification": random.choice([
                        "Level 1", "Level 2", "Level 3", "Level 4"
                    ]),
                    "hours": random.choice([7.6, 8.0, 9.5]),
                    "day": random.choice(["weekday", "saturday", "sunday"]) ,
                }
            )
        return workers

    @log_function_call
    def run_smoke_tests(
        self,
        family_id: str,
        family_rates: List[Dict[str, Any]],
        family_rules: List[Dict[str, Any]],
        synthetic_workers: List[Dict[str, Any]],
    ) -> List[QATestResult]:
        # Build simple rate lookup
        rate_map: Dict[str, float] = {}
        for r in family_rates:
            key = (r.get("classification") or "").strip()
            if not key:
                continue
            base = float(r.get("base_rate", 0.0))
            unit = (r.get("unit") or "hourly").lower()
            # Normalize to hourly for simple calc
            if unit == "weekly":
                base = base / 38.0
            elif unit == "annual":
                base = base / (52.0 * 38.0)
            rate_map[key.lower()] = base

        overtime_mult = 1.5
        for rule in family_rules:
            if (rule.get("rule_type") == "overtime") and isinstance(rule.get("rule_data"), dict):
                try:
                    overtime_mult = float(rule["rule_data"].get("multiplier", 1.5))
                except Exception:
                    overtime_mult = 1.5

        results: List[QATestResult] = []
        for i, w in enumerate(synthetic_workers):
            test_id = f"SMOKE_{i+1:03d}"
            cls = (w.get("classification") or "").lower()
            base = rate_map.get(cls, 0.0)
            hours = float(w.get("hours", 0.0))
            day = (w.get("day") or "weekday").lower()
            anomalies: List[str] = []
            warnings: List[str] = []

            if base <= 0:
                anomalies.append("Missing base rate for classification")

            hourly = base
            if day in ("saturday", "sunday"):
                hourly = base * overtime_mult

            pay = hourly * hours
            if pay <= 0:
                warnings.append("Computed pay is zero or negative")

            results.append(
                QATestResult(
                    test_id=test_id,
                    family_id=family_id,
                    worker_scenario=w,
                    test_results={"hourly_rate": hourly, "hours": hours, "computed_pay": pay},
                    anomalies=anomalies,
                    warnings=warnings,
                    passed=len(anomalies) == 0,
                    execution_time_seconds=0.0,
                )
            )

        return results

    @log_function_call
    def export_qa_results(self, results: List[QATestResult], output_path: str) -> None:
        import json
        from pathlib import Path

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(
                [
                    {
                        "test_id": r.test_id,
                        "family_id": r.family_id,
                        "worker_scenario": r.worker_scenario,
                        "test_results": r.test_results,
                        "anomalies": r.anomalies,
                        "warnings": r.warnings,
                        "passed": r.passed,
                        "execution_time_seconds": r.execution_time_seconds,
                    }
                    for r in results
                ],
                f,
                indent=2,
            )


__all__ = ["QACalculator", "WorkerScenario"]

"""
QA Calculator for EA Importer - Smoke testing with synthetic scenarios.
"""

import logging
import random
from typing import List, Dict, Any, Optional
from datetime import datetime, time
import json

from ..core.logging import get_logger, log_function_call
from ..models import QATestResult

logger = get_logger(__name__)


class QACalculator:
    """Quality assurance calculator for smoke testing EA families."""
    
    def __init__(self):
        self.classifications = ["Level 1", "Level 2", "Level 3", "Level 4"]
        self.shift_types = ["day", "afternoon", "night"]
        
    @log_function_call
    def generate_synthetic_workers(self, family_id: str, count: int = 20) -> List[Dict[str, Any]]:
        """Generate synthetic worker scenarios for testing."""
        
        workers = []
        
        for i in range(count):
            worker = {
                'worker_id': f"TEST_{family_id}_{i+1:03d}",
                'classification': random.choice(self.classifications),
                'hours_per_week': random.choice([38, 40, 35]),
                'shift_type': random.choice(self.shift_types),
                'overtime_hours': random.randint(0, 10),
                'weekend_hours': random.randint(0, 16),
                'allowances': {
                    'tool_allowance': random.choice([True, False]),
                    'travel_allowance': random.choice([0, 25, 50]),
                    'meal_allowance': random.choice([True, False])
                },
                'start_time': random.choice([time(7, 0), time(8, 0), time(9, 0)]),
                'days_worked': random.randint(4, 6)
            }
            workers.append(worker)
        
        logger.info(f"Generated {len(workers)} synthetic workers for family {family_id}")
        return workers
    
    def run_smoke_tests(
        self,
        family_id: str,
        family_rates: List[Dict[str, Any]],
        family_rules: List[Dict[str, Any]],
        synthetic_workers: List[Dict[str, Any]]
    ) -> List[QATestResult]:
        """Run smoke tests on synthetic worker scenarios."""
        
        results = []
        
        for worker in synthetic_workers:
            start_time = datetime.now()
            
            try:
                # Calculate pay for the worker
                calculation_result = self._calculate_worker_pay(worker, family_rates, family_rules)
                
                # Validate results
                anomalies = self._detect_anomalies(calculation_result)
                warnings = self._detect_warnings(calculation_result)
                
                test_result = QATestResult(
                    test_id=f"SMOKE_{worker['worker_id']}",
                    family_id=family_id,
                    worker_scenario=worker,
                    test_results=calculation_result,
                    anomalies=anomalies,
                    warnings=warnings,
                    passed=len(anomalies) == 0,
                    execution_time_seconds=(datetime.now() - start_time).total_seconds()
                )
                
                results.append(test_result)
                
            except Exception as e:
                # Test failed
                test_result = QATestResult(
                    test_id=f"SMOKE_{worker['worker_id']}",
                    family_id=family_id,
                    worker_scenario=worker,
                    test_results={},
                    anomalies=[f"Calculation failed: {str(e)}"],
                    warnings=[],
                    passed=False,
                    execution_time_seconds=(datetime.now() - start_time).total_seconds()
                )
                
                results.append(test_result)
        
        logger.info(f"Completed smoke tests: {sum(1 for r in results if r.passed)}/{len(results)} passed")
        return results
    
    def _calculate_worker_pay(
        self,
        worker: Dict[str, Any],
        family_rates: List[Dict[str, Any]],
        family_rules: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate pay for a synthetic worker."""
        
        # Find base rate for worker's classification
        base_rate = self._find_base_rate(worker['classification'], family_rates)
        
        if base_rate is None:
            raise ValueError(f"No base rate found for classification: {worker['classification']}")
        
        # Calculate ordinary hours pay
        ordinary_hours = min(worker['hours_per_week'], 38)  # Standard full-time hours
        ordinary_pay = ordinary_hours * base_rate
        
        # Calculate overtime pay
        overtime_multiplier = self._find_overtime_multiplier(family_rules)
        overtime_pay = worker['overtime_hours'] * base_rate * overtime_multiplier
        
        # Calculate penalty rates
        penalty_pay = 0
        if worker['weekend_hours'] > 0:
            weekend_multiplier = self._find_weekend_multiplier(family_rules)
            penalty_pay += worker['weekend_hours'] * base_rate * weekend_multiplier
        
        # Calculate allowances
        allowance_pay = self._calculate_allowances(worker, family_rules)
        
        # Total pay
        total_pay = ordinary_pay + overtime_pay + penalty_pay + allowance_pay
        
        return {
            'base_rate': base_rate,
            'ordinary_hours': ordinary_hours,
            'ordinary_pay': round(ordinary_pay, 2),
            'overtime_hours': worker['overtime_hours'],
            'overtime_pay': round(overtime_pay, 2),
            'penalty_pay': round(penalty_pay, 2),
            'allowance_pay': round(allowance_pay, 2),
            'total_pay': round(total_pay, 2),
            'effective_hourly_rate': round(total_pay / (ordinary_hours + worker['overtime_hours'] + worker['weekend_hours']) if (ordinary_hours + worker['overtime_hours'] + worker['weekend_hours']) > 0 else 0, 2)
        }
    
    def _find_base_rate(self, classification: str, family_rates: List[Dict[str, Any]]) -> Optional[float]:
        """Find base rate for classification."""
        
        for rate in family_rates:
            if classification.lower() in rate.get('classification', '').lower():
                return rate.get('base_rate')
        
        # Fallback: use a default rate
        return 25.0  # Minimum wage approximation
    
    def _find_overtime_multiplier(self, family_rules: List[Dict[str, Any]]) -> float:
        """Find overtime multiplier."""
        
        for rule in family_rules:
            if rule.get('rule_type') == 'overtime':
                return rule.get('rule_data', {}).get('multiplier', 1.5)
        
        return 1.5  # Standard overtime rate
    
    def _find_weekend_multiplier(self, family_rules: List[Dict[str, Any]]) -> float:
        """Find weekend penalty multiplier."""
        
        for rule in family_rules:
            if 'weekend' in rule.get('rule_type', '') or 'penalty' in rule.get('rule_type', ''):
                return rule.get('rule_data', {}).get('multiplier', 1.5)
        
        return 1.5  # Standard weekend penalty
    
    def _calculate_allowances(self, worker: Dict[str, Any], family_rules: List[Dict[str, Any]]) -> float:
        """Calculate allowances for worker."""
        
        total_allowances = 0
        allowances = worker.get('allowances', {})
        
        # Tool allowance
        if allowances.get('tool_allowance'):
            tool_amount = self._find_allowance_amount('tool', family_rules)
            total_allowances += tool_amount
        
        # Travel allowance
        travel_km = allowances.get('travel_allowance', 0)
        if travel_km > 0:
            travel_rate = self._find_allowance_rate('travel', family_rules)
            total_allowances += travel_km * travel_rate
        
        # Meal allowance
        if allowances.get('meal_allowance'):
            meal_amount = self._find_allowance_amount('meal', family_rules)
            total_allowances += meal_amount
        
        return total_allowances
    
    def _find_allowance_amount(self, allowance_type: str, family_rules: List[Dict[str, Any]]) -> float:
        """Find allowance amount."""
        
        for rule in family_rules:
            if allowance_type in rule.get('rule_type', ''):
                return rule.get('rule_data', {}).get('amount', 0)
        
        # Default allowance amounts
        defaults = {'tool': 25.0, 'meal': 15.0}
        return defaults.get(allowance_type, 0)
    
    def _find_allowance_rate(self, allowance_type: str, family_rules: List[Dict[str, Any]]) -> float:
        """Find allowance rate (per km, etc.)."""
        
        for rule in family_rules:
            if allowance_type in rule.get('rule_type', ''):
                return rule.get('rule_data', {}).get('rate', 0)
        
        # Default rates
        defaults = {'travel': 0.5}  # 50 cents per km
        return defaults.get(allowance_type, 0)
    
    def _detect_anomalies(self, calculation_result: Dict[str, Any]) -> List[str]:
        """Detect critical anomalies in calculation results."""
        
        anomalies = []
        
        # Check for negative values
        if calculation_result.get('total_pay', 0) < 0:
            anomalies.append("Negative total pay")
        
        if calculation_result.get('base_rate', 0) < 10:
            anomalies.append("Base rate below minimum wage")
        
        # Check for unreasonably high values
        if calculation_result.get('total_pay', 0) > 5000:
            anomalies.append("Total pay unreasonably high")
        
        if calculation_result.get('effective_hourly_rate', 0) > 200:
            anomalies.append("Effective hourly rate unreasonably high")
        
        return anomalies
    
    def _detect_warnings(self, calculation_result: Dict[str, Any]) -> List[str]:
        """Detect potential issues in calculation results."""
        
        warnings = []
        
        # Check for unusual values
        if calculation_result.get('overtime_pay', 0) > calculation_result.get('ordinary_pay', 0):
            warnings.append("Overtime pay exceeds ordinary pay")
        
        if calculation_result.get('penalty_pay', 0) > calculation_result.get('ordinary_pay', 0) * 0.5:
            warnings.append("High penalty payments")
        
        if calculation_result.get('effective_hourly_rate', 0) < calculation_result.get('base_rate', 0):
            warnings.append("Effective rate below base rate")
        
        return warnings
    
    def export_qa_results(self, results: List[QATestResult], output_path: str):
        """Export QA results to JSON."""
        
        results_data = []
        for result in results:
            results_data.append({
                'test_id': result.test_id,
                'family_id': result.family_id,
                'passed': result.passed,
                'execution_time_seconds': result.execution_time_seconds,
                'anomalies': result.anomalies,
                'warnings': result.warnings,
                'worker_scenario': result.worker_scenario,
                'test_results': result.test_results
            })
        
        with open(output_path, 'w') as f:
            json.dump({
                'summary': {
                    'total_tests': len(results),
                    'passed_tests': sum(1 for r in results if r.passed),
                    'failed_tests': sum(1 for r in results if not r.passed),
                    'total_anomalies': sum(len(r.anomalies) for r in results),
                    'total_warnings': sum(len(r.warnings) for r in results)
                },
                'results': results_data
            }, f, indent=2)
        
        logger.info(f"Exported QA results to {output_path}")


__all__ = ['QACalculator']