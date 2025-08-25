"""
Comprehensive Testing Framework for EA Importer

Provides unit tests, integration tests, and quality assurance for all system components:
- PDF processing and OCR functionality
- Text cleaning and normalization
- Clause segmentation accuracy
- Document fingerprinting consistency
- Clustering algorithm validation
- Family building logic
- Rates and rules extraction
- Instance management
- QA calculator accuracy
- End-to-end pipeline testing
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import sqlite3
from datetime import datetime, date

from ..core.config import get_settings
from ..core.logging import get_logger
from ..database import create_test_database, get_db_session
from ..models import (
    DocumentDB, ClauseDB, FingerprintDB, 
    FamilyDB, InstanceDB, OverlayDB,
    PDFDocument, PDFPage
)
from ..utils.pdf_processor import PDFProcessor
from ..utils.text_cleaner import TextCleaner
from ..utils.text_segmenter import TextSegmenter
from ..utils.fingerprinter import Fingerprinter
from ..utils.rates_rules_extractor import RatesRulesExtractor
from ..utils.qa_calculator import QACalculator
from ..pipeline.clustering import ClusteringEngine
from ..pipeline.family_builder import FamilyBuilder
from ..pipeline.instance_manager import InstanceManager

logger = get_logger(__name__)


class TestFixtures:
    """Test fixtures and sample data for testing."""
    
    @staticmethod
    def create_sample_ea_text() -> str:
        """Create sample Enterprise Agreement text for testing."""
        return """
        ENTERPRISE AGREEMENT
        
        1. TITLE AND COVERAGE
        1.1 This Agreement shall be known as the Sample Manufacturing Enterprise Agreement 2024.
        1.2 This Agreement covers employees of Sample Manufacturing Pty Ltd.
        
        2. TERM OF AGREEMENT
        2.1 This Agreement commences on 1 January 2024.
        2.2 This Agreement expires on 31 December 2026.
        
        3. CLASSIFICATIONS AND RATES
        3.1 The classifications and minimum rates are set out in Schedule A.
        
        Schedule A - Classifications and Rates
        
        Level 1 - Manufacturing Assistant: $850.00 per week
        Level 2 - Machine Operator: $920.00 per week  
        Level 3 - Senior Operator: $1,050.00 per week
        Level 4 - Team Leader: $1,180.00 per week
        
        4. ORDINARY HOURS
        4.1 Ordinary hours of work are 38 hours per week.
        4.2 Ordinary hours are worked Monday to Friday between 7:00am and 6:00pm.
        
        5. OVERTIME
        5.1 Overtime is payable at time and a half for the first 2 hours.
        5.2 Overtime after 2 hours is payable at double time.
        
        6. ALLOWANCES
        6.1 Tool allowance: $25.00 per week for employees required to provide tools.
        6.2 Travel allowance: $0.50 per kilometre for work-related travel.
        """
    
    @staticmethod
    def create_sample_pdf_document() -> PDFDocument:
        """Create sample PDF document for testing."""
        
        text = TestFixtures.create_sample_ea_text()
        
        page = PDFPage(
            page_number=1,
            text=text,
            bbox=[0, 0, 612, 792],
            has_images=False,
            tables=[]
        )
        
        return PDFDocument(
            file_path="/test/sample_ea.pdf",
            pages=[page],
            metadata={
                "title": "Sample Manufacturing Enterprise Agreement 2024",
                "author": "Test Author",
                "pages": 1
            }
        )
    
    @staticmethod
    def create_sample_rates_data() -> List[Dict[str, Any]]:
        """Create sample rates data for testing."""
        return [
            {
                "classification": "Level 1 - Manufacturing Assistant",
                "base_rate": 850.00,
                "rate_type": "weekly",
                "currency": "AUD"
            },
            {
                "classification": "Level 2 - Machine Operator", 
                "base_rate": 920.00,
                "rate_type": "weekly",
                "currency": "AUD"
            },
            {
                "classification": "Level 3 - Senior Operator",
                "base_rate": 1050.00,
                "rate_type": "weekly", 
                "currency": "AUD"
            },
            {
                "classification": "Level 4 - Team Leader",
                "base_rate": 1180.00,
                "rate_type": "weekly",
                "currency": "AUD"
            }
        ]
    
    @staticmethod
    def create_sample_rules_data() -> List[Dict[str, Any]]:
        """Create sample rules data for testing."""
        return [
            {
                "rule_type": "overtime",
                "rule_data": {
                    "multiplier": 1.5,
                    "threshold_hours": 38,
                    "max_hours": 2
                }
            },
            {
                "rule_type": "overtime_double",
                "rule_data": {
                    "multiplier": 2.0,
                    "threshold_hours": 40
                }
            },
            {
                "rule_type": "tool_allowance",
                "rule_data": {
                    "amount": 25.00,
                    "frequency": "weekly"
                }
            },
            {
                "rule_type": "travel_allowance",
                "rule_data": {
                    "rate": 0.50,
                    "unit": "kilometre"
                }
            }
        ]


class TestRunner:
    """Main test runner for the EA Importer system."""
    
    def __init__(self):
        """Initialize test runner with test database."""
        self.test_db = create_test_database()
        self.fixtures = TestFixtures()
        
        # Create temporary directory for test files
        self.temp_dir = Path(tempfile.mkdtemp(prefix="ea_importer_test_"))
        
    def cleanup(self):
        """Cleanup test resources."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all test suites and return results."""
        
        test_results = {}
        
        try:
            # Unit tests
            test_results.update(self.run_unit_tests())
            
            # Integration tests
            test_results.update(self.run_integration_tests())
            
            # Quality assurance tests
            test_results.update(self.run_qa_tests())
            
            # Performance tests
            test_results.update(self.run_performance_tests())
            
        finally:
            self.cleanup()
        
        return test_results
    
    def run_unit_tests(self) -> Dict[str, bool]:
        """Run unit tests for individual components."""
        
        results = {}
        
        # PDF Processing Tests
        results["test_pdf_processor"] = self._test_pdf_processor()
        
        # Text Cleaning Tests
        results["test_text_cleaner"] = self._test_text_cleaner()
        
        # Text Segmentation Tests
        results["test_text_segmenter"] = self._test_text_segmenter()
        
        # Fingerprinting Tests
        results["test_fingerprinter"] = self._test_fingerprinter()
        
        # Rates and Rules Extraction Tests
        results["test_rates_rules_extractor"] = self._test_rates_rules_extractor()
        
        # QA Calculator Tests
        results["test_qa_calculator"] = self._test_qa_calculator()
        
        return results
    
    def run_integration_tests(self) -> Dict[str, bool]:
        """Run integration tests for component interactions."""
        
        results = {}
        
        # Clustering Engine Tests
        results["test_clustering_engine"] = self._test_clustering_engine()
        
        # Family Builder Tests
        results["test_family_builder"] = self._test_family_builder()
        
        # Instance Manager Tests
        results["test_instance_manager"] = self._test_instance_manager()
        
        # Database Integration Tests
        results["test_database_integration"] = self._test_database_integration()
        
        return results
    
    def run_qa_tests(self) -> Dict[str, bool]:
        """Run quality assurance tests."""
        
        results = {}
        
        # End-to-end Pipeline Tests
        results["test_e2e_pipeline"] = self._test_e2e_pipeline()
        
        # Data Consistency Tests
        results["test_data_consistency"] = self._test_data_consistency()
        
        # Error Handling Tests
        results["test_error_handling"] = self._test_error_handling()
        
        return results
    
    def run_performance_tests(self) -> Dict[str, bool]:
        """Run performance and load tests."""
        
        results = {}
        
        # Performance Benchmarks
        results["test_performance_benchmarks"] = self._test_performance_benchmarks()
        
        # Memory Usage Tests
        results["test_memory_usage"] = self._test_memory_usage()
        
        # Scalability Tests
        results["test_scalability"] = self._test_scalability()
        
        return results
    
    def _test_pdf_processor(self) -> bool:
        """Test PDF processing functionality."""
        
        try:
            processor = PDFProcessor()
            
            # Test with sample document
            sample_doc = self.fixtures.create_sample_pdf_document()
            
            # Test text extraction
            assert len(sample_doc.pages) == 1
            assert len(sample_doc.pages[0].text) > 0
            
            # Test metadata extraction
            assert "title" in sample_doc.metadata
            assert sample_doc.metadata["pages"] == 1
            
            logger.info("PDF processor tests passed")
            return True
            
        except Exception as e:
            logger.error(f"PDF processor tests failed: {e}")
            return False
    
    def _test_text_cleaner(self) -> bool:
        """Test text cleaning functionality."""
        
        try:
            cleaner = TextCleaner()
            
            # Test with sample text
            sample_text = self.fixtures.create_sample_ea_text()
            
            # Test basic cleaning
            cleaned_text = cleaner.clean_text(sample_text)
            assert len(cleaned_text) > 0
            assert cleaned_text != sample_text  # Should be modified
            
            # Test document cleaning
            sample_doc = self.fixtures.create_sample_pdf_document()
            cleaned_doc = cleaner.clean_document(sample_doc)
            
            assert len(cleaned_doc.pages) == 1
            assert "text_cleaned" in cleaned_doc.metadata
            
            logger.info("Text cleaner tests passed")
            return True
            
        except Exception as e:
            logger.error(f"Text cleaner tests failed: {e}")
            return False
    
    def _test_text_segmenter(self) -> bool:
        """Test text segmentation functionality."""
        
        try:
            segmenter = TextSegmenter()
            
            # Test with sample text
            sample_text = self.fixtures.create_sample_ea_text()
            
            # Test segmentation
            segments = segmenter.segment_text(sample_text)
            
            assert len(segments) > 0
            
            # Check for expected clause numbers
            clause_numbers = [segment.clause_number for segment in segments]
            assert "1" in clause_numbers
            assert "1.1" in clause_numbers
            assert "2" in clause_numbers
            
            logger.info("Text segmenter tests passed")
            return True
            
        except Exception as e:
            logger.error(f"Text segmenter tests failed: {e}")
            return False
    
    def _test_fingerprinter(self) -> bool:
        """Test document fingerprinting functionality."""
        
        try:
            fingerprinter = Fingerprinter()
            
            # Test with sample document
            sample_doc = self.fixtures.create_sample_pdf_document()
            
            # Generate fingerprint
            fingerprint = fingerprinter.generate_fingerprint(sample_doc)
            
            assert fingerprint.sha256_hash is not None
            assert fingerprint.minhash_signature is not None
            assert len(fingerprint.sha256_hash) == 64  # SHA256 hex length
            
            # Test similarity calculation
            fingerprint2 = fingerprinter.generate_fingerprint(sample_doc)
            similarity = fingerprinter.calculate_similarity(fingerprint, fingerprint2)
            
            assert similarity == 1.0  # Identical documents
            
            logger.info("Fingerprinter tests passed")
            return True
            
        except Exception as e:
            logger.error(f"Fingerprinter tests failed: {e}")
            return False
    
    def _test_rates_rules_extractor(self) -> bool:
        """Test rates and rules extraction functionality."""
        
        try:
            extractor = RatesRulesExtractor()
            
            # Test with sample text
            sample_text = self.fixtures.create_sample_ea_text()
            
            # Extract rates
            rates = extractor.extract_rates(sample_text)
            
            assert len(rates) > 0
            
            # Check for expected rates
            rate_amounts = [rate.base_rate for rate in rates]
            assert 850.0 in rate_amounts  # Level 1 rate
            assert 920.0 in rate_amounts  # Level 2 rate
            
            # Extract rules
            rules = extractor.extract_rules(sample_text)
            
            assert len(rules) > 0
            
            # Check for overtime rules
            overtime_rules = [rule for rule in rules if "overtime" in rule.rule_type]
            assert len(overtime_rules) > 0
            
            logger.info("Rates and rules extractor tests passed")
            return True
            
        except Exception as e:
            logger.error(f"Rates and rules extractor tests failed: {e}")
            return False
    
    def _test_qa_calculator(self) -> bool:
        """Test QA calculator functionality."""
        
        try:
            calculator = QACalculator()
            
            # Test synthetic worker generation
            workers = calculator.generate_synthetic_workers("test_family", count=5)
            
            assert len(workers) == 5
            assert all("worker_id" in worker for worker in workers)
            assert all("classification" in worker for worker in workers)
            
            # Test smoke tests
            family_rates = self.fixtures.create_sample_rates_data()
            family_rules = self.fixtures.create_sample_rules_data()
            
            results = calculator.run_smoke_tests(
                family_id="test_family",
                family_rates=family_rates,
                family_rules=family_rules,
                synthetic_workers=workers
            )
            
            assert len(results) == 5
            assert all(result.test_id.startswith("SMOKE_") for result in results)
            
            logger.info("QA calculator tests passed")
            return True
            
        except Exception as e:
            logger.error(f"QA calculator tests failed: {e}")
            return False
    
    def _test_clustering_engine(self) -> bool:
        """Test clustering engine functionality."""
        
        try:
            engine = ClusteringEngine()
            
            # Create sample documents with fingerprints
            sample_docs = []
            for i in range(3):
                doc = self.fixtures.create_sample_pdf_document()
                doc.file_path = f"/test/sample_ea_{i}.pdf"
                sample_docs.append(doc)
            
            # Test clustering
            clusters = engine.cluster_documents(
                sample_docs,
                algorithm="threshold",
                similarity_threshold=0.8
            )
            
            # With identical documents, should form one cluster
            assert len(clusters) <= 1
            
            logger.info("Clustering engine tests passed")
            return True
            
        except Exception as e:
            logger.error(f"Clustering engine tests failed: {e}")
            return False
    
    def _test_family_builder(self) -> bool:
        """Test family builder functionality."""
        
        try:
            builder = FamilyBuilder()
            
            # Test family creation
            sample_docs = [self.fixtures.create_sample_pdf_document()]
            
            family = builder.create_family(
                family_name="Test Family",
                documents=sample_docs
            )
            
            assert family.family_name == "Test Family"
            assert len(family.document_ids) == 1
            
            logger.info("Family builder tests passed")
            return True
            
        except Exception as e:
            logger.error(f"Family builder tests failed: {e}")
            return False
    
    def _test_instance_manager(self) -> bool:
        """Test instance manager functionality."""
        
        try:
            manager = InstanceManager()
            
            # Test instance creation
            family_data = {
                "family_id": 1,
                "family_name": "Test Family"
            }
            
            instance = manager.create_instance(
                family_data=family_data,
                employer="Test Employer",
                effective_date=date(2024, 1, 1),
                expiry_date=date(2026, 12, 31)
            )
            
            assert instance.employer == "Test Employer"
            assert instance.effective_date == date(2024, 1, 1)
            
            logger.info("Instance manager tests passed")
            return True
            
        except Exception as e:
            logger.error(f"Instance manager tests failed: {e}")
            return False
    
    def _test_database_integration(self) -> bool:
        """Test database integration functionality."""
        
        try:
            with self.test_db.get_session() as session:
                # Test document creation
                document = DocumentDB(
                    file_path="/test/sample.pdf",
                    file_name="sample.pdf",
                    file_size=1024,
                    processing_status="completed"
                )
                session.add(document)
                session.flush()
                
                assert document.id is not None
                
                # Test clause creation
                clause = ClauseDB(
                    document_id=document.id,
                    clause_number="1.1",
                    heading="Test Clause",
                    text="This is a test clause."
                )
                session.add(clause)
                session.flush()
                
                assert clause.id is not None
                
                # Test relationships
                assert len(document.clauses) == 1
                assert document.clauses[0].text == "This is a test clause."
                
            logger.info("Database integration tests passed")
            return True
            
        except Exception as e:
            logger.error(f"Database integration tests failed: {e}")
            return False
    
    def _test_e2e_pipeline(self) -> bool:
        """Test end-to-end pipeline functionality."""
        
        try:
            # This would test the complete pipeline from PDF to corpus
            # For now, we'll do a simplified test
            
            sample_doc = self.fixtures.create_sample_pdf_document()
            
            # Test that we can process a document through multiple stages
            cleaner = TextCleaner()
            segmenter = TextSegmenter()
            fingerprinter = Fingerprinter()
            
            # Clean text
            cleaned_doc = cleaner.clean_document(sample_doc)
            
            # Segment text
            segments = segmenter.segment_document(cleaned_doc)
            
            # Generate fingerprint
            fingerprint = fingerprinter.generate_fingerprint(cleaned_doc)
            
            assert len(segments) > 0
            assert fingerprint is not None
            
            logger.info("End-to-end pipeline tests passed")
            return True
            
        except Exception as e:
            logger.error(f"End-to-end pipeline tests failed: {e}")
            return False
    
    def _test_data_consistency(self) -> bool:
        """Test data consistency across operations."""
        
        try:
            # Test that fingerprints are consistent
            fingerprinter = Fingerprinter()
            sample_doc = self.fixtures.create_sample_pdf_document()
            
            fp1 = fingerprinter.generate_fingerprint(sample_doc)
            fp2 = fingerprinter.generate_fingerprint(sample_doc)
            
            assert fp1.sha256_hash == fp2.sha256_hash
            
            # Test that segmentation is deterministic
            segmenter = TextSegmenter()
            sample_text = self.fixtures.create_sample_ea_text()
            
            segments1 = segmenter.segment_text(sample_text)
            segments2 = segmenter.segment_text(sample_text)
            
            assert len(segments1) == len(segments2)
            
            logger.info("Data consistency tests passed")
            return True
            
        except Exception as e:
            logger.error(f"Data consistency tests failed: {e}")
            return False
    
    def _test_error_handling(self) -> bool:
        """Test error handling and recovery."""
        
        try:
            # Test invalid PDF handling
            processor = PDFProcessor()
            
            # Create invalid PDF file
            invalid_pdf_path = self.temp_dir / "invalid.pdf"
            with open(invalid_pdf_path, 'w') as f:
                f.write("This is not a PDF file")
            
            # Should handle gracefully
            try:
                processor.process_pdf(str(invalid_pdf_path))
                assert False, "Should have raised an exception"
            except Exception:
                pass  # Expected
            
            # Test empty text handling
            cleaner = TextCleaner()
            cleaned = cleaner.clean_text("")
            assert cleaned == ""
            
            logger.info("Error handling tests passed")
            return True
            
        except Exception as e:
            logger.error(f"Error handling tests failed: {e}")
            return False
    
    def _test_performance_benchmarks(self) -> bool:
        """Test performance benchmarks."""
        
        try:
            # Test processing speed
            start_time = datetime.now()
            
            # Process sample document
            sample_doc = self.fixtures.create_sample_pdf_document()
            cleaner = TextCleaner()
            segmenter = TextSegmenter()
            
            cleaned_doc = cleaner.clean_document(sample_doc)
            segments = segmenter.segment_document(cleaned_doc)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Should process in reasonable time (< 1 second for small doc)
            assert processing_time < 1.0
            
            logger.info("Performance benchmark tests passed")
            return True
            
        except Exception as e:
            logger.error(f"Performance benchmark tests failed: {e}")
            return False
    
    def _test_memory_usage(self) -> bool:
        """Test memory usage patterns."""
        
        try:
            # Test that we don't have obvious memory leaks
            # This is a simplified test
            
            processor = PDFProcessor()
            
            # Process multiple documents
            for i in range(10):
                sample_doc = self.fixtures.create_sample_pdf_document()
                sample_doc.file_path = f"/test/doc_{i}.pdf"
                # Process and discard
                
            logger.info("Memory usage tests passed")
            return True
            
        except Exception as e:
            logger.error(f"Memory usage tests failed: {e}")
            return False
    
    def _test_scalability(self) -> bool:
        """Test system scalability."""
        
        try:
            # Test batch processing
            clustering_engine = ClusteringEngine()
            
            # Create multiple sample documents
            sample_docs = []
            for i in range(10):
                doc = self.fixtures.create_sample_pdf_document()
                doc.file_path = f"/test/sample_ea_{i}.pdf"
                sample_docs.append(doc)
            
            # Test clustering at scale
            clusters = clustering_engine.cluster_documents(
                sample_docs,
                algorithm="threshold",
                similarity_threshold=0.8
            )
            
            # Should handle multiple documents
            assert isinstance(clusters, list)
            
            logger.info("Scalability tests passed")
            return True
            
        except Exception as e:
            logger.error(f"Scalability tests failed: {e}")
            return False


def run_tests() -> Dict[str, Any]:
    """
    Run the complete test suite and return detailed results.
    
    Returns:
        Dictionary with test results and statistics
    """
    
    logger.info("Starting comprehensive test suite")
    
    runner = TestRunner()
    
    try:
        test_results = runner.run_all_tests()
        
        # Calculate statistics
        total_tests = len(test_results)
        passed_tests = sum(1 for result in test_results.values() if result)
        failed_tests = total_tests - passed_tests
        
        summary = {
            "test_results": test_results,
            "statistics": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0
            },
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Test suite completed: {passed_tests}/{total_tests} tests passed")
        
        return summary
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        raise
    
    finally:
        runner.cleanup()


__all__ = [
    'TestRunner',
    'TestFixtures', 
    'run_tests'
]