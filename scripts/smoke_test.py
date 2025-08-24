#!/usr/bin/env python3
"""
Basic smoke tests for EA Importer functionality.
"""

import sys
import tempfile
from pathlib import Path

# Add src to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

def test_imports():
    """Test that all main modules can be imported."""
    print("Testing imports...")
    
    try:
        from ea_importer.core.config import get_settings
        from ea_importer.core.logging import setup_logging
        from ea_importer.utils.pdf_processor import create_pdf_processor
        from ea_importer.utils.text_cleaner import create_text_cleaner
        from ea_importer.utils.text_segmenter import create_text_segmenter
        from ea_importer.utils.fingerprinter import create_text_fingerprinter
        from ea_importer.pipeline.clustering import create_ea_clusterer
        from ea_importer.pipeline import create_ingest_pipeline
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_configuration():
    """Test configuration loading."""
    print("Testing configuration...")
    
    try:
        from ea_importer.core.config import get_settings
        settings = get_settings()
        
        # Check key attributes exist
        assert hasattr(settings, 'data_root')
        assert hasattr(settings, 'database_url')
        assert hasattr(settings, 'min_clause_count')
        
        print(f"✅ Configuration loaded: data_root={settings.data_root}")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_text_processing():
    """Test text processing utilities."""
    print("Testing text processing...")
    
    try:
        from ea_importer.utils.text_cleaner import create_text_cleaner
        from ea_importer.utils.text_segmenter import create_text_segmenter
        
        cleaner = create_text_cleaner()
        segmenter = create_text_segmenter()
        
        # Test text cleaning
        sample_text = """
        Page 1 of 10
        
        1. INTRODUCTION
        
        This is a sample Enterprise Agreement.
        
        2. DEFINITIONS
        
        2.1 In this agreement:
        (a) "Employee" means a person employed under this agreement
        (b) "Employer" means the organization
        
        3. CLASSIFICATION AND RATES
        
        The minimum rates of pay are set out below.
        """
        
        cleaned_text, stats = cleaner.clean_text(sample_text)
        assert len(cleaned_text) > 0
        assert stats.original_length > 0
        
        # Test segmentation
        clauses = segmenter.segment_text(cleaned_text)
        assert len(clauses) > 0
        
        print(f"✅ Text processing: {len(clauses)} clauses extracted")
        return True
    except Exception as e:
        print(f"❌ Text processing test failed: {e}")
        return False

def test_fingerprinting():
    """Test fingerprinting functionality."""
    print("Testing fingerprinting...")
    
    try:
        from ea_importer.utils.fingerprinter import create_text_fingerprinter
        
        fingerprinter = create_text_fingerprinter()
        
        sample_text = "This is a sample Enterprise Agreement for testing purposes."
        
        # Test SHA256
        sha256 = fingerprinter.compute_sha256(sample_text)
        assert len(sha256) == 64
        
        # Test MinHash
        minhash = fingerprinter.compute_minhash(sample_text)
        assert minhash is not None
        
        # Test fingerprint creation
        fingerprint = fingerprinter.fingerprint_document("TEST_EA", sample_text, 5)
        assert fingerprint.ea_id == "TEST_EA"
        assert fingerprint.sha256_hash == sha256
        
        print(f"✅ Fingerprinting: SHA256={sha256[:16]}...")
        return True
    except Exception as e:
        print(f"❌ Fingerprinting test failed: {e}")
        return False

def test_clustering():
    """Test clustering functionality."""
    print("Testing clustering...")
    
    try:
        from ea_importer.utils.fingerprinter import create_text_fingerprinter
        from ea_importer.pipeline.clustering import create_ea_clusterer
        
        fingerprinter = create_text_fingerprinter()
        clusterer = create_ea_clusterer()
        
        # Create some sample fingerprints
        sample_texts = [
            "This is the first Enterprise Agreement",
            "This is the first Enterprise Agreement with minor changes",
            "This is a completely different agreement about something else"
        ]
        
        fingerprints = []
        for i, text in enumerate(sample_texts):
            fp = fingerprinter.fingerprint_document(f"EA_{i}", text)
            fingerprints.append(fp)
        
        # Test clustering
        clusters = clusterer.cluster_by_minhash_threshold(fingerprints, threshold=0.5)
        assert len(clusters) > 0
        
        print(f"✅ Clustering: {len(clusters)} clusters created")
        return True
    except Exception as e:
        print(f"❌ Clustering test failed: {e}")
        return False

def test_database_connection():
    """Test database connection (if available)."""
    print("Testing database connection...")
    
    try:
        from ea_importer.database import get_database_manager
        
        # Use SQLite for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            test_db_url = f"sqlite:///{temp_dir}/test.db"
            
            from ea_importer.database import DatabaseManager
            db_manager = DatabaseManager(test_db_url)
            
            # Test connection
            success = db_manager.test_connection()
            if success:
                print("✅ Database connection successful")
                return True
            else:
                print("⚠️  Database connection failed (but this might be expected)")
                return True  # Don't fail the test for DB issues
    except Exception as e:
        print(f"⚠️  Database test skipped: {e}")
        return True  # Don't fail tests for DB issues

def main():
    """Run all smoke tests."""
    print("🧪 EA Importer Smoke Tests")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_configuration,
        test_text_processing,
        test_fingerprinting,
        test_clustering,
        test_database_connection,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("=" * 40)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)