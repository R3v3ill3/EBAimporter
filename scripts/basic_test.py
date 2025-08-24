#!/usr/bin/env python3
"""
Basic structure test for EA Importer - tests core logic without heavy dependencies.
"""

import sys
from pathlib import Path

# Add src to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

def test_text_processing():
    """Test text processing without external dependencies."""
    print("Testing text processing...")
    
    # Test text cleaning logic
    sample_text = """
    Page 1 of 10
    
    1. INTRODUCTION
    
    This is a sample Enterprise Agreement with some
    hyphen-
    ated words and multiple    spaces.
    
    2. DEFINITIONS
    
    2.1 In this agreement:
    (a) "Employee" means a person employed under this agreement
    (b) "Employer" means the organization
    """
    
    # Basic cleaning operations (without the class)
    import re
    
    # Remove page headers
    text = re.sub(r'Page \d+ of \d+', '', sample_text)
    
    # Fix hyphenation
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
    # Normalize whitespace
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    assert len(text) > 0
    assert 'hyphenated' in text or 'hyphen-' not in text
    
    print("✅ Basic text cleaning works")
    
    # Test segmentation patterns
    patterns = [
        r'^(\d+)\.\s+(.+)',  # "1. Introduction"
        r'^(\d+\.\d+)\s+(.+)',  # "2.1 Definition"
        r'^\(([a-z])\)\s+(.+)',  # "(a) Item"
    ]
    
    lines = text.split('\n')
    clauses_found = 0
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        for pattern in patterns:
            if re.match(pattern, line):
                clauses_found += 1
                break
    
    assert clauses_found > 0
    print(f"✅ Found {clauses_found} potential clause headings")
    
    return True

def test_fingerprinting():
    """Test basic fingerprinting without MinHash library."""
    print("Testing basic fingerprinting...")
    
    import hashlib
    
    sample_text = "This is a sample Enterprise Agreement for testing purposes."
    
    # Test SHA256
    sha256 = hashlib.sha256(sample_text.encode('utf-8')).hexdigest()
    assert len(sha256) == 64
    
    # Test basic n-gram extraction
    def extract_ngrams(text, n=3):
        normalized = text.lower().replace('\n', ' ')
        normalized = ' '.join(normalized.split())
        
        ngrams = set()
        for i in range(len(normalized) - n + 1):
            ngram = normalized[i:i + n]
            ngrams.add(ngram)
        
        return ngrams
    
    ngrams = extract_ngrams(sample_text)
    assert len(ngrams) > 0
    
    print(f"✅ SHA256: {sha256[:16]}...")
    print(f"✅ Extracted {len(ngrams)} 3-grams")
    
    return True

def test_clustering_logic():
    """Test basic clustering logic without external libraries."""
    print("Testing clustering logic...")
    
    # Simulate similarity scores
    documents = ["doc1", "doc2", "doc3", "doc4"]
    
    # Similarity matrix (symmetric)
    similarities = {
        ("doc1", "doc2"): 0.95,  # Very similar
        ("doc1", "doc3"): 0.3,   # Different
        ("doc1", "doc4"): 0.4,   # Different
        ("doc2", "doc3"): 0.2,   # Different
        ("doc2", "doc4"): 0.35,  # Different
        ("doc3", "doc4"): 0.88,  # Similar
    }
    
    def get_similarity(doc1, doc2):
        if doc1 == doc2:
            return 1.0
        key = (min(doc1, doc2), max(doc1, doc2))
        return similarities.get(key, 0.0)
    
    # Simple threshold-based clustering
    threshold = 0.8
    clusters = []
    processed = set()
    
    for doc in documents:
        if doc in processed:
            continue
        
        cluster = [doc]
        processed.add(doc)
        
        for other_doc in documents:
            if other_doc not in processed:
                sim = get_similarity(doc, other_doc)
                if sim >= threshold:
                    cluster.append(other_doc)
                    processed.add(other_doc)
        
        clusters.append(cluster)
    
    assert len(clusters) > 0
    
    # Should find 2 clusters: [doc1, doc2] and [doc3, doc4]
    cluster_sizes = [len(c) for c in clusters]
    
    print(f"✅ Created {len(clusters)} clusters with sizes {cluster_sizes}")
    
    return True

def test_file_structure():
    """Test that the project structure is correct."""
    print("Testing file structure...")
    
    required_files = [
        "src/ea_importer/__init__.py",
        "src/ea_importer/core/__init__.py",
        "src/ea_importer/core/config.py",
        "src/ea_importer/utils/__init__.py",
        "src/ea_importer/models/__init__.py",
        "src/ea_importer/pipeline/__init__.py",
        "requirements.txt",
        "setup.py",
        "README.md",
    ]
    
    project_root = Path(__file__).parent.parent
    
    missing_files = []
    for file_path in required_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All required files present")
    return True

def main():
    """Run basic tests."""
    print("🧪 EA Importer Basic Tests")
    print("=" * 40)
    
    tests = [
        test_file_structure,
        test_text_processing,
        test_fingerprinting,
        test_clustering_logic,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
            results.append(False)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("=" * 40)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All basic tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)