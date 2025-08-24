#!/usr/bin/env python3
"""
Demo script for EA Importer system.

This script demonstrates the complete pipeline:
1. Initialize database
2. Process sample PDFs
3. Run clustering
4. Generate reports
"""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

from ea_importer.core.config import get_settings
from ea_importer.core.logging import setup_logging
from ea_importer.database import init_database
from ea_importer.pipeline import create_ingest_pipeline
from ea_importer.pipeline.clustering import create_ea_clusterer
from ea_importer.utils.fingerprinter import create_text_fingerprinter

def main():
    """Run the demo."""
    print("🚀 EA Importer Demo")
    print("=" * 50)
    
    # Set up logging
    setup_logging(log_level="INFO")
    logger = setup_logging().getLogger("demo")
    
    # Get settings
    settings = get_settings()
    print(f"📁 Data directory: {settings.data_root}")
    
    # Ensure directories exist
    settings.ensure_directories()
    
    # Step 1: Initialize database (if needed)
    print("\n🗄️  Initializing database...")
    try:
        init_database(drop_existing=False)
        print("✅ Database initialized")
    except Exception as e:
        print(f"⚠️  Database already exists or error: {e}")
    
    # Step 2: Check for sample PDFs
    print(f"\n📄 Checking for PDFs in {settings.raw_eas_dir}...")
    pdf_files = list(settings.raw_eas_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("⚠️  No PDF files found. Please add some PDFs to the data/eas/raw/ directory.")
        print("   You can download sample EAs from: https://www.fwc.gov.au/")
        return
    
    print(f"📊 Found {len(pdf_files)} PDF files")
    for pdf_file in pdf_files[:5]:  # Show first 5
        print(f"   • {pdf_file.name}")
    if len(pdf_files) > 5:
        print(f"   ... and {len(pdf_files) - 5} more")
    
    # Step 3: Run ingestion pipeline
    print(f"\n🔄 Running ingestion pipeline...")
    pipeline = create_ingest_pipeline()
    
    # Limit to first 3 files for demo
    demo_files = pdf_files[:3]
    
    stats = pipeline.batch_ingest(
        input_dir=settings.raw_eas_dir,
        max_files=len(demo_files),
        force_ocr=False
    )
    
    print(f"✅ Ingestion complete:")
    print(f"   • Files processed: {stats.files_processed}")
    print(f"   • Success rate: {stats.success_rate:.1%}")
    print(f"   • Total clauses: {stats.total_clauses}")
    print(f"   • Duration: {stats.duration_seconds:.1f}s")
    
    # Step 4: Load fingerprints for clustering
    if stats.files_succeeded >= 2:
        print(f"\n🔍 Running clustering analysis...")
        
        fingerprinter = create_text_fingerprinter()
        clusterer = create_ea_clusterer()
        
        # Load fingerprints
        fingerprint_files = list(settings.fingerprints_dir.glob("*.minhash"))
        fingerprints = []
        
        for fp_file in fingerprint_files:
            try:
                fingerprint = fingerprinter.load_fingerprint(fp_file)
                fingerprints.append(fingerprint)
            except Exception as e:
                print(f"⚠️  Could not load {fp_file}: {e}")
        
        if len(fingerprints) >= 2:
            # Run clustering
            result = clusterer.adaptive_clustering(fingerprints)
            
            print(f"✅ Clustering complete:")
            print(f"   • Documents: {result.num_documents}")
            print(f"   • Clusters: {result.num_clusters}")
            print(f"   • Outliers: {len(result.outliers)}")
            print(f"   • Algorithm: {result.algorithm.value}")
            
            # Save clustering results
            cluster_output_dir = settings.reports_dir / "clusters" / "demo"
            clusterer.save_clustering_result(result, cluster_output_dir)
            print(f"   • Results saved to: {cluster_output_dir}")
            
        else:
            print("⚠️  Need at least 2 documents for clustering")
    
    else:
        print("⚠️  Need at least 2 successful ingestions for clustering demo")
    
    # Step 5: Show file outputs
    print(f"\n📁 Generated outputs:")
    output_dirs = [
        ("Text files", settings.text_dir, "*.txt"),
        ("Clause files", settings.clauses_dir, "*.jsonl"),
        ("Fingerprints", settings.fingerprints_dir, "*.sha256"),
        ("Reports", settings.reports_dir, "*")
    ]
    
    for name, directory, pattern in output_dirs:
        if directory.exists():
            files = list(directory.glob(pattern))
            print(f"   • {name}: {len(files)} files in {directory}")
        else:
            print(f"   • {name}: Directory not found")
    
    print(f"\n🎉 Demo completed!")
    print(f"💡 Next steps:")
    print(f"   • Add more PDFs to {settings.raw_eas_dir}")
    print(f"   • Run: python -m ea_importer.cli ingest run {settings.raw_eas_dir}")
    print(f"   • Run: python -m ea_importer.cli cluster run")
    print(f"   • Explore the web interface (coming soon)")

if __name__ == "__main__":
    main()