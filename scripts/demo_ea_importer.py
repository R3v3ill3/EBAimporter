#!/usr/bin/env python3
"""
EA Importer System Demonstration

This script demonstrates the core functionality of the EA Importer system
including PDF processing, text cleaning, clause segmentation, and fingerprinting.

Usage:
    python demo_ea_importer.py [pdf_file_path]
"""

import sys
import logging
from pathlib import Path
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from ea_importer import setup_logging, get_logger
    from ea_importer.utils import (
        PDFProcessor, TextCleaner, TextSegmenter, Fingerprinter,
        get_system_info, validate_dependencies
    )
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure the EA Importer package is properly installed")
    print("Run: pip install -e . from the project root directory")
    sys.exit(1)


def main():
    """Main demonstration function"""
    
    # Setup logging
    setup_logging(log_level="INFO")
    logger = get_logger(__name__)
    
    print("🚀 EA Importer System Demonstration")
    print("=" * 50)
    
    # 1. System Information
    print("\n📊 System Information:")
    system_info = get_system_info()
    
    print(f"✅ Core utilities: {len(system_info['utilities_available'])} available")
    
    dependencies = system_info['dependencies']
    print(f"📦 Dependencies:")
    for dep, available in dependencies.items():
        status = "✅" if available else "❌"
        print(f"  {status} {dep}")
    
    print(f"\n🎯 Optional features:")
    features = system_info['optional_features']
    for feature, available in features.items():
        status = "✅" if available else "❌"
        print(f"  {status} {feature}")
    
    # 2. Initialize processors
    print("\n🔧 Initializing processors...")
    
    try:
        pdf_processor = PDFProcessor()
        text_cleaner = TextCleaner()
        text_segmenter = TextSegmenter()
        fingerprinter = Fingerprinter()
        print("✅ All processors initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing processors: {e}")
        return
    
    # 3. Process sample text if no PDF provided
    pdf_file = sys.argv[1] if len(sys.argv) > 1 else None
    
    if pdf_file and Path(pdf_file).exists():
        print(f"\n📄 Processing PDF file: {pdf_file}")
        try:
            # Process PDF
            document = pdf_processor.process_pdf(pdf_file)
            print(f"✅ PDF processed: {document.total_pages} pages, {len(document.full_text)} characters")
            
            # Clean text
            cleaned_document = text_cleaner.clean_document(document)
            print(f"✅ Text cleaned")
            
            # Segment clauses
            clauses = text_segmenter.segment_document(cleaned_document)
            print(f"✅ Text segmented: {len(clauses)} clauses identified")
            
            # Create fingerprint
            fingerprint = fingerprinter.fingerprint_document(document)
            print(f"✅ Document fingerprinted")
            
            # Display sample results
            print(f"\n📋 Sample Results:")
            print(f"  EA ID: {document.metadata.get('ea_id')}")
            print(f"  Total pages: {document.total_pages}")
            print(f"  Total clauses: {len(clauses)}")
            print(f"  Text length: {len(document.full_text):,} characters")
            
            if clauses:
                print(f"\n📝 Sample clauses:")
                for i, clause in enumerate(clauses[:3]):  # Show first 3
                    print(f"  {i+1}. [{clause.clause_id}] {clause.heading or 'No heading'}")
                    print(f"     Text preview: {clause.text[:100]}...")
                
                if len(clauses) > 3:
                    print(f"  ... and {len(clauses) - 3} more clauses")
        
        except Exception as e:
            print(f"❌ Error processing PDF: {e}")
            logger.error(f"PDF processing failed: {e}")
    
    else:
        print(f"\n📝 Processing sample text (no PDF file provided)")
        
        # Sample EA text for demonstration
        sample_text = """
        ENTERPRISE AGREEMENT
        
        1. TITLE
        This Agreement shall be known as the Sample Enterprise Agreement 2024.
        
        2. PARTIES
        2.1 This Agreement is made between Sample Company Pty Ltd and its employees.
        2.2 The Agreement covers all employees in the classifications set out in Schedule A.
        
        3. TERM
        3.1 This Agreement commences on 1 January 2024.
        3.2 This Agreement will have a nominal expiry date of 31 December 2026.
        
        4. WAGES AND ALLOWANCES
        4.1 Base Rates
        The minimum weekly rates of pay are set out in Schedule B.
        
        4.2 Allowances
        a) Tool allowance: $25.00 per week
        b) Travel allowance: As per Schedule C
        c) Meal allowance: $15.00 per shift when working overtime
        
        5. HOURS OF WORK
        5.1 Ordinary hours shall be 38 hours per week, Monday to Friday.
        5.2 Overtime rates apply for work outside ordinary hours.
        
        SCHEDULE A - CLASSIFICATIONS
        Level 1: General Employee
        Level 2: Experienced Employee  
        Level 3: Senior Employee
        
        SCHEDULE B - WAGE RATES
        Level 1: $800.00 per week
        Level 2: $900.00 per week
        Level 3: $1000.00 per week
        """
        
        try:
            # Clean the sample text
            cleaned_text = text_cleaner.clean_text(sample_text)
            print(f"✅ Sample text cleaned")
            
            # Segment the text
            clauses = text_segmenter.segment_text(cleaned_text, ea_id="SAMPLE-DEMO")
            print(f"✅ Sample text segmented: {len(clauses)} clauses identified")
            
            # Display results
            print(f"\n📋 Segmentation Results:")
            for clause in clauses:
                print(f"  [{clause.clause_id}] {clause.heading or 'No heading'}")
                print(f"    Path: {' > '.join(clause.path) if clause.path else 'No path'}")
                print(f"    Length: {len(clause.text)} chars, {clause.token_count} tokens")
                print(f"    Preview: {clause.text[:100]}...")
                print()
        
        except Exception as e:
            print(f"❌ Error processing sample text: {e}")
            logger.error(f"Sample processing failed: {e}")
    
    # 4. Show next steps
    print(f"\n🔜 Next Steps:")
    print("  1. Install optional dependencies for full functionality:")
    print("     pip install spacy datasketch")
    print("     python -m spacy download en_core_web_sm")
    print("  2. Set up PostgreSQL database for persistence")
    print("  3. Configure environment variables in .env file")
    print("  4. Run full pipeline with: python -m ea_importer.cli ingest --help")
    
    print(f"\n✅ Demonstration completed!")
    print("📚 See README.md for full documentation and usage examples")


if __name__ == "__main__":
    main()