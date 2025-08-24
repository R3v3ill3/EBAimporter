#!/usr/bin/env python3
"""
Complete EA Importer Demo - Including Web Interface

This script demonstrates the full EA Importer system capabilities:
1. Basic setup and structure validation
2. Core processing pipeline demonstration
3. Web interface showcase
4. CLI command examples

Run this after setting up the system to see all features in action.
"""

import sys
import time
import subprocess
from pathlib import Path

# Add src to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def print_section(title: str):
    """Print a section header."""
    print(f"\n📋 {title}")
    print("-" * 60)


def run_command(cmd: str, description: str = None):
    """Run a command and show output."""
    if description:
        print(f"\n▶ {description}")
    print(f"Command: {cmd}")
    print("-" * 40)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=project_root)
        if result.stdout:
            print(result.stdout)
        if result.stderr and result.returncode != 0:
            print(f"Error: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"Failed to run command: {e}")
        return False


def demo_system_overview():
    """Show system overview and structure."""
    print_header("EA IMPORTER SYSTEM DEMO")
    
    print("""
🎯 EA Importer - Australian Enterprise Agreement Ingestion & Corpus Builder

A comprehensive system for processing, clustering, and querying Enterprise Agreements
with human-in-the-loop validation and quality assurance.

Key Features:
• Batch PDF ingestion with OCR support
• Intelligent text cleaning and clause segmentation  
• Document fingerprinting and similarity analysis
• Multi-algorithm clustering into EA families
• Human-in-the-loop validation workflows
• Structured data extraction (rates, rules, allowances)
• Quality assurance with synthetic testing
• Version control and audit trails
• Web interface for review and management
    """)
    
    print_section("Project Structure")
    
    # Show key directories
    key_paths = [
        "src/ea_importer/",
        "src/ea_importer/core/",
        "src/ea_importer/utils/",
        "src/ea_importer/pipeline/",
        "src/ea_importer/models/", 
        "src/ea_importer/web/",
        "src/ea_importer/web/templates/",
        "src/ea_importer/cli/",
        "scripts/",
        "data/",
        "requirements.txt",
        "setup.py",
        "README.md",
    ]
    
    print("Key project components:")
    for path in key_paths:
        full_path = project_root / path
        if full_path.exists():
            if full_path.is_dir():
                count = len(list(full_path.iterdir()))
                print(f"  ✓ {path} ({count} items)")
            else:
                print(f"  ✓ {path}")
        else:
            print(f"  ✗ {path} (missing)")


def demo_cli_interface():
    """Demonstrate CLI commands."""
    print_section("CLI Interface Demonstration")
    
    print("The EA Importer provides a comprehensive CLI interface:")
    
    # Show main help
    run_command("python -m ea_importer.cli --help", "Main CLI help")
    
    # Show subcommand help
    commands_to_demo = [
        ("python -m ea_importer.cli ingest --help", "Ingestion commands"),
        ("python -m ea_importer.cli cluster --help", "Clustering commands"),
        ("python -m ea_importer.cli web --help", "Web interface commands"),
        ("python -m ea_importer.cli db --help", "Database commands"),
    ]
    
    for cmd, desc in commands_to_demo:
        run_command(cmd, desc)
        time.sleep(1)


def demo_web_interface():
    """Demonstrate web interface capabilities."""
    print_section("Web Interface Features")
    
    print("""
🌐 Web Interface Capabilities:

The web interface provides human-in-the-loop functionality for:

1. Dashboard Overview
   • System statistics and health monitoring
   • Recent family activity
   • Version information
   
2. Family Management
   • Browse all EA families
   • View detailed family information
   • Inspect clauses and instances
   • Family statistics and quality metrics
   
3. Clustering Review
   • Review clustering algorithm results
   • Approve or reject cluster formations
   • Quality assessment with confidence scoring
   • Interactive cluster visualization
   
4. Version Control
   • Manage corpus versions
   • Lock versions for production use
   • Version timeline and history
   • Audit trail management

5. Quality Assurance
   • Health checks and system status
   • Processing statistics
   • Error reporting and diagnostics
    """)
    
    print_section("Web Interface Templates")
    
    templates_dir = project_root / "src/ea_importer/web/templates"
    if templates_dir.exists():
        templates = list(templates_dir.glob("*.html"))
        print(f"Available templates ({len(templates)}):")
        for template in templates:
            print(f"  ✓ {template.name}")
            
            # Show what each template does
            template_descriptions = {
                "base.html": "Base template with navigation and styling",
                "dashboard.html": "Main dashboard with system overview",
                "families.html": "List all EA families with statistics",
                "family_detail.html": "Detailed view of individual families",
                "clustering.html": "Review clustering results and algorithms",
                "clustering_detail.html": "Detailed cluster analysis and approval",
                "versions.html": "Version control and corpus management"
            }
            
            if template.name in template_descriptions:
                print(f"    {template_descriptions[template.name]}")
    
    print_section("Starting Web Interface")
    
    print("To start the web interface, use one of these methods:")
    print("\n1. Using CLI command:")
    print("   python -m ea_importer.cli web start")
    print("   python -m ea_importer.cli web start --host 0.0.0.0 --port 8080 --reload")
    
    print("\n2. Using start script:")
    print("   python scripts/start_web.py")
    
    print("\n3. Direct uvicorn:")
    print("   uvicorn ea_importer.web:app --host 127.0.0.1 --port 8000 --reload")
    
    print("\nOnce started, access the interface at: http://127.0.0.1:8000")


def demo_data_flow():
    """Show the data processing flow."""
    print_section("Data Processing Flow")
    
    print("""
📊 End-to-End Processing Pipeline:

1. PDF Ingestion
   └─ Raw PDFs (data/eas/raw/) 
   └─ Text extraction with OCR fallback
   └─ Cleaned text files (data/eas/text/)

2. Text Processing  
   └─ Clause segmentation with pattern recognition
   └─ Hierarchical numbering detection
   └─ Clause files (data/eas/clauses/)

3. Fingerprinting
   └─ SHA256 and MinHash generation
   └─ LSH for similarity queries
   └─ Fingerprint files (data/eas/fp/)

4. Clustering Analysis
   └─ Multi-algorithm clustering (MinHash, HDBSCAN, DBSCAN)
   └─ Adaptive threshold selection
   └─ Cluster reports (data/reports/clusters/)

5. Family Building
   └─ Human-in-the-loop cluster review (Web Interface)
   └─ Gold text selection and scoring
   └─ Family definitions (data/families/)

6. Instance Management
   └─ Employer-specific parameters
   └─ Overlay generation and validation
   └─ Instance data (data/instances/)

7. Quality Assurance
   └─ Synthetic worker scenario testing
   └─ Rate calculation validation
   └─ QA reports (data/reports/qa/)

8. Version Control
   └─ Corpus versioning and locking
   └─ Audit trail generation
   └─ Version manifests (data/versions/)
    """)


def demo_usage_examples():
    """Show practical usage examples."""
    print_section("Practical Usage Examples")
    
    print("""
🔧 Common Workflows:

1. Initial Setup:
   pip install -r requirements.txt
   python -m ea_importer.cli db init
   python -m ea_importer.cli config

2. Process New EAs:
   # Place PDFs in data/eas/raw/
   python -m ea_importer.cli ingest run data/eas/raw/ --force-ocr
   python -m ea_importer.cli ingest status

3. Run Clustering Analysis:
   python -m ea_importer.cli cluster run --algorithm adaptive
   python -m ea_importer.cli web start  # Review results in web interface

4. Family Management:
   # Use web interface at http://127.0.0.1:8000/families
   # Review clustering at http://127.0.0.1:8000/clustering
   # Approve families and manage versions

5. Batch Import from CSV:
   # Using built-in CSV importer for FWC documents
   python -c "
   from ea_importer.utils.csv_importer import create_csv_importer
   importer = create_csv_importer()
   # Process CSV with PDF URLs
   "

6. Quality Assurance:
   python -c "
   from ea_importer.utils.qa_calculator import create_qa_calculator
   calculator = create_qa_calculator()
   # Run smoke tests on processed families
   "
    """)


def demo_advanced_features():
    """Show advanced features and configurations."""
    print_section("Advanced Features")
    
    print("""
⚙️ Advanced Configuration:

1. Environment Variables:
   DATABASE_URL=postgresql://user:pass@localhost:5432/ea_importer
   OCR_LANGUAGE=eng
   CLUSTER_THRESHOLD_HIGH=0.95
   WEB_HOST=0.0.0.0
   WEB_PORT=8000
   DEBUG=true

2. Custom Clustering:
   # Fine-tune clustering parameters
   from ea_importer.pipeline.clustering import EAClusterer
   clusterer = EAClusterer()
   # Use specific algorithms with custom thresholds

3. Rate Extraction:
   # Extract pay rates and business rules
   from ea_importer.utils.rates_rules_extractor import create_rates_extractor
   extractor = create_rates_extractor()
   # Process specific family for rates

4. Instance Overlays:
   # Create employer-specific modifications
   from ea_importer.utils.instance_manager import create_instance_manager
   manager = create_instance_manager()
   # Generate overlays for specific employers

5. Audit and Compliance:
   # Full audit trail with version control
   from ea_importer.utils.version_control import create_version_manager
   version_manager = create_version_manager()
   # Lock corpus versions for production use
    """)


def main():
    """Run the complete demo."""
    try:
        demo_system_overview()
        demo_cli_interface()
        demo_web_interface()
        demo_data_flow()
        demo_usage_examples()
        demo_advanced_features()
        
        print_header("DEMO COMPLETE")
        print("""
🎉 EA Importer Demo Complete!

The system is ready for use. Key next steps:

1. Set up your environment variables in .env file
2. Initialize the database: python -m ea_importer.cli db init
3. Place some EA PDFs in data/eas/raw/
4. Run ingestion: python -m ea_importer.cli ingest run data/eas/raw/
5. Start web interface: python -m ea_importer.cli web start
6. Access web interface: http://127.0.0.1:8000

For detailed documentation, see README.md

Happy processing! 🚀
        """)
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)