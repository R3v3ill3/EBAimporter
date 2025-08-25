"""CLI interface for EA Importer system."""

import typer
from pathlib import Path
from typing import Optional
import logging

from ..core import setup_logging, get_logger
from ..utils import (
    PDFProcessor, TextCleaner, TextSegmenter, 
    Fingerprinter, RatesRulesExtractor, QACalculator
)
from ..pipeline import ClusteringEngine, FamilyBuilder, InstanceManager
from ..utils.version_control import VersionController
from .csv_commands import csv_app
from .test_commands import test_app

app = typer.Typer(help="EA Importer - Enterprise Agreement Processing System")
logger = get_logger(__name__)

# Subcommands
ingest_app = typer.Typer(help="PDF ingestion commands")
cluster_app = typer.Typer(help="Clustering commands")
family_app = typer.Typer(help="Family management commands")
qa_app = typer.Typer(help="Quality assurance commands")
version_app = typer.Typer(help="Version control commands")

app.add_typer(ingest_app, name="ingest")
app.add_typer(cluster_app, name="cluster")
app.add_typer(family_app, name="family")
app.add_typer(qa_app, name="qa")
app.add_typer(version_app, name="version")
app.add_typer(csv_app, name="csv")
app.add_typer(test_app, name="test")

@app.command()
def setup(
    database_url: Optional[str] = typer.Option(None, "--db", help="Database URL"),
    create_dirs: bool = typer.Option(True, help="Create data directories")
):
    """Initialize EA Importer system."""
    setup_logging()
    logger.info("Setting up EA Importer system")
    
    if create_dirs:
        from ..core.config import get_settings
        settings = get_settings()
        settings.create_directories()
        typer.echo("✅ Created data directories")
    
    if database_url:
        typer.echo(f"Database URL configured: {database_url}")
    
    typer.echo("✅ EA Importer setup completed")

@ingest_app.command()
def run(
    input_dir: Path = typer.Argument(..., help="Directory containing PDF files"),
    output_dir: Optional[Path] = typer.Option(None, "--output", help="Output directory"),
    force_ocr: bool = typer.Option(False, "--force-ocr", help="Force OCR processing"),
    max_files: Optional[int] = typer.Option(None, "--max-files", help="Maximum files to process")
):
    """Run PDF ingestion pipeline."""
    setup_logging()
    
    if not input_dir.exists():
        typer.echo(f"❌ Input directory not found: {input_dir}")
        raise typer.Exit(1)
    
    output_dir = output_dir or Path("data/eas")
    
    # Process PDFs
    processor = PDFProcessor()
    cleaner = TextCleaner()
    segmenter = TextSegmenter()
    fingerprinter = Fingerprinter()
    
    pdf_files = list(input_dir.glob("*.pdf"))
    if max_files:
        pdf_files = pdf_files[:max_files]
    
    typer.echo(f"Processing {len(pdf_files)} PDF files...")
    
    processed_count = 0
    failed_count = 0
    
    for pdf_file in pdf_files:
        try:
            # Process PDF
            document = processor.process_pdf(pdf_file, force_ocr=force_ocr)
            
            # Clean text
            cleaned_document = cleaner.clean_document(document)
            
            # Segment clauses
            clauses = segmenter.segment_document(cleaned_document)
            
            # Generate fingerprint
            fingerprint = fingerprinter.fingerprint_document(document)
            
            # Save outputs
            ea_id = document.metadata['ea_id']
            
            # Save text
            text_dir = output_dir / "text"
            text_dir.mkdir(parents=True, exist_ok=True)
            with open(text_dir / f"{ea_id}.txt", 'w') as f:
                f.write(document.full_text)
            
            # Save clauses
            clauses_dir = output_dir / "clauses"
            clauses_dir.mkdir(parents=True, exist_ok=True)
            with open(clauses_dir / f"{ea_id}.jsonl", 'w') as f:
                import json
                for clause in clauses:
                    clause_data = {
                        'ea_id': clause.ea_id,
                        'clause_id': clause.clause_id,
                        'heading': clause.heading,
                        'text': clause.text,
                        'path': clause.path,
                        'hash_sha256': clause.hash_sha256,
                        'token_count': clause.token_count
                    }
                    f.write(json.dumps(clause_data) + '\n')
            
            # Save fingerprint
            fp_dir = output_dir / "fp"
            fingerprinter.save_fingerprint(fingerprint, fp_dir)
            
            processed_count += 1
            typer.echo(f"✅ Processed: {pdf_file.name} -> {ea_id}")
            
        except Exception as e:
            failed_count += 1
            typer.echo(f"❌ Failed: {pdf_file.name} - {e}")
    
    typer.echo(f"\n📊 Results: {processed_count} processed, {failed_count} failed")

@cluster_app.command()
def run(
    clauses_dir: Path = typer.Argument(..., help="Directory containing clause files"),
    algorithm: str = typer.Option("adaptive", help="Clustering algorithm"),
    threshold: float = typer.Option(0.9, help="Similarity threshold"),
    output_dir: Optional[Path] = typer.Option(None, help="Output directory")
):
    """Run document clustering."""
    setup_logging()
    
    output_dir = output_dir or Path("reports/clusters")
    
    # Load fingerprints
    fp_dir = clauses_dir.parent / "fp"
    fingerprints = []
    
    for fp_file in fp_dir.glob("*.fingerprint"):
        try:
            fingerprinter = Fingerprinter()
            fingerprint = fingerprinter.load_fingerprint(fp_file)
            fingerprints.append(fingerprint)
        except Exception as e:
            typer.echo(f"❌ Failed to load fingerprint {fp_file}: {e}")
    
    typer.echo(f"Loaded {len(fingerprints)} fingerprints")
    
    # Run clustering
    clustering_engine = ClusteringEngine({'algorithm': algorithm})
    clusters = clustering_engine.cluster_documents(fingerprints, algorithm)
    
    # Generate candidates
    candidates = clustering_engine.generate_cluster_candidates(clusters, fingerprints)
    
    # Save results
    clustering_engine.save_clustering_results(clusters, output_dir)
    
    # Display summary
    high_conf = sum(1 for c in candidates if c.confidence_level == 'high')
    medium_conf = sum(1 for c in candidates if c.confidence_level == 'medium')
    low_conf = sum(1 for c in candidates if c.confidence_level == 'low')
    
    typer.echo(f"\n📊 Clustering Results:")
    typer.echo(f"  Total clusters: {len(clusters)}")
    typer.echo(f"  High confidence: {high_conf}")
    typer.echo(f"  Medium confidence: {medium_conf}")
    typer.echo(f"  Low confidence: {low_conf}")

@family_app.command()
def build(
    cluster_file: Path = typer.Argument(..., help="Clustering results file"),
    clauses_dir: Path = typer.Argument(..., help="Directory containing clause files"),
    output_dir: Optional[Path] = typer.Option(None, help="Output directory")
):
    """Build agreement families from clusters."""
    setup_logging()
    
    output_dir = output_dir or Path("data/families")
    
    # Load clustering results
    import json
    with open(cluster_file, 'r') as f:
        cluster_data = json.load(f)
    
    clusters = cluster_data['clusters']
    
    # Load clauses
    document_clauses = {}
    for clause_file in clauses_dir.glob("*.jsonl"):
        ea_id = clause_file.stem
        clauses = []
        
        with open(clause_file, 'r') as f:
            for line in f:
                clause_data = json.loads(line)
                from ..models import ClauseSegment
                clause = ClauseSegment(
                    ea_id=clause_data['ea_id'],
                    clause_id=clause_data['clause_id'],
                    heading=clause_data['heading'],
                    text=clause_data['text'],
                    path=clause_data['path'],
                    hash_sha256=clause_data['hash_sha256'],
                    token_count=clause_data['token_count']
                )
                clauses.append(clause)
        
        document_clauses[ea_id] = clauses
    
    # Build families
    family_builder = FamilyBuilder()
    
    # Convert clusters to candidates
    from ..models import ClusterCandidate
    candidates = []
    for cluster_id, doc_ids in clusters.items():
        candidate = ClusterCandidate(
            cluster_id=cluster_id,
            document_ids=doc_ids,
            similarity_scores=[],
            confidence_level="high" if len(doc_ids) > 1 else "singleton"
        )
        candidates.append(candidate)
    
    families = family_builder.build_families_from_clusters(candidates, document_clauses)
    
    # Save families
    for family_id, family_data in families.items():
        family_builder.save_family(family_data, output_dir)
    
    typer.echo(f"✅ Built {len(families)} families")

@qa_app.command()
def smoketest(
    family_id: str = typer.Argument(..., help="Family ID to test"),
    scenarios: int = typer.Option(20, help="Number of test scenarios"),
    output_dir: Optional[Path] = typer.Option(None, help="Output directory")
):
    """Run smoke tests on a family."""
    setup_logging()
    
    output_dir = output_dir or Path("reports/qa")
    
    # Generate synthetic workers
    qa_calculator = QACalculator()
    workers = qa_calculator.generate_synthetic_workers(family_id, scenarios)
    
    # Mock family data for testing
    family_rates = [
        {'classification': 'Level 1', 'base_rate': 25.0, 'unit': 'hourly'},
        {'classification': 'Level 2', 'base_rate': 28.0, 'unit': 'hourly'},
        {'classification': 'Level 3', 'base_rate': 32.0, 'unit': 'hourly'}
    ]
    
    family_rules = [
        {'rule_type': 'overtime', 'rule_data': {'multiplier': 1.5}},
        {'rule_type': 'penalty_weekend', 'rule_data': {'multiplier': 1.5}},
        {'rule_type': 'allowance', 'rule_data': {'amount': 25.0}}
    ]
    
    # Run tests
    results = qa_calculator.run_smoke_tests(family_id, family_rates, family_rules, workers)
    
    # Save results
    output_file = output_dir / family_id / "smoketest_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    qa_calculator.export_qa_results(results, str(output_file))
    
    # Display summary
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    total_anomalies = sum(len(r.anomalies) for r in results)
    total_warnings = sum(len(r.warnings) for r in results)
    
    typer.echo(f"\n📊 QA Results for {family_id}:")
    typer.echo(f"  Tests passed: {passed}/{len(results)}")
    typer.echo(f"  Tests failed: {failed}")
    typer.echo(f"  Total anomalies: {total_anomalies}")
    typer.echo(f"  Total warnings: {total_warnings}")

@version_app.command()
def create(
    version_name: str = typer.Argument(..., help="Version name"),
    families_dir: Path = typer.Argument(..., help="Families directory"),
    instances_dir: Optional[Path] = typer.Option(None, help="Instances directory"),
    notes: Optional[str] = typer.Option(None, help="Version notes")
):
    """Create a new corpus version."""
    setup_logging()
    
    # Load families data
    families_data = {}
    for family_dir in families_dir.iterdir():
        if family_dir.is_dir() and (family_dir / "family.json").exists():
            with open(family_dir / "family.json", 'r') as f:
                import json
                family_data = json.load(f)
                families_data[family_dir.name] = family_data
    
    # Load instances data
    instances_data = {}
    if instances_dir and instances_dir.exists():
        for instance_dir in instances_dir.iterdir():
            if instance_dir.is_dir() and (instance_dir / "instance.json").exists():
                with open(instance_dir / "instance.json", 'r') as f:
                    import json
                    instance_data = json.load(f)
                    instances_data[instance_dir.name] = instance_data
    
    # Create version
    version_controller = VersionController()
    manifest = version_controller.create_version(version_name, families_data, instances_data, notes)
    
    # Save version
    versions_dir = Path("versions")
    version_controller.save_version(manifest, families_data, instances_data, versions_dir)
    
    typer.echo(f"✅ Created version {version_name} with {manifest.families_count} families and {manifest.instances_count} instances")

@version_app.command()
def list_versions():
    """List all available versions."""
    setup_logging()
    
    version_controller = VersionController()
    versions = version_controller.list_versions(Path("versions"))
    
    if not versions:
        typer.echo("No versions found")
        return
    
    typer.echo("📋 Available Versions:")
    for version in versions:
        locked_status = "🔒" if version.get('locked', False) else "🔓"
        typer.echo(f"  {locked_status} {version['version']} - {version['created_at']} ({version.get('families_count', 0)} families, {version.get('instances_count', 0)} instances)")

def main():
    """Main CLI entry point."""
    app()

if __name__ == "__main__":
    main()