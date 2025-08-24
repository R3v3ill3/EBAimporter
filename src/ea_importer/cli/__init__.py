"""
Command-line interface for EA Importer.
"""

import sys
from pathlib import Path
from typing import Optional, List
import json

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from rich import print as rprint

from ..core.config import get_settings
from ..core.logging import setup_logging
from ..database import init_database, get_database_manager
from ..pipeline import create_ingest_pipeline
from ..pipeline.clustering import create_ea_clusterer
from ..utils.fingerprinter import create_text_fingerprinter


# Create the main app
app = typer.Typer(
    name="ea-agent",
    help="Australian Enterprise Agreement Ingestion & Corpus Builder",
    add_completion=False
)

# Create subcommands
ingest_app = typer.Typer(help="Document ingestion commands")
cluster_app = typer.Typer(help="Clustering and family detection commands")
family_app = typer.Typer(help="Family building and gold text selection commands")
qa_app = typer.Typer(help="Quality assurance and testing commands")
corpus_app = typer.Typer(help="Corpus management and versioning commands")
db_app = typer.Typer(help="Database management commands")
web_app = typer.Typer(help="Web interface commands")

app.add_typer(ingest_app, name="ingest")
app.add_typer(cluster_app, name="cluster")
app.add_typer(family_app, name="family")
app.add_typer(qa_app, name="qa")
app.add_typer(corpus_app, name="corpus")
app.add_typer(db_app, name="db")
app.add_typer(web_app, name="web")

console = Console()


def setup_cli_logging(verbose: bool = False, log_file: Optional[Path] = None):
    """Set up logging for CLI operations."""
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(log_level=log_level, log_file=log_file)


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    log_file: Optional[Path] = typer.Option(None, "--log-file", help="Log file path"),
    config_file: Optional[Path] = typer.Option(None, "--config", help="Configuration file path")
):
    """
    EA Agent - Australian Enterprise Agreement Ingestion & Corpus Builder
    
    A comprehensive system for processing, clustering, and querying Enterprise Agreements.
    """
    setup_cli_logging(verbose, log_file)
    
    if config_file and config_file.exists():
        # TODO: Load configuration from file
        pass


# ============================================================================
# INGESTION COMMANDS
# ============================================================================

@ingest_app.command("run")
def ingest_run(
    input_dir: Path = typer.Argument(..., help="Directory containing PDF files"),
    output_dir: Optional[Path] = typer.Option(None, "--out", help="Output directory (default: data/eas)"),
    force_ocr: bool = typer.Option(False, "--force-ocr", help="Force OCR even if text layer exists"),
    max_files: Optional[int] = typer.Option(None, "--max-files", help="Maximum number of files to process"),
    file_pattern: str = typer.Option("*.pdf", "--pattern", help="File pattern to match"),
):
    """
    Run the main ingestion pipeline on a directory of PDF files.
    
    This will:
    1. Extract text from PDFs (with OCR if needed)
    2. Clean and normalize text
    3. Segment into clauses
    4. Generate fingerprints
    5. Save outputs and update database
    """
    settings = get_settings()
    
    if not input_dir.exists():
        rprint(f"[red]Error: Input directory does not exist: {input_dir}[/red]")
        raise typer.Exit(1)
    
    if output_dir:
        settings.data_root = output_dir
    
    # Ensure directories exist
    settings.ensure_directories()
    
    # Create and run pipeline
    pipeline = create_ingest_pipeline()
    
    rprint(f"[bold blue]Starting EA ingestion pipeline[/bold blue]")
    rprint(f"Input directory: [cyan]{input_dir}[/cyan]")
    rprint(f"Output directory: [cyan]{settings.data_root}[/cyan]")
    rprint(f"File pattern: [cyan]{file_pattern}[/cyan]")
    
    if force_ocr:
        rprint(f"[yellow]OCR will be forced for all documents[/yellow]")
    
    if max_files:
        rprint(f"[yellow]Processing limited to {max_files} files[/yellow]")
    
    # Run the pipeline
    stats = pipeline.batch_ingest(
        input_dir=input_dir,
        file_pattern=file_pattern,
        force_ocr=force_ocr,
        max_files=max_files
    )
    
    # Display results
    display_ingest_results(stats)


def display_ingest_results(stats):
    """Display ingestion results in a formatted table."""
    table = Table(title="Ingestion Results")
    
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    
    table.add_row("Run ID", stats.run_id)
    table.add_row("Files Found", str(stats.files_found))
    table.add_row("Files Processed", str(stats.files_processed))
    table.add_row("Files Succeeded", str(stats.files_succeeded))
    table.add_row("Files Failed", str(stats.files_failed))
    table.add_row("Success Rate", f"{stats.success_rate:.1%}")
    table.add_row("Total Pages", str(stats.total_pages))
    table.add_row("Total Clauses", str(stats.total_clauses))
    table.add_row("Total Characters", f"{stats.total_text_chars:,}")
    
    if stats.duration_seconds:
        table.add_row("Duration", f"{stats.duration_seconds:.1f} seconds")
    
    console.print(table)
    
    # Show failures if any
    if stats.files_failed > 0:
        rprint(f"\n[red]{stats.files_failed} files failed processing:[/red]")
        for stat in stats.processing_stats:
            if not stat.success:
                rprint(f"  • {stat.ea_id}: {stat.error_message}")


@ingest_app.command("status")
def ingest_status():
    """Show status of the ingestion pipeline."""
    settings = get_settings()
    
    # Count files in various directories
    stats = {
        "Raw PDFs": len(list(settings.raw_eas_dir.glob("*.pdf"))) if settings.raw_eas_dir.exists() else 0,
        "Processed Text": len(list(settings.text_dir.glob("*.txt"))) if settings.text_dir.exists() else 0,
        "Clause Files": len(list(settings.clauses_dir.glob("*.jsonl"))) if settings.clauses_dir.exists() else 0,
        "Fingerprints": len(list(settings.fingerprints_dir.glob("*.sha256"))) if settings.fingerprints_dir.exists() else 0,
    }
    
    table = Table(title="Ingestion Status")
    table.add_column("Category", style="cyan")
    table.add_column("Count", style="magenta")
    
    for category, count in stats.items():
        table.add_row(category, str(count))
    
    console.print(table)


# ============================================================================
# CLUSTERING COMMANDS
# ============================================================================

@cluster_app.command("run")
def cluster_run(
    clauses_dir: Optional[Path] = typer.Option(None, "--clauses", help="Directory containing clause files"),
    output_dir: Optional[Path] = typer.Option(None, "--out", help="Output directory for cluster reports"),
    threshold: float = typer.Option(0.9, "--threshold", help="Similarity threshold for clustering"),
    algorithm: str = typer.Option("adaptive", "--algorithm", help="Clustering algorithm (adaptive, minhash, hdbscan, dbscan)"),
):
    """
    Run clustering to group similar EAs into families.
    
    This analyzes document fingerprints to identify groups of similar
    Enterprise Agreements that should be treated as the same family.
    """
    settings = get_settings()
    
    if clauses_dir is None:
        clauses_dir = settings.clauses_dir
    
    if output_dir is None:
        output_dir = settings.reports_dir / "clusters"
    
    if not clauses_dir.exists():
        rprint(f"[red]Error: Clauses directory does not exist: {clauses_dir}[/red]")
        raise typer.Exit(1)
    
    rprint(f"[bold blue]Starting EA clustering[/bold blue]")
    rprint(f"Clauses directory: [cyan]{clauses_dir}[/cyan]")
    rprint(f"Output directory: [cyan]{output_dir}[/cyan]")
    rprint(f"Algorithm: [cyan]{algorithm}[/cyan]")
    
    if algorithm != "adaptive":
        rprint(f"Threshold: [cyan]{threshold}[/cyan]")
    
    # Load fingerprints
    fingerprinter = create_text_fingerprinter()
    clusterer = create_ea_clusterer()
    
    # Find all fingerprint files
    fingerprint_files = list(settings.fingerprints_dir.glob("*.minhash"))
    
    if not fingerprint_files:
        rprint(f"[red]Error: No fingerprint files found in {settings.fingerprints_dir}[/red]")
        rprint(f"[yellow]Run 'ea-agent ingest run' first to process PDFs[/yellow]")
        raise typer.Exit(1)
    
    rprint(f"Loading {len(fingerprint_files)} document fingerprints...")
    
    fingerprints = []
    with Progress() as progress:
        task = progress.add_task("Loading fingerprints...", total=len(fingerprint_files))
        
        for fp_file in fingerprint_files:
            try:
                with open(fp_file, 'r') as f:
                    fp_data = json.load(f)
                fingerprint = fingerprinter.load_fingerprint(str(fp_file))
                fingerprints.append(fingerprint)
                progress.advance(task)
            except Exception as e:
                rprint(f"[red]Failed to load {fp_file}: {e}[/red]")
    
    if not fingerprints:
        rprint(f"[red]Error: No valid fingerprints loaded[/red]")
        raise typer.Exit(1)
    
    rprint(f"Loaded {len(fingerprints)} fingerprints")
    
    # Run clustering
    if algorithm == "adaptive":
        result = clusterer.adaptive_clustering(fingerprints)
    elif algorithm == "minhash":
        clusters = clusterer.cluster_by_minhash_threshold(fingerprints, threshold)
        # Create result object (simplified for this example)
        result = None  # Would need to create proper ClusteringResult
    else:
        rprint(f"[red]Error: Unknown algorithm: {algorithm}[/red]")
        raise typer.Exit(1)
    
    if result:
        # Save results
        output_dir.mkdir(parents=True, exist_ok=True)
        clusterer.save_clustering_result(result, output_dir)
        
        # Display results
        display_clustering_results(result)
    else:
        rprint(f"[red]Clustering failed[/red]")


def display_clustering_results(result):
    """Display clustering results."""
    table = Table(title="Clustering Results")
    
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Algorithm", result.algorithm.value)
    table.add_row("Documents", str(result.num_documents))
    table.add_row("Clusters", str(result.num_clusters))
    table.add_row("Outliers", str(len(result.outliers)))
    table.add_row("Execution Time", f"{result.execution_time_seconds:.2f}s")
    
    console.print(table)
    
    # Show cluster details
    if result.clusters:
        cluster_table = Table(title="Cluster Details")
        cluster_table.add_column("Cluster ID", style="cyan")
        cluster_table.add_column("Size", style="magenta")
        cluster_table.add_column("Confidence", style="green")
        cluster_table.add_column("Centroid", style="blue")
        
        for cluster in result.clusters[:10]:  # Show first 10 clusters
            cluster_table.add_row(
                cluster.cluster_id[:8] + "...",
                str(cluster.size),
                f"{cluster.confidence_score:.3f}",
                cluster.centroid_ea_id
            )
        
        if len(result.clusters) > 10:
            cluster_table.add_row("...", "...", "...", "...")
        
        console.print(cluster_table)


# ============================================================================
# DATABASE COMMANDS
# ============================================================================

@db_app.command("init")
def db_init(
    drop_existing: bool = typer.Option(False, "--drop", help="Drop existing tables first"),
    database_url: Optional[str] = typer.Option(None, "--url", help="Database URL override")
):
    """Initialize the database with required tables."""
    rprint(f"[bold blue]Initializing database...[/bold blue]")
    
    if drop_existing:
        rprint(f"[yellow]Warning: This will drop all existing tables![/yellow]")
        if not typer.confirm("Are you sure you want to continue?"):
            rprint("Cancelled")
            raise typer.Exit()
    
    try:
        init_database(database_url=database_url, drop_existing=drop_existing)
        rprint(f"[green]Database initialized successfully[/green]")
    except Exception as e:
        rprint(f"[red]Database initialization failed: {e}[/red]")
        raise typer.Exit(1)


@db_app.command("status")
def db_status():
    """Check database connection and show status."""
    db_manager = get_database_manager()
    
    rprint(f"[bold blue]Database Status[/bold blue]")
    
    # Test connection
    if db_manager.test_connection():
        rprint(f"[green]✓ Database connection successful[/green]")
        
        # Get some basic stats
        try:
            with db_manager.session_scope() as session:
                # Count records in key tables
                from ..models import IngestRun, DocumentFingerprint as DBDocumentFingerprint
                
                ingest_runs = session.query(IngestRun).count()
                fingerprints = session.query(DBDocumentFingerprint).count()
                
                table = Table()
                table.add_column("Table", style="cyan")
                table.add_column("Records", style="magenta")
                
                table.add_row("Ingest Runs", str(ingest_runs))
                table.add_row("Document Fingerprints", str(fingerprints))
                
                console.print(table)
                
        except Exception as e:
            rprint(f"[yellow]Could not retrieve database statistics: {e}[/yellow]")
    
    else:
        rprint(f"[red]✗ Database connection failed[/red]")
        raise typer.Exit(1)


# ============================================================================
# UTILITY COMMANDS
# ============================================================================

@app.command("version")
def version():
    """Show version information."""
    from .. import VERSION_INFO
    
    table = Table()
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="magenta")
    
    for key, value in VERSION_INFO.items():
        table.add_row(key.title(), str(value))
    
    console.print(table)


@app.command("config")
def show_config():
    """Show current configuration."""
    settings = get_settings()
    
    table = Table(title="EA Importer Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="magenta")
    
    # Key settings to display
    config_items = [
        ("Data Root", str(settings.data_root)),
        ("Database URL", settings.database_url[:50] + "..." if len(settings.database_url) > 50 else settings.database_url),
        ("OCR Language", settings.ocr_language),
        ("OCR DPI", str(settings.ocr_dpi)),
        ("Min Clause Count", str(settings.min_clause_count)),
        ("Jurisdiction", settings.jurisdiction),
        ("Target Version", settings.target_version),
        ("Debug Mode", str(settings.debug)),
    ]
    
    for setting, value in config_items:
        table.add_row(setting, value)
    
    console.print(table)




# ============================================================================
# WEB INTERFACE COMMANDS
# ============================================================================

@web_app.command("start")
def web_start(
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind the web server to"),
    port: int = typer.Option(8000, "--port", help="Port to bind the web server to"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode")
):
    """
    Start the web interface for human-in-the-loop review and approval.
    
    This provides a web-based interface for:
    - Reviewing clustering results
    - Managing EA families 
    - Approving family formations
    - Version control and corpus locking
    """
    import uvicorn
    from ..web import app as web_app
    
    rprint(f"[bold blue]Starting EA Importer Web Interface[/bold blue]")
    rprint(f"Server: [cyan]http://{host}:{port}[/cyan]")
    
    if debug:
        rprint(f"[yellow]Debug mode enabled[/yellow]")
    
    if reload:
        rprint(f"[yellow]Auto-reload enabled for development[/yellow]")
    
    try:
        uvicorn.run(
            "ea_importer.web:app",
            host=host,
            port=port,
            reload=reload,
            log_level="debug" if debug else "info"
        )
    except KeyboardInterrupt:
        rprint(f"\n[yellow]Web server stopped[/yellow]")
    except Exception as e:
        rprint(f"[red]Failed to start web server: {e}[/red]")
        raise typer.Exit(1)


@web_app.command("status")
def web_status():
    """
    Check the status of web interface components.
    """
    import requests
    from ..core.config import get_settings
    
    settings = get_settings()
    url = f"http://{settings.web_host}:{settings.web_port}/health"
    
    rprint(f"[bold blue]Web Interface Status[/bold blue]")
    rprint(f"Expected URL: [cyan]{url}[/cyan]")
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            rprint(f"[green]✓ Web interface is running[/green]")
            
            table = Table()
            table.add_column("Component", style="cyan")
            table.add_column("Status", style="magenta")
            
            table.add_row("Overall", data.get('status', 'unknown'))
            table.add_row("Database", data.get('database', 'unknown'))
            table.add_row("Data Directories", data.get('data_directories', 'unknown'))
            table.add_row("Version", data.get('version', 'unknown'))
            
            console.print(table)
        else:
            rprint(f"[red]✗ Web interface returned status code {response.status_code}[/red]")
    
    except requests.exceptions.ConnectionError:
        rprint(f"[red]✗ Web interface is not running or not accessible[/red]")
        rprint(f"[yellow]Start it with: ea-agent web start[/yellow]")
    except Exception as e:
        rprint(f"[red]✗ Error checking web interface: {e}[/red]")


if __name__ == "__main__":
    app()