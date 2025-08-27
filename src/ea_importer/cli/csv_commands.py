"""
CLI commands for CSV batch import functionality.
"""

import asyncio
import csv
import json
from pathlib import Path
from typing import Optional, List
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

from ..utils.csv_batch_importer import CSVBatchImporter, import_fwc_search_results, import_url_list
try:
    from ..database import get_db_session  # type: ignore
    from ..models import BatchImportJob, BatchImportResult  # type: ignore
    _DB_AVAILABLE = True
except Exception:
    get_db_session = None  # type: ignore
    BatchImportJob = None  # type: ignore
    BatchImportResult = None  # type: ignore
    _DB_AVAILABLE = False
from ..core.logging import get_logger

logger = get_logger(__name__)
console = Console()

# Create CSV batch import CLI group
csv_app = typer.Typer(name="csv", help="CSV batch import commands")


@csv_app.command("import")
def import_csv(
    csv_file: str = typer.Argument(..., help="Path to CSV file with document URLs"),
    job_name: Optional[str] = typer.Option(None, "--name", "-n", help="Job name"),
    auto_process: bool = typer.Option(True, "--auto-process/--no-auto-process", help="Auto-process downloaded PDFs"),
    max_concurrent: int = typer.Option(5, "--concurrent", "-c", help="Maximum concurrent downloads"),
    resume_job: Optional[int] = typer.Option(None, "--resume", "-r", help="Resume existing job ID")
):
    """Import Enterprise Agreements from CSV file with URLs."""
    
    csv_path = Path(csv_file)
    if not csv_path.exists():
        console.print(f"[red]Error: CSV file not found: {csv_file}[/red]")
        raise typer.Exit(1)
    
    console.print(Panel(f"Starting CSV batch import from: {csv_file}", title="CSV Batch Import"))
    
    async def run_import():
        importer = CSVBatchImporter(max_concurrent=max_concurrent)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Importing documents...", total=None)
            
            try:
                job = await importer.import_from_csv(
                    csv_file_path=str(csv_path),
                    job_name=job_name,
                    auto_process=auto_process,
                    resume_job_id=resume_job
                )
                
                progress.update(task, description="Import completed")
                
                # Display results
                console.print("\n[green]✓ Import completed successfully![/green]")
                
                # Create results table
                table = Table(title="Import Results")
                table.add_column("Metric", style="cyan")
                table.add_column("Count", justify="right", style="magenta")
                
                job_id = getattr(job, 'id', 0)
                total_items = getattr(job, 'total_items', 0)
                processed_items = getattr(job, 'processed_items', total_items)
                successful_items = getattr(job, 'successful_items', processed_items)
                failed_items = getattr(job, 'failed_items', 0)
                table.add_row("Job ID", str(job_id))
                table.add_row("Job Name", getattr(job, 'job_name', 'CSV Import'))
                table.add_row("Total Items", str(total_items))
                table.add_row("Processed", str(processed_items))
                table.add_row("Successful", str(successful_items))
                table.add_row("Failed", str(failed_items))
                table.add_row("Success Rate", f"{(successful_items/total_items*100):.1f}%" if total_items > 0 else "0%")
                
                console.print(table)
                
                if failed_items > 0 and _DB_AVAILABLE:
                    console.print(f"\n[yellow]Warning: {failed_items} items failed. Use 'ea-importer csv status {job_id}' for details.[/yellow]")
                
            except Exception as e:
                progress.update(task, description="Import failed")
                console.print(f"\n[red]Error: {e}[/red]")
                raise typer.Exit(1)
    
    asyncio.run(run_import())


@csv_app.command("fwc")
def import_fwc(
    csv_file: str = typer.Argument(..., help="Path to FWC search results CSV"),
    job_name: Optional[str] = typer.Option(None, "--name", "-n", help="Job name")
):
    """Import Enterprise Agreements from FWC search results CSV."""
    
    csv_path = Path(csv_file)
    if not csv_path.exists():
        console.print(f"[red]Error: CSV file not found: {csv_file}[/red]")
        raise typer.Exit(1)
    
    console.print(Panel(f"Importing FWC search results from: {csv_file}", title="FWC Import"))
    
    async def run_fwc_import():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Processing FWC data...", total=None)
            
            try:
                job = await import_fwc_search_results(
                    csv_file_path=str(csv_path),
                    job_name=job_name
                )
                
                progress.update(task, description="FWC import completed")
                console.print(f"\n[green]✓ FWC import completed! Job ID: {job.id}[/green]")
                
            except Exception as e:
                progress.update(task, description="FWC import failed")
                console.print(f"\n[red]Error: {e}[/red]")
                raise typer.Exit(1)
    
    asyncio.run(run_fwc_import())


@csv_app.command("urls")
def import_urls(
    urls: List[str] = typer.Argument(..., help="List of PDF URLs to import"),
    job_name: Optional[str] = typer.Option(None, "--name", "-n", help="Job name")
):
    """Import Enterprise Agreements from a list of URLs."""
    
    if not urls:
        console.print("[red]Error: No URLs provided[/red]")
        raise typer.Exit(1)
    
    console.print(Panel(f"Importing {len(urls)} URLs", title="URL Import"))
    
    for i, url in enumerate(urls):
        console.print(f"  {i+1}. {url}")
    
    async def run_url_import():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Processing URLs...", total=None)
            
            try:
                job = await import_url_list(urls=urls, job_name=job_name)
                
                progress.update(task, description="URL import completed")
                console.print(f"\n[green]✓ URL import completed! Job ID: {job.id}[/green]")
                
            except Exception as e:
                progress.update(task, description="URL import failed")
                console.print(f"\n[red]Error: {e}[/red]")
                raise typer.Exit(1)
    
    asyncio.run(run_url_import())


@csv_app.command("status")
def job_status(
    job_id: int = typer.Argument(..., help="Batch import job ID")
):
    """Show status of a batch import job."""
    
    if not _DB_AVAILABLE:
        console.print("[yellow]Status is only available when DB is configured.[/yellow]")
        raise typer.Exit(0)
    importer = CSVBatchImporter()
    job = importer.get_job_status(job_id)
    
    if not job:
        console.print(f"[red]Error: Job {job_id} not found[/red]")
        raise typer.Exit(1)
    
    # Job overview
    table = Table(title=f"Job Status: {job.job_name}")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Job ID", str(job.id))
    table.add_row("Name", job.job_name)
    table.add_row("Status", job.status)
    table.add_row("Source Type", job.source_type)
    table.add_row("Source Path", job.source_path or "N/A")
    table.add_row("Created", job.created_at.strftime("%Y-%m-%d %H:%M:%S"))
    
    if job.completed_at:
        table.add_row("Completed", job.completed_at.strftime("%Y-%m-%d %H:%M:%S"))
        duration = (job.completed_at - job.created_at).total_seconds()
        table.add_row("Duration", f"{duration:.1f} seconds")
    
    table.add_row("Total Items", str(job.total_items))
    table.add_row("Processed", str(job.processed_items))
    table.add_row("Successful", str(job.successful_items))
    table.add_row("Failed", str(job.failed_items))
    
    if job.total_items > 0:
        progress_pct = (job.processed_items / job.total_items) * 100
        success_rate = (job.successful_items / job.total_items) * 100
        table.add_row("Progress", f"{progress_pct:.1f}%")
        table.add_row("Success Rate", f"{success_rate:.1f}%")
    
    if job.error_message:
        table.add_row("Error", job.error_message)
    
    console.print(table)
    
    # Show failed items if any
    if job.failed_items > 0:
        with get_db_session() as session:  # type: ignore[arg-type]
            failed_results = session.query(BatchImportResult).filter(
                BatchImportResult.job_id == job_id,
                BatchImportResult.status == 'failed'
            ).limit(10).all()
            
            if failed_results:
                console.print("\n[red]Failed Items (showing first 10):[/red]")
                
                failed_table = Table()
                failed_table.add_column("Row", justify="right")
                failed_table.add_column("URL", style="blue")
                failed_table.add_column("Error", style="red")
                
                for result in failed_results:
                    error_msg = result.error_message[:50] + "..." if len(result.error_message) > 50 else result.error_message
                    failed_table.add_row(
                        str(result.row_number),
                        result.source_url[:50] + "..." if len(result.source_url) > 50 else result.source_url,
                        error_msg
                    )
                
                console.print(failed_table)


@csv_app.command("list")
def list_jobs(
    limit: int = typer.Option(20, "--limit", "-l", help="Maximum number of jobs to show")
):
    """List recent batch import jobs."""
    
    importer = CSVBatchImporter()
    jobs = importer.list_jobs(limit=limit)
    
    if not jobs:
        console.print("[yellow]No batch import jobs found[/yellow]")
        return
    
    table = Table(title="Recent Batch Import Jobs")
    table.add_column("ID", justify="right")
    table.add_column("Name", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Progress", justify="right")
    table.add_column("Success Rate", justify="right")
    table.add_column("Created", style="blue")
    
    for job in jobs:
        # Calculate progress
        if job.total_items > 0:
            progress = f"{job.processed_items}/{job.total_items}"
            success_rate = f"{(job.successful_items/job.total_items*100):.1f}%"
        else:
            progress = "0/0"
            success_rate = "0%"
        
        # Status styling
        status_style = {
            'completed': '[green]',
            'running': '[yellow]',
            'failed': '[red]',
            'cancelled': '[dim]'
        }.get(job.status, '')
        
        status_display = f"{status_style}{job.status}[/]" if status_style else job.status
        
        table.add_row(
            str(job.id),
            job.job_name,
            status_display,
            progress,
            success_rate,
            job.created_at.strftime("%m-%d %H:%M")
        )
    
    console.print(table)


@csv_app.command("cancel")
def cancel_job(
    job_id: int = typer.Argument(..., help="Job ID to cancel")
):
    """Cancel a running batch import job."""
    
    importer = CSVBatchImporter()
    
    if importer.cancel_job(job_id):
        console.print(f"[green]✓ Job {job_id} cancelled successfully[/green]")
    else:
        console.print(f"[red]Error: Could not cancel job {job_id} (not found or not running)[/red]")
        raise typer.Exit(1)


@csv_app.command("template")
def create_template(
    output_file: str = typer.Argument("ea_import_template.csv", help="Output CSV template file")
):
    """Create a CSV template file for batch imports."""
    
    template_data = [
        [
            "url",
            "title", 
            "employer",
            "union",
            "industry",
            "effective_date",
            "expiry_date",
            "status",
            "fwc_code",
            "metadata"
        ],
        [
            "https://example.com/agreement1.pdf",
            "Example Enterprise Agreement 2024",
            "Example Company Pty Ltd",
            "Example Union",
            "Manufacturing",
            "2024-01-01",
            "2026-12-31",
            "active",
            "EA24001",
            '{"region": "NSW", "employees": 150}'
        ],
        [
            "https://example.com/agreement2.pdf",
            "Another Agreement",
            "Another Company",
            "",
            "Retail",
            "2023-07-01",
            "2025-06-30",
            "active",
            "",
            "{}"
        ]
    ]
    
    output_path = Path(output_file)
    
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(template_data)
        
        console.print(f"[green]✓ CSV template created: {output_file}[/green]")
        console.print("\nTemplate includes:")
        console.print("• Required fields: url, title")
        console.print("• Optional fields: employer, union, industry, dates, status, fwc_code")
        console.print("• metadata field for additional JSON data")
        console.print("\nEdit the template and use 'ea-importer csv import' to process it.")
        
    except Exception as e:
        console.print(f"[red]Error creating template: {e}[/red]")
        raise typer.Exit(1)


__all__ = ['csv_app']
 
# ------------------ Enhancements ------------------
@csv_app.command("validate")
def validate_csv(
    csv_file: str = typer.Argument(..., help="Path to CSV to validate")
):
    """Validate CSV headers and URL formats without importing."""
    from ..utils.csv_batch_importer import CSVBatchImporter
    import asyncio as _asyncio
    console.print(Panel(f"Validating CSV: {csv_file}", title="CSV Validate"))
    async def _run():
        importer = CSVBatchImporter()
        records = await importer._parse_csv_file(csv_file)  # type: ignore[attr-defined]
        bad_urls = [r for r in records if not importer._is_valid_url(r['url'])]  # type: ignore[attr-defined]
        console.print(f"[green]✓ Parsed {len(records)} records[/green]")
        if bad_urls:
            console.print(f"[yellow]Found {len(bad_urls)} invalid URLs[/yellow]")
    _asyncio.run(_run())


@csv_app.command("retry-failed")
def retry_failed_cli(
    job_id: int = typer.Argument(..., help="Existing job ID to retry failed rows from"),
    job_name: Optional[str] = typer.Option(None, "--name", "-n", help="New job name")
):
    """Create a new job from failed rows of an existing job."""
    if not _DB_AVAILABLE:
        console.print("[red]Database not available. Cannot retry failed without DB.[/red]")
        raise typer.Exit(1)
    from ..database import get_db_session as _get_db_session
    from ..models import BatchImportResult as _BatchImportResult
    import tempfile as _tempfile

    with _get_db_session() as session:  # type: ignore[arg-type]
        failed = session.query(_BatchImportResult).filter(
            _BatchImportResult.job_id == job_id, _BatchImportResult.status == 'failed'
        ).all()
        if not failed:
            console.print("[yellow]No failed items to retry[/yellow]")
            raise typer.Exit(0)

    with _tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='w', newline='', encoding='utf-8') as tmp:
        writer = csv.writer(tmp)
        writer.writerow(['url', 'title'])
        for i, r in enumerate(failed, start=1):
            writer.writerow([r.source_url, f'Retry_{i}'])
        csv_path = tmp.name

    console.print(Panel(f"Starting retry job from {len(failed)} failed rows", title="Retry Failed"))
    import asyncio as _asyncio
    async def _run():
        importer = CSVBatchImporter()
        job = await importer.import_from_csv(csv_file_path=csv_path, job_name=job_name or f"Retry {job_id}")
        console.print(f"[green]✓ Retry job started. Job ID: {getattr(job, 'id', 0)}[/green]")
    _asyncio.run(_run())