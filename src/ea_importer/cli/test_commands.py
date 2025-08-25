"""
CLI commands for running tests and quality assurance.
"""

import json
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text

from ..testing import run_tests, TestRunner
from ..core.logging import get_logger

logger = get_logger(__name__)
console = Console()

# Create test CLI group
test_app = typer.Typer(name="test", help="Testing and quality assurance commands")


@test_app.command("all")
def run_all_tests(
    output_file: Optional[str] = typer.Option(None, "--output", "-o", help="Save results to JSON file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Run the complete test suite."""
    
    console.print(Panel("EA Importer - Comprehensive Test Suite", title="Testing Framework"))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task("Running comprehensive tests...", total=None)
        
        try:
            # Run all tests
            results = run_tests()
            
            progress.update(task, description="Tests completed")
            
            # Display results summary
            stats = results["statistics"]
            
            console.print(f"\n[green]✓ Test Suite Completed![/green]")
            
            # Create summary table
            summary_table = Table(title="Test Results Summary")
            summary_table.add_column("Metric", style="cyan")
            summary_table.add_column("Value", justify="right", style="magenta")
            
            summary_table.add_row("Total Tests", str(stats["total_tests"]))
            summary_table.add_row("Passed", str(stats["passed_tests"]))
            summary_table.add_row("Failed", str(stats["failed_tests"]))
            summary_table.add_row("Success Rate", f"{stats['success_rate']:.1f}%")
            
            console.print(summary_table)
            
            # Detailed results table
            if verbose or stats["failed_tests"] > 0:
                console.print("\n[bold]Detailed Test Results:[/bold]")
                
                results_table = Table()
                results_table.add_column("Test Name", style="blue")
                results_table.add_column("Status", justify="center")
                results_table.add_column("Category", style="cyan")
                
                test_categories = {
                    "test_pdf_processor": "Unit Tests",
                    "test_text_cleaner": "Unit Tests", 
                    "test_text_segmenter": "Unit Tests",
                    "test_fingerprinter": "Unit Tests",
                    "test_rates_rules_extractor": "Unit Tests",
                    "test_qa_calculator": "Unit Tests",
                    "test_clustering_engine": "Integration Tests",
                    "test_family_builder": "Integration Tests",
                    "test_instance_manager": "Integration Tests",
                    "test_database_integration": "Integration Tests",
                    "test_e2e_pipeline": "QA Tests",
                    "test_data_consistency": "QA Tests",
                    "test_error_handling": "QA Tests",
                    "test_performance_benchmarks": "Performance Tests",
                    "test_memory_usage": "Performance Tests",
                    "test_scalability": "Performance Tests"
                }
                
                for test_name, passed in results["test_results"].items():
                    status = "[green]✓ PASS[/green]" if passed else "[red]✗ FAIL[/red]"
                    category = test_categories.get(test_name, "Unknown")
                    clean_name = test_name.replace("test_", "").replace("_", " ").title()
                    
                    results_table.add_row(clean_name, status, category)
                
                console.print(results_table)
            
            # Save results to file if requested
            if output_file:
                output_path = Path(output_file)
                
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)
                
                console.print(f"\n[blue]Results saved to: {output_file}[/blue]")
            
            # Exit with appropriate code
            if stats["failed_tests"] > 0:
                console.print(f"\n[red]Warning: {stats['failed_tests']} test(s) failed![/red]")
                raise typer.Exit(1)
            else:
                console.print("\n[green]All tests passed successfully! ✨[/green]")
                
        except Exception as e:
            progress.update(task, description="Tests failed")
            console.print(f"\n[red]Error running tests: {e}[/red]")
            raise typer.Exit(1)


@test_app.command("unit")
def run_unit_tests():
    """Run unit tests only."""
    
    console.print(Panel("Unit Tests", title="Testing Framework"))
    
    runner = TestRunner()
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Running unit tests...", total=None)
            
            results = runner.run_unit_tests()
            
            progress.update(task, description="Unit tests completed")
            
            # Display results
            _display_test_results(results, "Unit Tests")
            
    except Exception as e:
        console.print(f"[red]Error running unit tests: {e}[/red]")
        raise typer.Exit(1)
    finally:
        runner.cleanup()


@test_app.command("integration") 
def run_integration_tests():
    """Run integration tests only."""
    
    console.print(Panel("Integration Tests", title="Testing Framework"))
    
    runner = TestRunner()
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Running integration tests...", total=None)
            
            results = runner.run_integration_tests()
            
            progress.update(task, description="Integration tests completed")
            
            # Display results
            _display_test_results(results, "Integration Tests")
            
    except Exception as e:
        console.print(f"[red]Error running integration tests: {e}[/red]")
        raise typer.Exit(1)
    finally:
        runner.cleanup()


@test_app.command("qa")
def run_qa_tests():
    """Run quality assurance tests only."""
    
    console.print(Panel("Quality Assurance Tests", title="Testing Framework"))
    
    runner = TestRunner()
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Running QA tests...", total=None)
            
            results = runner.run_qa_tests()
            
            progress.update(task, description="QA tests completed")
            
            # Display results
            _display_test_results(results, "QA Tests")
            
    except Exception as e:
        console.print(f"[red]Error running QA tests: {e}[/red]")
        raise typer.Exit(1)
    finally:
        runner.cleanup()


@test_app.command("performance")
def run_performance_tests():
    """Run performance tests only."""
    
    console.print(Panel("Performance Tests", title="Testing Framework"))
    
    runner = TestRunner()
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Running performance tests...", total=None)
            
            results = runner.run_performance_tests()
            
            progress.update(task, description="Performance tests completed")
            
            # Display results
            _display_test_results(results, "Performance Tests")
            
    except Exception as e:
        console.print(f"[red]Error running performance tests: {e}[/red]")
        raise typer.Exit(1)
    finally:
        runner.cleanup()


@test_app.command("component")
def test_component(
    component: str = typer.Argument(..., help="Component to test (pdf, text, fingerprint, clustering, etc.)")
):
    """Test a specific component."""
    
    component_tests = {
        "pdf": "test_pdf_processor",
        "text": "test_text_cleaner",
        "segmentation": "test_text_segmenter", 
        "fingerprint": "test_fingerprinter",
        "rates": "test_rates_rules_extractor",
        "qa": "test_qa_calculator",
        "clustering": "test_clustering_engine",
        "family": "test_family_builder",
        "instance": "test_instance_manager",
        "database": "test_database_integration"
    }
    
    if component not in component_tests:
        console.print(f"[red]Unknown component: {component}[/red]")
        console.print("Available components:")
        for comp in component_tests.keys():
            console.print(f"  • {comp}")
        raise typer.Exit(1)
    
    console.print(Panel(f"Testing Component: {component.title()}", title="Component Test"))
    
    runner = TestRunner()
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task(f"Testing {component}...", total=None)
            
            # Run specific test method
            test_method = getattr(runner, f"_{component_tests[component]}")
            result = test_method()
            
            progress.update(task, description=f"{component} test completed")
            
            # Display result
            if result:
                console.print(f"\n[green]✓ {component.title()} test passed![/green]")
            else:
                console.print(f"\n[red]✗ {component.title()} test failed![/red]")
                raise typer.Exit(1)
                
    except Exception as e:
        console.print(f"[red]Error testing {component}: {e}[/red]")
        raise typer.Exit(1)
    finally:
        runner.cleanup()


@test_app.command("validate")
def validate_system():
    """Run basic system validation checks."""
    
    console.print(Panel("System Validation", title="Testing Framework"))
    
    checks = [
        ("Database Connection", _check_database),
        ("Core Dependencies", _check_dependencies),
        ("Configuration", _check_configuration),
        ("File Permissions", _check_file_permissions)
    ]
    
    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task("Running validation checks...", total=len(checks))
        
        for check_name, check_func in checks:
            progress.update(task, description=f"Checking {check_name.lower()}...")
            
            try:
                check_result = check_func()
                results.append((check_name, check_result, None))
            except Exception as e:
                results.append((check_name, False, str(e)))
            
            progress.advance(task)
    
    # Display validation results
    console.print("\n[bold]Validation Results:[/bold]")
    
    validation_table = Table()
    validation_table.add_column("Check", style="blue")
    validation_table.add_column("Status", justify="center")
    validation_table.add_column("Notes", style="yellow")
    
    all_passed = True
    
    for check_name, passed, error in results:
        status = "[green]✓ PASS[/green]" if passed else "[red]✗ FAIL[/red]"
        notes = error if error else "OK"
        
        validation_table.add_row(check_name, status, notes)
        
        if not passed:
            all_passed = False
    
    console.print(validation_table)
    
    if all_passed:
        console.print("\n[green]✓ All validation checks passed![/green]")
    else:
        console.print("\n[red]✗ Some validation checks failed![/red]")
        raise typer.Exit(1)


def _display_test_results(results: dict, test_type: str):
    """Display test results in a formatted table."""
    
    passed_count = sum(1 for result in results.values() if result)
    total_count = len(results)
    
    console.print(f"\n[bold]{test_type} Results:[/bold]")
    
    table = Table()
    table.add_column("Test", style="blue")
    table.add_column("Status", justify="center")
    
    for test_name, passed in results.items():
        status = "[green]✓ PASS[/green]" if passed else "[red]✗ FAIL[/red]"
        clean_name = test_name.replace("test_", "").replace("_", " ").title()
        table.add_row(clean_name, status)
    
    console.print(table)
    
    console.print(f"\nResults: {passed_count}/{total_count} tests passed")
    
    if passed_count < total_count:
        raise typer.Exit(1)


def _check_database() -> bool:
    """Check database connectivity."""
    try:
        from ..database import get_database
        db = get_database()
        return db.test_connection()
    except Exception:
        return False


def _check_dependencies() -> bool:
    """Check that core dependencies are available."""
    try:
        import pandas
        import numpy
        import scikit_learn
        import pdfplumber
        import tesseract
        return True
    except ImportError:
        return False


def _check_configuration() -> bool:
    """Check system configuration."""
    try:
        from ..core.config import get_settings
        settings = get_settings()
        return settings is not None
    except Exception:
        return False


def _check_file_permissions() -> bool:
    """Check file system permissions."""
    try:
        from ..core.config import get_settings
        settings = get_settings()
        
        # Check if we can write to upload directory
        upload_dir = Path(settings.paths.upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to create a test file
        test_file = upload_dir / "test_write.tmp"
        test_file.write_text("test")
        test_file.unlink()
        
        return True
    except Exception:
        return False


__all__ = ['test_app']