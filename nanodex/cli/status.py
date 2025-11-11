"""
Status command to display pipeline progress.

Shows the current state of all pipeline steps and provides
helpful information about what to do next.
"""

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from pathlib import Path
import os

from ..utils.pipeline_state import PipelineState, StepStatus

console = Console()


@click.command("status")
@click.option("--config", default="config.yaml", help="Configuration file path")
def status_cmd(config):
    """
    Show pipeline status and progress

    Displays the current state of all pipeline steps including
    configuration, analysis, data generation, training, and RAG indexing.

    \b
    Examples:
        nanodex status
        nanodex status --config my_config.yaml
    """
    try:
        console.print()
        console.print(
            Panel.fit("[bold cyan]📊 nanodex Pipeline Status[/bold cyan]", border_style="cyan")
        )
        console.print()

        # Initialize pipeline state
        pipeline = PipelineState()

        # Check each step (using the specified config path)
        config_path = Path(config)
        if config_path.exists():
            pipeline.set_step_status("configuration", StepStatus.COMPLETED, {"config_file": config})
        pipeline.check_analysis()
        pipeline.check_data_generation()
        pipeline.check_training()
        pipeline.check_rag_indexing()

        # Show project info
        console.print(f"[bold]Project:[/bold] {os.getcwd()}")
        console.print(f"[bold]Config:[/bold] {config} " f"{'✓' if config_path.exists() else '✗'}\n")

        # Create progress table
        table = Table(title="Pipeline Progress", box=box.ROUNDED)
        table.add_column("Step", style="cyan", no_wrap=True)
        table.add_column("Status", style="white", no_wrap=True)
        table.add_column("Details", style="dim")

        # Define steps with their status
        steps = [
            ("1. Configuration", "configuration", "Setup config.yaml"),
            ("2. Analysis", "analysis", "Analyze codebase structure"),
            ("3. Data Generation", "data_generation", "Generate training data"),
            ("4. Training", "training", "Fine-tune model"),
            ("5. RAG Indexing", "rag_indexing", "Build semantic search index"),
        ]

        for step_name, step_key, description in steps:
            status = pipeline.get_step_status(step_key)

            if status == StepStatus.COMPLETED:
                status_icon = "[green]✅[/green]"
                status_text = "[green]Completed[/green]"
            elif status == StepStatus.IN_PROGRESS:
                status_icon = "[yellow]🔄[/yellow]"
                status_text = "[yellow]In Progress[/yellow]"
            elif status == StepStatus.FAILED:
                status_icon = "[red]❌[/red]"
                status_text = "[red]Failed[/red]"
            else:
                status_icon = "[dim]⏸️[/dim]"
                status_text = "[dim]Pending[/dim]"

            table.add_row(f"{status_icon}  {step_name}", status_text, description)

        console.print(table)
        console.print()

        # Show progress percentage
        progress = pipeline.get_progress_percentage()
        console.print(f"[bold]Overall Progress:[/bold] {progress:.0f}%")
        console.print()

        # Show next step
        next_command = pipeline.get_next_step()
        if next_command:
            console.print("[bold yellow]💡 Next Step:[/bold yellow]")
            console.print(f"   Run: [cyan]{next_command}[/cyan]")
        else:
            console.print("[bold green]✨ Pipeline Complete![/bold green]")
            console.print("   All steps have been completed successfully.")

        console.print()

        # Show resource info if available
        try:
            import psutil

            _show_resource_info()
        except ImportError:
            pass  # psutil not available, skip resource info

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        import traceback

        console.print(f"\n[dim]{traceback.format_exc()}[/dim]")
        raise click.Abort()


def _show_resource_info():
    """Show system resource information."""
    try:
        import psutil

        console.print("[bold]System Resources:[/bold]")

        # RAM
        ram = psutil.virtual_memory()
        ram_used_gb = ram.used / (1024**3)
        ram_total_gb = ram.total / (1024**3)
        console.print(
            f"  RAM: {ram_used_gb:.1f}GB / {ram_total_gb:.1f}GB " f"({ram.percent:.0f}% used)"
        )

        # Disk
        disk = psutil.disk_usage(".")
        disk_free_gb = disk.free / (1024**3)
        console.print(f"  Disk: {disk_free_gb:.1f}GB available")

        # GPU (if nvidia-smi available)
        try:
            import subprocess

            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                gpu_info = result.stdout.strip().split(",")
                if len(gpu_info) == 3:
                    gpu_name = gpu_info[0].strip()
                    gpu_used = float(gpu_info[1])
                    gpu_total = float(gpu_info[2])
                    console.print(
                        f"  GPU: {gpu_name} - " f"{gpu_used:.0f}MB / {gpu_total:.0f}MB used"
                    )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass  # nvidia-smi not available

        console.print()

    except Exception:
        pass  # If resource info fails, silently continue
