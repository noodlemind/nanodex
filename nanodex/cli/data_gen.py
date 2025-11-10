"""
Data generation commands.
"""

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box
import json
from pathlib import Path

from ..utils import Config
from ..analyzers import CodeAnalyzer
from ..data_generators.orchestrator import DataGenerationOrchestrator

console = Console()


@click.group()
def data_cmd():
    """
    📦 Data generation commands

    Generate training data from your codebase using various strategies:
    - Self-supervised (free)
    - Synthetic via APIs (paid)
    """
    pass


@data_cmd.command('generate')
@click.option('--config', default='config.yaml', help='Configuration file path')
@click.option('--output', default='./data/training_examples.json', help='Output file path')
@click.option('--limit', type=int, help='Limit number of files to process')
def generate(config, output, limit):
    """
    Generate training data from codebase.

    This command:
    1. Analyzes your codebase
    2. Generates training examples using configured mode (free/hybrid/full)
    3. Filters and deduplicates examples
    4. Saves results to JSON file
    """
    try:
        console.print("\n[bold cyan]Generating Training Data...[/bold cyan]\n")

        # Load configuration
        cfg = Config(config)
        repo_config = cfg.get_repository_config()

        # Analyze codebase
        console.print("Step 1: Analyzing codebase...")
        analyzer = CodeAnalyzer(repo_config)
        code_samples = analyzer.analyze()

        if limit:
            code_samples = code_samples[:limit]
            console.print(f"[dim]Limited to {limit} files[/dim]")

        stats = analyzer.get_statistics(code_samples)
        console.print(f"  Found {stats['total_files']} files with {stats['total_lines']:,} lines\n")

        # Initialize orchestrator
        console.print("Step 2: Initializing data generators...")
        orchestrator = DataGenerationOrchestrator(cfg.config.model_dump())

        # Estimate cost
        cost_estimate = orchestrator.estimate_total_cost(len(code_samples))
        console.print(f"  Mode: [cyan]{orchestrator.mode}[/cyan]")
        console.print(f"  Estimated cost: [yellow]${cost_estimate['total']:.2f}[/yellow]\n")

        if cost_estimate['total'] > 0:
            if not click.confirm('Proceed with data generation?'):
                console.print("[yellow]Cancelled[/yellow]")
                return

        # Generate training data
        console.print("Step 3: Generating training examples...")
        examples = orchestrator.generate_from_codebase(code_samples, show_progress=True)

        # Save results
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(examples, f, indent=2)

        # Display statistics
        gen_stats = orchestrator.get_stats()

        console.print("\n")
        console.print(Panel.fit(
            f"[bold green]✓ Data Generation Complete[/bold green]\n\n"
            f"Examples generated: [cyan]{len(examples)}[/cyan]\n"
            f"Examples filtered: [yellow]{gen_stats['total_filtered']}[/yellow]\n"
            f"Duplicates removed: [yellow]{gen_stats['total_duplicates']}[/yellow]\n"
            f"Total cost: [green]${gen_stats['total_cost_usd']:.2f}[/green]\n\n"
            f"Saved to: [cyan]{output}[/cyan]",
            title="Generation Complete",
            border_style="green"
        ))

        # Next steps
        console.print("\n[bold]Next steps:[/bold]")
        console.print("  nanodex data stats      # View detailed statistics")
        console.print("  nanodex train           # Start training\n")

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise click.Abort()


@data_cmd.command('stats')
@click.argument('data_file', type=click.Path(exists=True))
def stats(data_file):
    """
    Show statistics about generated training data.

    \b
    Example:
        nanodex data stats ./data/training_examples.json
    """
    try:
        console.print(f"\n[bold cyan]Analyzing {data_file}...[/bold cyan]\n")

        # Load data
        with open(data_file, 'r') as f:
            examples = json.load(f)

        # Calculate statistics
        total = len(examples)

        # Group by type
        types = {}
        for ex in examples:
            ex_type = ex.get('metadata', {}).get('type', 'unknown')
            types[ex_type] = types.get(ex_type, 0) + 1

        # Length statistics
        input_lengths = [len(ex.get('input', '')) for ex in examples]
        output_lengths = [len(ex.get('output', '')) for ex in examples]

        avg_input = sum(input_lengths) / len(input_lengths) if input_lengths else 0
        avg_output = sum(output_lengths) / len(output_lengths) if output_lengths else 0

        # Display results
        table = Table(title="Dataset Statistics", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")

        table.add_row("Total Examples", str(total))
        table.add_row("Average Input Length", f"{avg_input:.0f} chars")
        table.add_row("Average Output Length", f"{avg_output:.0f} chars")

        console.print(table)

        # Example types
        if types:
            table = Table(title="\nExample Types", box=box.ROUNDED)
            table.add_column("Type", style="cyan")
            table.add_column("Count", justify="right")
            table.add_column("Percentage", justify="right", style="green")

            for ex_type, count in sorted(types.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total * 100) if total > 0 else 0
                table.add_row(ex_type, str(count), f"{percentage:.1f}%")

            console.print(table)

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise click.Abort()


@data_cmd.command('preview')
@click.argument('data_file', type=click.Path(exists=True))
@click.option('--limit', default=3, help='Number of examples to preview')
def preview(data_file, limit):
    """
    Preview training examples.

    \b
    Example:
        nanodex data preview ./data/training_examples.json --limit 5
    """
    try:
        # Load data
        with open(data_file, 'r') as f:
            examples = json.load(f)

        console.print(f"\n[bold cyan]Previewing {min(limit, len(examples))} examples from {data_file}[/bold cyan]\n")

        for i, ex in enumerate(examples[:limit]):
            console.print(Panel(
                f"[bold]Instruction:[/bold]\n{ex.get('instruction', 'N/A')}\n\n"
                f"[bold]Input:[/bold]\n{ex.get('input', 'N/A')[:200]}{'...' if len(ex.get('input', '')) > 200 else ''}\n\n"
                f"[bold]Output:[/bold]\n{ex.get('output', 'N/A')[:200]}{'...' if len(ex.get('output', '')) > 200 else ''}\n\n"
                f"[dim]Type: {ex.get('metadata', {}).get('type', 'unknown')}[/dim]",
                title=f"Example {i+1}/{min(limit, len(examples))}",
                border_style="cyan"
            ))

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise click.Abort()
