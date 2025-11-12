"""
Analyze command for codebase analysis.
"""

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
from rich.markdown import Markdown
from rich import box
import random

from ..utils import Config
from ..analyzers import CodeAnalyzer

console = Console()

# Educational tips for codebase analysis
ANALYZE_TIPS = [
    "💡 Deep parsing extracts functions and classes for better training data",
    "💡 Larger codebases create more diverse training examples",
    "💡 Code complexity metrics help identify patterns worth learning",
    "💡 Analysis happens once - results are cached for training",
]


@click.command()
@click.option('--config', default='config.yaml', help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def analyze_cmd(config, verbose):
    """
    📊 Analyze your codebase

    Scans your repository and provides detailed statistics about:
    - File counts and sizes
    - Language distribution
    - Code complexity
    - Dependencies
    """
    try:
        console.print("\n[bold cyan]Analyzing Codebase...[/bold cyan]\n")

        # Load configuration
        cfg = Config(config)
        repo_config = cfg.get_repository_config()

        # Analyze repository
        analyzer = CodeAnalyzer(repo_config)
        code_samples = analyzer.analyze()
        stats = analyzer.get_statistics(code_samples)

        # Display results
        console.print(Panel.fit(
            f"[bold green]✓ Analysis Complete[/bold green]\n\n"
            f"Repository: [cyan]{repo_config['path']}[/cyan]",
            border_style="green"
        ))

        # Show a random educational tip
        tip = random.choice(ANALYZE_TIPS)
        console.print()
        console.print(Markdown(tip))
        console.print()

        # File statistics table
        table = Table(title="\nFile Statistics", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")

        table.add_row("Total Files", str(stats['total_files']))
        table.add_row("Total Lines", f"{stats['total_lines']:,}")
        table.add_row("Average Lines/File", f"{stats['avg_lines_per_file']:.1f}")

        console.print(table)

        # Language distribution table
        if stats['languages']:
            table = Table(title="\nLanguage Distribution", box=box.ROUNDED)
            table.add_column("Language", style="cyan")
            table.add_column("Files", justify="right")
            table.add_column("Lines", justify="right")
            table.add_column("Percentage", justify="right", style="green")

            for lang, lang_stats in sorted(
                stats['languages'].items(),
                key=lambda x: x[1]['lines'],
                reverse=True
            ):
                percentage = (lang_stats['lines'] / stats['total_lines'] * 100) if stats['total_lines'] > 0 else 0
                table.add_row(
                    lang,
                    str(lang_stats['files']),
                    f"{lang_stats['lines']:,}",
                    f"{percentage:.1f}%"
                )

            console.print(table)

        # Deep parsing results (if enabled)
        if repo_config.get('deep_parsing', {}).get('enabled'):
            console.print("\n[bold cyan]Deep Parsing Results[/bold cyan]\n")

            total_functions = sum(
                len(sample.get('parsed', {}).get('functions', []))
                for sample in code_samples
            )
            total_classes = sum(
                len(sample.get('parsed', {}).get('classes', []))
                for sample in code_samples
            )

            console.print(f"  Functions extracted: [green]{total_functions}[/green]")
            console.print(f"  Classes extracted: [green]{total_classes}[/green]")

        # Recommendations
        console.print("\n[bold cyan]Recommendations[/bold cyan]\n")

        if stats['total_files'] < 10:
            console.print("  [yellow]⚠[/yellow] Small codebase detected (<10 files)")
            console.print("    Consider adding more code or using synthetic data generation\n")
        elif stats['total_files'] > 1000:
            console.print("  [yellow]⚠[/yellow] Large codebase detected (>1000 files)")
            console.print("    Training may take longer. Consider filtering specific directories\n")
        else:
            console.print("  [green]✓[/green] Codebase size looks good for training\n")

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        if verbose:
            import traceback
            console.print(f"\n[dim]{traceback.format_exc()}[/dim]")
        raise click.Abort()
