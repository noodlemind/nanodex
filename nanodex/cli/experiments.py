"""
Experiment tracking and management commands.

Track, compare, and analyze training experiments to find the best configurations.
"""

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from pathlib import Path
import json

from ..utils.experiment_tracker import ExperimentTracker
from ..utils.visualizations import MetricsVisualizer

console = Console()


@click.group()
def experiments_cmd():
    """
    🧪 Experiment tracking and management

    Track training runs, compare results, and analyze what works best.
    """
    pass


@experiments_cmd.command('list')
@click.option('--status', type=click.Choice(['running', 'completed', 'failed']), help='Filter by status')
@click.option('--limit', type=int, default=10, help='Maximum number to show')
def list_experiments(status, limit):
    """
    List all experiments.

    Shows tracked training runs with their configurations and results.
    """
    tracker = ExperimentTracker()
    experiments = tracker.list_experiments(status=status, limit=limit)

    if not experiments:
        console.print("\n[yellow]No experiments found[/yellow]\n")
        return

    console.print(f"\n[bold cyan]Experiments ({len(experiments)} found)[/bold cyan]\n")

    table = Table(box=box.ROUNDED)
    table.add_column("ID", style="cyan", width=30)
    table.add_column("Name", style="green")
    table.add_column("Status", justify="center")
    table.add_column("Final Loss", justify="right")
    table.add_column("Time", justify="right")

    for exp in experiments:
        exp_id = exp['id']
        name = exp['name']
        exp_status = exp['status']
        results = exp.get('results', {})

        # Format status with color
        if exp_status == 'completed':
            status_str = "[green]✓ completed[/green]"
        elif exp_status == 'running':
            status_str = "[yellow]⚠ running[/yellow]"
        else:
            status_str = "[red]✗ failed[/red]"

        final_loss = results.get('final_loss', 'N/A')
        if isinstance(final_loss, (int, float)):
            final_loss = f"{final_loss:.4f}"

        training_time = results.get('training_time', 'N/A')
        if isinstance(training_time, (int, float)):
            training_time = f"{training_time:.1f}s"

        table.add_row(exp_id, name, status_str, str(final_loss), str(training_time))

    console.print(table)
    console.print(f"\n[dim]Use: nanodex experiments show <id> for details[/dim]\n")


@experiments_cmd.command('show')
@click.argument('exp_id')
def show_experiment(exp_id):
    """
    Show detailed information about an experiment.

    Displays configuration, metrics, and results for a specific experiment.
    """
    tracker = ExperimentTracker()
    exp = tracker.get_experiment(exp_id)

    if not exp:
        console.print(f"\n[red]Experiment not found: {exp_id}[/red]\n")
        return

    console.print(f"\n[bold cyan]Experiment: {exp['name']}[/bold cyan]\n")

    # Configuration
    config_table = Table(title="Configuration", box=box.ROUNDED)
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="green")

    for key, value in exp['config'].items():
        config_table.add_row(str(key), str(value))

    console.print(config_table)

    # Results
    if exp['status'] == 'completed':
        console.print()
        results_table = Table(title="Results", box=box.ROUNDED)
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="green")

        results = exp['results']
        for key, value in results.items():
            if isinstance(value, float):
                value = f"{value:.4f}"
            results_table.add_row(str(key), str(value))

        console.print(results_table)

        # Loss history visualization
        metrics = exp['metrics']
        loss_history = [m['value'] for m in metrics.get('loss_history', [])]

        if loss_history:
            console.print()
            losses = {"Training Loss": loss_history}

            eval_loss_history = [m['value'] for m in metrics.get('eval_loss_history', [])]
            if eval_loss_history:
                losses["Evaluation Loss"] = eval_loss_history

            MetricsVisualizer.show_loss_comparison(losses, title="Training Progress")

    # Notes
    if exp.get('notes'):
        console.print()
        console.print(Panel(exp['notes'], title="Notes", border_style="dim"))

    console.print()


@experiments_cmd.command('compare')
@click.argument('exp_ids', nargs=-1, required=True)
def compare_experiments(exp_ids):
    """
    Compare multiple experiments.

    Shows side-by-side comparison of configurations and results.
    """
    tracker = ExperimentTracker()
    summaries = tracker.compare_experiments(list(exp_ids))

    if not summaries:
        console.print("\n[yellow]No experiments to compare[/yellow]\n")
        return

    console.print(f"\n[bold cyan]Comparing {len(summaries)} Experiments[/bold cyan]\n")

    table = Table(box=box.ROUNDED)
    table.add_column("Experiment", style="cyan")
    table.add_column("Model", style="yellow")
    table.add_column("LoRA Rank", justify="right")
    table.add_column("Learning Rate", justify="right")
    table.add_column("Final Loss", justify="right", style="green")
    table.add_column("Best Loss", justify="right", style="green")
    table.add_column("Time", justify="right")

    for summary in summaries:
        name = summary['name']
        model = str(summary['model'])
        lora_rank = str(summary['lora_rank'])

        learning_rate = summary['learning_rate']
        if isinstance(learning_rate, float):
            learning_rate = f"{learning_rate:.2e}"

        final_loss = summary['final_loss']
        if isinstance(final_loss, float):
            final_loss = f"{final_loss:.4f}"

        best_loss = summary['best_loss']
        if isinstance(best_loss, float):
            best_loss = f"{best_loss:.4f}"

        training_time = summary['training_time']
        if isinstance(training_time, (int, float)):
            training_time = f"{training_time:.1f}s"

        table.add_row(
            name,
            model,
            lora_rank,
            str(learning_rate),
            str(final_loss),
            str(best_loss),
            str(training_time)
        )

    console.print(table)
    console.print()


@experiments_cmd.command('best')
@click.option('--metric', default='final_loss', help='Metric to compare')
def best_experiment(metric):
    """
    Find best experiment by metric.

    Shows the experiment with the best performance for a given metric.
    """
    tracker = ExperimentTracker()
    best = tracker.get_best_experiment(metric=metric)

    if not best:
        console.print("\n[yellow]No completed experiments found[/yellow]\n")
        return

    console.print(f"\n[bold green]Best Experiment (by {metric})[/bold green]\n")

    console.print(Panel.fit(
        f"[bold]Name:[/bold] {best['name']}\n"
        f"[bold]ID:[/bold] {best['id']}\n"
        f"[bold]Final Loss:[/bold] {best['results']['final_loss']:.4f}\n"
        f"[bold]Best Loss:[/bold] {best['results']['best_loss']:.4f}",
        border_style="green"
    ))

    console.print(f"\n[dim]Use: nanodex experiments show {best['id']} for details[/dim]\n")


@experiments_cmd.command('delete')
@click.argument('exp_id')
@click.option('--yes', '-y', is_flag=True, help='Skip confirmation')
def delete_experiment(exp_id, yes):
    """
    Delete an experiment.

    Removes experiment from tracking database.
    """
    tracker = ExperimentTracker()

    # Confirm deletion
    if not yes:
        if not click.confirm(f'Delete experiment {exp_id}?'):
            console.print("[yellow]Cancelled[/yellow]")
            return

    tracker.delete_experiment(exp_id)


@experiments_cmd.command('export')
@click.option('--output', default='./experiments_export.json', help='Output file path')
@click.option('--ids', multiple=True, help='Specific experiment IDs to export')
def export_experiments(output, ids):
    """
    Export experiments to JSON file.

    Saves experiment data for backup or sharing.
    """
    tracker = ExperimentTracker()
    exp_ids = list(ids) if ids else None

    tracker.export_experiments(output, exp_ids=exp_ids)


@experiments_cmd.command('visualize')
@click.argument('exp_id')
def visualize_experiment(exp_id):
    """
    Visualize experiment training progress.

    Shows detailed training curves and metrics.
    """
    tracker = ExperimentTracker()
    exp = tracker.get_experiment(exp_id)

    if not exp:
        console.print(f"\n[red]Experiment not found: {exp_id}[/red]\n")
        return

    console.print(f"\n[bold cyan]Training Progress: {exp['name']}[/bold cyan]\n")

    metrics = exp['metrics']
    loss_history = [m['value'] for m in metrics.get('loss_history', [])]
    eval_loss_history = [m['value'] for m in metrics.get('eval_loss_history', [])]

    if not loss_history:
        console.print("[yellow]No training data available[/yellow]\n")
        return

    # Show loss curves
    losses = {"Training Loss": loss_history}
    if eval_loss_history:
        losses["Evaluation Loss"] = eval_loss_history

    MetricsVisualizer.show_loss_comparison(losses, title="Loss Curves")

    # Show learning rate schedule
    lr_history = [m['value'] for m in metrics.get('learning_rate_history', [])]
    if lr_history:
        console.print()
        table = Table(title="Learning Rate Schedule", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Trend", style="yellow", width=30)

        table.add_row(
            "Learning Rate",
            f"{lr_history[-1]:.2e}",
            MetricsVisualizer.sparkline(lr_history, width=30)
        )

        console.print(table)

    console.print()
