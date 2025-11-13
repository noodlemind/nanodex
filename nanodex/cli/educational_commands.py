"""
Educational commands for the interactive shell.

These commands provide learning support: explain concepts, compare configurations,
visualize metrics, and show context.
"""

from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich import box

from ..utils.educational import ConceptExplainer, ConfigPresets

console = Console()


@click.group()
def educational():
    """Educational commands (explain, compare, presets)"""
    pass


@educational.command("explain")
@click.argument("concept", required=False)
@click.option("--detailed", "-d", is_flag=True, help="Show detailed explanation")
def explain_cmd(concept, detailed):
    """
    Explain machine learning concepts.

    Available concepts: lora, quantization, rag, training, embeddings

    \b
    Examples:
        explain lora
        explain quantization --detailed
        explain
    """
    if not concept:
        # Show available concepts
        console.print("\n[bold cyan]Available Concepts:[/bold cyan]\n")

        for name, info in ConceptExplainer.CONCEPTS.items():
            console.print(f"  [cyan]{name:15}[/cyan] - {info['summary']}")

        console.print("\n[dim]Usage: explain <concept> [--detailed][/dim]\n")
        return

    ConceptExplainer.explain(concept, detailed=detailed)
    if not detailed:
        console.print("\n[dim]Tip: Use --detailed for full explanation[/dim]\n")


@educational.command("presets")
@click.argument("preset_name", required=False)
def presets_cmd(preset_name):
    """
    Show configuration presets.

    \b
    Available presets:
    - quick: Fast testing (1 epoch, small LoRA)
    - balanced: Good defaults (3 epochs, rank 16)
    - quality: Best results (5 epochs, rank 32)

    \b
    Examples:
        presets
        presets balanced
    """
    if not preset_name:
        # Show all presets
        ConfigPresets.list_presets()
        return

    # Show specific preset
    ConfigPresets.explain_preset(preset_name)


@educational.command("compare")
@click.argument("param1")
@click.argument("param2")
def compare_cmd(param1, param2):
    """
    Compare two parameters or configurations.

    \b
    Examples:
        compare 4bit 8bit
        compare lora full-finetune
        compare quick quality
    """
    comparisons = {
        ("4bit", "8bit"): _compare_quantization,
        ("8bit", "4bit"): _compare_quantization,
        ("lora", "full"): _compare_training_methods,
        ("lora", "full-finetune"): _compare_training_methods,
        ("full", "lora"): _compare_training_methods,
        ("quick", "balanced"): _compare_presets,
        ("quick", "quality"): _compare_presets,
        ("balanced", "quality"): _compare_presets,
    }

    key = (param1.lower(), param2.lower())
    if key in comparisons:
        comparisons[key](param1, param2)
    else:
        console.print(f"\n[yellow]Don't know how to compare '{param1}' and '{param2}'[/yellow]")
        console.print("\n[dim]Available comparisons:[/dim]")
        console.print("  • 4bit vs 8bit")
        console.print("  • lora vs full-finetune")
        console.print("  • quick vs balanced vs quality\n")


@educational.command("context")
@click.pass_context
def context_cmd(ctx):
    """Show current session context and state."""
    root_ctx = ctx.find_root()

    console.print("\n[bold cyan]Current Session Context[/bold cyan]\n")

    # Configuration
    if "config" in root_ctx.obj:
        console.print("[green]✓[/green] Configuration loaded")
        cfg = root_ctx.obj["config"]
        console.print(f"  Model: {cfg.get_model_source()}")
    else:
        console.print("[yellow]⚠[/yellow] No configuration loaded")

    # Last results
    if "last_results" in root_ctx.obj and root_ctx.obj["last_results"]:
        console.print(f"\n[green]✓[/green] Last results available")
        console.print(f"  {root_ctx.obj['last_results']}")

    # Training state
    if "training_state" in root_ctx.obj and root_ctx.obj["training_state"]:
        console.print(f"\n[green]✓[/green] Training state: {root_ctx.obj['training_state']}")

    # Command count
    if "command_count" in root_ctx.obj:
        console.print(f"\n📊 Commands run: {root_ctx.obj['command_count']}")

    console.print()


# Helper functions for comparisons


def _compare_quantization(param1: str, param2: str) -> None:
    """Compare 4-bit vs 8-bit quantization.

    Args:
        param1: First quantization method (4bit/8bit)
        param2: Second quantization method (4bit/8bit)
    """
    table = Table(title="Quantization Comparison", box=box.ROUNDED)
    table.add_column("Aspect", style="cyan")
    table.add_column("4-bit", style="green")
    table.add_column("8-bit", style="yellow")

    table.add_row("Memory Savings", "75%", "50%")
    table.add_row("Accuracy Loss", "~1-2%", "~0.5-1%")
    table.add_row("Speed", "Faster", "Medium")
    table.add_row("Model Size (7B)", "3.5GB", "7GB")
    table.add_row("Best For", "Limited GPU memory", "Better quality")

    console.print()
    console.print(table)
    console.print("\n[bold]Recommendation:[/bold]")
    console.print("  • Use 4-bit for most cases (best memory/quality trade-off)")
    console.print("  • Use 8-bit if you need maximum quality and have GPU memory\n")


def _compare_training_methods(param1: str, param2: str) -> None:
    """Compare LoRA vs full fine-tuning.

    Args:
        param1: First training method (lora/full)
        param2: Second training method (lora/full)
    """
    table = Table(title="Training Method Comparison", box=box.ROUNDED)
    table.add_column("Aspect", style="cyan")
    table.add_column("LoRA", style="green")
    table.add_column("Full Fine-Tuning", style="yellow")

    table.add_row("Parameters Trained", "0.1% (4M)", "100% (6.7B)")
    table.add_row("Saved Model Size", "~50MB", "~13GB")
    table.add_row("Training Time", "Fast (30-60 min)", "Slow (hours)")
    table.add_row("Memory Required", "Low (8GB GPU)", "High (80GB+ GPU)")
    table.add_row("Quality", "Excellent", "Slightly better")

    console.print()
    console.print(table)
    console.print("\n[bold]Recommendation:[/bold]")
    console.print("  • Use LoRA for almost all cases (efficient, great results)")
    console.print("  • Use full fine-tuning only if you have resources & need max quality\n")


def _compare_presets(param1: str, param2: str) -> None:
    """Compare configuration presets.

    Args:
        param1: First preset name (quick/balanced/quality)
        param2: Second preset name (quick/balanced/quality)
    """
    presets = [p.lower() for p in [param1, param2]]

    table = Table(title="Preset Comparison", box=box.ROUNDED)
    table.add_column("Aspect", style="cyan")

    for preset in presets:
        table.add_column(preset.title(), style="green" if preset == "quick" else "yellow")

    from ..utils.educational import ConfigPresets

    aspects = [
        ("Epochs", lambda p: ConfigPresets.PRESETS[p]["config"]["training"]["num_epochs"]),
        ("LoRA Rank", lambda p: ConfigPresets.PRESETS[p]["config"]["training"]["lora"]["r"]),
        ("Batch Size", lambda p: ConfigPresets.PRESETS[p]["config"]["training"]["batch_size"]),
        ("Time", lambda p: ConfigPresets.PRESETS[p]["time"]),
        ("Use Case", lambda p: ConfigPresets.PRESETS[p]["use_case"]),
    ]

    for aspect_name, getter in aspects:
        row = [aspect_name]
        for preset in presets:
            row.append(str(getter(preset)))
        table.add_row(*row)

    console.print()
    console.print(table)
    console.print()
