"""
Interactive learning tutorials.

Provides hands-on, interactive tutorials to understand fine-tuning, LoRA,
quantization, RAG, and other machine learning concepts.
"""

import click
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich import box

from ..utils.educational import ConceptExplainer

console = Console()


@click.group()
def learn_cmd():
    """
    🎓 Interactive learning tutorials

    Learn machine learning concepts hands-on with guided tutorials.

    \b
    Available tutorials:
    1. Fine-Tuning Basics
    2. Understanding LoRA
    3. Quantization Deep Dive
    4. RAG From Scratch

    \b
    Example:
        nanodex learn
        nanodex learn finetuning
        nanodex learn lora
    """
    pass


@learn_cmd.command("list")
def list_tutorials():
    """List all available tutorials."""
    console.print("\n[bold cyan]Available Tutorials:[/bold cyan]\n")

    tutorials = [
        (
            "finetuning",
            "Fine-Tuning Basics",
            "Understand how fine-tuning works",
            "Beginner",
            "~10 min",
        ),
        ("lora", "Understanding LoRA", "Learn Low-Rank Adaptation", "Beginner", "~15 min"),
        (
            "quantization",
            "Quantization Deep Dive",
            "Master 4-bit and 8-bit quantization",
            "Intermediate",
            "~12 min",
        ),
        (
            "rag",
            "RAG From Scratch",
            "Build retrieval-augmented generation",
            "Intermediate",
            "~20 min",
        ),
    ]

    table = Table(box=box.ROUNDED)
    table.add_column("Command", style="cyan")
    table.add_column("Tutorial", style="green")
    table.add_column("Description")
    table.add_column("Level")
    table.add_column("Time")

    for cmd, title, desc, level, duration in tutorials:
        table.add_row(cmd, title, desc, level, duration)

    console.print(table)
    console.print("\n[dim]Run: nanodex learn <command>[/dim]\n")


@learn_cmd.command("finetuning")
def tutorial_finetuning():
    """
    Interactive tutorial: Fine-Tuning Basics

    Learn what fine-tuning does and how it works.
    """
    console.print(
        Panel.fit(
            "[bold cyan]Tutorial: Fine-Tuning Basics[/bold cyan]\n\n"
            "You'll learn:\n"
            "• What fine-tuning does\n"
            "• How models learn from your code\n"
            "• The role of training data\n"
            "• Loss, epochs, and hyperparameters\n\n"
            "Duration: ~10 minutes",
            border_style="cyan",
        )
    )

    if not Confirm.ask("\n[bold]Ready to start?[/bold]"):
        console.print("[dim]Tutorial cancelled[/dim]\n")
        return

    # Step 1: The Problem
    console.print("\n" + "=" * 70)
    console.print("[bold]Step 1: The Problem[/bold]")
    console.print("=" * 70 + "\n")

    console.print("Imagine you have a base model that knows general coding:")

    code = """
# Base model (before fine-tuning)
prompt = "Write a function to sort a list"
response = model.generate(prompt)
# → Gives generic sorting code
"""
    console.print(Syntax(code, "python", theme="monokai"))

    console.print("\n[dim]But what if your codebase has specific patterns?[/dim]")
    console.print("[dim]• Custom frameworks[/dim]")
    console.print("[dim]• Domain-specific logic[/dim]")
    console.print("[dim]• Company conventions[/dim]\n")

    if not Confirm.ask("Understand the problem?"):
        console.print("\n[yellow]The base model doesn't know YOUR code patterns yet![/yellow]\n")

    # Step 2: The Solution
    console.print("\n" + "=" * 70)
    console.print("[bold]Step 2: The Solution - Fine-Tuning[/bold]")
    console.print("=" * 70 + "\n")

    console.print("Fine-tuning teaches the model YOUR code patterns:\n")

    steps = [
        ("1. Collect Examples", "Extract patterns from your codebase"),
        ("2. Format Data", "Create instruction-response pairs"),
        ("3. Train", "Model learns your patterns"),
        ("4. Deploy", "Use specialized model"),
    ]

    for step, desc in steps:
        console.print(f"[cyan]{step:20}[/cyan] {desc}")

    console.print()

    # Step 3: Training Data
    console.print("\n" + "=" * 70)
    console.print("[bold]Step 3: Training Data Format[/bold]")
    console.print("=" * 70 + "\n")

    example = """{
  "instruction": "Explain this authentication code",
  "input": "def login(user, password): ...",
  "output": "This function handles user login by..."
}"""
    console.print(Syntax(example, "json", theme="monokai"))

    console.print("\n[dim]Each example teaches one pattern![/dim]\n")

    # Step 4: Key Concepts
    console.print("\n" + "=" * 70)
    console.print("[bold]Step 4: Key Concepts[/bold]")
    console.print("=" * 70 + "\n")

    concepts = [
        ("Loss", "Measures prediction error (lower = better)"),
        ("Epoch", "One pass through all training data"),
        ("Learning Rate", "How big are weight updates?"),
        ("Batch Size", "How many examples at once?"),
    ]

    table = Table(box=box.SIMPLE)
    table.add_column("Concept", style="cyan")
    table.add_column("Explanation")

    for concept, explanation in concepts:
        table.add_row(concept, explanation)

    console.print(table)
    console.print()

    # Summary
    console.print("\n" + "=" * 70)
    console.print("[bold green]✓ Tutorial Complete![/bold green]")
    console.print("=" * 70 + "\n")

    console.print("[bold]What you learned:[/bold]")
    console.print("✓ Fine-tuning adapts models to YOUR code")
    console.print("✓ Training data = instruction + input + output")
    console.print("✓ Loss measures how well the model learns")
    console.print("✓ Lower loss = better predictions\n")

    console.print("[bold]Next steps:[/bold]")
    console.print("• Try: nanodex learn lora (Learn efficient fine-tuning)")
    console.print("• Try: nanodex analyze (Analyze your codebase)")
    console.print("• Try: nanodex data generate (Create training data)\n")


@learn_cmd.command("lora")
def tutorial_lora():
    """
    Interactive tutorial: Understanding LoRA

    Learn Low-Rank Adaptation and why it's efficient.
    """
    console.print(
        Panel.fit(
            "[bold cyan]Tutorial: Understanding LoRA[/bold cyan]\n\n"
            "You'll learn:\n"
            "• Why training all parameters is wasteful\n"
            "• How LoRA uses low-rank decomposition\n"
            "• The math (optional but enlightening)\n"
            "• Practical benefits\n\n"
            "Duration: ~15 minutes",
            border_style="cyan",
        )
    )

    if not Confirm.ask("\n[bold]Ready to start?[/bold]"):
        console.print("[dim]Tutorial cancelled[/dim]\n")
        return

    # Show detailed LoRA explanation
    ConceptExplainer.explain("lora", detailed=True)

    console.print("\n[bold]The Chef Analogy:[/bold]\n")
    console.print("Think of the base model as a professional chef:")
    console.print("• ❌ Full fine-tuning: Teach them cooking from scratch")
    console.print("• ✅ LoRA: Give them a specialty cookbook\n")

    console.print("The chef keeps their knowledge, just adds new recipes!\n")

    # Show the math
    show_math = Confirm.ask("[bold]Want to see the math?[/bold] (optional)")

    if show_math:
        console.print("\n[bold]The Mathematics:[/bold]\n")

        math = """
Traditional: Train weight matrix W
  W ∈ R^(d×d)  # 6.7 billion parameters

LoRA: Add decomposition W' = W + BA
  B ∈ R^(d×r)  # r << d (e.g., r=16)
  A ∈ R^(r×d)

Parameters to train:
  Traditional: d² = 6,700,000,000
  LoRA: 2dr = 2 × 4096 × 16 ≈ 4,000,000

Result: 0.06% of parameters! 🎉
        """
        console.print(Syntax(math, "python", theme="monokai"))
        console.print()

    # Practical example
    console.print("[bold]Practical Impact:[/bold]\n")

    comparison = [
        ("", "Full Fine-Tuning", "LoRA"),
        ("Parameters Trained", "6.7B (100%)", "4M (0.06%)"),
        ("Saved Model Size", "~13GB", "~50MB"),
        ("Training Time", "Hours", "30-60 min"),
        ("GPU Memory", "80GB+", "8-16GB"),
    ]

    table = Table(box=box.ROUNDED)
    for i, row in enumerate(comparison):
        if i == 0:
            table.add_column(row[0], style="cyan")
            table.add_column(row[1], style="yellow")
            table.add_column(row[2], style="green")
        else:
            table.add_row(*row)

    console.print(table)
    console.print()

    # Summary
    console.print("[bold green]✓ Tutorial Complete![/bold green]\n")
    console.print("[bold]Key Takeaways:[/bold]")
    console.print("✓ LoRA trains only 0.06% of parameters")
    console.print("✓ Saved models are tiny (~50MB vs 13GB)")
    console.print("✓ Works by low-rank decomposition: W + BA")
    console.print("✓ Perfect for consumer GPUs\n")

    console.print("[bold]Next steps:[/bold]")
    console.print("• Try: nanodex learn quantization")
    console.print("• Try: nanodex train --preset quick\n")


@learn_cmd.command("quantization")
def tutorial_quantization():
    """
    Interactive tutorial: Quantization Deep Dive

    Master 4-bit and 8-bit quantization.
    """
    console.print(
        Panel.fit(
            "[bold cyan]Tutorial: Quantization Deep Dive[/bold cyan]\n\n"
            "You'll learn:\n"
            "• What quantization does\n"
            "• 4-bit vs 8-bit vs 16-bit\n"
            "• Trade-offs (memory vs quality)\n"
            "• When to use what\n\n"
            "Duration: ~12 minutes",
            border_style="cyan",
        )
    )

    if not Confirm.ask("\n[bold]Ready to start?[/bold]"):
        console.print("[dim]Tutorial cancelled[/dim]\n")
        return

    # Show detailed quantization explanation
    ConceptExplainer.explain("quantization", detailed=True)

    console.print("\n[bold]Visual Comparison:[/bold]\n")

    # Show precision comparison
    comparison = [
        ("Precision", "Example Value", "Bits", "Memory (7B model)"),
        ("FP32", "1.234567890", "32", "28GB"),
        ("FP16", "1.234567", "16", "14GB"),
        ("INT8", "1.234", "8", "7GB"),
        ("NF4", "1.2", "4", "3.5GB"),
    ]

    table = Table(title="Precision Levels", box=box.ROUNDED)
    for i, row in enumerate(comparison):
        if i == 0:
            for col in row:
                table.add_column(col, style="cyan" if i == 0 else None)
        else:
            table.add_row(*row)

    console.print(table)
    console.print()

    # Trade-offs
    console.print("[bold]The Trade-Off:[/bold]\n")
    console.print("• More precision = More memory, Better quality")
    console.print("• Less precision = Less memory, Slight quality loss")
    console.print("\n[dim]Sweet spot: 4-bit (75% savings, <1% loss)[/dim]\n")

    # Comparison command
    console.print("[bold]Compare quantization methods:[/bold]")
    console.print("[dim]Run: nanodex edu compare 4bit 8bit[/dim]\n")

    # Summary
    console.print("[bold green]✓ Tutorial Complete![/bold green]\n")
    console.print("[bold]Key Takeaways:[/bold]")
    console.print("✓ Quantization reduces precision to save memory")
    console.print("✓ 4-bit: 75% savings, ~1% quality loss")
    console.print("✓ 8-bit: 50% savings, ~0.5% quality loss")
    console.print("✓ Essential for running large models\n")


@learn_cmd.command("rag")
def tutorial_rag():
    """
    Interactive tutorial: RAG From Scratch

    Build retrieval-augmented generation from the ground up.
    """
    console.print(
        Panel.fit(
            "[bold cyan]Tutorial: RAG From Scratch[/bold cyan]\n\n"
            "You'll learn:\n"
            "• What RAG is and why it's useful\n"
            "• How embeddings capture meaning\n"
            "• How FAISS enables fast search\n"
            "• Building a RAG system\n\n"
            "Duration: ~20 minutes",
            border_style="cyan",
        )
    )

    if not Confirm.ask("\n[bold]Ready to start?[/bold]"):
        console.print("[dim]Tutorial cancelled[/dim]\n")
        return

    # Show detailed RAG explanation
    ConceptExplainer.explain("rag", detailed=True)
    console.print()
    ConceptExplainer.explain("embeddings", detailed=True)

    # Summary
    console.print("\n[bold green]✓ Tutorial Complete![/bold green]\n")
    console.print("[bold]Key Takeaways:[/bold]")
    console.print("✓ RAG combines search + generation")
    console.print("✓ Embeddings convert code to vectors")
    console.print("✓ FAISS enables fast similarity search")
    console.print("✓ No retraining needed - just update index\n")

    console.print("[bold]Try it:[/bold]")
    console.print("• nanodex rag index")
    console.print("• nanodex rag search 'authentication'\n")


# Main entry point when run without subcommand
@learn_cmd.command("start", hidden=True)
@click.pass_context
def start(ctx):
    """Start interactive tutorial selection."""
    if ctx.parent.invoked_subcommand is None:
        list_tutorials()


# Make list the default command
learn_cmd.add_command(list_tutorials, name="__default__")
