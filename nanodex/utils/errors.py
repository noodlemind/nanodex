"""
Enhanced error formatting and messaging utilities.

Provides helpful error messages with actionable suggestions
to improve user experience when things go wrong.
"""

from rich.console import Console
from rich.panel import Panel

console = Console()


def format_error_with_suggestions(error_message: str, suggestions: list, examples: list = None):
    """
    Format an error message with helpful suggestions and examples.

    Args:
        error_message: The main error message to display
        suggestions: List of actionable suggestions
        examples: Optional list of example commands or fixes

    Returns:
        Formatted error string with suggestions
    """
    output = f"[bold red]Error:[/bold red] {error_message}\n\n"

    if suggestions:
        output += "[bold yellow]💡 Suggestions:[/bold yellow]\n"
        for i, suggestion in enumerate(suggestions, 1):
            output += f"   {i}. {suggestion}\n"
        output += "\n"

    if examples:
        output += "[bold cyan]Examples:[/bold cyan]\n"
        for example in examples:
            output += f"   {example}\n"

    return output


def show_config_not_found_error(config_path: str = "config.yaml"):
    """Show helpful error when config file is not found."""
    console.print()
    console.print(
        Panel(
            format_error_with_suggestions(
                f"Config file '{config_path}' not found",
                [
                    "Run 'nanodex init' to create a new config interactively",
                    f"Specify a different config with '--config path/to/config.yaml'",
                    "Check that you're in the correct directory",
                ],
                ["nanodex init  # Create config", "nanodex analyze  # Then analyze"],
            ),
            title="[bold red]Configuration Error[/bold red]",
            border_style="red",
        )
    )
    console.print()


def show_model_not_found_error(model_path: str):
    """Show helpful error when model is not found."""
    console.print()
    console.print(
        Panel(
            format_error_with_suggestions(
                f"Model not found at '{model_path}'",
                [
                    "Train a model first with 'nanodex train'",
                    "Verify the model path is correct",
                    "Check if training completed successfully",
                ],
                [
                    "nanodex train  # Train a new model",
                    "nanodex chat  # Use without model (RAG only)",
                ],
            ),
            title="[bold red]Model Error[/bold red]",
            border_style="red",
        )
    )
    console.print()


def show_rag_index_not_found_error(index_path: str):
    """Show helpful error when RAG index is not found."""
    console.print()
    console.print(
        Panel(
            format_error_with_suggestions(
                f"RAG index not found at '{index_path}'",
                [
                    "Build a RAG index first with 'nanodex rag index'",
                    "Verify the index path is correct",
                    "Check if indexing completed successfully",
                ],
                [
                    "nanodex rag index  # Build RAG index",
                    'nanodex rag search "your query"  # Then search',
                ],
            ),
            title="[bold red]RAG Index Error[/bold red]",
            border_style="red",
        )
    )
    console.print()


def show_data_not_found_error():
    """Show helpful error when training data is not found."""
    console.print()
    console.print(
        Panel(
            format_error_with_suggestions(
                "Training data not found",
                [
                    "Generate training data first with 'nanodex data generate'",
                    "Check the data output directory",
                    "Verify data generation completed successfully",
                ],
                [
                    "nanodex data generate --mode free  # Generate data",
                    "nanodex data preview  # Preview generated data",
                    "nanodex train  # Then train",
                ],
            ),
            title="[bold red]Data Error[/bold red]",
            border_style="red",
        )
    )
    console.print()
