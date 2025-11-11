"""
Interactive shell command for nanodex.

Provides a REPL interface where users can execute multiple nanodex commands
in a persistent session without reloading configuration and resources.
"""

import click
from click_repl import repl
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from rich.console import Console
from rich.panel import Panel

console = Console()


@click.command("shell")
@click.pass_context
@click.option("--no-history", is_flag=True, help="Disable command history")
def shell_cmd(ctx, no_history):
    """
    Start interactive shell mode

    All nanodex commands available in shell mode with persistent context.
    Models and indexes stay loaded for faster iteration.

    \b
    Built-in commands:
    - help     - Show available commands
    - exit     - Exit shell (or Ctrl+D)

    \b
    Examples:
        nanodex shell
        nanodex shell --no-history

    \b
    Usage in shell:
        nanodex> analyze
        nanodex> data generate --mode free
        nanodex> train --epochs 3
        nanodex> rag index
        nanodex> exit
    """

    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]🐚 nanodex Interactive Shell[/bold cyan]\n\n"
            "All commands available. Type 'help' for list.\n"
            "Context persists between commands.\n\n"
            "[dim]Type 'exit' to quit or press Ctrl+D[/dim]",
            border_style="cyan",
        )
    )
    console.print()

    # Initialize persistent context
    ctx.ensure_object(dict)
    ctx.obj["shell_mode"] = True

    # Load config once (instead of per-command)
    try:
        from ..utils.config import Config

        ctx.obj["config"] = Config("config.yaml")
        console.print("[green]✓[/green] Configuration loaded\n")
    except Exception as e:
        console.print(f"[yellow]⚠[/yellow] Config not loaded: {e}")
        console.print("[dim]You can still use commands that don't require config[/dim]\n")

    # Setup prompt kwargs
    prompt_kwargs = {
        "message": "nanodex> ",
    }

    if not no_history:
        prompt_kwargs["history"] = FileHistory(".nanodex_history")
        prompt_kwargs["auto_suggest"] = AutoSuggestFromHistory()

    # Start REPL
    try:
        repl(ctx, prompt_kwargs=prompt_kwargs)
    except (KeyboardInterrupt, EOFError):
        console.print("\n[cyan]👋 Exiting shell...[/cyan]\n")
