"""
Shell-specific commands for the interactive REPL.

These commands are available within the shell and handle session management,
checkpoints, and other interactive features.
"""

import click
from rich.console import Console
from rich.table import Table
from rich import box
from pathlib import Path
import json

console = Console()


@click.group()
def shell_commands():
    """Special shell commands"""
    pass


@shell_commands.command('sessions')
def sessions_cmd():
    """
    List all saved sessions.

    \b
    Shows named sessions saved with /save command.
    Use /load <name> to restore a session.
    """
    from .shell import list_sessions
    list_sessions()


@shell_commands.command('checkpoints')
@click.option('--dir', default='./models/fine-tuned', help='Model directory')
def checkpoints_cmd(dir):
    """
    List available training checkpoints.

    \b
    Shows checkpoints saved during training.
    Use --resume <checkpoint> with train command to continue.

    \b
    Example:
        checkpoints
        checkpoints --dir ./my-model
    """
    model_dir = Path(dir)

    if not model_dir.exists():
        console.print(f"[yellow]⚠ Directory not found: {model_dir}[/yellow]")
        return

    # Find checkpoint directories
    checkpoints = []
    for path in model_dir.iterdir():
        if path.is_dir() and path.name.startswith('checkpoint-'):
            try:
                step = int(path.name.split('-')[1])
                checkpoints.append((step, path))
            except (IndexError, ValueError):
                continue

    if not checkpoints:
        console.print(f"[dim]No checkpoints found in {model_dir}[/dim]")
        return

    # Sort by step number
    checkpoints.sort()

    console.print(f"\n[bold cyan]Checkpoints in {model_dir}:[/bold cyan]\n")

    table = Table(box=box.ROUNDED)
    table.add_column("Step", style="cyan")
    table.add_column("Path", style="green")
    table.add_column("Size", justify="right")

    for step, path in checkpoints:
        # Calculate directory size
        size_mb = sum(f.stat().st_size for f in path.rglob('*') if f.is_file()) / (1024 * 1024)
        table.add_row(
            str(step),
            str(path.name),
            f"{size_mb:.1f}MB"
        )

    console.print(table)
    console.print(f"\n[dim]Use: nanodex train --resume {model_dir}/checkpoint-XXXX[/dim]\n")


@shell_commands.command('history')
@click.option('--limit', '-n', default=20, help='Number of commands to show')
def history_cmd(limit):
    """
    Show command history.

    \b
    Displays recent commands run in the shell.

    \b
    Example:
        history
        history --limit 50
    """
    history_file = Path('.nanodex_history')

    if not history_file.exists():
        console.print("[dim]No command history found[/dim]")
        return

    try:
        with open(history_file, 'r') as f:
            lines = f.readlines()

        # Show last N commands
        recent = lines[-limit:] if len(lines) > limit else lines

        console.print(f"\n[bold cyan]Recent Commands (last {len(recent)}):[/bold cyan]\n")

        for i, cmd in enumerate(recent, 1):
            console.print(f"  {i:3d}  {cmd.strip()}")

        console.print()

    except IOError as e:
        console.print(f"[yellow]⚠ Could not read history: {e}[/yellow]")


@shell_commands.command('clear')
def clear_cmd():
    """Clear the terminal screen."""
    import os
    os.system('clear' if os.name != 'nt' else 'cls')
    console.print("[dim]Screen cleared[/dim]\n")


@shell_commands.command('info')
@click.pass_context
def info_cmd(ctx):
    """
    Show session information.

    \b
    Displays current context, loaded config, and session state.
    """
    root_ctx = ctx.find_root()

    console.print("\n[bold cyan]Session Information[/bold cyan]\n")

    # Configuration
    if 'config' in root_ctx.obj:
        console.print("[green]✓[/green] Configuration loaded")
        cfg = root_ctx.obj['config']
        console.print(f"  Model source: {cfg.get_model_source()}")
        model_config = cfg.get_model_config()
        console.print(f"  Model: {model_config.get('model_name', 'N/A')}")
    else:
        console.print("[yellow]⚠[/yellow] No configuration loaded")

    # Session stats
    console.print(f"\n📊 Session Statistics:")
    console.print(f"  Commands run: {root_ctx.obj.get('command_count', 0)}")

    # Last results
    if 'last_results' in root_ctx.obj and root_ctx.obj['last_results']:
        console.print(f"\n[green]✓[/green] Last results available")

    # Training state
    if 'training_state' in root_ctx.obj and root_ctx.obj['training_state']:
        console.print(f"[green]✓[/green] Training state: {root_ctx.obj['training_state']}")

    console.print()


@shell_commands.command('help')
def help_cmd():
    """
    Show available shell commands.

    \b
    Displays all special shell commands and how to use them.
    """
    console.print("\n[bold cyan]Shell Commands:[/bold cyan]\n")

    commands = [
        ("sessions", "List saved sessions"),
        ("checkpoints", "List training checkpoints"),
        ("history", "Show command history"),
        ("clear", "Clear screen"),
        ("info", "Show session information"),
        ("help", "Show this help"),
    ]

    for cmd, desc in commands:
        console.print(f"  [cyan]{cmd:15}[/cyan] {desc}")

    console.print("\n[bold cyan]Regular Commands:[/bold cyan]\n")
    console.print("  All nanodex commands are available:")
    console.print("    analyze, data, train, rag, chat, edu, etc.")

    console.print("\n[dim]Tip: Use 'nanodex --help' for full command list[/dim]\n")
