"""
Interactive shell command for nanodex.

Provides a REPL interface where users can execute multiple nanodex commands
in a persistent session without reloading configuration and resources.

Educational Shell Features:
- Persistent context: Config, results, and state maintained across commands
- Command history: Navigate with arrow keys (↑↓)
- Tab completion: Smart completion for commands and options
- Session auto-save: State saved to .nanodex_session.json
- Educational tips: Learn while you work
"""

import click
from click_repl import repl
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

console = Console()

# Educational tips shown during shell usage
SHELL_TIPS = [
    "💡 **Tip**: Commands run faster in shell mode - config is loaded once!",
    "💡 **Tip**: LoRA fine-tuning trains only 0.1% of model parameters",
    "💡 **Tip**: 4-bit quantization reduces memory by 75% with minimal accuracy loss",
    "💡 **Tip**: RAG adds semantic search without modifying your model",
    "💡 **Tip**: Free mode generates training data from your codebase - no API costs!",
]

SESSION_FILE = ".nanodex_session.json"


def load_session() -> Dict[str, Any]:
    """
    Load previous session state if available.

    Session maintains:
    - Last command results
    - Training progress
    - Analysis statistics
    - Timestamp of last session
    """
    session_path = Path(SESSION_FILE)
    if session_path.exists():
        try:
            with open(session_path, 'r') as f:
                session = json.load(f)
            console.print(f"[dim]📂 Restored session from {session.get('last_saved', 'unknown time')}[/dim]")
            return session
        except (json.JSONDecodeError, IOError) as e:
            console.print(f"[yellow]⚠ Could not load session: {e}[/yellow]")
    return {
        "last_results": None,
        "training_state": None,
        "analysis_stats": None,
        "command_count": 0,
    }


def save_session(context: Dict[str, Any]) -> None:
    """
    Save current session state for next time.

    Persists context across shell sessions so you can pick up where you left off.
    """
    try:
        session = {
            "last_results": context.get("last_results"),
            "training_state": context.get("training_state"),
            "analysis_stats": context.get("analysis_stats"),
            "command_count": context.get("command_count", 0),
            "last_saved": datetime.now().isoformat(),
        }
        with open(SESSION_FILE, 'w') as f:
            json.dump(session, f, indent=2)
        console.print(f"\n[dim]💾 Session saved to {SESSION_FILE}[/dim]")
    except IOError as e:
        console.print(f"[yellow]⚠ Could not save session: {e}[/yellow]")


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

    # Show a random educational tip on startup
    import random
    tip = random.choice(SHELL_TIPS)
    console.print(Markdown(tip))
    console.print()

    # Initialize persistent context on root CLI context
    root_ctx = ctx.find_root()
    root_ctx.ensure_object(dict)
    root_ctx.obj["shell_mode"] = True

    # Load previous session if available
    session_data = load_session()
    root_ctx.obj.update(session_data)

    # Load config once (instead of per-command) and store on root context
    try:
        from ..utils.config import Config

        root_ctx.obj["config"] = Config("config.yaml")
        console.print("[green]✓[/green] Configuration loaded\n")
    except FileNotFoundError as e:
        console.print(f"[yellow]⚠[/yellow] Config file not found: {e}")
        console.print("[dim]You can still use commands that don't require config[/dim]\n")
    except ImportError as e:
        console.print(f"[yellow]⚠[/yellow] Failed to import Config module: {e}")
        console.print("[dim]You can still use commands that don't require config[/dim]\n")
    except yaml.YAMLError as e:
        console.print(f"[yellow]⚠[/yellow] Config file has invalid YAML: {e}")
        console.print("[dim]You can still use commands that don't require config[/dim]\n")

    # Setup prompt kwargs
    prompt_kwargs = {
        "message": "nanodex> ",
    }

    if not no_history:
        prompt_kwargs["history"] = FileHistory(".nanodex_history")
        prompt_kwargs["auto_suggest"] = AutoSuggestFromHistory()

    # Start REPL with root context so commands see shared state
    try:
        repl(root_ctx, prompt_kwargs=prompt_kwargs)
    except (KeyboardInterrupt, EOFError):
        console.print("\n[cyan]👋 Exiting shell...[/cyan]")
    finally:
        # Save session state on exit
        save_session(root_ctx.obj)
        console.print()
