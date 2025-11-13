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

# Suppress noisy warnings from bitsandbytes and PyTorch
import warnings

warnings.filterwarnings("ignore", message=".*bitsandbytes.*")
warnings.filterwarnings("ignore", message=".*PyTorch.*")
warnings.filterwarnings("ignore", message=".*Redirects.*")

import os

os.environ["BITSANDBYTES_NOWELCOME"] = "1"

import click
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from pydantic import BaseModel, Field, validator
import yaml
import json
import shlex
import re
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from contextlib import contextmanager

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
SESSIONS_DIR = Path(".nanodex_sessions")


@contextmanager
def file_lock(lock_path: Path, timeout: float = 5.0):
    """
    Simple file-based lock for concurrent access protection.

    Args:
        lock_path: Path to lock file
        timeout: Maximum time to wait for lock (seconds)

    Raises:
        TimeoutError: If lock cannot be acquired within timeout

    Uses lock file creation as atomic operation. Works across platforms.
    """
    lock_file = lock_path.with_suffix(".lock")
    start_time = time.time()

    # Try to acquire lock
    while True:
        try:
            # Exclusive creation (atomic on most filesystems)
            lock_file.touch(exist_ok=False)
            break
        except FileExistsError:
            # Lock file exists, check if it's stale (>30 seconds old)
            if lock_file.exists():
                age = time.time() - lock_file.stat().st_mtime
                if age > 30:
                    # Stale lock, remove it
                    try:
                        lock_file.unlink()
                        continue
                    except FileNotFoundError:
                        continue

            # Check timeout
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Could not acquire lock on {lock_path} within {timeout}s")

            # Wait a bit before retrying
            time.sleep(0.1)

    try:
        yield
    finally:
        # Release lock
        try:
            lock_file.unlink()
        except FileNotFoundError:
            pass


def atomic_write_json(file_path: Path, data: Dict[str, Any]) -> None:
    """
    Atomically write JSON data to a file.

    Args:
        file_path: Target file path
        data: Data to write

    Uses write-to-temp-then-rename pattern to ensure atomic updates
    and prevent data corruption from interrupted writes.
    """
    temp_file = file_path.with_suffix(".tmp")

    try:
        # Write to temporary file
        with open(temp_file, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()  # Ensure data is written to disk
            os.fsync(f.fileno())  # Force write to disk

        # Atomic rename (POSIX guarantees atomicity)
        temp_file.replace(file_path)
    except Exception:
        # Clean up temp file on error
        if temp_file.exists():
            temp_file.unlink()
        raise


class SessionData(BaseModel):
    """
    Validated session data schema using Pydantic.

    Ensures session data integrity and prevents JSON injection attacks.
    All fields are validated for type safety and reasonable constraints.

    Size Limits:
    - Each dict field: max 10KB when serialized to JSON
    - Total session size: max 100KB
    """

    last_results: Optional[Dict[str, Any]] = Field(None, description="Last command results")
    training_state: Optional[Dict[str, Any]] = Field(None, description="Training progress state")
    analysis_stats: Optional[Dict[str, Any]] = Field(None, description="Analysis statistics")
    command_count: int = Field(0, ge=0, le=1_000_000, description="Commands run in session")
    last_saved: str = Field(..., description="ISO format timestamp of last save")

    @validator("last_results", "training_state", "analysis_stats")
    def validate_dict_size(cls, v, field):
        """Ensure dict fields don't exceed size limits."""
        if v is None:
            return v

        # Check JSON serialized size (max 10KB per field)
        try:
            serialized = json.dumps(v)
            size_kb = len(serialized) / 1024
            if size_kb > 10:
                raise ValueError(f"{field.name} exceeds 10KB limit ({size_kb:.1f}KB)")
        except (TypeError, ValueError) as e:
            if "exceeds 10KB limit" in str(e):
                raise
            raise ValueError(f"{field.name} contains non-serializable data: {e}")

        return v

    @validator("command_count")
    def validate_command_count(cls, v):
        """Ensure command count is reasonable."""
        if v < 0:
            raise ValueError("command_count cannot be negative")
        if v > 1_000_000:
            raise ValueError("command_count exceeds maximum (1,000,000)")
        return v

    @validator("last_saved")
    def validate_timestamp(cls, v):
        """Ensure timestamp is valid ISO format."""
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError(f"Invalid ISO timestamp: {v}")
        return v

    class Config:
        extra = "forbid"  # Reject unknown fields to prevent injection


def validate_session_name(session_name: str) -> bool:
    """
    Validate session name to prevent path traversal attacks.

    Args:
        session_name: Name to validate

    Returns:
        True if valid, False otherwise

    Security:
        - Only allows alphanumeric, underscore, and hyphen characters
        - Prevents directory traversal attempts (../, absolute paths, etc.)
        - Limits length to reasonable maximum (100 chars)
    """
    if not session_name:
        return False

    # Length check
    if len(session_name) > 100:
        return False

    # Pattern check: only alphanumeric, underscore, hyphen
    pattern = re.compile(r"^[a-zA-Z0-9_-]+$")
    if not pattern.match(session_name):
        return False

    # Additional safety: ensure resolved path stays within sessions directory
    try:
        SESSIONS_DIR.mkdir(exist_ok=True)
        session_path = SESSIONS_DIR / f"{session_name}.json"
        resolved_path = session_path.resolve()
        sessions_dir_resolved = SESSIONS_DIR.resolve()

        # Check if the resolved path is within the sessions directory
        if not resolved_path.is_relative_to(sessions_dir_resolved):
            return False
    except (ValueError, OSError):
        return False

    return True


def load_session(session_name: str = None) -> Dict[str, Any]:
    """
    Load previous session state if available.

    Args:
        session_name: Optional name of saved session to load

    Session maintains:
    - Last command results
    - Training progress
    - Analysis statistics
    - Timestamp of last session
    """
    if session_name:
        # Validate session name for security
        if not validate_session_name(session_name):
            console.print(f"[red]✗ Invalid session name: '{session_name}'[/red]")
            console.print(
                "[dim]Session names must contain only letters, numbers, hyphens, and underscores[/dim]"
            )
            return _empty_session()

        # Load named session
        session_path = SESSIONS_DIR / f"{session_name}.json"
        if not session_path.exists():
            console.print(f"[yellow]⚠ Session '{session_name}' not found[/yellow]")
            return _empty_session()
    else:
        # Load default session
        session_path = Path(SESSION_FILE)

    if session_path.exists():
        try:
            # Use file lock to prevent concurrent access issues
            with file_lock(session_path):
                with open(session_path, "r") as f:
                    raw_data = json.load(f)

                # Validate with Pydantic schema
                try:
                    validated_session = SessionData(**raw_data)
                    session_dict = validated_session.dict()

                    if session_name:
                        console.print(
                            f"[dim]📂 Loaded session '{session_name}' from {session_dict['last_saved']}[/dim]"
                        )
                    else:
                        console.print(
                            f"[dim]📂 Restored session from {session_dict['last_saved']}[/dim]"
                        )
                    return session_dict
                except Exception as e:
                    console.print(f"[yellow]⚠ Session data validation failed: {e}[/yellow]")
                    console.print("[dim]Starting with empty session[/dim]")
                    return _empty_session()

        except TimeoutError:
            console.print(
                f"[yellow]⚠ Could not acquire lock on session file, using empty session[/yellow]"
            )
            return _empty_session()
        except (json.JSONDecodeError, IOError) as e:
            console.print(f"[yellow]⚠ Could not load session: {e}[/yellow]")

    return _empty_session()


def _empty_session() -> Dict[str, Any]:
    """Return an empty session dictionary."""
    return {
        "last_results": None,
        "training_state": None,
        "analysis_stats": None,
        "command_count": 0,
    }


def save_session(context: Dict[str, Any], session_name: str = None) -> None:
    """
    Save current session state for next time.

    Args:
        context: Session context to save
        session_name: Optional name for the session

    Persists context across shell sessions so you can pick up where you left off.
    """
    try:
        # Validate session name if provided
        if session_name and not validate_session_name(session_name):
            console.print(f"\n[red]✗ Cannot save session: Invalid name '{session_name}'[/red]")
            console.print(
                "[dim]Session names must contain only letters, numbers, hyphens, and underscores[/dim]"
            )
            return

        # Create and validate session data with Pydantic
        try:
            session_data = SessionData(
                last_results=context.get("last_results"),
                training_state=context.get("training_state"),
                analysis_stats=context.get("analysis_stats"),
                command_count=context.get("command_count", 0),
                last_saved=datetime.now().isoformat(),
            )
        except Exception as e:
            console.print(f"\n[red]✗ Session data validation failed: {e}[/red]")
            console.print("[dim]Session not saved[/dim]")
            return

        # Convert to dict for JSON serialization
        session = session_data.dict()

        if session_name:
            # Save named session
            SESSIONS_DIR.mkdir(exist_ok=True)
            session_path = SESSIONS_DIR / f"{session_name}.json"
        else:
            # Save default session
            session_path = Path(SESSION_FILE)

        # Use file lock to prevent concurrent access issues
        with file_lock(session_path):
            atomic_write_json(session_path, session)

        if session_name:
            console.print(f"\n[dim]💾 Session saved as '{session_name}'[/dim]")
        else:
            console.print(f"\n[dim]💾 Session saved to {SESSION_FILE}[/dim]")

    except TimeoutError:
        console.print(
            f"\n[yellow]⚠ Could not acquire lock on session file, session not saved[/yellow]"
        )
    except IOError as e:
        console.print(f"[yellow]⚠ Could not save session: {e}[/yellow]")


def list_sessions() -> None:
    """List all saved named sessions."""
    if not SESSIONS_DIR.exists():
        console.print("[dim]No saved sessions found[/dim]")
        return

    sessions = list(SESSIONS_DIR.glob("*.json"))
    if not sessions:
        console.print("[dim]No saved sessions found[/dim]")
        return

    console.print("\n[bold cyan]Saved Sessions:[/bold cyan]\n")
    for session_path in sorted(sessions):
        try:
            with open(session_path, "r") as f:
                session = json.load(f)
            name = session_path.stem
            last_saved = session.get("last_saved", "unknown")
            console.print(f"  [cyan]{name}[/cyan] - saved {last_saved}")
        except Exception:
            continue
    console.print()


def rotate_history(history_file: Path, max_lines: int = 1000) -> None:
    """
    Rotate history file to prevent unbounded growth.

    Args:
        history_file: Path to history file
        max_lines: Maximum lines to keep (default: 1000)

    Keeps only the most recent max_lines entries.
    """
    if not history_file.exists():
        return

    try:
        with open(history_file, "r") as f:
            lines = f.readlines()

        if len(lines) <= max_lines:
            return

        # Keep only the most recent max_lines
        recent_lines = lines[-max_lines:]

        # Write back atomically
        temp_file = history_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            f.writelines(recent_lines)

        temp_file.replace(history_file)

        console.print(f"[dim]History rotated: kept {len(recent_lines)} most recent entries[/dim]")

    except IOError as e:
        console.print(f"[yellow]⚠ Could not rotate history: {e}[/yellow]")


def custom_repl(ctx: click.Context, prompt_kwargs: Dict[str, Any]) -> None:
    """
    Custom REPL implementation without click-repl dependency.

    Args:
        ctx: Click context object
        prompt_kwargs: Prompt session configuration (message, history, auto_suggest)

    Avoids compatibility issues with Click 8.1+.
    """
    # Get available commands
    cli = ctx.find_root().command
    available_commands = list(cli.commands.keys()) if hasattr(cli, "commands") else []

    # Create completer
    completer = WordCompleter(words=available_commands + ["help", "exit", "quit"], ignore_case=True)

    # Create prompt session
    session = PromptSession(
        message=prompt_kwargs.get("message", "> "),
        history=prompt_kwargs.get("history"),
        auto_suggest=prompt_kwargs.get("auto_suggest"),
        completer=completer,
    )

    while True:
        try:
            # Get input
            text = session.prompt()

            # Skip empty lines
            if not text.strip():
                continue

            # Handle exit commands
            if text.strip().lower() in ("exit", "quit"):
                break

            # Handle help command
            if text.strip().lower() == "help":
                console.print("\n[bold cyan]Available Commands:[/bold cyan]\n")
                for cmd in sorted(available_commands):
                    console.print(f"  [cyan]{cmd}[/cyan]")
                console.print("\n[bold cyan]Built-in Commands:[/bold cyan]")
                console.print("  [cyan]help[/cyan] - Show this help")
                console.print("  [cyan]exit[/cyan] or [cyan]quit[/cyan] - Exit shell\n")
                continue

            # Parse and invoke command
            try:
                args = shlex.split(text)

                # Invoke command through Click's main() method
                # standalone_mode=False prevents SystemExit from being raised
                cli.main(args=args, standalone_mode=False, obj=ctx.obj)

            except click.ClickException as e:
                e.show()
            except click.Abort:
                console.print("[yellow]Command aborted[/yellow]")
            except Exception as e:
                console.print(f"[red]Error:[/red] {e}")

            # Increment command counter
            if "command_count" in ctx.obj:
                ctx.obj["command_count"] += 1

        except KeyboardInterrupt:
            console.print("\n[dim]Use 'exit' or Ctrl+D to quit[/dim]")
            continue
        except EOFError:
            break


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
        # Rotate history file if needed (keep last 1000 lines)
        history_path = Path(".nanodex_history")
        rotate_history(history_path, max_lines=1000)

        prompt_kwargs["history"] = FileHistory(".nanodex_history")
        prompt_kwargs["auto_suggest"] = AutoSuggestFromHistory()

    # Start REPL with root context so commands see shared state
    try:
        custom_repl(root_ctx, prompt_kwargs=prompt_kwargs)
    except (KeyboardInterrupt, EOFError):
        console.print("\n[cyan]👋 Exiting shell...[/cyan]")
    finally:
        # Save session state on exit
        save_session(root_ctx.obj)
        console.print()
