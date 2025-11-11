"""
Interactive chat CLI command.
"""

import click
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich import box
from pathlib import Path
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

console = Console()


@click.command("chat")
@click.option("--index", default="./models/rag_index", help="Path to RAG index")
@click.option("--model", default=None, help="Path to fine-tuned model (optional)")
@click.option("--session", default=None, help="Session file to load/save")
@click.option("--no-rag", is_flag=True, help="Disable RAG context retrieval")
@click.option("--temperature", default=0.7, type=float, help="Generation temperature")
def chat_cmd(index, model, session, no_rag, temperature):
    """
    💬 Interactive chat with your codebase

    Start an interactive chat session where you can ask questions about your code,
    request explanations, or get coding help with RAG-augmented responses.

    \\b
    Features:
    - Multi-turn conversations with history
    - RAG-powered context retrieval
    - Code-aware responses
    - Session persistence

    \\b
    Example:
        nanodex chat
        nanodex chat --model ./models/fine-tuned
        nanodex chat --session my_session.json

    \\b
    Commands during chat:
    - Type your message and press Enter
    - Use ↑/↓ arrow keys to browse command history
    - Ctrl+R for reverse history search
    - /help     - Show help
    - /history  - Show conversation history
    - /clear    - Clear conversation history
    - /stats    - Show session statistics
    - /save     - Save session
    - /exit     - Exit chat
    """
    try:
        # Welcome message
        console.print()
        console.print(
            Panel.fit(
                "[bold cyan]💬 nanodex - Interactive Chat[/bold cyan]\n\n"
                "Ask questions about your codebase, request code explanations,\n"
                "or get coding help with AI-powered responses.\n\n"
                "[dim]Type /help for commands or /exit to quit[/dim]",
                border_style="cyan",
            )
        )
        console.print()

        # Load RAG index
        console.print("🔍 Loading RAG index...")
        from ..rag import SemanticRetriever
        from ..inference import RAGInference, ChatSession

        retriever = SemanticRetriever()

        # Check if index exists
        index_path = Path(index)
        if not index_path.exists():
            console.print(f"[yellow]⚠ RAG index not found at {index}[/yellow]")
            console.print("[dim]Build an index first with: nanodex rag index[/dim]\n")

            if not click.confirm("Continue without RAG?"):
                return

            retriever = None
            use_rag = False
        else:
            retriever.load(index)
            console.print(f"  ✓ Loaded {len(retriever.indexer.chunks)} code chunks\n")
            use_rag = not no_rag

        # Load model if provided
        model_obj = None
        tokenizer_obj = None

        if model:
            console.print(f"🤖 Loading model from {model}...")
            try:
                from ..models import ModelLoader
                from ..utils import Config

                # Load config
                cfg = Config("config.yaml")
                model_config = cfg.get_model_config()

                # Load model
                loader = ModelLoader(model_config, {})
                model_obj, tokenizer_obj = loader.load_huggingface_model()

                console.print("  ✓ Model loaded\n")
            except Exception as e:
                console.print(f"[yellow]⚠ Could not load model: {e}[/yellow]")
                console.print("[dim]Continuing without model (RAG search only)[/dim]\n")
        else:
            console.print("[dim]💡 No model specified. Using RAG search only.[/dim]")
            console.print("[dim]   Train a model with: nanodex train[/dim]\n")

        # Create inference engine
        rag_inference = RAGInference(retriever=retriever, model=model_obj, tokenizer=tokenizer_obj)

        # Load or create session
        if session and Path(session).exists():
            console.print(f"📂 Loading session from {session}...")
            chat_session = ChatSession.load_session(session, rag_inference)
            console.print(f"  ✓ Loaded session with {len(chat_session.messages)} messages\n")
        else:
            chat_session = ChatSession(rag_inference=rag_inference, use_rag=use_rag)
            console.print(f"📝 Created new chat session: {chat_session.session_id}\n")

        # Create prompt session with history and auto-suggestions
        prompt_session = PromptSession(
            history=FileHistory(".nanodex_chat_history"),
            auto_suggest=AutoSuggestFromHistory(),
            enable_history_search=True,
        )

        # Chat loop
        console.print(
            "[bold green]Ready to chat! Type your message or /help for commands.[/bold green]\n"
        )
        console.print("[dim]💡 Tip: Use arrow keys for history, Ctrl+R for reverse search[/dim]\n")

        while True:
            try:
                # Get user input with enhanced prompt
                console.print()
                user_input = prompt_session.prompt("You: ")

                if not user_input.strip():
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    command = user_input.lower().strip()

                    if command == "/exit" or command == "/quit":
                        console.print("\n[cyan]👋 Goodbye![/cyan]\n")
                        break

                    elif command == "/help":
                        _show_help()

                    elif command == "/history":
                        _show_history(chat_session)

                    elif command == "/clear":
                        chat_session.clear_history()
                        console.print("[green]✓ Conversation history cleared[/green]")

                    elif command == "/stats":
                        _show_stats(chat_session)

                    elif command == "/save":
                        save_path = session or f"{chat_session.session_id}.json"
                        chat_session.save_session(save_path)
                        console.print(f"[green]✓ Session saved to {save_path}[/green]")

                    else:
                        console.print(f"[yellow]Unknown command: {command}[/yellow]")
                        console.print("[dim]Type /help for available commands[/dim]")

                    continue

                # Send message and get response
                console.print()
                console.print("[dim]Thinking...[/dim]")

                response = chat_session.send_message(user_input, temperature=temperature)

                # Display response
                console.print()
                console.print(
                    Panel(
                        Markdown(response),
                        title="[bold green]Assistant[/bold green]",
                        border_style="green",
                        padding=(1, 2),
                    )
                )

            except KeyboardInterrupt:
                console.print("\n\n[cyan]Use /exit to quit or continue chatting...[/cyan]")
                continue
            except EOFError:
                break

        # Auto-save session if specified
        if session:
            chat_session.save_session(session)
            console.print(f"\n[dim]Session auto-saved to {session}[/dim]\n")

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        import traceback

        console.print(f"\n[dim]{traceback.format_exc()}[/dim]")
        raise click.Abort()


def _show_help():
    """Show help message."""
    help_table = Table(title="Chat Commands", box=box.ROUNDED)
    help_table.add_column("Command", style="cyan")
    help_table.add_column("Description")

    help_table.add_row("/help", "Show this help message")
    help_table.add_row("/history", "Show conversation history")
    help_table.add_row("/clear", "Clear conversation history")
    help_table.add_row("/stats", "Show session statistics")
    help_table.add_row("/save", "Save current session")
    help_table.add_row("/exit", "Exit chat")

    console.print()
    console.print(help_table)
    console.print()


def _show_history(chat_session):
    """Show conversation history."""
    messages = chat_session.get_history()

    if not messages:
        console.print("[yellow]No conversation history yet[/yellow]")
        return

    console.print()
    console.print("[bold cyan]Conversation History[/bold cyan]\n")

    for i, msg in enumerate(messages, 1):
        role_color = "cyan" if msg.role == "user" else "green"
        role_label = "You" if msg.role == "user" else "Assistant"

        console.print(f"[{role_color}]{i}. {role_label}:[/{role_color}]")
        console.print(f"   {msg.content[:100]}{'...' if len(msg.content) > 100 else ''}")
        console.print()


def _show_stats(chat_session):
    """Show session statistics."""
    stats = chat_session.get_stats()

    console.print()
    table = Table(title="Session Statistics", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Session ID", stats["session_id"])
    table.add_row("Created", stats["created_at"])
    table.add_row("Duration", f"{stats['duration_seconds']:.0f} seconds")
    table.add_row("Total Messages", str(stats["total_messages"]))
    table.add_row("User Messages", str(stats["user_messages"]))
    table.add_row("Assistant Messages", str(stats["assistant_messages"]))
    table.add_row("RAG Enabled", "Yes" if stats["use_rag"] else "No")

    console.print(table)
    console.print()
