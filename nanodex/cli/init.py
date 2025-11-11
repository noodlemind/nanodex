"""
Interactive initialization wizard for nanodex.
"""

import sys
import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich import box
from pathlib import Path
import yaml
import questionary
from questionary import Style

console = Console()

# Custom style for questionary
custom_style = Style(
    [
        ("qmark", "fg:cyan bold"),
        ("question", "bold"),
        ("answer", "fg:green bold"),
        ("pointer", "fg:cyan bold"),
        ("highlighted", "fg:cyan bold"),
        ("selected", "fg:green"),
        ("separator", "fg:black"),
        ("instruction", ""),
        ("text", ""),
    ]
)


def get_default_config(repo_path="."):
    """Generate default configuration without user prompts."""
    return {
        "mode": "free",
        "repository": {
            "path": repo_path,
            "include_extensions": [".py"],
            "exclude_dirs": ["node_modules", "venv", ".git", "__pycache__", "build", "dist"],
            "max_file_size": 1048576,
            "deep_parsing": {
                "enabled": True,
                "extract_functions": True,
                "extract_classes": True,
                "extract_imports": True,
                "extract_docstrings": True,
                "calculate_complexity": True,
            },
        },
        "model_source": "huggingface",
        "model": {
            "huggingface": {
                "model_name": "deepseek-ai/deepseek-coder-6.7b-base",
                "use_4bit": True,
                "use_8bit": False,
                "trust_remote_code": False,
            },
            "ollama": {"model_name": "codellama", "base_url": "http://localhost:11434"},
        },
        "training": {
            "output_dir": "./models/fine-tuned",
            "num_epochs": 3,
            "batch_size": 4,
            "learning_rate": 2.0e-5,
            "max_seq_length": 2048,
            "gradient_accumulation_steps": 4,
            "warmup_steps": 100,
            "logging_steps": 10,
            "save_steps": 500,
            "lora": {
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            },
        },
        "data": {
            "output_dir": "./data/processed",
            "train_split": 0.9,
            "validation_split": 0.1,
            "context_window": 4096,
            "random_seed": 42,
        },
        "export": {
            "output_dir": "./models/exported",
            "format": "gguf",
            "quantization": "q4_k_m",
        },
    }


@click.command()
@click.option("--output", default="config.yaml", help="Output configuration file")
@click.option("--non-interactive", is_flag=True, help="Generate default config without prompts")
@click.option("--repo-path", default=".", help="Repository path (for non-interactive mode)")
def init_cmd(output, non_interactive, repo_path):
    """
    🎯 Interactive setup wizard

    Creates a configuration file through an interactive questionnaire.
    Helps you choose the right mode, model, and settings for your use case.

    Use --non-interactive for automated/CI environments.
    """
    # Check if running in non-interactive environment
    if not sys.stdin.isatty() and not non_interactive:
        console.print("[yellow]Warning: Not running in an interactive terminal.[/yellow]")
        console.print("Use --non-interactive flag for automated environments.\n")
        non_interactive = True

    # Non-interactive mode
    if non_interactive:
        config = get_default_config(repo_path)
        output_path = Path(output)
        with open(output_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        console.print(f"[green]✓[/green] Default configuration saved to {output}")
        console.print(
            f"[dim]Mode: free, Model: deepseek-coder-6.7b-base, Repository: {repo_path}[/dim]"
        )
        return

    console.clear()

    # Welcome banner
    console.print(
        Panel.fit(
            "[bold cyan]nanodex Setup Wizard[/bold cyan]\n\n"
            "Let's configure your codebase-specific coding assistant!\n"
            "This wizard will help you choose the right settings.",
            border_style="cyan",
        )
    )

    try:
        config = {}

        # Step 1: Choose mode
        console.print("\n[bold cyan]Step 1: Choose Training Mode[/bold cyan]\n")
        mode = questionary.select(
            "Which mode do you want to use?",
            choices=[
                {
                    "name": "🆓 Free Mode - Zero cost self-supervised learning (recommended for start)",
                    "value": "free",
                },
                {
                    "name": "🔄 Hybrid Mode - Self-supervised + optional paid API (balanced)",
                    "value": "hybrid",
                },
                {
                    "name": "💎 Full Mode - Maximum quality with paid APIs (best results)",
                    "value": "full",
                },
            ],
            style=custom_style,
        ).ask()

        config["mode"] = mode

        # Step 2: Repository path
        console.print("\n[bold cyan]Step 2: Repository Configuration[/bold cyan]\n")

        repo_path = questionary.path(
            "Path to your codebase:", default=".", style=custom_style
        ).ask()

        if not Path(repo_path).exists():
            console.print(f"[red]Error: Path does not exist: {repo_path}[/red]")
            raise click.Abort()

        # File extensions
        extensions = questionary.checkbox(
            "Which file types should we analyze?",
            choices=[
                {"name": "🐍 Python (.py)", "value": ".py", "checked": True},
                {"name": "🟨 JavaScript (.js)", "value": ".js"},
                {"name": "🔷 TypeScript (.ts)", "value": ".ts"},
                {"name": "☕ Java (.java)", "value": ".java"},
                {"name": "⚙️  C++ (.cpp, .hpp)", "value": ".cpp"},
                {"name": "🦀 Rust (.rs)", "value": ".rs"},
                {"name": "🐹 Go (.go)", "value": ".go"},
            ],
            style=custom_style,
        ).ask()

        config["repository"] = {
            "path": repo_path,
            "include_extensions": extensions,
            "exclude_dirs": ["node_modules", "venv", ".git", "__pycache__", "build", "dist"],
            "max_file_size": 1048576,
            "deep_parsing": {
                "enabled": True,
                "extract_functions": True,
                "extract_classes": True,
                "extract_imports": True,
                "extract_docstrings": True,
                "calculate_complexity": True,
            },
        }

        # Step 3: Model selection
        console.print("\n[bold cyan]Step 3: Model Selection[/bold cyan]\n")

        model_source = questionary.select(
            "Where should we get the model from?",
            choices=[
                {"name": "🤗 HuggingFace (recommended)", "value": "huggingface"},
                {"name": "🦙 Ollama (local models)", "value": "ollama"},
            ],
            style=custom_style,
        ).ask()

        config["model_source"] = model_source

        if model_source == "huggingface":
            model_name = questionary.select(
                "Which base model?",
                choices=[
                    {
                        "name": "deepseek-ai/deepseek-coder-6.7b-base (recommended)",
                        "value": "deepseek-ai/deepseek-coder-6.7b-base",
                    },
                    {"name": "codellama/CodeLlama-7b-hf", "value": "codellama/CodeLlama-7b-hf"},
                    {"name": "bigcode/starcoder2-7b", "value": "bigcode/starcoder2-7b"},
                    {"name": "Custom (enter manually)", "value": "custom"},
                ],
                style=custom_style,
            ).ask()

            if model_name == "custom":
                model_name = questionary.text(
                    "Enter HuggingFace model name:", style=custom_style
                ).ask()

            use_4bit = questionary.confirm(
                "Use 4-bit quantization? (recommended for GPUs with <24GB RAM)",
                default=True,
                style=custom_style,
            ).ask()

            config["model"] = {
                "huggingface": {
                    "model_name": model_name,
                    "use_4bit": use_4bit,
                    "use_8bit": False,
                    "trust_remote_code": False,
                },
                "ollama": {"model_name": "codellama", "base_url": "http://localhost:11434"},
            }
        else:
            # Ollama
            model_name = questionary.text(
                "Ollama model name:", default="codellama", style=custom_style
            ).ask()

            config["model"] = {
                "huggingface": {
                    "model_name": "deepseek-ai/deepseek-coder-6.7b-base",
                    "use_4bit": True,
                    "use_8bit": False,
                    "trust_remote_code": False,
                },
                "ollama": {"model_name": model_name, "base_url": "http://localhost:11434"},
            }

        # Step 4: API Keys (if needed)
        if mode in ["hybrid", "full"]:
            console.print("\n[bold cyan]Step 4: API Configuration[/bold cyan]\n")
            console.print("[dim]For synthetic data generation[/dim]\n")

            api_provider = questionary.select(
                "Which API provider?",
                choices=[
                    {"name": "OpenAI (GPT-4, GPT-3.5)", "value": "openai"},
                    {"name": "Anthropic (Claude)", "value": "anthropic"},
                    {"name": "Skip for now", "value": "skip"},
                ],
                style=custom_style,
            ).ask()

            if api_provider != "skip":
                api_key = questionary.password(
                    f"Enter your {api_provider.upper()} API key:", style=custom_style
                ).ask()

                if api_provider == "openai":
                    api_model = questionary.select(
                        "Which model?",
                        choices=[
                            {"name": "gpt-3.5-turbo (cheaper)", "value": "gpt-3.5-turbo"},
                            {"name": "gpt-4-turbo (better quality)", "value": "gpt-4-turbo"},
                        ],
                        style=custom_style,
                    ).ask()
                else:
                    api_model = questionary.select(
                        "Which model?",
                        choices=[
                            {"name": "claude-3-haiku (cheapest)", "value": "claude-3-haiku"},
                            {"name": "claude-3-sonnet (balanced)", "value": "claude-3-sonnet"},
                            {"name": "claude-3-opus (best quality)", "value": "claude-3-opus"},
                        ],
                        style=custom_style,
                    ).ask()

                budget = questionary.text(
                    "Budget in USD for synthetic data:", default="10.0", style=custom_style
                ).ask()

                config["synthetic_api"] = {
                    "provider": api_provider,
                    "model": api_model,
                    "api_key": api_key,
                    "budget_usd": float(budget),
                    "examples_per_file": 5,
                    "temperature": 0.7,
                    "max_tokens": 1000,
                    "requests_per_minute": 10,
                }

        # Step 5: Training configuration
        console.print("\n[bold cyan]Step 5: Training Configuration[/bold cyan]\n")

        epochs = questionary.text(
            "Number of training epochs:", default="3", style=custom_style
        ).ask()

        batch_size = questionary.select(
            "Batch size:",
            choices=[
                {"name": "2 (for low GPU memory)", "value": 2},
                {"name": "4 (recommended)", "value": 4},
                {"name": "8 (for high GPU memory)", "value": 8},
            ],
            style=custom_style,
        ).ask()

        config["training"] = {
            "output_dir": "./models/fine-tuned",
            "num_epochs": int(epochs),
            "batch_size": batch_size,
            "learning_rate": 2.0e-5,
            "max_seq_length": 2048,
            "gradient_accumulation_steps": 4,
            "warmup_steps": 100,
            "logging_steps": 10,
            "save_steps": 500,
            "lora": {
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            },
        }

        # Data configuration
        config["data"] = {
            "output_dir": "./data/processed",
            "train_split": 0.9,
            "validation_split": 0.1,
            "context_window": 4096,
            "random_seed": 42,
        }

        # Export configuration
        config["export"] = {
            "output_dir": "./models/exported",
            "format": "gguf",
            "quantization": "q4_k_m",
        }

        # Save configuration
        output_path = Path(output)
        with open(output_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        # Success message
        console.print("\n")
        console.print(
            Panel.fit(
                f"[bold green]✓ Configuration saved to {output}![/bold green]\n\n"
                f"Mode: [cyan]{mode}[/cyan]\n"
                f"Model: [cyan]{config.get('model', {}).get(model_source, {}).get('model_name', 'N/A')}[/cyan]\n"
                f"Repository: [cyan]{repo_path}[/cyan]\n\n"
                "[bold]Next steps:[/bold]\n"
                "1. nanodex analyze          # Analyze your codebase\n"
                "2. nanodex data generate    # Generate training data\n"
                "3. nanodex train            # Fine-tune the model",
                title="Setup Complete",
                border_style="green",
            )
        )

    except KeyboardInterrupt:
        console.print("\n[yellow]Setup cancelled[/yellow]")
        raise click.Abort()
    except Exception as e:
        console.print(f"\n[red]Error during setup: {e}[/red]")
        raise click.Abort()
