"""
Main CLI entry point using Click framework.
"""

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
import logging
from pathlib import Path

from ..utils import Config

# Create rich console
console = Console()

# Setup logging with rich
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler()]
)


@click.group()
@click.version_option(version='0.2.0', prog_name='turbo-code-gpt')
@click.pass_context
def cli(ctx):
    """
    🚀 Turbo Code GPT - Fine-tune LLMs on your codebase

    A flexible system for creating specialized coding assistants by fine-tuning
    open-source models on your codebase.

    \b
    Quick Start:
      turbo-code-gpt init          # Interactive setup wizard
      turbo-code-gpt analyze       # Analyze your codebase
      turbo-code-gpt data generate # Generate training data
      turbo-code-gpt train         # Fine-tune the model
      turbo-code-gpt chat          # Chat with your model

    \b
    Learn more:
      turbo-code-gpt --help
      turbo-code-gpt COMMAND --help
    """
    ctx.ensure_object(dict)


@cli.command()
@click.option('--config', default='config.yaml', help='Configuration file path')
def config_show(config):
    """Show current configuration."""
    try:
        cfg = Config(config)

        console.print("\n[bold cyan]Configuration Summary[/bold cyan]\n")

        # Model configuration
        table = Table(title="Model Settings", box=box.ROUNDED)
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Source", cfg.get_model_source())
        model_cfg = cfg.get_model_config()
        for key, value in model_cfg.items():
            table.add_row(key, str(value))

        console.print(table)

        # Repository configuration
        table = Table(title="Repository Settings", box=box.ROUNDED)
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")

        repo_cfg = cfg.get_repository_config()
        table.add_row("Path", repo_cfg['path'])
        table.add_row("Extensions", ", ".join(repo_cfg['include_extensions']))
        table.add_row("Deep Parsing", str(repo_cfg['deep_parsing']['enabled']))

        console.print(table)

    except Exception as e:
        console.print(f"[bold red]Error loading configuration:[/bold red] {e}")
        raise click.Abort()


@cli.command()
@click.option('--config', default='config.yaml', help='Configuration file path')
def config_validate(config):
    """Validate configuration file."""
    try:
        cfg = Config(config)
        console.print(f"\n[bold green]✓[/bold green] Configuration is valid: {config}\n")
    except Exception as e:
        console.print(f"\n[bold red]✗[/bold red] Configuration validation failed:\n")
        console.print(f"[red]{e}[/red]\n")
        raise click.Abort()


@cli.command()
@click.argument('key')
@click.argument('value')
@click.option('--config', default='config.yaml', help='Configuration file path')
def config_set(key, value, config):
    """Set a configuration value."""
    console.print(f"[yellow]Setting {key} = {value} in {config}[/yellow]")
    console.print("[dim]Note: This feature is not yet implemented[/dim]")
    console.print("[dim]Please edit config.yaml directly for now[/dim]")


# Import other command modules
from .init import init_cmd
from .analyze import analyze_cmd
from .data_gen import data_cmd
from .train import train_cmd

# Register commands
cli.add_command(init_cmd, name='init')
cli.add_command(analyze_cmd, name='analyze')
cli.add_command(data_cmd, name='data')
cli.add_command(train_cmd, name='train')


if __name__ == '__main__':
    cli()
