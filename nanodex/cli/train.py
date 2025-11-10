"""
Training command.
"""

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich import box
import json
from pathlib import Path

from ..utils import Config
from ..trainers import DataPreparer, ModelTrainer
from ..models import ModelLoader

console = Console()


@click.command()
@click.option('--config', default='config.yaml', help='Configuration file path')
@click.option('--data', help='Path to training data JSON file (optional)')
@click.option('--resume', help='Resume from checkpoint path')
def train_cmd(config, data, resume):
    """
    🚀 Train/fine-tune the model

    Fine-tunes the selected base model on your generated training data using LoRA.

    \b
    Steps:
    1. Load configuration
    2. Prepare datasets (or load from file)
    3. Load base model
    4. Apply LoRA
    5. Fine-tune
    6. Save checkpoints

    \b
    Example:
        nanodex train
        nanodex train --data ./data/training_examples.json
        nanodex train --resume ./models/fine-tuned/checkpoint-500
    """
    try:
        console.print("\n[bold cyan]Starting Training Pipeline...[/bold cyan]\n")

        # Load configuration
        cfg = Config(config)

        # Display training configuration
        training_config = cfg.get_training_config()
        model_source = cfg.get_model_source()
        model_config = cfg.get_model_config()

        table = Table(title="Training Configuration", box=box.ROUNDED)
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Model Source", model_source)
        table.add_row("Model", model_config.get('model_name', 'N/A'))
        table.add_row("Epochs", str(training_config['num_epochs']))
        table.add_row("Batch Size", str(training_config['batch_size']))
        table.add_row("Learning Rate", str(training_config['learning_rate']))
        table.add_row("LoRA Rank (r)", str(training_config['lora']['r']))

        console.print(table)
        console.print()

        if model_source != 'huggingface':
            console.print("[yellow]⚠ Training is currently only supported for HuggingFace models[/yellow]")
            console.print("[dim]For Ollama models, use the Ollama CLI tools[/dim]\n")
            return

        # Prepare training data
        console.print("Step 1: Preparing training data...")

        if data:
            # Load from file
            console.print(f"  Loading data from {data}...")
            with open(data, 'r') as f:
                examples = json.load(f)

            # Convert to dataset format (simplified for now)
            from datasets import Dataset

            # Create train/val split
            data_config = cfg.get_data_config()
            split_idx = int(len(examples) * data_config['train_split'])

            train_data = examples[:split_idx]
            val_data = examples[split_idx:]

            train_dataset = Dataset.from_list(train_data)
            val_dataset = Dataset.from_list(val_data) if val_data else None

            console.print(f"  Training examples: {len(train_dataset)}")
            console.print(f"  Validation examples: {len(val_dataset) if val_dataset else 0}\n")
        else:
            # Generate from codebase
            console.print("  [yellow]No data file specified, will analyze codebase...[/yellow]")
            console.print("  [dim]Tip: Use 'nanodex data generate' to pre-generate data[/dim]\n")

            from ..analyzers import CodeAnalyzer

            repo_config = cfg.get_repository_config()
            analyzer = CodeAnalyzer(repo_config)
            code_samples = analyzer.analyze()

            data_config = cfg.get_data_config()
            preparer = DataPreparer(data_config)
            train_dataset, val_dataset = preparer.prepare_data(code_samples)

            console.print(f"  Training examples: {len(train_dataset)}")
            console.print(f"  Validation examples: {len(val_dataset)}\n")

        # Load model
        console.print("Step 2: Loading model...")
        console.print(f"  Model: [cyan]{model_config['model_name']}[/cyan]")

        if model_config.get('use_4bit'):
            console.print("  Quantization: [cyan]4-bit[/cyan]")
        elif model_config.get('use_8bit'):
            console.print("  Quantization: [cyan]8-bit[/cyan]")

        loader = ModelLoader(
            model_config,
            training_config,
            trust_remote_code=model_config.get('trust_remote_code', False)
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Loading model...", total=None)
            model, tokenizer = loader.load_huggingface_model()
            progress.update(task, completed=True)

        console.print("  [green]✓[/green] Model loaded\n")

        # Apply LoRA
        console.print("Step 3: Applying LoRA...")
        console.print(f"  LoRA rank: [cyan]{training_config['lora']['r']}[/cyan]")
        console.print(f"  Target modules: [cyan]{', '.join(training_config['lora']['target_modules'])}[/cyan]")

        model = loader.apply_lora(model)
        console.print("  [green]✓[/green] LoRA applied\n")

        # Train
        console.print("Step 4: Fine-tuning...")
        console.print("[dim]This may take a while depending on your hardware...[/dim]\n")

        trainer = ModelTrainer(model, tokenizer, training_config)

        if resume:
            console.print(f"  Resuming from checkpoint: {resume}\n")

        # Start training
        console.print(Panel.fit(
            "[bold green]Training Started[/bold green]\n\n"
            "Monitor progress in the logs below.\n"
            "Checkpoints will be saved periodically.",
            border_style="green"
        ))

        trainer.train(train_dataset, val_dataset, resume_from_checkpoint=resume)

        # Success
        console.print("\n")
        console.print(Panel.fit(
            f"[bold green]✓ Training Complete![/bold green]\n\n"
            f"Model saved to: [cyan]{training_config['output_dir']}[/cyan]\n\n"
            "[bold]Next steps:[/bold]\n"
            "1. nanodex export          # Export to GGUF/ONNX\n"
            "2. nanodex chat            # Chat with your model",
            title="Success",
            border_style="green"
        ))

    except Exception as e:
        console.print(f"\n[bold red]Training failed:[/bold red] {e}")
        import traceback
        console.print(f"\n[dim]{traceback.format_exc()}[/dim]")
        raise click.Abort()
