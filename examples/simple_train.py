#!/usr/bin/env python3
"""
Simple Training Example - Educational Walkthrough

This example demonstrates the complete process of fine-tuning a coding model
on your codebase using nanodex. Perfect for learning how everything works!

What you'll learn:
1. How to load and quantize a model (4-bit compression)
2. How to apply LoRA adapters (train only 0.1% of parameters)
3. How to prepare training data from your code
4. How to train and save the fine-tuned model

Requirements:
- 16GB RAM minimum (8GB GPU VRAM recommended)
- 50GB disk space
- CUDA-capable GPU (or CPU with patience!)
"""

import logging
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table

# Configure logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

console = Console()


def main():
    """
    Complete training workflow from start to finish.

    Steps:
    1. Configuration: Define what to train and how
    2. Model Loading: Load base model with 4-bit quantization
    3. LoRA Application: Add trainable adapter layers
    4. Data Preparation: Create training examples from code
    5. Training: Fine-tune the model on your code
    6. Saving: Export the trained adapters
    """

    console.print(Panel.fit(
        "[bold cyan]nanodex Simple Training Example[/bold cyan]\n\n"
        "This example walks through the complete fine-tuning process.\n"
        "Educational comments explain each step.\n\n"
        "[dim]Press Ctrl+C to stop at any time[/dim]",
        border_style="cyan"
    ))
    console.print()

    # ============================================================================
    # STEP 1: Configuration
    # ============================================================================
    console.print("[bold]Step 1: Configuration[/bold]\n")
    console.print("💡 Configuration defines what model to use and how to train it\n")

    config = {
        # Model configuration
        "model": {
            "huggingface": {
                "model_name": "deepseek-ai/deepseek-coder-1.3b-base",  # Smaller model for example
                "use_4bit": True,  # 75% memory reduction
                "use_8bit": False,
                "trust_remote_code": False,
            },
        },
        "model_source": "huggingface",

        # Repository to learn from
        "repository": {
            "path": ".",  # Current directory
            "include_extensions": [".py"],  # Python files only for this example
            "exclude_dirs": ["venv", ".git", "__pycache__", "build", "dist"],
            "max_file_size": 1048576,  # 1MB max per file
            "deep_parsing": {
                "enabled": True,  # Extract functions and classes
                "extract_functions": True,
                "extract_classes": True,
                "extract_docstrings": True,
            }
        },

        # Training configuration
        "training": {
            "output_dir": "./models/simple-example",
            "num_epochs": 1,  # Quick example - use 3-5 for real training
            "batch_size": 2,  # Small batch for low memory
            "learning_rate": 2e-5,
            "max_seq_length": 1024,  # Shorter sequences = less memory
            "gradient_accumulation_steps": 4,  # Simulate larger batch
            "warmup_steps": 10,
            "logging_steps": 5,
            "save_steps": 100,

            # LoRA configuration
            "lora": {
                "r": 8,  # Smaller rank = fewer parameters (faster, less memory)
                "lora_alpha": 16,  # Scaling factor
                "lora_dropout": 0.05,
                "target_modules": ["q_proj", "v_proj"],  # Just attention layers
            }
        },

        # Data configuration
        "data": {
            "output_dir": "./data/simple-example",
            "train_split": 0.9,
            "validation_split": 0.1,
            "context_window": 1024,
            "random_seed": 42,
        },

        "mode": "free",  # Free mode: no API costs!
    }

    console.print(f"✓ Using model: [cyan]{config['model']['huggingface']['model_name']}[/cyan]")
    console.print(f"✓ Training on: [cyan]{config['repository']['path']}[/cyan]")
    console.print(f"✓ Mode: [cyan]{config['mode']}[/cyan] (no API costs)\n")

    # ============================================================================
    # STEP 2: Analyze Codebase
    # ============================================================================
    console.print("[bold]Step 2: Analyzing Codebase[/bold]\n")
    console.print("💡 Analysis extracts code samples and metadata from your repository\n")

    from nanodex.analyzers import CodeAnalyzer

    analyzer = CodeAnalyzer(config['repository'])
    console.print("Scanning files...")
    code_samples = analyzer.analyze()
    stats = analyzer.get_statistics(code_samples)

    console.print(f"✓ Found {stats['total_files']} files")
    console.print(f"✓ Total lines: {stats['total_lines']:,}")
    console.print(f"✓ Average lines/file: {stats['avg_lines_per_file']:.1f}\n")

    if stats['total_files'] == 0:
        console.print("[red]No files found! Check your repository path and extensions.[/red]")
        return

    # ============================================================================
    # STEP 3: Generate Training Data
    # ============================================================================
    console.print("[bold]Step 3: Generating Training Data[/bold]\n")
    console.print("💡 Convert code samples into instruction-response pairs\n")
    console.print("Format: {instruction: ..., input: ..., output: ...}\n")

    from nanodex.data_generators.orchestrator import DataGenerationOrchestrator

    orchestrator = DataGenerationOrchestrator(config)
    console.print(f"Mode: [cyan]{orchestrator.mode}[/cyan]")
    console.print("Generating examples...")

    training_examples = orchestrator.generate_from_codebase(
        code_samples[:20],  # Limit to 20 files for quick example
        show_progress=True
    )

    console.print(f"✓ Generated {len(training_examples)} training examples\n")

    if len(training_examples) == 0:
        console.print("[red]No training examples generated! Check your codebase.[/red]")
        return

    # Show example
    if training_examples:
        example = training_examples[0]
        console.print("[bold]Example Training Data:[/bold]")
        console.print(f"Instruction: {example['instruction'][:100]}...")
        console.print(f"Input: {example['input'][:100]}...")
        console.print(f"Output: {example['output'][:100]}...\n")

    # ============================================================================
    # STEP 4: Prepare Dataset
    # ============================================================================
    console.print("[bold]Step 4: Preparing Dataset[/bold]\n")
    console.print("💡 Convert examples to HuggingFace Dataset format\n")

    from nanodex.trainers import DataPreparer

    preparer = DataPreparer(config['data'])
    console.print("Splitting into train/validation...")
    train_dataset, val_dataset = preparer.prepare_datasets(training_examples)

    console.print(f"✓ Training examples: {len(train_dataset)}")
    console.print(f"✓ Validation examples: {len(val_dataset)}\n")

    # ============================================================================
    # STEP 5: Load Model
    # ============================================================================
    console.print("[bold]Step 5: Loading Model[/bold]\n")
    console.print("💡 Load model with 4-bit quantization to save memory\n")
    console.print("This will download ~500MB on first run\n")

    from nanodex.models import ModelLoader

    loader = ModelLoader(
        model_config=config['model']['huggingface'],
        training_config=config['training']
    )

    console.print("Loading model and tokenizer...")
    console.print("[dim]This may take a few minutes on first run...[/dim]\n")

    model, tokenizer = loader.load_huggingface_model()
    console.print("✓ Model loaded\n")

    # ============================================================================
    # STEP 6: Apply LoRA
    # ============================================================================
    console.print("[bold]Step 6: Applying LoRA Adapters[/bold]\n")
    console.print("💡 LoRA adds small trainable layers instead of training entire model\n")
    console.print("Result: Train 0.1% of parameters instead of 100%!\n")

    model = loader.apply_lora(model)
    console.print("✓ LoRA applied\n")

    # Show trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    console.print(f"Trainable: {trainable:,} parameters ({100*trainable/total:.2f}%)")
    console.print(f"Total: {total:,} parameters\n")

    # ============================================================================
    # STEP 7: Train Model
    # ============================================================================
    console.print("[bold]Step 7: Training Model[/bold]\n")
    console.print("💡 Fine-tune the model on your code patterns\n")
    console.print("This is where the learning happens!\n")

    from nanodex.trainers import ModelTrainer

    trainer_obj = ModelTrainer(model, tokenizer, config['training'])

    console.print("Starting training...")
    console.print("[dim]You'll see loss decrease as the model learns[/dim]\n")
    console.print("=" * 60)

    trainer = trainer_obj.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset
    )

    console.print("=" * 60)
    console.print()

    # ============================================================================
    # STEP 8: Save Model
    # ============================================================================
    console.print("[bold]Step 8: Saving Fine-Tuned Model[/bold]\n")
    console.print("💡 Save only the LoRA adapters (~50MB instead of 2GB+)\n")

    output_dir = Path(config['training']['output_dir'])
    console.print(f"✓ Model saved to: [cyan]{output_dir}[/cyan]\n")

    # ============================================================================
    # Summary
    # ============================================================================
    console.print(Panel.fit(
        "[bold green]✓ Training Complete![/bold green]\n\n"
        f"Fine-tuned model saved to: {output_dir}\n\n"
        "[bold]What's next?[/bold]\n"
        "1. Use the model: python -m nanodex chat\n"
        "2. Build RAG index: python -m nanodex rag index\n"
        "3. Export model: python -m nanodex export\n\n"
        "[dim]The model has learned patterns from your codebase!\n"
        "It can now answer questions about your code.[/dim]",
        border_style="green"
    ))
    console.print()

    # Show model size
    if output_dir.exists():
        size_mb = sum(f.stat().st_size for f in output_dir.rglob('*') if f.is_file()) / (1024*1024)
        console.print(f"Model size: [cyan]{size_mb:.1f}MB[/cyan]")
        console.print(f"[dim](Compare to {total / 1e9:.1f}B full parameters!)[/dim]\n")

    console.print("[bold]Key Takeaways:[/bold]")
    console.print("• 4-bit quantization: 75% memory savings")
    console.print("• LoRA: Train only 0.1% of parameters")
    console.print("• Free mode: No API costs")
    console.print("• Result: ~50MB fine-tuned model that understands your code\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Training stopped by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        logger.exception("Training failed")
        raise
