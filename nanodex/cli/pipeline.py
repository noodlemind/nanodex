"""
Pipeline command for guided workflow execution.

Provides guidance through the nanodex pipeline with helpful tips
and command suggestions at each step.
"""

import click
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from ..utils.pipeline_state import PipelineState

console = Console()


@click.group("pipeline")
def pipeline_cmd():
    """
    Pipeline workflow helpers

    Provides guidance and automation for executing the full nanodex pipeline
    from initialization through training and deployment.

    \b
    Examples:
        nanodex pipeline guide    # Show pipeline guide
        nanodex pipeline check    # Check current progress
    """
    pass


@pipeline_cmd.command("guide")
def pipeline_guide():
    """
    Show complete pipeline guide

    Displays a step-by-step guide for the complete nanodex pipeline
    with commands and explanations for each step.
    """
    console.print()
    console.print(
        Panel.fit("[bold cyan]🔄 nanodex Pipeline Guide[/bold cyan]", border_style="cyan")
    )
    console.print()

    guide_text = """
# Complete nanodex Pipeline

## Phase 1: Setup & Configuration
```bash
# 1. Initialize configuration
nanodex init

# This creates config.yaml with your project settings
# Configure your codebase path and model preferences
```

## Phase 2: Analysis
```bash
# 2. Analyze your codebase
nanodex analyze

# Understand code structure, complexity, and patterns
# Helps determine optimal training parameters
```

## Phase 3: Data Generation
```bash
# 3. Generate training data
nanodex data generate --mode free

# Modes:
#   --mode free     : Extract from codebase (no API costs)
#   --mode hybrid   : Mix codebase + synthetic examples
#   --mode full     : API-powered with OpenAI/Claude

# Preview generated data
nanodex data preview
nanodex data stats
```

## Phase 4: Model Training
```bash
# 4. Train your model
nanodex train --epochs 3

# This is the longest step (2-8 hours depending on hardware)
# Uses LoRA fine-tuning with 4-bit quantization for efficiency
# Progress is saved in checkpoints

# Optional: Resume from checkpoint
nanodex train --resume
```

## Phase 5: RAG Index (Optional but Recommended)
```bash
# 5. Build RAG index for semantic search
nanodex rag index

# Enables context-aware responses by searching your codebase
# Greatly improves answer quality

# Test semantic search
nanodex rag search "authentication logic"
```

## Phase 6: Interactive Use
```bash
# 6. Chat with your assistant
nanodex chat

# Or use shell mode for iterative workflows
nanodex shell
```

---

## 💡 Pro Tips

- **Use shell mode** for faster iteration (resources stay loaded)
- **Check progress** anytime with `nanodex status`
- **Start small** with free mode data generation to test the pipeline
- **Monitor resources** during training (GPU/RAM usage)
- **Save checkpoints** regularly during long training runs

## 🔍 Troubleshooting

- **Out of memory?** Reduce batch size in config.yaml
- **Training too slow?** Check GPU utilization with `nvidia-smi`
- **Poor quality responses?** Generate more training data or build RAG index
- **Config errors?** Run `nanodex config-validate` to check

## 📖 Documentation

Full documentation: https://github.com/noodlemind/nanodex/tree/main/docs
"""

    console.print(Markdown(guide_text))
    console.print()


@pipeline_cmd.command("check")
def pipeline_check():
    """
    Check pipeline progress

    Shows current progress and suggests the next step.
    (This is an alias for `nanodex status`)
    """
    # Import and call status command
    from .status import status_cmd

    ctx = click.get_current_context()
    ctx.invoke(status_cmd)
