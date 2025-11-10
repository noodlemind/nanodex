# CLI Implementation Plan

## Overview

Build a professional CLI experience similar to GitHub Copilot CLI, Claude Code, and Amp Code that makes Turbo-Code-GPT easy and delightful to use.

---

## CLI Framework Choice

**Recommended: Click + Rich + Questionary**

```python
# Stack:
click          # Command structure and argument parsing
rich           # Beautiful terminal output, tables, progress bars
questionary    # Interactive prompts and wizards
typer          # Alternative: Type-hint based CLI (optional)
```

**Why this stack:**
- Click: Industry standard, powerful, well-documented
- Rich: Beautiful output that users love
- Questionary: Best interactive prompts with great UX

---

## Implementation Phases

### Phase 1: CLI Foundation (Add to existing Phase 1)
**Time: 6-8 hours**

**New files:**
- `turbo_code_gpt/cli/__init__.py`
- `turbo_code_gpt/cli/main.py` - Main CLI entry point
- `turbo_code_gpt/cli/utils.py` - Shared CLI utilities
- `turbo_code_gpt/cli/console.py` - Rich console setup
- `turbo_code_gpt/__main__.py` - Package entry point

**Commands to implement:**
- `turbo-code-gpt --help` - Show help
- `turbo-code-gpt --version` - Show version
- `turbo-code-gpt config show` - Display config
- `turbo-code-gpt config validate` - Validate config

**Deliverable:**
- Working CLI framework
- Beautiful help output
- Config commands working

---

### Phase 2: Interactive Setup (Add to existing Phase 2)
**Time: 8-10 hours**

**New files:**
- `turbo_code_gpt/cli/commands/init.py` - Interactive wizard
- `turbo_code_gpt/cli/commands/config.py` - Config management
- `turbo_code_gpt/cli/wizards/setup_wizard.py` - Setup flow
- `turbo_code_gpt/cli/wizards/mode_selector.py` - Mode selection

**Commands to implement:**
- `turbo-code-gpt init` - Full interactive setup
  - Codebase path selection
  - Mode selection (free/hybrid/full/rag-only)
  - Model selection with recommendations
  - API key configuration
  - Budget setup
  - Hardware detection
  - Config generation

- `turbo-code-gpt config set <key> <value>` - Set config values
- `turbo-code-gpt config edit` - Open in editor
- `turbo-code-gpt models list` - List available models
- `turbo-code-gpt models info <model>` - Model details

**User Experience:**
```bash
$ turbo-code-gpt init

🚀 Welcome to Turbo-Code-GPT!

Let's set up your codebase-specific AI assistant.

? Where is your codebase? [.]: ./my-project
  ✓ Found 127 Python files

? Choose your training mode:
  ❯ 🆓 Free - Self-supervised only ($0 API, ~$10-30 GPU)
    💎 Hybrid - Best quality/cost (~$20-100 total) [Recommended]
    🚀 Full Training - Maximum quality (~$100-1000)
    🔍 RAG-only - No training, retrieval only ($0)

? Choose base model:
  ❯ DeepSeek Coder 6.7B [Recommended for code]
    CodeLlama 7B
    StarCoder2 7B
    Custom...

? Use 4-bit quantization? (Saves memory) [Y/n]:

💾 Creating configuration...
✓ Configuration saved to config.yaml

📝 Next steps:
  1. turbo-code-gpt analyze - Analyze your codebase
  2. turbo-code-gpt data generate - Generate training data
  3. turbo-code-gpt train - Start training

Ready to analyze your codebase? [Y/n]:
```

**Deliverable:**
- Zero-friction setup
- Smart defaults
- Interactive mode selection
- API key management

---

### Phase 3: Analysis & Data Commands (Add to existing Phase 2)
**Time: 6-8 hours**

**New files:**
- `turbo_code_gpt/cli/commands/analyze.py`
- `turbo_code_gpt/cli/commands/data.py`
- `turbo_code_gpt/cli/formatters/analysis_formatter.py`
- `turbo_code_gpt/cli/formatters/data_formatter.py`

**Commands:**
- `turbo-code-gpt analyze repo` - Full repository analysis
- `turbo-code-gpt analyze dependencies` - Dependency graph
- `turbo-code-gpt analyze stats` - Code statistics
- `turbo-code-gpt data generate` - Generate training data
- `turbo-code-gpt data preview` - Preview examples
- `turbo-code-gpt data stats` - Data statistics

**Output example:**
```bash
$ turbo-code-gpt analyze repo

Analyzing codebase... ━━━━━━━━━━━━━━━━━━ 127/127

📊 Repository Analysis

┏━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┓
┃ Language     ┃ Files  ┃ Lines    ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━┩
│ Python       │ 85     │ 14,234   │
│ JavaScript   │ 32     │ 3,102    │
│ TypeScript   │ 10     │ 1,120    │
└──────────────┴────────┴──────────┘

Quality:
  ├─ Docstrings: 234 found (67% coverage) ✓
  ├─ Type hints: 356 found (78% coverage) ✓
  ├─ Tests: 23 files (189 test cases) ✓
  └─ Complexity: 4.2 avg (Good)

Estimated training data:
  ├─ Free mode: ~2,500 examples
  └─ Hybrid mode: ~8,000 examples
```

**Deliverable:**
- Beautiful analysis output
- Progress indicators
- Rich tables and formatting
- Actionable insights

---

### Phase 4: Training Commands (Add to existing Phase 4)
**Time: 8-10 hours**

**New files:**
- `turbo_code_gpt/cli/commands/train.py`
- `turbo_code_gpt/cli/commands/evaluate.py`
- `turbo_code_gpt/cli/formatters/training_formatter.py`
- `turbo_code_gpt/cli/progress/training_progress.py`

**Commands:**
- `turbo-code-gpt train start` - Start training
- `turbo-code-gpt train resume` - Resume from checkpoint
- `turbo-code-gpt train status` - Show current status
- `turbo-code-gpt evaluate run` - Run evaluation
- `turbo-code-gpt evaluate report` - Generate report

**Features:**
- Real-time progress bars
- Loss tracking visualization
- ETA calculations
- Checkpoint management
- GPU utilization display

**Output example:**
```bash
$ turbo-code-gpt train start

🚀 Starting Training

Model: deepseek-coder-6.7b-base
Examples: 7,227 train / 804 validation

Loading model... ━━━━━━━━━━━━━━━━━ 100%

Epoch 1/3
  Step 500/1807 ━━━━━━━━━━━━━━━━━ 27% 0:15:32
  Loss: 0.234 → 0.189
  Speed: 2.3 steps/sec
  GPU: 12.4GB / 16GB (78%)
  ETA: 0:42:15

[Press Ctrl+C to pause, 's' to save checkpoint]
```

**Deliverable:**
- Professional training UI
- Real-time feedback
- Easy checkpoint management
- Training visualization

---

### Phase 5: Interactive Chat & Debug (New Phase)
**Time: 10-12 hours**

**New files:**
- `turbo_code_gpt/cli/commands/chat.py`
- `turbo_code_gpt/cli/commands/ask.py`
- `turbo_code_gpt/cli/commands/debug.py`
- `turbo_code_gpt/cli/chat/session.py`
- `turbo_code_gpt/cli/chat/rag_integration.py`

**Commands:**
- `turbo-code-gpt chat` - Interactive chat session
- `turbo-code-gpt ask "<question>"` - One-shot question
- `turbo-code-gpt debug locate "<error>"` - Find error source
- `turbo-code-gpt debug explain "<error>"` - Explain error

**Chat experience:**
```bash
$ turbo-code-gpt chat

Loading model... ✓
Loading RAG index... ✓

🤖 Turbo-Code-GPT (my-project)

Commands: /help /context /clear /exit

You: What does the ModelLoader class do?

🔍 Searching codebase...
  ├─ models/model_loader.py
  ├─ main.py
  └─ config.yaml

🤖 The ModelLoader class (models/model_loader.py:17) is responsible
   for loading and configuring HuggingFace models for fine-tuning.

   Key features:
   - Loads models with optional 4-bit/8-bit quantization
   - Applies LoRA adapters for efficient training
   - Handles model validation and error checking
   - Supports trust_remote_code for custom models

   It's used in main.py (line 102) during the training pipeline.

You: Show me an example of using it

🤖 Here's how it's used in main.py:

   ```python
   model_config = config.get_model_config()
   training_config = config.get_training_config()

   loader = ModelLoader(model_config, training_config)
   model, tokenizer = loader.load_huggingface_model()
   model = loader.apply_lora(model)
   ```

   Would you like me to explain any specific part?

You: /exit

👋 Goodbye! Your chat session has been saved.
```

**Debug experience:**
```bash
$ turbo-code-gpt debug locate "TypeError: 'NoneType' object is not subscriptable"

🔍 Analyzing error...

Likely locations:
  1. turbo_code_gpt/trainers/data_preparer.py:93
     └─ Missing null check on examples['input']

  2. turbo_code_gpt/analyzers/code_analyzer.py:76
     └─ File read returned None

  3. main.py:157
     └─ Config value not set

Suggested fixes:
  ├─ Check config has all required values
  ├─ Add null checks before subscripting
  └─ Validate dataset before training

Run with --verbose for detailed analysis
```

**Deliverable:**
- Interactive chat with context
- RAG-powered responses
- Code-aware debugging
- Error localization

---

### Phase 6: Export & Serve (Future)
**Time: 6-8 hours**

**Commands:**
- `turbo-code-gpt export gguf` - Export to GGUF
- `turbo-code-gpt export onnx` - Export to ONNX
- `turbo-code-gpt serve` - Start API server (future)
- `turbo-code-gpt serve status` - Check server status

---

## Technical Implementation Details

### 1. CLI Structure

```python
# turbo_code_gpt/cli/main.py

import click
from rich.console import Console
from turbo_code_gpt import __version__

console = Console()

@click.group()
@click.version_option(version=__version__)
@click.option('--config', default='config.yaml', help='Config file path')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.pass_context
def cli(ctx, config, verbose):
    """Turbo-Code-GPT: Fine-tune models on your codebase."""
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = config
    ctx.obj['verbose'] = verbose

# Register command groups
cli.add_command(init_command)
cli.add_command(config_group)
cli.add_command(models_group)
cli.add_command(analyze_group)
cli.add_command(data_group)
cli.add_command(train_group)
cli.add_command(evaluate_group)
cli.add_command(chat_command)
cli.add_command(ask_command)
cli.add_command(debug_group)
cli.add_command(export_group)

if __name__ == '__main__':
    cli()
```

### 2. Console Utilities

```python
# turbo_code_gpt/cli/console.py

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

console = Console()

def print_header(title: str):
    """Print a beautiful header."""
    console.print(Panel(f"[bold blue]{title}[/bold blue]"))

def print_success(message: str):
    """Print success message."""
    console.print(f"[green]✓[/green] {message}")

def print_error(message: str):
    """Print error message."""
    console.print(f"[red]✗[/red] {message}")

def print_warning(message: str):
    """Print warning."""
    console.print(f"[yellow]⚠️[/yellow]  {message}")

def create_table(title: str, columns: list) -> Table:
    """Create a formatted table."""
    table = Table(title=title, show_header=True, header_style="bold magenta")
    for col in columns:
        table.add_column(col)
    return table

def create_progress() -> Progress:
    """Create a progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    )
```

### 3. Interactive Wizards

```python
# turbo_code_gpt/cli/wizards/setup_wizard.py

import questionary
from pathlib import Path

def run_setup_wizard():
    """Run interactive setup wizard."""

    print_header("🚀 Welcome to Turbo-Code-GPT!")

    # Codebase path
    repo_path = questionary.path(
        "Where is your codebase?",
        default=".",
        only_directories=True
    ).ask()

    # Mode selection
    mode = questionary.select(
        "Choose your training mode:",
        choices=[
            questionary.Choice(
                title="🆓 Free - Self-supervised only ($0 API, ~$10-30 GPU)",
                value="free"
            ),
            questionary.Choice(
                title="💎 Hybrid - Best quality/cost (~$20-100) [Recommended]",
                value="hybrid"
            ),
            questionary.Choice(
                title="🚀 Full Training - Maximum quality (~$100-1000)",
                value="full"
            ),
            questionary.Choice(
                title="🔍 RAG-only - No training ($0)",
                value="rag-only"
            ),
        ]
    ).ask()

    # API configuration (if hybrid/full)
    if mode in ['hybrid', 'full']:
        api_provider = questionary.select(
            "API Provider:",
            choices=["OpenAI (GPT-4)", "Anthropic (Claude)", "Together AI"]
        ).ask()

        api_key = questionary.password("API Key:").ask()

        max_cost = questionary.text(
            "Maximum API spending ($):",
            default="50",
            validate=lambda x: x.isdigit() and int(x) > 0
        ).ask()

    # Model selection
    model = questionary.select(
        "Choose base model:",
        choices=[
            "DeepSeek Coder 6.7B [Recommended]",
            "CodeLlama 7B",
            "StarCoder2 7B",
            "Custom..."
        ]
    ).ask()

    # Hardware config
    use_4bit = questionary.confirm(
        "Use 4-bit quantization? (Saves memory)",
        default=True
    ).ask()

    return {
        'repo_path': repo_path,
        'mode': mode,
        'model': model,
        'use_4bit': use_4bit,
        # ... more config
    }
```

### 4. Update setup.py

```python
# setup.py

entry_points={
    "console_scripts": [
        "turbo-code-gpt=turbo_code_gpt.cli.main:cli",
    ],
}
```

### 5. Dependencies to Add

```bash
# CLI dependencies
click>=8.1.0
rich>=13.0.0
questionary>=2.0.0
prompt-toolkit>=3.0.0

# Optional: For advanced features
typer>=0.9.0  # Alternative CLI framework
textual>=0.40.0  # For TUI (future)
```

---

## Updated Phase Breakdown

### Phase 1 (Foundation) - NOW INCLUDES CLI
**Time: 22-32 hours (was 16-24)**

**Added tasks:**
1.1.5: Create CLI foundation
1.1.6: Implement basic commands (config, version)
1.1.7: Setup Rich console utilities
1.1.8: Update setup.py entry point

### Phase 2 (Data Generation) - NOW INCLUDES SETUP WIZARD
**Time: 40-54 hours (was 26-36)**

**Added tasks:**
2.4: Interactive setup wizard (`turbo-code-gpt init`)
2.5: Model selection and management
2.6: Analysis commands with beautiful output
2.7: Data generation commands with progress

### Phase 5 (Integration) - NOW INCLUDES CHAT/DEBUG
**Time: 42-54 hours (was 32-42)**

**Added tasks:**
5.6: Interactive chat command
5.7: One-shot ask command
5.8: Debug localization commands
5.9: Export commands

---

## Success Metrics

### Phase 1 Success:
✅ `turbo-code-gpt --help` shows beautiful help
✅ `turbo-code-gpt config show` displays formatted config
✅ CLI installs correctly via pip

### Phase 2 Success:
✅ `turbo-code-gpt init` guides user through setup
✅ Config generated automatically with defaults
✅ Mode selection is intuitive
✅ API keys stored securely

### Phase 5 Success:
✅ `turbo-code-gpt chat` provides interactive experience
✅ `turbo-code-gpt debug locate` finds errors accurately
✅ RAG integration works seamlessly
✅ Beautiful, professional output throughout

---

## User Flow: Complete Example

```bash
# 1. Install
pip install turbo-code-gpt

# 2. Interactive setup (first time)
turbo-code-gpt init
# Guides through: path, mode, model, API keys, etc.

# 3. Optional: Review config
turbo-code-gpt config show

# 4. Analyze codebase
turbo-code-gpt analyze repo

# 5. Generate training data
turbo-code-gpt data generate
# Shows progress, cost tracking, examples generated

# 6. Train model
turbo-code-gpt train start
# Real-time progress, loss tracking, ETA

# 7. Evaluate
turbo-code-gpt evaluate run

# 8. Use it!
turbo-code-gpt chat
# OR
turbo-code-gpt ask "How does authentication work?"
# OR
turbo-code-gpt debug locate "AttributeError in line 42"
```

---

## CLI vs Library Usage

The CLI is the primary interface, but also support library usage:

```python
# Library usage (for advanced users)
from turbo_code_gpt import TurboCodeGPT

tcg = TurboCodeGPT(config_path="config.yaml")
tcg.analyze()
tcg.generate_data()
tcg.train()
tcg.chat("What does ModelLoader do?")
```

Both should work seamlessly!

---

## Next Steps

1. Review this CLI design
2. Approve additions to implementation phases
3. Start implementing Phase 1 with CLI foundation
4. Test each command as we build

This CLI will make Turbo-Code-GPT a joy to use! 🚀
