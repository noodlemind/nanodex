# Turbo-Code-GPT CLI Design

## Overview

A professional, user-friendly CLI similar to GitHub Copilot CLI, Claude Code, and Amp Code that guides users through the entire workflow from setup to inference.

## User Experience Goals

1. **Zero-config start** - `nanodex init` sets everything up
2. **Interactive wizards** - Guide users through complex choices
3. **Smart defaults** - Works out-of-box for common cases
4. **Progressive disclosure** - Simple commands with advanced options
5. **Beautiful output** - Rich formatting, progress bars, colors
6. **Helpful errors** - Actionable error messages with suggestions

---

## CLI Command Structure

```bash
nanodex
├── init              # Initialize project with interactive wizard
├── config            # Manage configuration
│   ├── show          # Display current configuration
│   ├── edit          # Open config in editor
│   ├── set           # Set specific values
│   └── validate      # Validate configuration
├── models            # Manage models
│   ├── list          # List available base models
│   ├── download      # Download a base model
│   └── info          # Show model details
├── analyze           # Analyze codebase
│   ├── repo          # Analyze repository structure
│   ├── dependencies  # Show dependency graph
│   └── stats         # Show code statistics
├── data              # Data generation and management
│   ├── generate      # Generate training data
│   ├── preview       # Preview generated examples
│   └── stats         # Show data statistics
├── train             # Training commands
│   ├── start         # Start training
│   ├── resume        # Resume from checkpoint
│   └── status        # Show training status
├── evaluate          # Evaluation commands
│   ├── run           # Run evaluation
│   └── report        # Generate evaluation report
├── chat              # Interactive chat with model
├── ask               # One-shot question answering
├── debug             # Debug assistance
│   ├── locate        # Locate error in codebase
│   └── explain       # Explain error message
├── export            # Export model
│   └── gguf          # Export to GGUF format
└── serve             # Start API server (future)
```

---

## Detailed Command Specifications

### 1. `nanodex init`

**Purpose:** Interactive setup wizard for new projects

**Flow:**
```
🚀 Welcome to Turbo-Code-GPT!

Let's set up your codebase-specific AI assistant.

📁 Codebase Selection
? Where is your codebase? [.]: /path/to/my/project
✓ Found 127 files across 15 directories

🎯 Training Mode Selection
? Choose your training mode:
  ❯ Free - Self-supervised learning only ($0 API cost)
    Hybrid - Self-supervised + AI-generated examples (~$20-100)
    Full Training - Train from scratch (~$100-1000)
    RAG-only - No training, use retrieval only ($0)

[If Free selected:]
✓ Great! This will cost ~$10-30 in GPU time.

[If Hybrid selected:]
💰 Budget Configuration
? Maximum API spending: [$50]: 75
? API Provider:
  ❯ OpenAI (GPT-4)
    Anthropic (Claude)
    Together AI
    Anyscale

? API Key: [Enter or paste]: ****************

[If Full Training selected:]
⚠️  Full training is advanced and expensive.
? Are you sure? (y/N):

🤖 Model Selection
? Choose your base model:
  ❯ DeepSeek Coder 6.7B (Recommended for code)
    CodeLlama 7B
    StarCoder2 7B
    Custom (enter model name)

⚙️  Hardware Configuration
? Use 4-bit quantization? (Saves memory) [Y/n]:
? GPU memory available: [Auto-detect]
  ✓ Detected: 16GB (suitable for selected model)

📊 Data Generation Options
? Include test files in analysis? [Y/n]:
? Include git history? [Y/n]:
? Run static analysis (mypy/pylint)? [Y/n]:

💾 Creating configuration...
✓ Configuration saved to config.yaml

📝 Next Steps:
1. Review config: nanodex config show
2. Generate data: nanodex data generate
3. Start training: nanodex train start

Ready to continue? [Y/n]:
```

**Implementation:**
```python
@click.command()
@click.option('--non-interactive', is_flag=True, help='Skip interactive prompts')
@click.option('--mode', type=click.Choice(['free', 'hybrid', 'full', 'rag-only']))
@click.option('--repo-path', type=click.Path(exists=True))
def init(non_interactive, mode, repo_path):
    """Initialize Turbo-Code-GPT for your codebase."""

    if not non_interactive:
        console.print(Panel("[bold blue]🚀 Welcome to Turbo-Code-GPT![/bold blue]"))

        # Interactive wizard
        repo_path = questionary.path(
            "Where is your codebase?",
            default=".",
            only_directories=True
        ).ask()

        # ... more interactive prompts

    # Generate config
    config = generate_config(repo_path, mode, ...)

    # Save config
    save_config(config)

    # Show next steps
    show_next_steps()
```

---

### 2. `nanodex config`

**Subcommands:**

#### `config show`
```bash
$ nanodex config show

📋 Current Configuration

Training Mode: Hybrid
Repository: /home/user/my-project (127 files)
Base Model: deepseek-ai/deepseek-coder-6.7b-base

Data Generation:
  ├─ Self-supervised: ✓ Enabled
  │  ├─ Docstrings: ✓
  │  ├─ Fill-in-middle: ✓
  │  ├─ Tests: ✓
  │  └─ Git history: ✓
  └─ Synthetic Data: ✓ Enabled
     ├─ Provider: OpenAI (GPT-4o-mini)
     ├─ Budget: $50
     └─ API Key: ✓ Set

Training:
  ├─ Epochs: 3
  ├─ Batch size: 4
  ├─ Learning rate: 2e-5
  └─ Output: ./models/fine-tuned

RAG:
  ├─ Enabled: ✓
  ├─ Index: ./models/rag_index
  └─ Top-k: 5

Configuration file: ./config.yaml
```

#### `config set`
```bash
$ nanodex config set data_generation.synthetic_data.max_cost_usd 100
✓ Updated data_generation.synthetic_data.max_cost_usd = 100

$ nanodex config set model.huggingface.model_name "codellama/CodeLlama-7b-hf"
✓ Updated model.huggingface.model_name = "codellama/CodeLlama-7b-hf"
⚠️  You may need to re-download the model.
```

#### `config validate`
```bash
$ nanodex config validate
Validating configuration...

✓ Model configuration is valid
✓ Repository path exists
✓ API key is set (OpenAI)
✗ Training batch_size (4) may be too small for good convergence
⚠️  GPU memory (8GB) may be insufficient for this model

2 warnings, 0 errors
Overall: Valid with warnings
```

---

### 3. `nanodex models`

```bash
$ nanodex models list

Available Base Models:

Code Models (Recommended):
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Model                          ┃ Size    ┃ Min GPU    ┃ Best For     ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ deepseek-coder-6.7b (default)  │ 6.7B    │ 8GB (4-bit)│ Code, Debug  │
│ codellama-7b                   │ 7B      │ 8GB (4-bit)│ Code, QA     │
│ starcoder2-7b                  │ 7B      │ 8GB (4-bit)│ Code         │
│ deepseek-coder-33b             │ 33B     │ 24GB       │ Advanced     │
└────────────────────────────────┴─────────┴────────────┴──────────────┘

$ nanodex models info deepseek-coder-6.7b

DeepSeek Coder 6.7B Base

Model Details:
  Name: deepseek-ai/deepseek-coder-6.7b-base
  Parameters: 6.7 billion
  Context: 16,384 tokens
  License: DeepSeek License

Hardware Requirements:
  Full Precision (fp32): 27GB
  Half Precision (fp16): 14GB
  4-bit Quantization: 8GB ✓ Recommended

Training Details:
  Trained on: 2T tokens (87% code, 13% docs)
  Languages: Python, JavaScript, Java, C++, Go, Rust, and more

Best For:
  ✓ Code understanding
  ✓ Code completion
  ✓ Debugging assistance
  ✓ Code explanation

Download: nanodex models download deepseek-coder-6.7b
```

---

### 4. `nanodex analyze`

```bash
$ nanodex analyze repo

Analyzing codebase at /home/user/my-project...

Scanning files... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 127/127 100%

📊 Repository Analysis

Code Statistics:
  Total files: 127
  Total lines: 18,456
  Languages:
    ├─ Python: 85 files (14,234 lines)
    ├─ JavaScript: 32 files (3,102 lines)
    └─ TypeScript: 10 files (1,120 lines)

Structure:
  ├─ Functions: 456
  ├─ Classes: 89
  ├─ Test files: 23
  └─ Documentation: 67%

Quality Indicators:
  ├─ Avg complexity: 4.2 (Good)
  ├─ Type coverage: 78% (Good)
  └─ Test coverage: 65% (Fair)

Dependency Graph:
  ├─ Max depth: 5
  ├─ Circular deps: 0
  └─ Orphan files: 3

Training Data Potential:
  ├─ Docstrings: 234 (High quality)
  ├─ Type hints: 356 (Excellent)
  ├─ Test cases: 189 (Good)
  └─ Estimated examples: ~2,500

Estimated Training:
  ├─ Free mode: ~2,500 examples
  ├─ Hybrid mode: ~8,000 examples
  └─ Time: 2-4 hours

$ nanodex analyze dependencies

Dependency Graph for /home/user/my-project

Top-level modules:
  main.py
  ├─ utils/config.py
  ├─ analyzers/code_analyzer.py
  │  └─ analyzers/ast_parser.py
  └─ trainers/model_trainer.py
     ├─ models/model_loader.py
     └─ trainers/data_preparer.py

Circular dependencies: None found ✓

Most imported files (hub files):
  1. utils/config.py (imported by 12 files)
  2. models/model_loader.py (imported by 8 files)
  3. analyzers/code_analyzer.py (imported by 6 files)

Export graph: nanodex analyze dependencies --export deps.json
```

---

### 5. `nanodex data generate`

```bash
$ nanodex data generate

🔍 Analyzing codebase...
✓ Found 127 files

📊 Generating training data...

Self-supervised generation:
  ├─ Extracting docstrings... ━━━━━━━━━━━━━ 234 found
  ├─ Creating FIM tasks... ━━━━━━━━━━━━━━━ 1,245 tasks
  ├─ Analyzing tests... ━━━━━━━━━━━━━━━━━ 189 test cases
  ├─ Processing git history... ━━━━━━━━━━ 156 commits
  └─ Running static analysis... ━━━━━━━━━ 23 issues

Generated 2,847 examples (free mode) ✓

Synthetic data generation:
  Provider: OpenAI (gpt-4o-mini)
  Budget: $50.00

  Generating examples...
  ├─ Code explanations ━━━━━━━━━━━━━━━━ 127/127 ($12.34)
  ├─ Debugging scenarios ━━━━━━━━━━━━━━ 127/127 ($8.76)
  ├─ Relationship Q&A ━━━━━━━━━━━━━━━━━ 95/127 ($6.89)
  └─ Edge cases ━━━━━━━━━━━━━━━━━━━━━━━ 127/127 ($9.45)

  Generated 5,184 examples ($37.44 / $50.00) ✓

💾 Saving datasets...
✓ Train: 7,227 examples
✓ Validation: 804 examples

📊 Quality Report:
  ├─ Avg length: 412 tokens
  ├─ Unique examples: 98.7%
  ├─ With context: 89.3%
  └─ Quality score: 8.9/10

Preview: nanodex data preview
```

---

### 6. `nanodex train`

```bash
$ nanodex train start

🚀 Starting Training

Configuration:
  ├─ Mode: Hybrid (free + synthetic)
  ├─ Model: deepseek-coder-6.7b-base
  ├─ Training examples: 7,227
  ├─ Validation examples: 804
  └─ Estimated time: 3-4 hours

Loading model... ━━━━━━━━━━━━━━━━━━━━━━━ 100%
Applying LoRA adapters... ✓

Training:
  Epoch 1/3 ━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:45:23
  ├─ Loss: 0.234 → 0.156
  ├─ Learning rate: 2e-5
  └─ Checkpoint saved: ./models/fine-tuned/checkpoint-500

  Epoch 2/3 ━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:43:12
  ├─ Loss: 0.156 → 0.098
  ├─ Validation loss: 0.112
  └─ Checkpoint saved: ./models/fine-tuned/checkpoint-1000

  Epoch 3/3 ━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:44:56
  ├─ Loss: 0.098 → 0.067
  ├─ Validation loss: 0.089 ✓ Best!
  └─ Final checkpoint saved

✓ Training complete! (2:13:31)

📊 Training Summary:
  ├─ Final loss: 0.067
  ├─ Best validation loss: 0.089
  ├─ Checkpoints: 3
  └─ Model saved: ./models/fine-tuned

🔍 Building RAG index...
✓ Indexed 127 files

Next steps:
  1. Evaluate: nanodex evaluate run
  2. Try it: nanodex chat
```

---

### 7. `nanodex chat`

```bash
$ nanodex chat

Loading model... ✓
Loading RAG index... ✓

🤖 Turbo-Code-GPT (my-project)

Type your questions about the codebase. Commands:
  /help - Show help
  /context - Show current context
  /clear - Clear conversation
  /exit - Exit chat

You: What does the ModelLoader class do?

🔍 Retrieving relevant code...
  ├─ models/model_loader.py (relevance: 0.95)
  ├─ main.py (relevance: 0.78)
  └─ config.yaml (relevance: 0.65)