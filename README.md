# 🚀 Turbo Code GPT

A comprehensive system for fine-tuning open-source LLMs on your codebase with RAG (Retrieval-Augmented Generation) support. Create specialized AI coding assistants that understand your specific codebase architecture, patterns, and conventions.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Features

### 🎯 Core Capabilities
- **Multiple Training Modes**: Free (data-only), Hybrid (synthetic), Full (API-powered)
- **Modern CLI**: Beautiful Click-based interface with Rich formatting
- **RAG Infrastructure**: Semantic code search with FAISS vector indexing
- **Interactive Chat**: Conversational interface with context-aware responses
- **Production Training**: Checkpoint recovery, early stopping, best model selection
- **Comprehensive Evaluation**: Multiple metrics (BLEU, F1, exact match, edit distance)

### 🤖 Supported Models
- **HuggingFace**: DeepSeek Coder, CodeLlama, StarCoder2, CodeGen
- **Efficient Fine-tuning**: LoRA (Low-Rank Adaptation)
- **Quantization**: 4-bit/8-bit for consumer hardware

### 🔍 RAG Features
- **Semantic Search**: Find code by meaning, not just keywords
- **Smart Chunking**: Function/class/file-level code segmentation
- **Fast Retrieval**: FAISS-powered similarity search (<100ms)
- **Context Assembly**: Automatically retrieve relevant code for queries

### 💻 Developer Experience
- **Setup Wizard**: Interactive configuration with `turbo-code-gpt init`
- **Validation**: Pydantic-powered config validation
- **Rich Output**: Beautiful tables, progress bars, and panels
- **Session Persistence**: Save and resume chat conversations

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended, CPU-only supported)
- 16GB+ RAM (32GB recommended)

### Install from Source

```bash
# Clone repository
git clone https://github.com/noodlemind/turbo-code-gpt.git
cd turbo-code-gpt

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### Verify Installation

```bash
turbo-code-gpt --version
turbo-code-gpt --help
```

## 🚀 Quick Start

### 1. Initialize Configuration

Run the interactive setup wizard:

```bash
turbo-code-gpt init
```

This will guide you through:
- Selecting data generation mode (free/hybrid/full)
- Choosing a base model
- Configuring training parameters
- Setting repository paths and filters

### 2. Analyze Your Codebase

Analyze your codebase to understand its structure:

```bash
turbo-code-gpt analyze
```

This shows:
- Total files and lines
- Language distribution
- File size statistics
- Code complexity metrics

### 3. Generate Training Data

Generate training examples from your code:

```bash
# Free mode (codebase-only, no API calls)
turbo-code-gpt data generate --mode free

# Hybrid mode (mixed synthetic + codebase)
turbo-code-gpt data generate --mode hybrid --count 100

# Full mode (API-powered, requires OpenAI key)
turbo-code-gpt data generate --mode full --count 500
```

### 4. Build RAG Index

Create a semantic search index:

```bash
turbo-code-gpt rag index
```

This enables:
- Fast semantic code search
- Context-aware Q&A
- RAG-augmented generation

### 5. Train Your Model

Fine-tune the model on your codebase:

```bash
turbo-code-gpt train
```

Features:
- Automatic checkpoint recovery
- Early stopping
- Best model selection
- Training metadata export

### 6. Start Chatting!

Launch the interactive chat interface:

```bash
turbo-code-gpt chat
```

Or search your codebase semantically:

```bash
turbo-code-gpt rag search "authentication logic"
turbo-code-gpt rag query "How does error handling work?"
```

## 📖 Usage Guide

### Configuration

Your `config.yaml` configures everything. Example:

```yaml
# Model selection
model_source: "huggingface"

model:
  huggingface:
    model_name: "deepseek-ai/deepseek-coder-6.7b-base"
    use_4bit: true
    device: "auto"

# Data generation
data_generation:
  mode: "free"  # or "hybrid" or "full"
  synthetic_count: 0
  openai_api_key: ""  # Only for full mode

# Training configuration
training:
  num_epochs: 3
  batch_size: 4
  learning_rate: 2.0e-5
  enable_early_stopping: true
  early_stopping_patience: 3
  save_best_model: true

  lora:
    r: 16
    lora_alpha: 32
    lora_dropout: 0.05

# Repository configuration
repository:
  path: "."
  include_extensions:
    - ".py"
    - ".js"
    - ".ts"
  exclude_dirs:
    - "node_modules"
    - ".git"
    - "__pycache__"
```

### CLI Commands Reference

#### Configuration
```bash
turbo-code-gpt init              # Interactive setup wizard
turbo-code-gpt config-show       # Display current configuration
turbo-code-gpt config-validate   # Validate configuration file
```

#### Analysis & Data
```bash
turbo-code-gpt analyze           # Analyze codebase
turbo-code-gpt data generate     # Generate training data
turbo-code-gpt data stats        # Show dataset statistics
turbo-code-gpt data validate     # Validate training data
```

#### Training
```bash
turbo-code-gpt train             # Train model
turbo-code-gpt train --resume    # Resume from checkpoint
```

#### RAG (Retrieval-Augmented Generation)
```bash
turbo-code-gpt rag index         # Build semantic search index
turbo-code-gpt rag search QUERY  # Search for code
turbo-code-gpt rag query QUESTION # Ask questions
turbo-code-gpt rag stats         # Show index statistics
```

#### Chat
```bash
turbo-code-gpt chat              # Interactive chat
turbo-code-gpt chat --model PATH # Chat with specific model
turbo-code-gpt chat --session FILE # Resume session
```

### Data Generation Modes

#### Free Mode (No API Required)
- **Cost**: $0
- **Quality**: Basic
- **Use Case**: Testing, small codebases

```bash
turbo-code-gpt data generate --mode free
```

Generates training examples from:
- Function/class docstrings
- Code structure analysis
- Pattern matching

#### Hybrid Mode (Mixed)
- **Cost**: Low ($0.01-0.10 depending on count)
- **Quality**: Good
- **Use Case**: Medium codebases, budget-conscious

```bash
turbo-code-gpt data generate --mode hybrid --count 200
```

Combines:
- Free mode examples
- Synthetic examples (LLM-generated)

#### Full Mode (API-Powered)
- **Cost**: Higher ($0.10-1.00 depending on count)
- **Quality**: Best
- **Use Case**: Production, large codebases

```bash
turbo-code-gpt data generate --mode full --count 500
```

Generates high-quality examples using OpenAI API.

### RAG Search Examples

#### Semantic Search
Find code by meaning, not keywords:

```bash
# Find authentication code
turbo-code-gpt rag search "user login and authentication"

# Find error handling
turbo-code-gpt rag search "exception handling and logging"

# Find specific patterns
turbo-code-gpt rag search "database connection pooling"
```

#### Q&A
Ask natural language questions:

```bash
turbo-code-gpt rag query "How does the caching system work?"
turbo-code-gpt rag query "Where are API endpoints defined?"
turbo-code-gpt rag query "What libraries are used for testing?"
```

#### Filtered Search
```bash
# Search only functions
turbo-code-gpt rag search "parse JSON" --type function

# Search only Python code
turbo-code-gpt rag search "async processing" --language python

# Get more results
turbo-code-gpt rag search "database query" -k 10
```

### Chat Interface

The chat interface provides an interactive experience:

```bash
turbo-code-gpt chat
```

**Features:**
- Multi-turn conversations
- Conversation history
- RAG-powered context retrieval
- Session save/load

**Chat Commands:**
- `/help` - Show available commands
- `/history` - View conversation history
- `/clear` - Clear history
- `/stats` - Session statistics
- `/save` - Save session
- `/exit` - Quit chat

**Example Session:**
```
You: Explain how authentication works in this codebase