# nanodex

**Fine-tune coding models on your codebase with RAG support.**

Train custom AI models that understand your specific codebase. Create specialized assistants that can answer questions about your code, help with debugging, and explain your architecture.

**Inspired by nanoGPT and nanochat** - bringing the "nano" philosophy to code understanding: simple, hackable, and efficient.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 What Is This?

nanodex fine-tunes large language models (like DeepSeek Coder, CodeLlama) on your codebase to create AI assistants that know YOUR code.

**Key Insight:** The model learns during training and works standalone after deployment - no code access needed!

📖 **[Read the Full Documentation →](docs/README.md)**

## ✨ Features

- 🚀 **Modern CLI** - Beautiful Click-based interface with Rich formatting
- 🎯 **State-of-the-Art Models** - DeepSeek Coder, CodeLlama, StarCoder2 support
- 💾 **Memory Efficient** - LoRA fine-tuning with 4-bit quantization
- 🔍 **RAG Infrastructure** - Semantic code search with FAISS indexing
- 💬 **Interactive Chat** - Conversational interface with context-aware responses
- 🤖 **Production Ready** - Deploy standalone models without code access
- 🐛 **Debug Assistant** - Models help identify error sources in your code
- 📊 **Comprehensive Evaluation** - Multiple metrics (BLEU, F1, exact match)

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/noodlemind/nanodex.git
cd nanodex

# Install package with all dependencies
pip install -e .

# Or install with dev dependencies for development
pip install -e ".[dev]"
```

### 2. Configure

Edit `config.yaml`:

```yaml
model_source: "huggingface"
repository:
  path: "/path/to/your/codebase"  # Your code here
  include_extensions:
    - ".py"
    - ".js"
```

### 3. Generate Data & Train

```bash
# Interactive setup wizard
nanodex init

# Generate training data (free mode - no API costs)
nanodex data generate --mode free

# Build RAG index for semantic search
nanodex rag index

# Train your model
nanodex train
```

**Time:** 2-8 hours depending on your codebase size and hardware.

### 4. Use Your Model

```bash
# Interactive chat
nanodex chat

# Semantic code search
nanodex rag search "authentication logic"

# Or use programmatically
python examples/inference_example.py
```

Ask questions like:
- "How does the authentication system work?"
- "Which module handles payment processing?"
- "What causes this error?"

## 📖 Documentation

**Complete documentation is available in the [`docs/`](docs/) directory:**

- **[Getting Started](docs/getting-started.md)** - Installation and first model
- **[How It Works](docs/how-it-works.md)** - Understanding the training pipeline
- **[Training vs Deployment](docs/training-vs-deployment.md)** - ⭐ Critical concept!
- **[Architecture](docs/architecture.md)** - System design and components

### User Guides
- **[Training Guide](docs/guides/training.md)** - Complete training reference
- **[Deployment Guide](docs/guides/deployment.md)** - Deploy your models
- **[Error Debugging](docs/guides/debugging.md)** - Use models for debugging
- **[Configuration Reference](docs/guides/configuration.md)** - All config options

### Reference
- **[Troubleshooting](docs/reference/troubleshooting.md)** - Common issues and solutions
- **[API Reference](docs/reference/api.md)** - Python API
- **[CLI Reference](docs/reference/cli.md)** - Command-line options

## 💡 Use Cases

- **Codebase Chatbots** - Answer questions about your specific code
- **Onboarding Tool** - Help new developers understand the codebase
- **Debug Assistant** - Identify which modules cause specific errors
- **Documentation** - Automated code explanation and documentation
- **Architecture Questions** - Explain how systems and modules interact
- **Semantic Search** - Find code by meaning, not just keywords

## 🎯 Example

After training on your codebase:

**You ask:** "I'm seeing 'AttributeError: repository is None' - which module?"

**Model responds:** "This error is in auth/manager.py. The UserManager class expects a repository to be injected during initialization. Check where UserManager is instantiated and ensure the repository dependency is provided."

The model knows YOUR codebase!

## 🛠️ Key Commands

```bash
# Setup & Configuration
nanodex init              # Interactive setup wizard
nanodex analyze           # Analyze your codebase

# Data Generation
nanodex data generate     # Generate training data
nanodex data stats        # Show dataset statistics

# Training
nanodex train             # Train model
nanodex train --resume    # Resume from checkpoint

# RAG & Search
nanodex rag index         # Build semantic search index
nanodex rag search QUERY  # Search for code
nanodex rag query "..."   # Ask questions

# Interactive Chat
nanodex chat              # Start chat session
nanodex chat --model PATH # Chat with specific model
```

## 📊 Data Generation Modes

- **Free Mode** ($0) - Extract from codebase only, no API calls
- **Hybrid Mode** (Low cost) - Mix codebase + synthetic examples
- **Full Mode** (Higher quality) - API-powered with OpenAI/Claude

## 🔧 Hardware Requirements

### Minimum (with 4-bit quantization)
- 16GB RAM
- 8GB GPU VRAM (NVIDIA with CUDA)
- 50GB disk space

### Recommended
- 32GB RAM
- 16GB+ GPU VRAM (RTX 3090, A100)
- 100GB NVMe SSD

**CPU-only mode is supported** but will be slower.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional data generation strategies
- More evaluation metrics
- UI/web interface
- Additional model support
- Performance optimizations

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

Built with ❤️ by developers, for developers.

- **Inspiration**: nanoGPT, nanochat - for the "nano" philosophy
- **HuggingFace** - Transformers library and model hub
- **DeepSeek, Meta, BigCode** - Open-source coding models
- **Click, Rich** - Excellent Python CLI libraries
- **FAISS** - Efficient vector similarity search

---

**[→ Get Started with the Full Documentation](docs/README.md)**
