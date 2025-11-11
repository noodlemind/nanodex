# Getting Started with nanodex

This guide will help you set up nanodex and train your first model.

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- At least 16GB RAM
- 50GB+ free disk space

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/noodlemind/nanodex.git
cd nanodex
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install package with all dependencies
pip install -e .

# Or install with dev dependencies for development
pip install -e ".[dev]"
```

### 4. (Optional) Install Ollama

For local model inference with Ollama:

```bash
# Linux/Mac
curl -fsSL https://ollama.com/install.sh | sh

# Or visit https://ollama.com for other installation methods
```

## Your First Model

### Step 1: Configure Your Project

Edit `config.yaml` to point to your codebase:

```yaml
model_source: "huggingface"

model:
  huggingface:
    model_name: "deepseek-ai/deepseek-coder-6.7b-base"
    use_4bit: true

repository:
  path: "/path/to/your/codebase"  # Change this!
  include_extensions:
    - ".py"
    - ".js"
    - ".ts"
    # Add more as needed

training:
  num_epochs: 3
  batch_size: 4
```

### Step 2: Analyze Your Codebase

Run a quick analysis to verify everything is configured correctly:

```bash
nanodex analyze
```

**Expected output:**
```
Found 150 files
- Python: 100 files, 15,000 lines
- JavaScript: 50 files, 8,000 lines
```

### Step 3: Train the Model

Start the full training pipeline:

```bash
nanodex train
```

**This will:**
1. Analyze your codebase
2. Create training examples
3. Download the base model
4. Fine-tune with LoRA
5. Save to `./models/fine-tuned/`

**Time:** 2-8 hours depending on your hardware and codebase size.

### Step 4: Test Your Model

```bash
nanodex chat
```

Try asking questions like:
- "What does this codebase do?"
- "Explain the main components"
- "How does [specific feature] work?"

## What's Next?

### Learn How It Works
Read [How It Works](how-it-works.md) to understand the training process.

### Understand Training vs Deployment
**Critical:** Read [Training vs Deployment](training-vs-deployment.md) to understand how the model learns during training and works standalone after deployment.

### Configure Advanced Options
See [Configuration Reference](guides/configuration.md) for all available options.

### Deploy Your Model
Follow the [Deployment Guide](guides/deployment.md) to use your model in production.

## Quick Troubleshooting

### "CUDA out of memory"
```yaml
# In config.yaml, reduce memory usage:
training:
  batch_size: 2  # Reduce from 4
  max_seq_length: 1024  # Reduce from 2048
```

### "No code files found"
- Verify `repository.path` is correct
- Check `include_extensions` matches your files
- Ensure directories aren't excluded

### "Model download failed"
- Check your internet connection
- Some models require HuggingFace approval
- Try a different model

### Training is Very Slow
```bash
# Check if CUDA is available:
python -c "import torch; print(torch.cuda.is_available())"

# Check GPU utilization:
nvidia-smi
```

For more help, see the [Troubleshooting Guide](reference/troubleshooting.md).

## Next Steps

- **Training Guide**: [Complete Training Guide](guides/training.md)
- **Deploy Your Model**: [Deployment Guide](guides/deployment.md)
- **Build a Chatbot**: See `examples/` directory
- **Debug Errors**: [Error Debugging Guide](guides/debugging.md)
