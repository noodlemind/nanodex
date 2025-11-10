# Getting Started with nanodex

This guide will walk you through setting up and using nanodex to fine-tune a model on your codebase.

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- At least 16GB RAM
- 50GB+ free disk space

## Step-by-Step Guide

### Step 1: Installation

```bash
# Clone the repository
git clone https://github.com/noodlemind/nanodex.git
cd nanodex

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Prepare Your Codebase

1. Identify the codebase you want to train on
2. Note the path to this codebase
3. Identify which file types are important (.py, .js, .ts, etc.)

### Step 3: Configure

Edit `config.yaml`:

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
    # Add others as needed
```

### Step 4: Analyze Your Codebase

First, do a dry run to see what will be analyzed:

```bash
python main.py --analyze-only
```

This will show you:
- Number of files found
- Total lines of code
- Breakdown by programming language

Review the output to ensure it's analyzing the right files.

### Step 5: Prepare Training Data

```bash
python main.py --prepare-only
```

This will:
1. Analyze your codebase
2. Create training examples
3. Save data to `./data/processed/`

Check the generated files in `./data/processed/train.json` to see sample training data.

### Step 6: Fine-tune the Model

```bash
python main.py
```

This will run the full pipeline:
1. Analyze code
2. Prepare data  
3. Download the base model
4. Fine-tune with LoRA
5. Save the fine-tuned model to `./models/fine-tuned/`

**Note**: This can take several hours depending on:
- Size of your codebase
- Number of training epochs
- Your hardware

Monitor the output for training progress.

### Step 7: Test Your Model

```bash
python examples/inference_example.py
```

This will:
- Load your fine-tuned model
- Run example queries
- Enter interactive mode for testing

Try asking questions like:
- "What does this codebase do?"
- "Explain the main components"
- "How does [specific feature] work?"

## Using with Ollama (Alternative)

If you prefer using Ollama for local inference:

### 1. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Update Configuration

Edit `config.yaml`:

```yaml
model_source: "ollama"

model:
  ollama:
    model_name: "deepseek-coder:6.7b"
    base_url: "http://localhost:11434"
```

### 3. Prepare Data

```bash
python main.py --prepare-only
```

### 4. Create Ollama Model

```bash
# Generate Modelfile
python examples/ollama_example.py > Modelfile

# Create custom model
ollama create my-code-expert -f Modelfile

# Test it
ollama run my-code-expert
```

## Advanced Configuration

### Memory Optimization

If you encounter out-of-memory errors:

```yaml
training:
  batch_size: 2  # Reduce from 4
  gradient_accumulation_steps: 8  # Increase from 4
  max_seq_length: 1024  # Reduce from 2048
```

### Training Quality

To improve model quality:

```yaml
training:
  num_epochs: 5  # Increase from 3
  learning_rate: 1.0e-5  # Reduce for more stable training
  
  lora:
    r: 32  # Increase from 16 for more capacity
    lora_alpha: 64  # Scale with r
```

### Multi-GPU Training

The code automatically uses all available GPUs via `device_map="auto"`. No additional configuration needed.

## Troubleshooting

### "CUDA out of memory"
- Reduce `batch_size`
- Enable 4-bit quantization
- Use smaller `max_seq_length`
- Close other GPU applications

### "Model download failed"
- Check internet connection
- Ensure you have HuggingFace access (some models require approval)
- Try a different model

### "No code files found"
- Check `repository.path` in config
- Verify `include_extensions` matches your files
- Check that directories aren't in `exclude_dirs`

### Training is very slow
- Ensure CUDA is properly installed: `python -c "import torch; print(torch.cuda.is_available())"`
- Check GPU utilization: `nvidia-smi`
- Consider using a smaller model for initial testing

## Next Steps

Once you have a fine-tuned model:

1. **Integrate with a chatbot framework**:
   - LangChain
   - LlamaIndex
   - Custom FastAPI/Flask server

2. **Export for production**:
   - Convert to GGUF for llama.cpp
   - Optimize with ONNX
   - Deploy with TensorRT

3. **Continuous improvement**:
   - Regularly retrain on updated code
   - Collect user feedback
   - Fine-tune on common questions

## Getting Help

- Check the main README.md for more details
- Review example scripts in `examples/`
- Check configuration comments in `config.yaml`
- Open an issue on GitHub
