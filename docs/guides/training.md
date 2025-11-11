# Training Guide

Complete guide to training your fine-tuned model.

## Overview

Training involves three main steps:
1. Configure your project
2. Analyze your codebase
3. Run the training pipeline

## Configuration

### Basic Configuration

Edit `config.yaml`:

```yaml
# Choose model source
model_source: "huggingface"  # or "ollama"

# Model settings
model:
  huggingface:
    model_name: "deepseek-ai/deepseek-coder-6.7b-base"
    use_4bit: true  # Enable 4-bit quantization
    use_8bit: false

# Repository to analyze
repository:
  path: "/path/to/your/codebase"
  include_extensions:
    - ".py"
    - ".js"
    - ".ts"
  exclude_dirs:
    - "node_modules"
    - "__pycache__"
    - ".git"

# Training settings
training:
  num_epochs: 3
  batch_size: 4
  learning_rate: 2.0e-5
  max_seq_length: 2048
  output_dir: "./models/fine-tuned"
```

### Advanced Configuration

#### LoRA Settings

```yaml
training:
  lora:
    r: 16              # Rank (higher = more capacity)
    lora_alpha: 32     # Scaling factor (usually 2x rank)
    lora_dropout: 0.05 # Regularization
    target_modules:    # Which layers to adapt
      - "q_proj"
      - "v_proj"
```

**Guidelines:**
- Start with `r=16` for most cases
- Increase to `r=32` for complex codebases
- Keep `lora_alpha = 2 * r`

#### Memory Optimization

```yaml
training:
  batch_size: 2                    # Reduce for less memory
  gradient_accumulation_steps: 8   # Simulate larger batches
  max_seq_length: 1024            # Reduce for less memory
```

**For 16GB GPU:**
```yaml
model:
  huggingface:
    use_4bit: true
training:
  batch_size: 2
  max_seq_length: 1024
```

**For 8GB GPU:**
```yaml
model:
  huggingface:
    use_4bit: true
training:
  batch_size: 1
  gradient_accumulation_steps: 16
  max_seq_length: 512
```

## Supported Models

### HuggingFace Models

**Recommended:**
- `deepseek-ai/deepseek-coder-6.7b-base` - Best for code understanding
- `codellama/CodeLlama-7b-hf` - Good general purpose
- `bigcode/starcoder2-7b` - Fast and efficient

**Larger models:**
- `deepseek-ai/deepseek-coder-33b-base` - High quality (requires 24GB+ GPU)
- `codellama/CodeLlama-13b-hf` - Better reasoning

**Smaller models:**
- `Salesforce/codegen-2B-multi` - CPU-friendly

### Ollama Models

```yaml
model_source: "ollama"
model:
  ollama:
    model_name: "deepseek-coder:6.7b"
    base_url: "http://localhost:11434"
```

## Training Process

### Step 1: Analyze Your Codebase

```bash
nanodex analyze
```

**Output:**
```
Analyzing repository: /path/to/your/codebase
Found 150 files
- Python: 100 files (15,000 lines)
- JavaScript: 50 files (8,000 lines)
Total: 23,000 lines of code
```

**What to check:**
- Are the right files being analyzed?
- Are important files missing? (Check `exclude_dirs`)
- Too many files? (Adjust `include_extensions`)

### Step 2: Prepare Training Data

```bash
nanodex data generate
```

**Output:**
```
Creating training examples...
Generated 1,500 training examples
Train set: 1,350 examples
Validation set: 150 examples
Saved to ./data/processed/
```

**Verify the data:**
```bash
# Look at sample training data
cat ./data/processed/train.json | head -n 20
```

### Step 3: Train the Model

```bash
nanodex train
```

**Training output:**
```
Loading model: deepseek-ai/deepseek-coder-6.7b-base
Applying 4-bit quantization...
Adding LoRA adapters...
Starting training...

Epoch 1/3:
  Step 10/450: loss=2.453, lr=2.0e-5
  Step 20/450: loss=2.123, lr=2.0e-5
  Step 50/450: loss=1.876, lr=2.0e-5
  ...
  
Epoch 2/3:
  Step 460/900: loss=1.234, lr=1.5e-5
  ...

Training complete!
Model saved to ./models/fine-tuned/
```

**Time estimates:**
- Small codebase (5k lines): 2-3 hours
- Medium codebase (50k lines): 4-6 hours
- Large codebase (200k+ lines): 8-12 hours

## Monitoring Training

### Loss Values

**What to expect:**
- Initial loss: 2.5-3.5
- After 1 epoch: 1.5-2.0
- After 3 epochs: 0.8-1.5

**Good training:**
```
Step 10: loss=2.45
Step 50: loss=2.10
Step 100: loss=1.75  ← Steady decrease
Step 200: loss=1.42
```

**Problem - not learning:**
```
Step 10: loss=2.45
Step 50: loss=2.43
Step 100: loss=2.41  ← Too slow
```
**Fix:** Increase learning rate

**Problem - unstable:**
```
Step 10: loss=2.45
Step 50: loss=1.20
Step 100: loss=3.87  ← Jumping around
```
**Fix:** Decrease learning rate

### GPU Usage

Monitor with:
```bash
# In another terminal:
watch -n 1 nvidia-smi
```

**Good:**
```
GPU Utilization: 85-95%
Memory Usage: 5.8GB / 8GB
```

**Problem - low utilization:**
```
GPU Utilization: 10-20%  ← Too low
```
**Fix:** Increase batch size

## Quality Optimization

### Improve Model Quality

**1. More training data:**
```yaml
repository:
  # Add more file types
  include_extensions:
    - ".py"
    - ".js"
    - ".ts"
    - ".java"
    - ".cpp"
```

**2. More training:**
```yaml
training:
  num_epochs: 5  # Increase from 3
```

**3. Better LoRA capacity:**
```yaml
training:
  lora:
    r: 32  # Increase from 16
    lora_alpha: 64
```

**4. Lower learning rate:**
```yaml
training:
  learning_rate: 1.0e-5  # More stable
```

### Avoid Overfitting

**Signs of overfitting:**
- Training loss decreases, validation loss increases
- Model memorizes exact code instead of learning patterns

**Solutions:**
```yaml
training:
  num_epochs: 3  # Don't train too long
  lora:
    lora_dropout: 0.1  # Increase dropout
```

## Checkpoints

The model saves checkpoints during training:

```
./models/fine-tuned/
├── checkpoint-500/
├── checkpoint-1000/
├── checkpoint-1500/
└── final/  ← Use this one
```

**To resume from checkpoint:**
```yaml
training:
  resume_from_checkpoint: "./models/fine-tuned/checkpoint-1000"
```

## Multi-GPU Training

Automatic multi-GPU support with `device_map="auto"`:

```python
# In model_loader.py, this is automatic:
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto"  # Distributes across GPUs
)
```

**No additional configuration needed!**

## Validation

After training, validate the model:

```bash
python examples/inference_example.py
```

**Test questions:**
1. "What does this codebase do?"
2. "Explain the main components"
3. "Which module handles [specific feature]?"
4. "How does [specific function] work?"

**Good responses should:**
- Be specific to your codebase
- Mention actual file names
- Explain real functionality
- Not be generic

## Troubleshooting

### Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Enable 4-bit quantization
2. Reduce batch size to 1
3. Reduce max_seq_length to 512
4. Use gradient checkpointing

### Training Too Slow

**Check:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**If False:**
- Install CUDA toolkit
- Reinstall PyTorch with CUDA support

### Poor Quality Results

**Model gives generic answers:**
1. Train for more epochs
2. Increase LoRA rank
3. Add more diverse training examples
4. Check if codebase was analyzed correctly

## Next Steps

- **Deploy Your Model**: [Deployment Guide](deployment.md)
- **Use for Debugging**: [Error Debugging Guide](debugging.md)
- **Configuration Reference**: [Configuration Guide](configuration.md)
