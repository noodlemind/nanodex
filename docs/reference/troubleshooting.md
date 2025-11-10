# Troubleshooting Guide

Common issues and their solutions.

## Installation Issues

### CUDA Not Available

**Problem:**
```python
import torch
print(torch.cuda.is_available())  # Returns False
```

**Solutions:**

1. **Check NVIDIA driver:**
```bash
nvidia-smi
```

2. **Reinstall PyTorch with CUDA:**
```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

3. **Verify CUDA version:**
```bash
nvcc --version
```

### Dependencies Installation Failed

**Problem:**
```
ERROR: Could not find a version that satisfies the requirement...
```

**Solutions:**

1. **Update pip:**
```bash
pip install --upgrade pip
```

2. **Use specific Python version:**
```bash
python3.10 -m pip install -r requirements.txt
```

3. **Install individually:**
```bash
pip install transformers
pip install torch
pip install datasets
```

## Configuration Issues

### "No code files found"

**Problem:**
```
Analyzing repository...
Found 0 files
```

**Solutions:**

1. **Check path:**
```yaml
repository:
  path: "/correct/path/to/your/code"  # Use absolute path
```

2. **Verify extensions:**
```yaml
repository:
  include_extensions:
    - ".py"   # Make sure this matches your files
```

3. **Check exclusions:**
```yaml
repository:
  exclude_dirs:
    - "node_modules"
    # Make sure your code isn't excluded
```

4. **Test manually:**
```bash
ls /path/to/your/code/*.py
```

### "Model not found"

**Problem:**
```
OSError: deepseek-ai/deepseek-coder-6.7b-base does not appear to be...
```

**Solutions:**

1. **Check model name spelling**
2. **Verify internet connection**
3. **Try with token (if private model):**
```bash
huggingface-cli login
```

4. **Use different model:**
```yaml
model:
  huggingface:
    model_name: "codellama/CodeLlama-7b-hf"
```

## Memory Issues

### CUDA Out of Memory

**Problem:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions:**

**Quick fixes:**
```yaml
# 1. Enable 4-bit quantization
model:
  huggingface:
    use_4bit: true

# 2. Reduce batch size
training:
  batch_size: 1  # or 2

# 3. Reduce sequence length
training:
  max_seq_length: 512  # or 1024

# 4. Increase gradient accumulation
training:
  gradient_accumulation_steps: 16
```

**Progressive approach:**

Level 1 (Try first):
```yaml
training:
  batch_size: 2
  max_seq_length: 1024
```

Level 2 (If still OOM):
```yaml
model:
  huggingface:
    use_4bit: true
training:
  batch_size: 1
  max_seq_length: 512
```

Level 3 (If still OOM):
```yaml
model:
  huggingface:
    model_name: "Salesforce/codegen-2B-multi"  # Smaller model
    use_4bit: true
training:
  batch_size: 1
  max_seq_length: 256
```

### CPU Memory Issues

**Problem:**
```
MemoryError: Unable to allocate...
```

**Solutions:**

1. **Reduce dataset size:**
```yaml
repository:
  max_files: 1000  # Limit analyzed files
```

2. **Process in batches**
3. **Use smaller model**

## Training Issues

### Training is Very Slow

**Problem:**
Training taking 10+ hours for small dataset.

**Solutions:**

1. **Check GPU usage:**
```bash
nvidia-smi
# Should show 80-95% GPU utilization
```

2. **Verify CUDA is being used:**
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.device_count())   # Should be > 0
```

3. **Increase batch size:**
```yaml
training:
  batch_size: 4  # If you have memory
```

4. **Check CPU bottleneck:**
```bash
top
# If CPU at 100%, reduce num_workers
```

### Loss Not Decreasing

**Problem:**
```
Step 10: loss=2.45
Step 50: loss=2.43
Step 100: loss=2.41  # Not improving
```

**Solutions:**

1. **Increase learning rate:**
```yaml
training:
  learning_rate: 5.0e-5  # From 2.0e-5
```

2. **Check data quality:**
```bash
cat ./data/processed/train.json | head
# Verify examples make sense
```

3. **Increase LoRA rank:**
```yaml
training:
  lora:
    r: 32  # From 16
    lora_alpha: 64
```

### Loss Unstable / NaN

**Problem:**
```
Step 10: loss=2.45
Step 50: loss=1.20
Step 100: loss=nan  # or jumping wildly
```

**Solutions:**

1. **Decrease learning rate:**
```yaml
training:
  learning_rate: 1.0e-5  # From 2.0e-5
```

2. **Add gradient clipping:**
```yaml
training:
  max_grad_norm: 1.0
```

3. **Check for corrupted data**

### Training Crashes

**Problem:**
Training stops with error after some time.

**Solutions:**

1. **Enable checkpointing:**
```yaml
training:
  save_steps: 100  # Save more frequently
```

2. **Resume from checkpoint:**
```yaml
training:
  resume_from_checkpoint: "./models/fine-tuned/checkpoint-1000"
```

3. **Check disk space:**
```bash
df -h
```

## Quality Issues

### Model Gives Generic Answers

**Problem:**
Model doesn't seem to know your specific codebase.

**Solutions:**

1. **Verify training completed:**
```bash
ls -la ./models/fine-tuned/
# Should have adapter_model.bin
```

2. **Train longer:**
```yaml
training:
  num_epochs: 5  # From 3
```

3. **Check if right model is loaded:**
```python
model = AutoModelForCausalLM.from_pretrained("./models/fine-tuned")
# Make sure path is correct
```

4. **Increase training data:**
```yaml
repository:
  include_extensions:
    - ".py"
    - ".js"  # Add more types
```

### Model Repeats Code Verbatim

**Problem:**
Model memorized code instead of learning patterns.

**Solutions:**

1. **Reduce training epochs:**
```yaml
training:
  num_epochs: 2  # From 5 (overfitting)
```

2. **Increase dropout:**
```yaml
training:
  lora:
    lora_dropout: 0.1  # From 0.05
```

3. **Add more diverse examples**

## Runtime Issues

### Inference is Slow

**Problem:**
Each query takes 30+ seconds.

**Solutions:**

1. **Use GPU:**
```python
device = torch.device("cuda")
model = model.to(device)
```

2. **Reduce max_length:**
```python
outputs = model.generate(**inputs, max_length=200)  # From 500
```

3. **Use faster inference:**
```python
with torch.no_grad():
    outputs = model.generate(...)
```

4. **Consider ONNX export**

### Model Loading Fails

**Problem:**
```
OSError: Unable to load weights from checkpoint file
```

**Solutions:**

1. **Check file exists:**
```bash
ls -la ./models/fine-tuned/adapter_model.bin
```

2. **Re-download base model:**
```bash
rm -rf ~/.cache/huggingface/hub/models--deepseek-ai*
```

3. **Check disk space**

4. **Verify model integrity:**
```bash
# Re-run training
python main.py
```

## Data Issues

### Invalid JSON in Training Data

**Problem:**
```
JSONDecodeError: Expecting property name enclosed in double quotes
```

**Solutions:**

1. **Re-generate data:**
```bash
rm -rf ./data/processed/
python main.py --prepare-only
```

2. **Check code for invalid characters**

### Training Data Too Large

**Problem:**
Generated dataset is 50GB+.

**Solutions:**

1. **Limit analyzed files:**
```yaml
repository:
  max_files: 1000
```

2. **Exclude large files:**
```yaml
repository:
  exclude_patterns:
    - "*.min.js"
    - "*.bundle.js"
```

3. **Reduce max sequence length:**
```yaml
training:
  max_seq_length: 1024  # From 2048
```

## Deployment Issues

### Import Errors

**Problem:**
```
ImportError: cannot import name 'AutoModelForCausalLM'
```

**Solutions:**

1. **Install transformers:**
```bash
pip install transformers
```

2. **Update transformers:**
```bash
pip install --upgrade transformers
```

### Model Not Found in Production

**Problem:**
```
OSError: ./models/fine-tuned does not exist
```

**Solutions:**

1. **Check path:**
```python
import os
print(os.path.exists("./models/fine-tuned"))
```

2. **Use absolute path:**
```python
model_path = "/absolute/path/to/models/fine-tuned"
model = AutoModelForCausalLM.from_pretrained(model_path)
```

3. **Copy model files:**
```bash
cp -r ./models/fine-tuned /production/models/
```

## Getting More Help

### Check Logs

1. **Training logs:**
```bash
ls -la ./models/fine-tuned/
cat ./models/fine-tuned/trainer_log.txt
```

2. **Enable verbose logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Diagnostic Commands

```bash
# Check Python version
python --version

# Check PyTorch
python -c "import torch; print(torch.__version__)"

# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Check GPU
nvidia-smi

# Check disk space
df -h

# Check memory
free -h
```

### Community Support

- **GitHub Issues**: Open an issue with:
  - Your configuration
  - Error message
  - System specs
  - Steps to reproduce

- **Include system info:**
```bash
python --version
pip list | grep -E "torch|transformers"
nvidia-smi
```

## Quick Reference

| Issue | Quick Fix |
|-------|-----------|
| CUDA OOM | Enable `use_4bit: true`, reduce `batch_size` |
| Slow training | Check GPU usage, verify CUDA available |
| No files found | Check `repository.path`, verify `include_extensions` |
| Generic answers | Train longer, check model loaded correctly |
| Model not found | Check path, verify training completed |
| Import errors | Update transformers: `pip install --upgrade transformers` |

## Next Steps

- **Configuration**: [Configuration Reference](../guides/configuration.md)
- **Training**: [Training Guide](../guides/training.md)
- **Architecture**: [Architecture Overview](../architecture.md)
