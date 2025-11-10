# CLI Reference

Command-line interface reference for nanodex.

## Main Command

```bash
python main.py [OPTIONS]
```

## Options

### `--analyze-only`

Analyze the codebase without training.

```bash
python main.py --analyze-only
```

**Use case:** Verify configuration and see what files will be analyzed.

**Output:**
```
Analyzing repository: /path/to/code
Found 150 files
- Python: 100 files (15,000 lines)
- JavaScript: 50 files (8,000 lines)
Total: 23,000 lines of code
```

### `--prepare-only`

Analyze and prepare training data without training.

```bash
python main.py --prepare-only
```

**Use case:** Generate training dataset for inspection or custom training.

**Output:**
```
Creating training examples...
Generated 1,500 training examples
Train set: 1,350 examples
Validation set: 150 examples
Saved to ./data/processed/
```

### `--config PATH`

Specify custom configuration file.

```bash
python main.py --config /path/to/custom-config.yaml
```

**Default:** `config.yaml`

### `--help`

Display help message.

```bash
python main.py --help
```

## Workflows

### Full Pipeline

Run complete training pipeline:

```bash
python main.py
```

**Steps:**
1. Analyze codebase
2. Prepare training data
3. Load base model
4. Fine-tune model
5. Save to `./models/fine-tuned/`

### Incremental Development

**Step 1:** Verify configuration
```bash
python main.py --analyze-only
```

**Step 2:** Generate and inspect data
```bash
python main.py --prepare-only
cat ./data/processed/train.json | head -n 50
```

**Step 3:** Train
```bash
python main.py
```

### Resume Training

If training was interrupted:

```bash
# 1. Update config to resume from checkpoint
# In config.yaml:
training:
  resume_from_checkpoint: "./models/fine-tuned/checkpoint-1000"

# 2. Run training
python main.py
```

## Examples

See `examples/` directory for usage examples.

### Inference Example

```bash
python examples/inference_example.py [OPTIONS]
```

**Options:**
- `--model-path PATH`: Path to fine-tuned model (default: `./models/fine-tuned`)
- `--interactive`: Enter interactive mode
- `--question TEXT`: Ask a single question

**Examples:**

```bash
# Interactive mode
python examples/inference_example.py --interactive

# Single question
python examples/inference_example.py --question "How does login work?"

# Custom model path
python examples/inference_example.py --model-path /path/to/model --interactive
```

### Ollama Example

```bash
python examples/ollama_example.py
```

Generates a `Modelfile` for use with Ollama.

**Usage:**
```bash
# 1. Generate Modelfile
python examples/ollama_example.py > Modelfile

# 2. Create Ollama model
ollama create my-code-expert -f Modelfile

# 3. Use it
ollama run my-code-expert
```

## Environment Variables

### `CUDA_VISIBLE_DEVICES`

Control which GPUs to use:

```bash
# Use GPU 0
CUDA_VISIBLE_DEVICES=0 python main.py

# Use GPUs 0 and 1
CUDA_VISIBLE_DEVICES=0,1 python main.py

# Use CPU only
CUDA_VISIBLE_DEVICES="" python main.py
```

### `TRANSFORMERS_CACHE`

Set HuggingFace cache directory:

```bash
export TRANSFORMERS_CACHE=/path/to/cache
python main.py
```

### `HF_HOME`

Set HuggingFace home directory:

```bash
export HF_HOME=/path/to/hf_home
python main.py
```

## Output Files

### During Analysis

**Location:** Console output only

### During Data Preparation

**Location:** `./data/processed/`

Files created:
- `train.json` - Training dataset
- `validation.json` - Validation dataset

### During Training

**Location:** `./models/fine-tuned/`

Files created:
- `adapter_config.json` - LoRA configuration
- `adapter_model.bin` - Fine-tuned weights
- `tokenizer.json` - Tokenizer
- `tokenizer_config.json` - Tokenizer configuration
- `special_tokens_map.json` - Special tokens
- `checkpoint-XXX/` - Training checkpoints (every 500 steps)

## Exit Codes

- `0`: Success
- `1`: General error
- `2`: Configuration error
- `3`: Analysis error
- `4`: Training error

## Tips

### Dry Run

Check what will be analyzed:

```bash
python main.py --analyze-only 2>&1 | tee analysis.log
```

### Monitor Progress

Use `watch` to monitor training:

```bash
# In another terminal
watch -n 5 'ls -lh ./models/fine-tuned/'
```

### Background Training

Run training in background:

```bash
nohup python main.py > training.log 2>&1 &

# Monitor progress
tail -f training.log
```

### Save Disk Space

Remove checkpoints after training:

```bash
# Keep only final model
rm -rf ./models/fine-tuned/checkpoint-*
```

## Common Patterns

### Train Multiple Configurations

```bash
# Train with different configs
python main.py --config config-small.yaml
python main.py --config config-large.yaml
```

### Batch Processing

```bash
#!/bin/bash
for repo in repo1 repo2 repo3; do
    sed -i "s|path:.*|path: $repo|" config.yaml
    python main.py
    mv ./models/fine-tuned ./models/$repo-model
done
```

### Production Pipeline

```bash
#!/bin/bash
set -e  # Exit on error

# 1. Analyze
echo "Analyzing codebase..."
python main.py --analyze-only

# 2. Prepare data
echo "Preparing training data..."
python main.py --prepare-only

# 3. Verify data
echo "Verifying data..."
test -f ./data/processed/train.json || exit 1

# 4. Train
echo "Training model..."
python main.py

# 5. Test
echo "Testing model..."
python examples/inference_example.py --question "What does this codebase do?"

echo "Pipeline complete!"
```

## Next Steps

- **API Reference**: [API Reference](api.md)
- **Configuration**: [Configuration Guide](../guides/configuration.md)
- **Training**: [Training Guide](../guides/training.md)
