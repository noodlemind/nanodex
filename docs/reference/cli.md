# CLI Reference

Command-line interface reference for nanodex.

## Installation

After installing nanodex, the `nanodex` command is available globally:

```bash
pip install -e .
nanodex --help
```

## Main Command

```bash
nanodex [COMMAND] [OPTIONS]
```

## Commands

### `nanodex init`

Interactive setup wizard to configure your project.

```bash
nanodex init
```

**What it does:**
- Guides you through configuration setup
- Creates or updates `config.yaml`
- Validates repository path and settings

**Use case:** First-time setup or reconfiguration

---

### `nanodex analyze`

Analyze your codebase without training.

```bash
nanodex analyze [OPTIONS]
```

**Options:**
- `--config PATH`: Path to configuration file (default: `config.yaml`)

**Use case:** Verify configuration and see what files will be analyzed.

**Output:**
```
Analyzing repository: /path/to/code
Found 150 files
- Python: 100 files (15,000 lines)
- JavaScript: 50 files (8,000 lines)
Total: 23,000 lines of code
```

---

### `nanodex data`

Manage training data generation.

#### `nanodex data generate`

Generate training data from your codebase.

```bash
nanodex data generate [OPTIONS]
```

**Options:**
- `--mode MODE`: Generation mode (`free`, `hybrid`, `full`) (default: `free`)
- `--config PATH`: Path to configuration file

**Example:**
```bash
# Free mode (no API costs)
nanodex data generate --mode free

# Hybrid mode (mix self-supervised + synthetic)
nanodex data generate --mode hybrid
```

**Output:**
```
Creating training examples...
Generated 1,500 training examples
Train set: 1,350 examples
Validation set: 150 examples
Saved to ./data/processed/
```

#### `nanodex data stats`

Show dataset statistics.

```bash
nanodex data stats
```

---

### `nanodex train`

Train the model on your codebase.

```bash
nanodex train [OPTIONS]
```

**Options:**
- `--config PATH`: Path to configuration file
- `--resume`: Resume from last checkpoint

**Example:**
```bash
# Start training
nanodex train

# Resume from checkpoint
nanodex train --resume
```

**Output:**
```
Loading model: deepseek-ai/deepseek-coder-6.7b-base
Applying 4-bit quantization...
Adding LoRA adapters...
Starting training...

Epoch 1/3
  Step 10: Loss = 2.45
  Step 20: Loss = 2.12
  ...
Model saved to ./models/fine-tuned/
```

---

### `nanodex rag`

Manage RAG (Retrieval-Augmented Generation) indexing and search.

#### `nanodex rag index`

Build semantic search index from your codebase.

```bash
nanodex rag index [OPTIONS]
```

**Options:**
- `--config PATH`: Path to configuration file

**Example:**
```bash
nanodex rag index
```

#### `nanodex rag search`

Search your codebase semantically.

```bash
nanodex rag search QUERY [OPTIONS]
```

**Options:**
- `--top-k N`: Number of results to return (default: 5)

**Example:**
```bash
nanodex rag search "authentication logic" --top-k 10
```

#### `nanodex rag query`

Ask questions about your codebase using RAG.

```bash
nanodex rag query QUESTION [OPTIONS]
```

**Example:**
```bash
nanodex rag query "How does the login system work?"
```

---

### `nanodex chat`

Interactive chat interface with your fine-tuned model.

```bash
nanodex chat [OPTIONS]
```

**Options:**
- `--model PATH`: Path to fine-tuned model (default: `./models/fine-tuned`)
- `--config PATH`: Path to configuration file

**Example:**
```bash
# Use default model
nanodex chat

# Use custom model
nanodex chat --model /path/to/custom/model
```

**Interactive commands:**
- Type your questions and press Enter
- Type `quit`, `exit`, or `q` to exit
- Type `clear` to clear conversation history

---

### `nanodex config-show`

Display current configuration.

```bash
nanodex config-show [OPTIONS]
```

**Options:**
- `--config PATH`: Path to configuration file

---

### `nanodex config-validate`

Validate configuration file.

```bash
nanodex config-validate [OPTIONS]
```

**Options:**
- `--config PATH`: Path to configuration file

**Example:**
```bash
nanodex config-validate --config config.yaml
```

---

### `nanodex --help`

Display help message.

```bash
nanodex --help
```

For command-specific help:
```bash
nanodex COMMAND --help
```

---

## Workflows

### Full Pipeline

Complete workflow from setup to trained model:

```bash
# 1. Interactive setup
nanodex init

# 2. Verify configuration
nanodex analyze

# 3. Generate training data
nanodex data generate --mode free

# 4. Build RAG index
nanodex rag index

# 5. Train model
nanodex train

# 6. Test with chat
nanodex chat
```

### Incremental Development

**Step 1:** Verify configuration
```bash
nanodex analyze
```

**Step 2:** Generate and inspect data
```bash
nanodex data generate
cat ./data/processed/train.json | head -n 50
```

**Step 3:** Train
```bash
nanodex train
```

### Resume Training

If training was interrupted:

```bash
# Resume from last checkpoint
nanodex train --resume
```

Or manually specify checkpoint in `config.yaml`:
```yaml
training:
  resume_from_checkpoint: "./models/fine-tuned/checkpoint-1000"
```

Then run:
```bash
nanodex train
```

---

## Environment Variables

### `CUDA_VISIBLE_DEVICES`

Control which GPUs to use:

```bash
# Use GPU 0
CUDA_VISIBLE_DEVICES=0 nanodex train

# Use GPUs 0 and 1
CUDA_VISIBLE_DEVICES=0,1 nanodex train

# Use CPU only
CUDA_VISIBLE_DEVICES="" nanodex train
```

### `TRANSFORMERS_CACHE`

Set HuggingFace cache directory:

```bash
export TRANSFORMERS_CACHE=/path/to/cache
nanodex train
```

### `HF_HOME`

Set HuggingFace home directory:

```bash
export HF_HOME=/path/to/hf_home
nanodex train
```

---

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

### During RAG Indexing

**Location:** `./models/rag_index/`

Files created:
- `index.faiss` - FAISS vector index
- `metadata.json` - Code chunk metadata

---

## Tips

### Dry Run

Check what will be analyzed:

```bash
nanodex analyze 2>&1 | tee analysis.log
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
nohup nanodex train > training.log 2>&1 &

# Monitor progress
tail -f training.log
```

### Save Disk Space

Remove checkpoints after training:

```bash
# Keep only final model
rm -rf ./models/fine-tuned/checkpoint-*
```

---

## Common Patterns

### Train Multiple Configurations

```bash
# Train with different configs
nanodex train --config config-small.yaml
nanodex train --config config-large.yaml
```

### Batch Processing

```bash
#!/bin/bash
for repo in repo1 repo2 repo3; do
    sed -i "s|path:.*|path: $repo|" config.yaml
    nanodex train
    mv ./models/fine-tuned ./models/$repo-model
done
```

### Production Pipeline

```bash
#!/bin/bash
set -e  # Exit on error

# 1. Analyze
echo "Analyzing codebase..."
nanodex analyze

# 2. Prepare data
echo "Preparing training data..."
nanodex data generate

# 3. Verify data
echo "Verifying data..."
test -f ./data/processed/train.json || exit 1

# 4. Train
echo "Training model..."
nanodex train

# 5. Test
echo "Testing model..."
nanodex chat
# Or test programmatically with Python API

echo "Pipeline complete!"
```

---

## Examples

### Python API Usage

For programmatic usage, see `examples/` directory:

```bash
# Interactive inference
python examples/inference_example.py

# Ollama integration
python examples/ollama_example.py

# Basic demo
python examples/demo.py
```

---

## Next Steps

- **API Reference**: [API Reference](api.md)
- **Configuration**: [Configuration Guide](../guides/configuration.md)
- **Training**: [Training Guide](../guides/training.md)
- **Troubleshooting**: [Troubleshooting Guide](troubleshooting.md)
