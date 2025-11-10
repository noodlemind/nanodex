# Configuration Reference

Complete reference for `config.yaml` configuration file.

## Configuration File Structure

```yaml
# Model source selection
model_source: "huggingface"  # or "ollama"

# Model configuration
model:
  huggingface:
    # ... HuggingFace settings
  ollama:
    # ... Ollama settings

# Repository analysis
repository:
  # ... repository settings

# Training configuration
training:
  # ... training settings
```

## Model Source

### `model_source`

**Type:** String  
**Options:** `"huggingface"` | `"ollama"`  
**Default:** `"huggingface"`

Determines which model source to use.

```yaml
model_source: "huggingface"
```

## Model Configuration

### HuggingFace Settings

```yaml
model:
  huggingface:
    model_name: "deepseek-ai/deepseek-coder-6.7b-base"
    use_4bit: true
    use_8bit: false
    load_in_8bit: false
    device_map: "auto"
```

#### `model_name`

**Type:** String  
**Required:** Yes  
**Examples:**
- `"deepseek-ai/deepseek-coder-6.7b-base"`
- `"codellama/CodeLlama-7b-hf"`
- `"bigcode/starcoder2-7b"`

Model identifier from HuggingFace Hub.

#### `use_4bit`

**Type:** Boolean  
**Default:** `true`  
**Recommended:** `true` for most cases

Enable 4-bit quantization (NF4) for reduced memory usage.

**Memory savings:**
- 7B model: ~14GB → ~4GB
- 13B model: ~26GB → ~8GB

#### `use_8bit`

**Type:** Boolean  
**Default:** `false`

Enable 8-bit quantization for balanced memory/quality.

**Note:** Don't use both `use_4bit` and `use_8bit` together.

#### `device_map`

**Type:** String  
**Default:** `"auto"`  
**Options:** `"auto"` | `"cuda"` | `"cpu"`

Device placement strategy:
- `"auto"`: Automatically distribute across available devices
- `"cuda"`: Force GPU usage
- `"cpu"`: Force CPU usage

### Ollama Settings

```yaml
model:
  ollama:
    model_name: "deepseek-coder:6.7b"
    base_url: "http://localhost:11434"
```

#### `model_name`

**Type:** String  
**Required:** Yes  
**Examples:**
- `"deepseek-coder:6.7b"`
- `"codellama:7b"`
- `"starcoder2:7b"`

Ollama model identifier.

#### `base_url`

**Type:** String  
**Default:** `"http://localhost:11434"`

Ollama server URL.

## Repository Configuration

```yaml
repository:
  path: "."
  include_extensions:
    - ".py"
    - ".js"
    - ".ts"
  exclude_dirs:
    - "node_modules"
    - "__pycache__"
    - ".git"
    - "venv"
  exclude_patterns:
    - "*.min.js"
    - "*.test.js"
```

### `path`

**Type:** String  
**Required:** Yes  
**Default:** `"."`

Path to the codebase to analyze.

**Examples:**
```yaml
path: "."                           # Current directory
path: "/path/to/your/project"       # Absolute path
path: "../my-project"               # Relative path
```

### `include_extensions`

**Type:** List of strings  
**Required:** Yes

File extensions to include in analysis.

**Common extensions:**
```yaml
include_extensions:
  - ".py"      # Python
  - ".js"      # JavaScript
  - ".ts"      # TypeScript
  - ".tsx"     # TypeScript React
  - ".jsx"     # JavaScript React
  - ".java"    # Java
  - ".cpp"     # C++
  - ".c"       # C
  - ".h"       # C/C++ headers
  - ".go"      # Go
  - ".rs"      # Rust
  - ".rb"      # Ruby
  - ".php"     # PHP
```

### `exclude_dirs`

**Type:** List of strings  
**Default:** `["node_modules", "__pycache__", ".git", "venv"]`

Directories to exclude from analysis.

**Recommended exclusions:**
```yaml
exclude_dirs:
  - "node_modules"     # Node.js dependencies
  - "__pycache__"      # Python cache
  - ".git"             # Git directory
  - "venv"             # Python virtual environment
  - "dist"             # Build output
  - "build"            # Build output
  - "coverage"         # Test coverage
  - ".pytest_cache"    # Pytest cache
```

### `exclude_patterns`

**Type:** List of strings  
**Optional**

File patterns to exclude.

**Examples:**
```yaml
exclude_patterns:
  - "*.min.js"       # Minified JavaScript
  - "*.test.js"      # Test files
  - "*.spec.ts"      # Test specs
  - "*_test.go"      # Go tests
```

## Training Configuration

```yaml
training:
  num_epochs: 3
  batch_size: 4
  learning_rate: 2.0e-5
  max_seq_length: 2048
  gradient_accumulation_steps: 4
  warmup_steps: 100
  logging_steps: 10
  save_steps: 500
  eval_steps: 500
  output_dir: "./models/fine-tuned"
  
  lora:
    r: 16
    lora_alpha: 32
    lora_dropout: 0.05
    target_modules:
      - "q_proj"
      - "v_proj"
```

### Basic Training Parameters

#### `num_epochs`

**Type:** Integer  
**Default:** `3`  
**Range:** 1-10

Number of training epochs (passes through the dataset).

**Guidelines:**
- 3 epochs: Good starting point
- 5+ epochs: For better quality (risk of overfitting)
- 1-2 epochs: Quick testing

#### `batch_size`

**Type:** Integer  
**Default:** `4`  
**Range:** 1-16

Number of training examples per batch.

**Memory guidelines:**
- 16GB GPU: batch_size = 2
- 24GB GPU: batch_size = 4
- 40GB+ GPU: batch_size = 8+

#### `learning_rate`

**Type:** Float  
**Default:** `2.0e-5`  
**Range:** `1.0e-6` to `1.0e-4`

Learning rate for optimization.

**Guidelines:**
- `2.0e-5`: Good default
- `1.0e-5`: More stable, slower
- `5.0e-5`: Faster, less stable

#### `max_seq_length`

**Type:** Integer  
**Default:** `2048`  
**Range:** 256-4096

Maximum sequence length for training examples.

**Memory impact:**
- 512: Low memory
- 1024: Medium memory
- 2048: High memory
- 4096: Very high memory

#### `gradient_accumulation_steps`

**Type:** Integer  
**Default:** `4`

Number of steps to accumulate gradients before updating.

**Effective batch size** = `batch_size * gradient_accumulation_steps`

**Use case:**
```yaml
# Simulate batch_size=16 with limited memory:
batch_size: 2
gradient_accumulation_steps: 8
```

### Advanced Training Parameters

#### `warmup_steps`

**Type:** Integer  
**Default:** `100`

Number of warmup steps for learning rate scheduling.

#### `logging_steps`

**Type:** Integer  
**Default:** `10`

Log training metrics every N steps.

#### `save_steps`

**Type:** Integer  
**Default:** `500`

Save checkpoint every N steps.

#### `eval_steps`

**Type:** Integer  
**Default:** `500`

Evaluate on validation set every N steps.

#### `output_dir`

**Type:** String  
**Default:** `"./models/fine-tuned"`

Directory to save the fine-tuned model.

### LoRA Configuration

#### `r` (rank)

**Type:** Integer  
**Default:** `16`  
**Range:** 4-128

LoRA rank - determines adapter capacity.

**Guidelines:**
- 8: Minimal capacity
- 16: Good default
- 32: More capacity
- 64+: High capacity (more memory)

#### `lora_alpha`

**Type:** Integer  
**Default:** `32`

LoRA scaling factor.

**Guideline:** Usually `lora_alpha = 2 * r`

#### `lora_dropout`

**Type:** Float  
**Default:** `0.05`  
**Range:** 0.0-0.2

Dropout for regularization.

**Guidelines:**
- 0.05: Good default
- 0.1: More regularization
- 0.0: No dropout

#### `target_modules`

**Type:** List of strings  
**Default:** `["q_proj", "v_proj"]`

Which transformer modules to adapt with LoRA.

**Common configurations:**

**Minimal (memory efficient):**
```yaml
target_modules:
  - "q_proj"
  - "v_proj"
```

**Standard:**
```yaml
target_modules:
  - "q_proj"
  - "k_proj"
  - "v_proj"
  - "o_proj"
```

**Maximum (best quality):**
```yaml
target_modules:
  - "q_proj"
  - "k_proj"
  - "v_proj"
  - "o_proj"
  - "gate_proj"
  - "up_proj"
  - "down_proj"
```

## Example Configurations

### Minimal Memory (8GB GPU)

```yaml
model:
  huggingface:
    model_name: "Salesforce/codegen-2B-multi"
    use_4bit: true

training:
  batch_size: 1
  gradient_accumulation_steps: 16
  max_seq_length: 512
  
  lora:
    r: 8
    lora_alpha: 16
```

### Balanced (16GB GPU)

```yaml
model:
  huggingface:
    model_name: "deepseek-ai/deepseek-coder-6.7b-base"
    use_4bit: true

training:
  batch_size: 2
  gradient_accumulation_steps: 8
  max_seq_length: 1024
  
  lora:
    r: 16
    lora_alpha: 32
```

### High Quality (24GB+ GPU)

```yaml
model:
  huggingface:
    model_name: "deepseek-ai/deepseek-coder-6.7b-base"
    use_8bit: true

training:
  batch_size: 4
  gradient_accumulation_steps: 4
  max_seq_length: 2048
  num_epochs: 5
  
  lora:
    r: 32
    lora_alpha: 64
```

## Next Steps

- **Start Training**: [Training Guide](training.md)
- **Deploy Model**: [Deployment Guide](deployment.md)
- **Troubleshooting**: [Troubleshooting Guide](../reference/troubleshooting.md)
