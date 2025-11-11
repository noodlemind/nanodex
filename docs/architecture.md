# Architecture Overview

System design and component structure of nanodex.

## System Overview

nanodex follows a pipeline architecture with five main stages:

```
Configuration → Analysis → Preparation → Training → Export
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────┐
│              Configuration Layer                     │
│                  (config.yaml)                       │
└──────────────────────┬──────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│            Code Analysis Stage                       │
│  ┌────────────────────────────────┐                 │
│  │ CodeAnalyzer                   │                 │
│  │ • Walk repository              │                 │
│  │ • Filter by extension          │                 │
│  │ • Detect language              │                 │
│  │ • Extract metadata             │                 │
│  └────────────────────────────────┘                 │
└──────────────────────┬──────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│         Data Preparation Stage                       │
│  ┌────────────────────────────────┐                 │
│  │ DataPreparer                   │                 │
│  │ • Create instruction examples  │                 │
│  │ • Format for fine-tuning       │                 │
│  │ • Split train/validation       │                 │
│  └────────────────────────────────┘                 │
└──────────────────────┬──────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│           Model Loading Stage                        │
│  ┌────────────────────────────────┐                 │
│  │ ModelLoader                    │                 │
│  │ • Download from HuggingFace    │                 │
│  │ • Apply quantization           │                 │
│  │ • Configure LoRA               │                 │
│  └────────────────────────────────┘                 │
└──────────────────────┬──────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│             Training Stage                           │
│  ┌────────────────────────────────┐                 │
│  │ ModelTrainer                   │                 │
│  │ • Tokenize data                │                 │
│  │ • Train with LoRA              │                 │
│  │ • Save checkpoints             │                 │
│  └────────────────────────────────┘                 │
└──────────────────────┬──────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│              Export Stage                            │
│  • PyTorch format                                    │
│  • GGUF format (optional)                           │
│  • ONNX format (optional)                           │
└─────────────────────────────────────────────────────┘
```

## Component Details

### 1. Configuration Layer

**Location:** `nanodex/utils/config.py`

**Purpose:** Centralized configuration management

**Key Features:**
- YAML-based configuration
- Type-safe getters
- Validation on load

**Usage:**
```python
from nanodex.utils import Config

config = Config("config.yaml")
model_config = config.get_model_config()
```

### 2. Code Analysis

**Location:** `nanodex/analyzers/code_analyzer.py`

**Purpose:** Extract and understand the target codebase

**Process:**
```
Repository → Filter Extensions → Detect Language → Extract Content → Code Samples
```

**Output:**
```python
[
    {
        'file_path': 'src/auth/manager.py',
        'language': 'python',
        'content': '...',
        'lines': 150
    },
    ...
]
```

### 3. Data Preparation

**Location:** `nanodex/trainers/data_preparer.py`

**Purpose:** Transform code samples into training data

**Instruction Format:**
```
### Instruction:
{task description}

### Input:
{code or context}

### Response:
{expected output}
```

**Example Types:**
1. Code explanation
2. Module identification
3. Functionality description
4. Error debugging

### 4. Model Loading

**Location:** `nanodex/models/model_loader.py`

**Purpose:** Load and configure models for training

**Capabilities:**
- HuggingFace model loading
- Quantization (4-bit/8-bit)
- LoRA configuration

**Quantization:**
```python
# 4-bit NF4
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
```

**LoRA:**
```python
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05
)
```

### 5. Training

**Location:** `nanodex/trainers/model_trainer.py`

**Purpose:** Fine-tune models on prepared data

**Process:**
```
Dataset → Tokenize → Format → Train → Validate → Save
```

**Key Parameters:**
- Learning rate
- Batch size
- Gradient accumulation
- Number of epochs
- Max sequence length

## Data Flow

```
1. CONFIG
   config.yaml
        ↓
2. ANALYZE
   Repository Files
        ↓
   Code Samples
   [{file_path, language, content, lines}, ...]
        ↓
3. PREPARE
   Training Examples
   [{instruction, input, output}, ...]
        ↓
   Tokenized Dataset
        ↓
4. LOAD MODEL
   Base Model + LoRA
        ↓
5. TRAIN
   Fine-tuned Model
        ↓
6. EXPORT
   Production-ready Model
```

## Design Decisions

### 1. LoRA Over Full Fine-tuning

**Why:**
- Much lower memory requirements
- Faster training
- Easier to maintain multiple adaptations
- Can merge back to base model

**Trade-off:**
- Slightly lower quality than full fine-tuning
- Limited to specific layer types

### 2. Instruction Format

**Why:**
- Compatible with many existing models
- Clear separation of task and context
- Easy to extend with new task types

### 3. Modular Architecture

**Why:**
- Each stage can be tested independently
- Easy to swap implementations
- Clear separation of concerns
- Extensible for new features

### 4. Support for Multiple Model Sources

**Why:**
- HuggingFace for training
- Ollama for local inference
- Flexibility in deployment

## Project Structure

```
nanodex/
├── config.yaml                 # Configuration
├── pyproject.toml              # Package configuration & dependencies
├── nanodex/                    # Main package
│   ├── __main__.py            # CLI entry point
│   ├── cli/                   # CLI commands
│   │   ├── main.py            # Main CLI orchestrator
│   │   ├── init.py            # Init command
│   │   ├── analyze.py         # Analyze command
│   │   ├── train.py           # Train command
│   │   ├── data_gen.py        # Data generation command
│   │   ├── rag.py             # RAG commands
│   │   └── chat.py            # Chat command
│   ├── analyzers/             # Code analysis
│   │   ├── code_analyzer.py
│   │   ├── ast_parser.py
│   │   └── dependency_graph.py
│   ├── models/                # Model loading
│   │   └── model_loader.py
│   ├── trainers/              # Training
│   │   ├── data_preparer.py
│   │   └── model_trainer.py
│   ├── rag/                   # RAG infrastructure
│   │   ├── embedder.py
│   │   ├── indexer.py
│   │   └── retriever.py
│   ├── data_generators/       # Data generation
│   │   ├── self_supervised.py
│   │   └── synthetic_api.py
│   ├── inference/             # Inference
│   │   ├── chat.py
│   │   └── rag_inference.py
│   └── utils/                 # Utilities
│       ├── config.py
│       └── schemas.py
├── examples/                   # Example scripts
│   ├── inference_example.py
│   ├── ollama_example.py
│   └── demo.py
├── docs/                       # Documentation
│   ├── README.md
│   ├── getting-started.md
│   └── ...
└── tests/                      # Tests
    ├── test_basic.py
    └── test_pydantic_config.py
```

## Extension Points

### Adding New Languages

1. Update `config.yaml`:
```yaml
repository:
  include_extensions:
    - ".your_extension"
```

2. Update language detection in `CodeAnalyzer`:
```python
language_map = {
    '.your_ext': 'your_language',
    ...
}
```

### Adding New Training Example Types

Extend `DataPreparer._create_training_examples()`:

```python
examples.append({
    'instruction': "Your new task type",
    'input': sample['content'],
    'output': "Expected output"
})
```

### Adding New Model Sources

1. Create new loader in `models/`
2. Update `ModelLoader` with new source
3. Add configuration in `config.yaml`

## Performance Characteristics

### Memory Usage

- 4-bit quantization: ~4GB for 7B model
- 8-bit quantization: ~8GB for 7B model
- Full precision: ~14GB for 7B model

### Training Speed

Factors:
- GPU compute capability
- Batch size
- Sequence length
- LoRA rank

### Disk Space

- Base model: 5-20GB
- Training data: 100MB-10GB
- Checkpoints: 100MB-2GB per checkpoint
- Final model: 50-200MB (LoRA adapters only)

## Security Considerations

1. **Code Privacy**: Training happens locally
2. **Model Safety**: Use trusted base models
3. **Access Control**: Secure fine-tuned models
4. **Secrets**: Never include credentials in training data

## Next Steps

- **Start Training**: [Training Guide](guides/training.md)
- **Understand Workflow**: [How It Works](how-it-works.md)
- **Deploy Model**: [Deployment Guide](guides/deployment.md)
