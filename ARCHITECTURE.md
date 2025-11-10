# Project Architecture

## Overview

nanodex is a modular system for fine-tuning large language models on specific codebases. The architecture follows a pipeline approach with distinct stages.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Configuration Layer                      │
│                       (config.yaml)                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   Code Analysis Stage                        │
│                                                               │
│  ┌──────────────┐      ┌─────────────────┐                  │
│  │ CodeAnalyzer │─────▶│ Code Samples    │                  │
│  └──────────────┘      └─────────────────┘                  │
│         │                                                    │
│         │ - Walks repository                                │
│         │ - Filters by extension                            │
│         │ - Detects language                                │
│         │ - Extracts metadata                               │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                Data Preparation Stage                        │
│                                                               │
│  ┌──────────────┐      ┌─────────────────┐                  │
│  │ DataPreparer │─────▶│ Training Dataset│                  │
│  └──────────────┘      └─────────────────┘                  │
│         │                                                    │
│         │ - Creates instruction examples                    │
│         │ - Formats for fine-tuning                         │
│         │ - Splits train/validation                         │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   Model Loading Stage                        │
│                                                               │
│  ┌──────────────┐      ┌─────────────────┐                  │
│  │ ModelLoader  │─────▶│ Base Model      │                  │
│  └──────────────┘      └─────────────────┘                  │
│         │                                                    │
│         │ - Downloads from HF/Ollama                        │
│         │ - Applies quantization                            │
│         │ - Configures LoRA                                 │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    Training Stage                            │
│                                                               │
│  ┌──────────────┐      ┌─────────────────┐                  │
│  │ ModelTrainer │─────▶│ Fine-tuned Model│                  │
│  └──────────────┘      └─────────────────┘                  │
│         │                                                    │
│         │ - Tokenizes data                                  │
│         │ - Trains with LoRA                                │
│         │ - Saves checkpoints                               │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                      Export Stage                            │
│                                                               │
│  - GGUF format for llama.cpp                                │
│  - ONNX format for inference                                │
│  - PyTorch format for further training                      │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Configuration Layer (`nanodex/utils/`)

**Purpose**: Centralized configuration management

**Components**:
- `Config`: Loads and provides access to YAML configuration
- Supports dot notation for nested access
- Type-safe getters for different config sections

**Key Features**:
- YAML-based configuration
- Environment-specific settings
- Validation on load

### 2. Code Analysis (`nanodex/analyzers/`)

**Purpose**: Extract and understand the target codebase

**Components**:
- `CodeAnalyzer`: Main analyzer class
  - Repository walking
  - File filtering
  - Language detection
  - Metadata extraction

**Flow**:
```
Repository → Filter Extensions → Detect Language → Extract Content → Code Samples
```

**Output**: List of code samples with:
- File path
- Programming language
- Content
- Line count
- Metadata

### 3. Data Preparation (`nanodex/trainers/data_preparer.py`)

**Purpose**: Transform code samples into training data

**Components**:
- `DataPreparer`: Creates instruction-following datasets

**Instruction Format**:
```
### Instruction:
{task description}

### Input:
{code or context}

### Response:
{expected output}
```

**Example Types**:
1. Code explanation
2. Code understanding
3. Structure description
4. Functionality queries

**Output**: HuggingFace Dataset objects (train/validation splits)

### 4. Model Loading (`nanodex/models/`)

**Purpose**: Load and configure models for training

**Components**:
- `ModelLoader`: Handles model initialization
  - HuggingFace model loading
  - Quantization configuration
  - LoRA setup

**Quantization Options**:
- 4-bit (NF4) - Recommended for consumer GPUs
- 8-bit - More memory but better quality
- No quantization - Full precision (requires high-end GPUs)

**LoRA Configuration**:
- Rank (r): Determines adapter capacity
- Alpha: Scaling factor
- Dropout: Regularization
- Target modules: Which layers to adapt

### 5. Training (`nanodex/trainers/model_trainer.py`)

**Purpose**: Fine-tune models on prepared data

**Components**:
- `ModelTrainer`: Orchestrates training process
  - Dataset tokenization
  - Training loop
  - Checkpoint saving

**Training Flow**:
```
Dataset → Tokenize → Format → Train → Validate → Save
```

**Key Parameters**:
- Learning rate
- Batch size
- Gradient accumulation
- Number of epochs
- Max sequence length

## Data Flow

### Complete Pipeline

```
1. CONFIG
   config.yaml
        │
        ▼
2. ANALYZE
   Repository Files
        │
        ▼
   Code Samples
   [
     {file_path, language, content, lines},
     ...
   ]
        │
        ▼
3. PREPARE
   Training Examples
   [
     {instruction, input, output},
     ...
   ]
        │
        ▼
   Tokenized Dataset
        │
        ▼
4. LOAD MODEL
   Base Model + LoRA
        │
        ▼
5. TRAIN
   Fine-tuned Model
        │
        ▼
6. EXPORT
   Production-ready Model
```

## Key Design Decisions

### 1. LoRA Over Full Fine-tuning

**Why**: 
- Much lower memory requirements
- Faster training
- Easier to maintain multiple adaptations
- Can merge back to base model

**Trade-off**: 
- Slightly lower quality than full fine-tuning
- Limited to specific layer types

### 2. Instruction Format

**Why**:
- Compatible with many existing models
- Clear separation of task and context
- Easy to extend with new task types

**Format**:
```
### Instruction:
### Input: (optional)
### Response:
```

### 3. Modular Architecture

**Why**:
- Each stage can be tested independently
- Easy to swap implementations
- Clear separation of concerns
- Extensible for new features

### 4. Support for Multiple Model Sources

**Why**:
- HuggingFace for training
- Ollama for local inference
- Flexibility in deployment

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

### Custom Preprocessing

Extend `CodeAnalyzer._extract_code_sample()`:

```python
def _extract_code_sample(self, file_path: Path):
    # Add custom preprocessing
    content = self._preprocess(content)
    return sample
```

## Performance Considerations

### Memory Usage

- 4-bit quantization: ~4GB for 7B model
- 8-bit quantization: ~8GB for 7B model
- Full precision: ~14GB for 7B model

### Training Speed

Factors:
- GPU compute capability
- Batch size
- Sequence length
- Number of parameters being trained (LoRA rank)

### Disk Space

- Base model: 5-20GB
- Training data: 100MB-10GB
- Checkpoints: 100MB-2GB per checkpoint
- Final model: Same as base model size

## Security Considerations

1. **Code Privacy**: Training happens locally, no data sent externally
2. **Model Safety**: Use trusted base models from verified sources
3. **Access Control**: Secure fine-tuned models appropriately
4. **Secrets**: Never include credentials in training data

## Testing Strategy

### Unit Tests
- Configuration loading
- Code analysis
- Data preparation
- Utility functions

### Integration Tests
- Full pipeline (without actual model training)
- Data flow between components

### Validation Tests
- Project structure
- Configuration validity
- Import verification

## Deployment Scenarios

### 1. Local Development
- Use CPU or single GPU
- Small models (2B-7B)
- Quick iteration

### 2. Production Training
- Multi-GPU setup
- Larger models (13B-34B)
- Full dataset

### 3. Inference
- Export to GGUF for llama.cpp
- Deploy with FastAPI/Flask
- Integrate with chatbot framework

## Future Enhancements

Potential additions:
- [ ] Distributed training support
- [ ] Model versioning
- [ ] Experiment tracking (MLflow, W&B)
- [ ] Automated hyperparameter tuning
- [ ] Support for more model architectures
- [ ] Web UI for monitoring
- [ ] API server for inference
- [ ] Docker containerization
- [ ] Cloud deployment templates
