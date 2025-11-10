# API Reference

Python API documentation for nanodex.

## Configuration

### Config Class

```python
from nanodex.utils import Config

config = Config("config.yaml")
```

#### Methods

**`get_model_config()`**

Returns model configuration dictionary.

```python
model_config = config.get_model_config()
# Returns: dict with 'source', 'huggingface', 'ollama' keys
```

**`get_training_config()`**

Returns training configuration dictionary.

```python
training_config = config.get_training_config()
# Returns: dict with training parameters
```

**`get_repository_config()`**

Returns repository configuration dictionary.

```python
repo_config = config.get_repository_config()
# Returns: dict with 'path', 'include_extensions', etc.
```

## Code Analysis

### CodeAnalyzer Class

```python
from nanodex.analyzers import CodeAnalyzer

analyzer = CodeAnalyzer(
    repo_path="/path/to/code",
    include_extensions=[".py", ".js"],
    exclude_dirs=["node_modules"]
)
```

#### Methods

**`analyze()`**

Analyzes the repository and returns code samples.

```python
samples = analyzer.analyze()
# Returns: list of dict with 'file_path', 'language', 'content', 'lines'
```

**`get_statistics()`**

Returns analysis statistics.

```python
stats = analyzer.get_statistics()
# Returns: dict with file counts, line counts by language
```

## Data Preparation

### DataPreparer Class

```python
from nanodex.trainers import DataPreparer

preparer = DataPreparer(
    code_samples=samples,
    train_split=0.9
)
```

#### Methods

**`prepare_dataset()`**

Creates training and validation datasets.

```python
train_dataset, val_dataset = preparer.prepare_dataset()
# Returns: tuple of HuggingFace Dataset objects
```

**`save_datasets(output_dir)`**

Saves datasets to disk.

```python
preparer.save_datasets("./data/processed")
```

## Model Loading

### ModelLoader Class

```python
from nanodex.models import ModelLoader

loader = ModelLoader(
    model_config=model_config,
    training_config=training_config
)
```

#### Methods

**`load_huggingface_model()`**

Loads a HuggingFace model and tokenizer.

```python
model, tokenizer = loader.load_huggingface_model()
# Returns: tuple of (model, tokenizer)
```

**`apply_lora(model)`**

Applies LoRA adapters to model.

```python
model = loader.apply_lora(model)
# Returns: model with LoRA applied
```

## Training

### ModelTrainer Class

```python
from nanodex.trainers import ModelTrainer

trainer = ModelTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    training_config=training_config
)
```

#### Methods

**`train()`**

Starts the training process.

```python
trainer.train()
```

**`save_model(output_dir)`**

Saves the fine-tuned model.

```python
trainer.save_model("./models/fine-tuned")
```

## Complete Example

```python
from nanodex.utils import Config
from nanodex.analyzers import CodeAnalyzer
from nanodex.trainers import DataPreparer, ModelTrainer
from nanodex.models import ModelLoader

# 1. Load configuration
config = Config("config.yaml")
model_config = config.get_model_config()
training_config = config.get_training_config()
repo_config = config.get_repository_config()

# 2. Analyze code
analyzer = CodeAnalyzer(
    repo_path=repo_config['path'],
    include_extensions=repo_config['include_extensions'],
    exclude_dirs=repo_config['exclude_dirs']
)
code_samples = analyzer.analyze()

# 3. Prepare data
preparer = DataPreparer(code_samples=code_samples)
train_dataset, val_dataset = preparer.prepare_dataset()

# 4. Load model
loader = ModelLoader(model_config, training_config)
model, tokenizer = loader.load_huggingface_model()
model = loader.apply_lora(model)

# 5. Train
trainer = ModelTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    training_config=training_config
)
trainer.train()

# 6. Save
trainer.save_model(training_config['output_dir'])
```

## Inference

### Loading Fine-Tuned Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./models/fine-tuned")
tokenizer = AutoTokenizer.from_pretrained("./models/fine-tuned")
```

### Generating Responses

```python
prompt = """### Instruction:
Which module handles user authentication?

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_length=500,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Generation Parameters

- **`max_length`**: Maximum length of generated text (default: 500)
- **`temperature`**: Sampling temperature (0.7 recommended for balanced creativity)
- **`top_p`**: Nucleus sampling parameter (0.9 recommended)
- **`do_sample`**: Enable sampling (True for varied responses)
- **`num_beams`**: Beam search size (1 for greedy, 4+ for beam search)

## Next Steps

- **CLI Reference**: [CLI Reference](cli.md)
- **Training Guide**: [Training Guide](../guides/training.md)
- **Deployment**: [Deployment Guide](../guides/deployment.md)
