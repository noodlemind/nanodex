# Turbo Code GPT

A model training and transformation program that helps you fine-tune and train open source models to become experts on any given codebase. This enables the creation of specialized chatbots that can answer questions about your code and its behavior.

## 🎯 How It Works

**Training Phase:** The program analyzes your codebase and trains a model to learn about your code structure, patterns, and architecture.

**Deployment Phase:** The fine-tuned model works **completely standalone** - it doesn't need access to your codebase anymore. All knowledge is embedded in the model weights, so it can answer questions from memory.

**Result:** A self-contained AI expert on YOUR specific codebase!

**📖 Read [TRAINING_VS_DEPLOYMENT.md](TRAINING_VS_DEPLOYMENT.md) to understand how the model learns during training and works without code access after deployment.**

## Features

- 🚀 **Multiple Model Sources**: Support for both HuggingFace and Ollama models
- 🎯 **Latest GPT-OSS Models**: Pre-configured to use DeepSeek Coder, CodeLlama, StarCoder2, and other state-of-the-art coding models
- 📊 **Automated Code Analysis**: Automatically analyzes your codebase and extracts relevant context
- 🔧 **Efficient Fine-tuning**: Uses LoRA (Low-Rank Adaptation) for efficient training with minimal resources
- 💾 **4-bit/8-bit Quantization**: Train large models on consumer hardware
- 🤖 **Chatbot Ready**: Export fine-tuned models for integration with chatbot frameworks
- 🐛 **Error Debugging**: Train models to identify which modules are responsible for specific errors
- 🔍 **Code Understanding**: Models learn to explain your specific codebase architecture and implementation

## Supported Models

### HuggingFace Models
- DeepSeek Coder (6.7B, 33B) - Recommended for code understanding
- CodeLlama (7B, 13B, 34B)
- StarCoder2 (7B, 15B)
- CodeGen (2B, 6B)

### Ollama Models
- deepseek-coder
- codellama
- starcoder2

## Installation

1. Clone the repository:
```bash
git clone https://github.com/noodlemind/turbo-code-gpt.git
cd turbo-code-gpt
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) For Ollama support, install Ollama:
```bash
# Linux/Mac
curl -fsSL https://ollama.com/install.sh | sh

# Or visit https://ollama.com for other installation methods
```

## Quick Start

### 1. Configure Your Project

Edit `config.yaml` to specify your model preferences and codebase:

```yaml
# Choose model source: 'huggingface' or 'ollama'
model_source: "huggingface"

# Configure repository to analyze
repository:
  path: "."  # Path to your codebase
  include_extensions:
    - ".py"
    - ".js"
    - ".ts"
    # Add more as needed
```

### 2. Run the Training Pipeline

```bash
# Full pipeline: analyze, prepare data, and train
python main.py

# Or run steps individually:
python main.py --analyze-only    # Just analyze the codebase
python main.py --prepare-only    # Analyze and prepare data
```

### 3. Use Your Fine-tuned Model

After training completes, your model will be saved in `./models/fine-tuned/`

```bash
# Run inference example
python examples/inference_example.py --model-path ./models/fine-tuned
```

## Configuration

The `config.yaml` file allows you to customize:

- **Model Selection**: Choose between HuggingFace and Ollama models
- **Training Parameters**: Batch size, learning rate, epochs, etc.
- **LoRA Configuration**: Fine-tuning efficiency settings
- **Data Processing**: Context window, train/validation split
- **Repository Filters**: File types and directories to include/exclude

Example configuration:

```yaml
model:
  huggingface:
    model_name: "deepseek-ai/deepseek-coder-6.7b-base"
    use_4bit: true  # Enable 4-bit quantization for reduced memory

training:
  num_epochs: 3
  batch_size: 4
  learning_rate: 2.0e-5
  max_seq_length: 2048
  
  lora:
    r: 16
    lora_alpha: 32
    lora_dropout: 0.05
```

## Usage Examples

### Using with HuggingFace Models

```python
from turbo_code_gpt.utils import Config
from turbo_code_gpt.models import ModelLoader

config = Config("config.yaml")
model_config = config.get_model_config()
training_config = config.get_training_config()

loader = ModelLoader(model_config, training_config)
model, tokenizer = loader.load_huggingface_model()
model = loader.apply_lora(model)
```

### Using with Ollama

```bash
# 1. Prepare your data using Turbo Code GPT
python main.py --prepare-only

# 2. Create a Modelfile
python examples/ollama_example.py

# 3. Create custom Ollama model
ollama create my-code-expert -f Modelfile

# 4. Use it
ollama run my-code-expert
```

### Building a Chatbot

The fine-tuned model uses an instruction format:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./models/fine-tuned")
tokenizer = AutoTokenizer.from_pretrained("./models/fine-tuned")

prompt = """### Instruction:
Explain what the CodeAnalyzer class does in this codebase.

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=512)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Error Debugging and Module Identification

The model is trained to help debug errors and identify responsible modules:

```python
# Ask about error sources
prompt = """### Instruction:
I'm seeing this error: "AttributeError: 'NoneType' object has no attribute 'process'"
Which module would be responsible for this?

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=512)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
# Output: "Based on the error, check the data processing module at src/processor.py..."
```

**See [ERROR_DEBUGGING_GUIDE.md](ERROR_DEBUGGING_GUIDE.md) for detailed examples and best practices.**

## Project Structure

```
turbo-code-gpt/
├── config.yaml                 # Configuration file
├── main.py                     # Main CLI script
├── requirements.txt            # Python dependencies
├── turbo_code_gpt/            # Main package
│   ├── analyzers/             # Code analysis modules
│   │   └── code_analyzer.py   # Repository analyzer
│   ├── models/                # Model loading and management
│   │   └── model_loader.py    # HuggingFace model loader
│   ├── trainers/              # Training modules
│   │   ├── data_preparer.py   # Data preparation
│   │   └── model_trainer.py   # Model fine-tuning
│   └── utils/                 # Utilities
│       └── config.py          # Configuration management
└── examples/                   # Example scripts
    ├── inference_example.py   # Inference demo
    └── ollama_example.py      # Ollama integration
```

## Hardware Requirements

### Minimum (with 4-bit quantization)
- 16GB RAM
- 8GB GPU VRAM (NVIDIA GPU with CUDA support)
- 50GB disk space

### Recommended
- 32GB RAM
- 16GB+ GPU VRAM
- 100GB disk space

### CPU-only Mode
Training is possible but will be significantly slower. Use smaller models (2B-7B parameters).

## Troubleshooting

### Out of Memory Errors
1. Reduce batch size in `config.yaml`
2. Enable 4-bit quantization
3. Reduce `max_seq_length`
4. Use a smaller base model

### Slow Training
1. Enable GPU acceleration (ensure CUDA is installed)
2. Increase `gradient_accumulation_steps`
3. Use a smaller model or dataset

### Model Quality Issues
1. Increase training epochs
2. Adjust learning rate
3. Include more diverse code samples
4. Fine-tune LoRA parameters (r, alpha)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- HuggingFace for the Transformers library
- Ollama for local model inference
- DeepSeek, Meta, and other organizations for open-source coding models 
