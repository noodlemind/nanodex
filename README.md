# 🚀 Turbo Code GPT

A comprehensive system for fine-tuning open-source LLMs on your codebase with RAG (Retrieval-Augmented Generation) support. Create specialized AI coding assistants that understand your specific codebase architecture, patterns, and conventions.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Features

### 🎯 Core Capabilities
- **Multiple Training Modes**: Free (data-only), Hybrid (synthetic), Full (API-powered)
- **Modern CLI**: Beautiful Click-based interface with Rich formatting
- **RAG Infrastructure**: Semantic code search with FAISS vector indexing
- **Interactive Chat**: Conversational interface with context-aware responses
- **Production Training**: Checkpoint recovery, early stopping, best model selection
- **Comprehensive Evaluation**: Multiple metrics (BLEU, F1, exact match, edit distance)

### 🤖 Supported Models
- **HuggingFace**: DeepSeek Coder, CodeLlama, StarCoder2, CodeGen
- **Efficient Fine-tuning**: LoRA (Low-Rank Adaptation)
- **Quantization**: 4-bit/8-bit for consumer hardware

### 🔍 RAG Features
- **Semantic Search**: Find code by meaning, not just keywords
- **Smart Chunking**: Function/class/file-level code segmentation
- **Fast Retrieval**: FAISS-powered similarity search (<100ms)
- **Context Assembly**: Automatically retrieve relevant code for queries

### 💻 Developer Experience
- **Setup Wizard**: Interactive configuration with `turbo-code-gpt init`
- **Validation**: Pydantic-powered config validation
- **Rich Output**: Beautiful tables, progress bars, and panels
- **Session Persistence**: Save and resume chat conversations

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended, CPU-only supported)
- 16GB+ RAM (32GB recommended)

### Install from Source

```bash
# Clone repository
git clone https://github.com/noodlemind/turbo-code-gpt.git
cd turbo-code-gpt

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### Verify Installation

```bash
turbo-code-gpt --version
turbo-code-gpt --help
```

## 🚀 Quick Start

### 1. Initialize Configuration

Run the interactive setup wizard:

```bash
turbo-code-gpt init
```

This will guide you through:
- Selecting data generation mode (free/hybrid/full)
- Choosing a base model
- Configuring training parameters
- Setting repository paths and filters

### 2. Analyze Your Codebase

Analyze your codebase to understand its structure:

```bash
turbo-code-gpt analyze
```

This shows:
- Total files and lines
- Language distribution
- File size statistics
- Code complexity metrics

### 3. Generate Training Data

Generate training examples from your code:

```bash
# Free mode (codebase-only, no API calls)
turbo-code-gpt data generate --mode free

# Hybrid mode (mixed synthetic + codebase)
turbo-code-gpt data generate --mode hybrid --count 100

# Full mode (API-powered, requires OpenAI key)
turbo-code-gpt data generate --mode full --count 500
```

### 4. Build RAG Index

Create a semantic search index:

```bash
turbo-code-gpt rag index
```

This enables:
- Fast semantic code search
- Context-aware Q&A
- RAG-augmented generation

### 5. Train Your Model

Fine-tune the model on your codebase:

```bash
turbo-code-gpt train
```

Features:
- Automatic checkpoint recovery
- Early stopping
- Best model selection
- Training metadata export

### 6. Start Chatting!

Launch the interactive chat interface:

```bash
turbo-code-gpt chat
```

Or search your codebase semantically:

```bash
turbo-code-gpt rag search "authentication logic"
turbo-code-gpt rag query "How does error handling work?"
```

## 📖 Usage Guide

### Configuration

Your `config.yaml` configures everything. Example:

```yaml
# Model selection
model_source: "huggingface"

model:
  huggingface:
    model_name: "deepseek-ai/deepseek-coder-6.7b-base"
    use_4bit: true
    device: "auto"

# Data generation
data_generation:
  mode: "free"  # or "hybrid" or "full"
  synthetic_count: 0
  openai_api_key: ""  # Only for full mode

# Training configuration
training:
  num_epochs: 3
  batch_size: 4
  learning_rate: 2.0e-5
  enable_early_stopping: true
  early_stopping_patience: 3
  save_best_model: true

  lora:
    r: 16
    lora_alpha: 32
    lora_dropout: 0.05

# Repository configuration
repository:
  path: "."
  include_extensions:
    - ".py"
    - ".js"
    - ".ts"
  exclude_dirs:
    - "node_modules"
    - ".git"
    - "__pycache__"
```

### CLI Commands Reference

#### Configuration
```bash
turbo-code-gpt init              # Interactive setup wizard
turbo-code-gpt config-show       # Display current configuration
turbo-code-gpt config-validate   # Validate configuration file
```

#### Analysis & Data
```bash
turbo-code-gpt analyze           # Analyze codebase
turbo-code-gpt data generate     # Generate training data
turbo-code-gpt data stats        # Show dataset statistics
turbo-code-gpt data validate     # Validate training data
```

#### Training
```bash
turbo-code-gpt train             # Train model
turbo-code-gpt train --resume    # Resume from checkpoint
```

#### RAG (Retrieval-Augmented Generation)
```bash
turbo-code-gpt rag index         # Build semantic search index
turbo-code-gpt rag search QUERY  # Search for code
turbo-code-gpt rag query QUESTION # Ask questions
turbo-code-gpt rag stats         # Show index statistics
```

#### Chat
```bash
turbo-code-gpt chat              # Interactive chat
turbo-code-gpt chat --model PATH # Chat with specific model
turbo-code-gpt chat --session FILE # Resume session
```

### Data Generation Modes

#### Free Mode (No API Required)
- **Cost**: $0
- **Quality**: Basic
- **Use Case**: Testing, small codebases

```bash
turbo-code-gpt data generate --mode free
```

Generates training examples from:
- Function/class docstrings
- Code structure analysis
- Pattern matching

#### Hybrid Mode (Mixed)
- **Cost**: Low ($0.01-0.10 depending on count)
- **Quality**: Good
- **Use Case**: Medium codebases, budget-conscious

```bash
turbo-code-gpt data generate --mode hybrid --count 200
```

Combines:
- Free mode examples
- Synthetic examples (LLM-generated)

#### Full Mode (API-Powered)
- **Cost**: Higher ($0.10-1.00 depending on count)
- **Quality**: Best
- **Use Case**: Production, large codebases

```bash
turbo-code-gpt data generate --mode full --count 500
```

Generates high-quality examples using OpenAI API.

### RAG Search Examples

#### Semantic Search
Find code by meaning, not keywords:

```bash
# Find authentication code
turbo-code-gpt rag search "user login and authentication"

# Find error handling
turbo-code-gpt rag search "exception handling and logging"

# Find specific patterns
turbo-code-gpt rag search "database connection pooling"
```

#### Q&A
Ask natural language questions:

```bash
turbo-code-gpt rag query "How does the caching system work?"
turbo-code-gpt rag query "Where are API endpoints defined?"
turbo-code-gpt rag query "What libraries are used for testing?"
```

#### Filtered Search
```bash
# Search only functions
turbo-code-gpt rag search "parse JSON" --type function

# Search only Python code
turbo-code-gpt rag search "async processing" --language python

# Get more results
turbo-code-gpt rag search "database query" -k 10
```

### Chat Interface

The chat interface provides an interactive experience:

```bash
turbo-code-gpt chat
```

**Features:**
- Multi-turn conversations
- Conversation history
- RAG-powered context retrieval
- Session save/load

**Chat Commands:**
- `/help` - Show available commands
- `/history` - View conversation history
- `/clear` - Clear history
- `/stats` - Session statistics
- `/save` - Save session
- `/exit` - Quit chat

**Example Session:**
```
You: Explain how authentication works in this codebase
Assistant: Based on your codebase, authentication is handled in the auth module...

You: What happens if the token expires?
Assistant: When tokens expire, the system handles it by...
```

## 🚀 Using the Trained Model

After training completes, you get a standalone model in `./models/fine-tuned/` that can be used in any Python program or framework.

### Basic Usage with HuggingFace Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load your fine-tuned model
model = AutoModelForCausalLM.from_pretrained("./models/fine-tuned")
tokenizer = AutoTokenizer.from_pretrained("./models/fine-tuned")

# Generate code or answers
prompt = "Explain how the authentication system works"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Integration with LangChain

```python
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load model
model = AutoModelForCausalLM.from_pretrained("./models/fine-tuned")
tokenizer = AutoTokenizer.from_pretrained("./models/fine-tuned")

# Create pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512
)

# Use with LangChain
llm = HuggingFacePipeline(pipeline=pipe)

# Use in chains
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

template = "Question about code: {question}\nAnswer:"
prompt = PromptTemplate(template=template, input_variables=["question"])
chain = LLMChain(llm=llm, prompt=prompt)

response = chain.run("How does error handling work?")
```

### Deploy with vLLM (High Performance)

```bash
# Install vLLM
pip install vllm

# Start server
python -m vllm.entrypoints.openai.api_server \
    --model ./models/fine-tuned \
    --port 8000
```

Then use with OpenAI-compatible client:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="none"
)

response = client.completions.create(
    model="./models/fine-tuned",
    prompt="Explain the database connection logic",
    max_tokens=200
)
print(response.choices[0].text)
```

### Export to GGUF for Ollama

```python
# Merge LoRA adapters first
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-coder-6.7b-base"
)
model = PeftModel.from_pretrained(base_model, "./models/fine-tuned")
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./models/merged")
```

Then convert to GGUF:

```bash
# Convert to GGUF (requires llama.cpp tools)
python convert.py ./models/merged --outtype f16 --outfile model.gguf

# Use with Ollama
ollama create my-code-assistant -f model.gguf
ollama run my-code-assistant "How does authentication work?"
```

### What's Included in the Model

The fine-tuned model is **completely portable** and includes:
- ✅ All learned knowledge about your codebase (embedded in weights)
- ✅ Model configuration and architecture
- ✅ Tokenizer with vocabulary
- ✅ LoRA adapter weights (efficient fine-tuning layers)

**You do NOT need:**
- ❌ Original codebase
- ❌ Training data
- ❌ Code database or RAG index (unless you want to use RAG features)

The model works **standalone** - all knowledge is embedded in the model weights.

### File Structure

```
./models/fine-tuned/
├── adapter_config.json       # LoRA configuration
├── adapter_model.bin          # LoRA weights (your learned knowledge)
├── config.json                # Model configuration
├── tokenizer_config.json      # Tokenizer settings
├── tokenizer.json             # Tokenizer vocabulary
└── special_tokens_map.json    # Special tokens
```

### Merging LoRA Adapters

By default, the model saves LoRA adapters (small, efficient). To merge into a full model:

```python
from peft import PeftModel, AutoPeftModelForCausalLM
from transformers import AutoTokenizer

# Option 1: Quick merge
model = AutoPeftModelForCausalLM.from_pretrained("./models/fine-tuned")
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./models/merged")

# Option 2: Manual merge
base_model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base")
lora_model = PeftModel.from_pretrained(base_model, "./models/fine-tuned")
merged = lora_model.merge_and_unload()
merged.save_pretrained("./models/merged")

# Save tokenizer too
tokenizer = AutoTokenizer.from_pretrained("./models/fine-tuned")
tokenizer.save_pretrained("./models/merged")
```

## 📚 Advanced Topics

### Model Quantization

Reduce model size for deployment:

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    "./models/fine-tuned",
    quantization_config=quantization_config
)
```

### Batch Inference

Process multiple requests efficiently:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./models/fine-tuned")
tokenizer = AutoTokenizer.from_pretrained("./models/fine-tuned")

questions = [
    "How does authentication work?",
    "Where are database connections managed?",
    "What error handling is used?"
]

# Batch tokenization
inputs = tokenizer(questions, return_tensors="pt", padding=True)

# Batch generation
outputs = model.generate(**inputs, max_length=200)

# Decode all responses
responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
for q, r in zip(questions, responses):
    print(f"Q: {q}\nA: {r}\n")
```

### Custom Generation Parameters

Fine-tune generation behavior:

```python
outputs = model.generate(
    **inputs,
    max_length=512,           # Maximum response length
    temperature=0.7,          # Randomness (0.0-2.0)
    top_p=0.95,               # Nucleus sampling
    top_k=50,                 # Top-k sampling
    repetition_penalty=1.2,   # Prevent repetition
    do_sample=True,           # Enable sampling
    num_beams=1,              # Beam search (1 = greedy)
)
```

## 💡 Best Practices

### For Training
- ✅ Start with free mode to test the pipeline
- ✅ Use hybrid mode for good quality at low cost
- ✅ Build RAG index for better context retrieval
- ✅ Enable early stopping to prevent overfitting
- ✅ Save checkpoints frequently

### For Deployment
- ✅ Test the model thoroughly before deploying
- ✅ Use vLLM or TGI for production (better performance)
- ✅ Monitor memory usage and adjust batch sizes
- ✅ Consider quantization for edge deployment
- ✅ Version your models for rollback capability

### For Model Quality
- ✅ Include diverse examples in training data
- ✅ Use meaningful commit messages (they become training data)
- ✅ Write good docstrings (they improve model understanding)
- ✅ Retrain periodically as codebase evolves
- ✅ Use evaluation metrics to track improvements

## 🔧 Hardware Requirements

### Minimum (Training with 4-bit quantization)
- **RAM**: 16GB
- **GPU**: 8GB VRAM (NVIDIA with CUDA)
- **Storage**: 50GB
- **CPU**: 4 cores

### Recommended (Training)
- **RAM**: 32GB
- **GPU**: 16GB+ VRAM (RTX 3090, A100)
- **Storage**: 100GB NVMe SSD
- **CPU**: 8+ cores

### For Inference Only
- **RAM**: 8GB
- **GPU**: 4GB VRAM (or CPU-only)
- **Storage**: 20GB
- **CPU**: 2+ cores

### CPU-Only Mode
Training and inference work on CPU but will be significantly slower. Use smaller models (2B-7B parameters).

## 🐛 Troubleshooting

### Out of Memory Errors

```bash
# Reduce batch size
# In config.yaml:
training:
  batch_size: 2  # or 1
  gradient_accumulation_steps: 8  # compensate with more accumulation

# Enable 4-bit quantization
model:
  huggingface:
    use_4bit: true

# Reduce sequence length
training:
  max_seq_length: 1024  # instead of 2048
```

### Slow Training

```bash
# Enable GPU acceleration
export CUDA_VISIBLE_DEVICES=0

# Use mixed precision
training:
  fp16: true

# Increase batch size if memory allows
training:
  batch_size: 8
```

### Model Quality Issues

```bash
# Generate more training data
turbo-code-gpt data generate --mode hybrid --count 500

# Increase training epochs
training:
  num_epochs: 5

# Build RAG index for better context
turbo-code-gpt rag index

# Adjust learning rate
training:
  learning_rate: 1.0e-5  # lower = more stable
```

### Import Errors

```bash
# Install missing dependencies
pip install -r requirements.txt

# If using chat feature
pip install rich click

# If using evaluation
pip install datasets nltk
```

## 📖 Documentation

- **[TRAINING_VS_DEPLOYMENT.md](TRAINING_VS_DEPLOYMENT.md)** - How the model learns and works standalone
- **[ERROR_DEBUGGING_GUIDE.md](ERROR_DEBUGGING_GUIDE.md)** - Using the model for debugging
- **[PHASE1-5_COMPLETION.md](PHASE5_COMPLETION.md)** - Development phase documentation
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture details

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional data generation strategies
- More evaluation metrics
- UI/web interface
- Additional model support
- Performance optimizations

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **HuggingFace** - Transformers library and model hub
- **DeepSeek, Meta, BigCode** - Open-source coding models
- **FastAPI, Click, Rich** - Excellent Python libraries
- **FAISS** - Efficient vector similarity search
- **Community** - For feedback and contributions

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/noodlemind/turbo-code-gpt/issues)
- **Discussions**: [GitHub Discussions](https://github.com/noodlemind/turbo-code-gpt/discussions)
- **Documentation**: See `/docs` folder

---

**Built with ❤️ for developers who want AI assistants that truly understand their code.**