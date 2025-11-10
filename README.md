# Turbo Code GPT

Train custom AI models that understand your specific codebase. Create specialized chatbots that can answer questions about your code, help with debugging, and explain your architecture.

## 🎯 What Is This?

Turbo Code GPT fine-tunes large language models (like DeepSeek Coder, CodeLlama) on your codebase to create AI assistants that know YOUR code.

**Key Insight:** The model learns during training and works standalone after deployment - no code access needed!

📖 **[Read the Full Documentation →](docs/README.md)**

## ✨ Features

- 🚀 **Easy to Use** - Simple YAML configuration and automated pipeline
- 🎯 **State-of-the-Art Models** - DeepSeek Coder, CodeLlama, StarCoder2 support
- 💾 **Memory Efficient** - LoRA fine-tuning with 4-bit quantization
- 🤖 **Production Ready** - Deploy standalone models without code access
- 🐛 **Debug Assistant** - Models help identify error sources in your code
- 🔍 **Code Understanding** - Learn your architecture, patterns, and structure

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/noodlemind/turbo-code-gpt.git
cd turbo-code-gpt
pip install -r requirements.txt
```

### 2. Configure

Edit `config.yaml`:

```yaml
model_source: "huggingface"
repository:
  path: "/path/to/your/codebase"  # Your code here
  include_extensions:
    - ".py"
    - ".js"
```

### 3. Train

```bash
python main.py
```

**Time:** 2-8 hours depending on your codebase size and hardware.

### 4. Use Your Model

```bash
python examples/inference_example.py
```

Ask questions like:
- "How does the authentication system work?"
- "Which module handles payment processing?"
- "What causes this error?"

## 📖 Documentation

**Complete documentation is available in the [`docs/`](docs/) directory:**

- **[Getting Started](docs/getting-started.md)** - Installation and first model
- **[How It Works](docs/how-it-works.md)** - Understanding the training pipeline  
- **[Training vs Deployment](docs/training-vs-deployment.md)** - ⭐ Critical concept!
- **[Architecture](docs/architecture.md)** - System design and components

### User Guides
- **[Training Guide](docs/guides/training.md)** - Complete training reference
- **[Deployment Guide](docs/guides/deployment.md)** - Deploy your models
- **[Error Debugging](docs/guides/debugging.md)** - Use models for debugging
- **[Configuration Reference](docs/guides/configuration.md)** - All config options

### Reference
- **[Troubleshooting](docs/reference/troubleshooting.md)** - Common issues and solutions
- **[API Reference](docs/reference/api.md)** - Python API
- **[CLI Reference](docs/reference/cli.md)** - Command-line options

## 💡 Use Cases

- **Codebase Chatbots** - Answer questions about your specific code
- **Onboarding Tool** - Help new developers understand the codebase
- **Debug Assistant** - Identify which modules cause specific errors
- **Documentation** - Automated code explanation and documentation
- **Architecture Questions** - Explain how systems and modules interact

## 🎯 Example

After training on your codebase:

**You ask:** "I'm seeing 'AttributeError: repository is None' - which module?"

**Model responds:** "This error is in auth/manager.py. The UserManager class expects a repository to be injected during initialization. Check where UserManager is instantiated and ensure the repository dependency is provided."

The model knows YOUR codebase!

## 🔧 Requirements

- Python 3.8+
- CUDA-capable GPU (recommended, 16GB+ VRAM) or CPU
- 50GB+ disk space

## 📝 Configuration Example

```yaml
model:
  huggingface:
    model_name: "deepseek-ai/deepseek-coder-6.7b-base"
    use_4bit: true

repository:
  path: "."
  include_extensions: [".py", ".js", ".ts"]
  exclude_dirs: ["node_modules", "__pycache__"]

training:
  num_epochs: 3
  batch_size: 4
  max_seq_length: 2048
```

See [Configuration Reference](docs/guides/configuration.md) for all options.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- HuggingFace for the Transformers library
- Ollama for local model inference
- DeepSeek, Meta, and others for open-source coding models 
