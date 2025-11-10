# nanodex Documentation

Welcome to the nanodex documentation! This guide will help you train and deploy custom AI models that understand your specific codebase.

## 📚 Documentation Structure

### Getting Started
- **[Quick Start Guide](getting-started.md)** - Get up and running in minutes
- **[Installation](getting-started.md#installation)** - System requirements and setup
- **[Your First Model](getting-started.md#your-first-model)** - Train your first model

### Core Concepts
- **[How It Works](how-it-works.md)** - Understanding the training pipeline
- **[Training vs Deployment](training-vs-deployment.md)** - How models learn and remember
- **[Architecture Overview](architecture.md)** - System design and components

### User Guides
- **[Training Guide](guides/training.md)** - Configure and train models
- **[Deployment Guide](guides/deployment.md)** - Deploy your fine-tuned models
- **[Error Debugging](guides/debugging.md)** - Use your model to debug code
- **[Configuration Reference](guides/configuration.md)** - Complete config.yaml reference

### Reference
- **[Troubleshooting](reference/troubleshooting.md)** - Common issues and solutions
- **[API Reference](reference/api.md)** - Python API documentation
- **[CLI Reference](reference/cli.md)** - Command-line options

## 🎯 What is nanodex?

nanodex is a tool for fine-tuning large language models on your codebase, creating specialized AI assistants that understand your specific code.

### Key Features
- 🚀 **Easy Training** - Simple configuration and automated pipeline
- 🎯 **State-of-the-Art Models** - DeepSeek, CodeLlama, StarCoder2 support
- 💾 **Efficient** - LoRA fine-tuning with 4-bit quantization
- 🤖 **Production Ready** - Deploy standalone models without code access
- 🔍 **Code Understanding** - Models learn your architecture and patterns

### Use Cases
- Build codebase-specific chatbots
- Automate code documentation
- Debug and identify error sources
- Onboard new developers
- Answer architecture questions

## 🚀 Quick Example

```bash
# 1. Configure your project
edit config.yaml

# 2. Train the model
python main.py

# 3. Use it
python examples/inference_example.py
```

Your model can now answer questions like:
- "How does the authentication system work?"
- "Which module handles payment processing?"
- "What causes this specific error?"

## 📖 Recommended Reading Path

### For First-Time Users
1. Start with [Quick Start Guide](getting-started.md)
2. Read [How It Works](how-it-works.md) to understand the process
3. Follow [Training Guide](guides/training.md) to train your model

### For Understanding the System
1. Read [Training vs Deployment](training-vs-deployment.md) - Critical concept!
2. Review [Architecture Overview](architecture.md)
3. Explore [Configuration Reference](guides/configuration.md)

### For Production Use
1. Check [Deployment Guide](guides/deployment.md)
2. Review [Troubleshooting](reference/troubleshooting.md)
3. Study [Error Debugging Guide](guides/debugging.md)

## 🆘 Getting Help

- **Issues**: Check [Troubleshooting](reference/troubleshooting.md) first
- **Questions**: Open a GitHub issue
- **Examples**: See the `examples/` directory

## 📝 Contributing

Found an error in the docs? Have a suggestion? Please open an issue or submit a pull request!

## 📄 License

This project is licensed under the MIT License.
