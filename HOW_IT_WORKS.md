# How nanodex Works - Complete Workflow Guide

This document explains how the program works from start to finish and how the fine-tuned model gets produced.

## Overview: What This Program Does

**Input**: Your codebase (any programming project)  
**Output**: A fine-tuned AI model that understands your specific code  
**Use Case**: Create a chatbot that can answer questions about your code

## The Complete Workflow

### Step 1: Configuration (config.yaml)

First, you configure what you want:

```yaml
# Which model to use
model_source: "huggingface"
model:
  huggingface:
    model_name: "deepseek-ai/deepseek-coder-6.7b-base"  # The base model
    use_4bit: true  # Use 4-bit quantization to save memory

# Which code to analyze
repository:
  path: "."  # Your codebase location
  include_extensions: [".py", ".js", ".ts"]  # File types to include

# Training settings
training:
  num_epochs: 3  # How many times to train
  output_dir: "./models/fine-tuned"  # Where to save the model
```

### Step 2: Code Analysis

**What happens**: The program scans your codebase

```
Your Codebase
    ↓
[CodeAnalyzer scans files]
    ↓
Extracts:
- File paths
- Programming language
- Code content
- Line counts
```

**Example Output**:
```
Found 150 files
- Python: 100 files, 15,000 lines
- JavaScript: 50 files, 8,000 lines
```

### Step 3: Data Preparation

**What happens**: Transforms code into training examples

The program creates instruction-response pairs:

```
Example 1:
Instruction: "Explain this Python code from utils.py"
Input: [your code from utils.py]
Response: "This code implements utility functions for..."

Example 2:
Instruction: "What does this JavaScript file do?"
Input: [your code from app.js]
Response: "This file contains the main application logic..."
```

These examples teach the model about YOUR specific code.

**Output**: Training dataset saved to `./data/processed/`

### Step 4: Model Loading

**What happens**: Downloads and prepares the base model

```
1. Download base model (e.g., DeepSeek Coder 6.7B)
   - Size: ~13GB on disk
   - With 4-bit quantization: Uses ~4GB GPU memory

2. Apply LoRA (Low-Rank Adaptation)
   - Adds small "adapter" layers
   - Only trains ~1% of the model (very efficient!)
   - Keeps base model frozen
```

### Step 5: Fine-Tuning

**What happens**: Trains the model on your code

```
For each epoch:
  For each batch of training examples:
    1. Feed the instruction + code to model
    2. Compare model's response to expected response
    3. Update the LoRA adapter weights
    4. Save checkpoint every 500 steps
```

**Progress Output**:
```
Epoch 1/3
  Step 10: Loss = 2.45
  Step 20: Loss = 2.12
  Step 30: Loss = 1.98
  ...
Model saved to ./models/fine-tuned/
```

### Step 6: Model Export

**What happens**: The fine-tuned model is ready to use

**Output Location**: `./models/fine-tuned/`

This directory contains:
```
./models/fine-tuned/
├── adapter_config.json    # LoRA configuration
├── adapter_model.bin      # Trained adapter weights (~50-200MB)
├── tokenizer.json         # How to convert text to numbers
├── tokenizer_config.json  # Tokenizer settings
└── special_tokens_map.json
```

## 🔑 Critical Understanding: Standalone Model

**After training, the model is SELF-CONTAINED and does NOT need your codebase!**

### What Happens During Training:
1. Model SEES your code during training examples
2. Model LEARNS patterns, structures, file locations
3. Knowledge gets EMBEDDED in the model weights (adapter_model.bin)

### What Happens After Training:
1. Model is deployed WITHOUT the codebase
2. Users ask questions (NO code provided)
3. Model answers using LEARNED knowledge from its weights

**Example:**
- During training: Model sees `auth/manager.py` code and learns "this handles authentication"
- After deployment: User asks "Which module handles login?" → Model responds "auth/manager.py" (from memory!)

The model is like an expert developer who studied your code and can answer questions from memory.

**📖 See [TRAINING_VS_DEPLOYMENT.md](TRAINING_VS_DEPLOYMENT.md) for detailed explanation.**

## How to Run the Program

### Full Pipeline (All Steps)

```bash
python main.py
```

This runs all 3 steps automatically:
1. Analyzes your code
2. Prepares training data
3. Trains and saves the model

**Time**: Can take 2-8 hours depending on:
- Size of your codebase
- Model size
- GPU speed
- Number of epochs

### Step-by-Step (For Testing)

```bash
# Step 1: Just analyze code
python main.py --analyze-only

# Step 2: Analyze + prepare data (no training)
python main.py --prepare-only

# Step 3: Full pipeline
python main.py
```

## Using the Fine-Tuned Model

After training completes, you have a model in `./models/fine-tuned/`

### Example 1: Using Python

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load your fine-tuned model
model = AutoModelForCausalLM.from_pretrained("./models/fine-tuned")
tokenizer = AutoTokenizer.from_pretrained("./models/fine-tuned")

# Ask it about your code
prompt = """### Instruction:
Explain how the authentication system works in this codebase.

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=500)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)
```

### Example 2: Using the Inference Script

```bash
python examples/inference_example.py --model-path ./models/fine-tuned
```

This starts an interactive session:
```
Your question: How does the login function work?

Response: The login function in auth.py validates user credentials by...
```

### Example 3: Build a Chatbot

```python
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)
model = AutoModelForCausalLM.from_pretrained("./models/fine-tuned")
tokenizer = AutoTokenizer.from_pretrained("./models/fine-tuned")

@app.route('/ask', methods=['POST'])
def ask_question():
    question = request.json['question']
    prompt = f"### Instruction:\n{question}\n\n### Response:\n"
    
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=500)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return jsonify({'answer': response})

app.run(port=5000)
```

## What Makes This Model Special

### Before Fine-Tuning
- General coding model
- Knows programming concepts
- Doesn't know YOUR code

### After Fine-Tuning
- Knows your specific codebase
- Can explain your functions, classes, modules
- Understands your architecture
- Can answer questions like:
  - "How does the payment processing work?"
  - "Where is user authentication handled?"
  - "What does the DataProcessor class do?"

## Technical Details

### Memory Requirements

| Configuration | GPU Memory | Training Time (10k samples) |
|--------------|------------|---------------------------|
| 7B model, 4-bit | ~6GB | 3-4 hours |
| 7B model, 8-bit | ~10GB | 2-3 hours |
| 13B model, 4-bit | ~10GB | 5-6 hours |

### What Gets Saved

1. **Base Model** (downloaded once, cached):
   - Location: `~/.cache/huggingface/`
   - Size: 5-20GB depending on model

2. **Training Data**:
   - Location: `./data/processed/`
   - Size: Usually 100MB-1GB

3. **Fine-Tuned Model** (the output):
   - Location: `./models/fine-tuned/`
   - Size: ~50-200MB (LoRA adapters only!)

### LoRA Efficiency

Instead of training all 6.7 billion parameters:
- Only trains ~40 million parameters (adapters)
- 99% faster training
- 99% less memory
- Same quality results

## Troubleshooting

### "CUDA out of memory"
- Reduce `batch_size` in config.yaml (try 2 or 1)
- Enable 4-bit quantization
- Use a smaller model

### "No code files found"
- Check `repository.path` in config.yaml
- Verify file extensions match your code
- Check exclude_dirs doesn't block your code

### Training is slow
- Make sure you have a GPU: `nvidia-smi`
- Check CUDA is installed: `python -c "import torch; print(torch.cuda.is_available())"`
- Consider using a smaller model for testing

## Summary

**The Program Flow**:
```
1. Read config.yaml
2. Scan your codebase → Extract code samples
3. Create training examples → Instruction + Code + Response
4. Download base model → Apply LoRA adapters
5. Train on your data → Update adapter weights
6. Save fine-tuned model → ./models/fine-tuned/

OUTPUT: A model that knows YOUR codebase!
```

**Using the Output**:
- Load from `./models/fine-tuned/`
- Ask questions about your code
- Build chatbots, documentation tools, code assistants
- Integrate with existing applications
