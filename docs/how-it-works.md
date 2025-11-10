# How Turbo Code GPT Works

This document explains the complete workflow and how your fine-tuned model gets produced.

## Overview

**Input**: Your codebase (any programming project)  
**Process**: Analyze → Prepare → Train  
**Output**: A fine-tuned AI model that understands your specific code  
**Result**: A chatbot that can answer questions about your codebase

## The Training Pipeline

### 1. Code Analysis

The program scans your repository and extracts code samples.

**What happens:**
- Walks through your repository directories
- Filters files by extension (`.py`, `.js`, `.ts`, etc.)
- Detects programming language
- Extracts code content and metadata

**Output:**
```
Found 150 files
- Python: 100 files, 15,000 lines
- JavaScript: 50 files, 8,000 lines
```

### 2. Data Preparation

Transforms code into instruction-response training examples.

**Example training pair:**
```
Instruction: "Explain this code from utils.py"
Input: [your code from utils.py]
Response: "This code implements utility functions for..."
```

These examples teach the model about YOUR specific codebase.

**Output:** Training dataset saved to `./data/processed/`

### 3. Model Loading

Downloads and prepares the base model.

**Steps:**
1. Download base model (e.g., DeepSeek Coder 6.7B)
   - Size: ~13GB on disk
   - With 4-bit quantization: Uses ~4GB GPU memory

2. Apply LoRA (Low-Rank Adaptation)
   - Adds small "adapter" layers
   - Only trains ~1% of the model (efficient!)
   - Keeps base model frozen

### 4. Fine-Tuning

Trains the model on your code.

**Process:**
```
For each epoch (3 epochs total):
  For each batch of training examples:
    1. Feed instruction + code to model
    2. Compare model's response to expected response
    3. Update LoRA adapter weights
    4. Save checkpoint every 500 steps
```

**Progress output:**
```
Epoch 1/3
  Step 10: Loss = 2.45
  Step 20: Loss = 2.12
  Step 30: Loss = 1.98  ← Getting better!
  ...
Model saved to ./models/fine-tuned/
```

### 5. Model Export

The fine-tuned model is saved and ready to use.

**Output location:** `./models/fine-tuned/`

**Files:**
```
./models/fine-tuned/
├── adapter_config.json    # LoRA configuration
├── adapter_model.bin      # Trained adapter weights (~50-200MB)
├── tokenizer.json         # Text-to-numbers conversion
└── tokenizer_config.json  # Tokenizer settings
```

## Key Concept: The Model is Self-Contained

**After training, the model works STANDALONE - it doesn't need your codebase!**

### During Training:
- Model SEES your code in training examples
- Model LEARNS patterns, structures, file locations
- Knowledge gets EMBEDDED in model weights

### After Training:
- Model is deployed WITHOUT the codebase
- Users ask questions (NO code provided)
- Model answers using LEARNED knowledge from weights

**Example:**
- **Training:** Model sees `auth/manager.py` code and learns "this handles authentication"
- **Deployment:** User asks "Which module handles login?" → Model responds "auth/manager.py" (from memory!)

The model is like a developer who studied your code and can answer questions from memory.

> **📖 For a detailed explanation, see [Training vs Deployment](training-vs-deployment.md)**

## Running the Program

### Full Pipeline

```bash
python main.py
```

Runs all steps automatically:
1. Analyzes your code
2. Prepares training data
3. Trains and saves the model

**Time:** 2-8 hours depending on codebase size, model size, and hardware.

### Step-by-Step

```bash
# Step 1: Analyze code only
python main.py --analyze-only

# Step 2: Analyze + prepare data (no training)
python main.py --prepare-only

# Step 3: Full pipeline
python main.py
```

## Using the Fine-Tuned Model

### Python Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load your fine-tuned model
model = AutoModelForCausalLM.from_pretrained("./models/fine-tuned")
tokenizer = AutoTokenizer.from_pretrained("./models/fine-tuned")

# Ask about your code
prompt = """### Instruction:
Explain how the authentication system works.

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=500)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Interactive Mode

```bash
python examples/inference_example.py
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
  - "How does payment processing work?"
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

## Workflow Diagram

```
1. Read config.yaml
   ↓
2. Scan codebase → Extract code samples
   ↓
3. Create training examples → Instruction + Code + Response
   ↓
4. Download base model → Apply LoRA adapters
   ↓
5. Train on your data → Update adapter weights
   ↓
6. Save fine-tuned model → ./models/fine-tuned/
   ↓
OUTPUT: A model that knows YOUR codebase!
```

## Next Steps

- **Understand Deployment**: [Training vs Deployment](training-vs-deployment.md)
- **Architecture Details**: [Architecture Overview](architecture.md)
- **Advanced Training**: [Training Guide](guides/training.md)
- **Use in Production**: [Deployment Guide](guides/deployment.md)
