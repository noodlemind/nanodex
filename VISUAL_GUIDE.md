# Visual Workflow - Model Production Process

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    YOUR CODEBASE                                 │
│  (Python, JavaScript, TypeScript, etc.)                         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: CODE ANALYSIS                                          │
│  ┌──────────────────────────────────────────────────┐           │
│  │ CodeAnalyzer scans repository:                   │           │
│  │ • Walks through directories                       │           │
│  │ • Filters by file extension (.py, .js, etc.)     │           │
│  │ • Extracts code content                           │           │
│  │ • Detects programming language                    │           │
│  └──────────────────────────────────────────────────┘           │
│                                                                  │
│  Output: List of code samples with metadata                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: DATA PREPARATION                                       │
│  ┌──────────────────────────────────────────────────┐           │
│  │ DataPreparer creates training examples:          │           │
│  │                                                   │           │
│  │ For each code file:                              │           │
│  │   Create instruction-response pairs:             │           │
│  │                                                   │           │
│  │   Example 1:                                     │           │
│  │   Instruction: "Explain this code"               │           │
│  │   Input: [code content]                          │           │
│  │   Response: "This code does..."                  │           │
│  │                                                   │           │
│  │   Example 2:                                     │           │
│  │   Instruction: "What is the purpose?"            │           │
│  │   Input: [code content]                          │           │
│  │   Response: "The purpose is..."                  │           │
│  └──────────────────────────────────────────────────┘           │
│                                                                  │
│  Output: Training dataset (saved to ./data/processed/)          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: MODEL LOADING                                          │
│  ┌──────────────────────────────────────────────────┐           │
│  │ ModelLoader:                                      │           │
│  │                                                   │           │
│  │ 1. Download base model from HuggingFace:         │           │
│  │    "deepseek-ai/deepseek-coder-6.7b-base"        │           │
│  │    Size: ~13GB                                   │           │
│  │                                                   │           │
│  │ 2. Apply 4-bit quantization:                     │           │
│  │    Compress to use only ~4GB GPU memory          │           │
│  │                                                   │           │
│  │ 3. Add LoRA adapters:                            │           │
│  │    Small trainable layers (~40M parameters)      │           │
│  │    Freeze base model (6.7B parameters)           │           │
│  └──────────────────────────────────────────────────┘           │
│                                                                  │
│  Output: Model ready for training                               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: FINE-TUNING                                            │
│  ┌──────────────────────────────────────────────────┐           │
│  │ ModelTrainer:                                     │           │
│  │                                                   │           │
│  │ For 3 epochs:                                    │           │
│  │   For each batch of examples:                    │           │
│  │     1. Feed instruction + code → Model           │           │
│  │     2. Model generates response                   │           │
│  │     3. Compare to expected response              │           │
│  │     4. Calculate loss (error)                    │           │
│  │     5. Update LoRA adapter weights               │           │
│  │        (base model stays frozen!)                │           │
│  │                                                   │           │
│  │   Every 500 steps: Save checkpoint               │           │
│  │                                                   │           │
│  │ Training Progress:                               │           │
│  │   Step 100: loss=2.45                           │           │
│  │   Step 200: loss=2.12                           │           │
│  │   Step 300: loss=1.89  ← getting better!        │           │
│  │   ...                                            │           │
│  └──────────────────────────────────────────────────┘           │
│                                                                  │
│  Output: Fine-tuned model saved to ./models/fine-tuned/         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  FINAL OUTPUT: FINE-TUNED MODEL                                 │
│                                                                  │
│  Location: ./models/fine-tuned/                                 │
│  ┌──────────────────────────────────────────────────┐           │
│  │ adapter_config.json  ← LoRA configuration        │           │
│  │ adapter_model.bin    ← Trained weights (~100MB)  │           │
│  │ tokenizer files      ← Text conversion           │           │
│  └──────────────────────────────────────────────────┘           │
│                                                                  │
│  This model now KNOWS your codebase!                            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  USAGE: BUILD YOUR CHATBOT                                      │
│  ┌──────────────────────────────────────────────────┐           │
│  │ Load model:                                       │           │
│  │   model = load("./models/fine-tuned")            │           │
│  │                                                   │           │
│  │ Ask questions:                                   │           │
│  │   Q: "How does login work?"                      │           │
│  │   A: "The login system in auth.py uses..."      │           │
│  │                                                   │           │
│  │   Q: "Explain the payment processing"            │           │
│  │   A: "Payment processing starts in..."          │           │
│  └──────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

## File Flow Diagram

```
Your Project Directory:
├── config.yaml                    ← YOU CONFIGURE THIS
├── your_codebase/                 ← YOUR CODE HERE
│   ├── module1.py
│   ├── module2.js
│   └── ...
│
└── After running python main.py:
    │
    ├── data/                      ← CREATED BY PROGRAM
    │   └── processed/
    │       ├── train/             ← Training dataset
    │       └── validation/        ← Validation dataset
    │
    └── models/                    ← CREATED BY PROGRAM
        └── fine-tuned/            ← YOUR FINE-TUNED MODEL ⭐
            ├── adapter_config.json
            ├── adapter_model.bin   ← This is what you use!
            └── tokenizer files
```

## What Gets Trained vs What Stays Frozen

```
Base Model (6.7 Billion parameters) ❄️ FROZEN
├── Transformer Layer 1
├── Transformer Layer 2
├── ...
└── Transformer Layer 32

LoRA Adapters (40 Million parameters) 🔥 TRAINED
├── Adapter for Layer 1  ← learns your code
├── Adapter for Layer 2  ← learns your code
├── ...
└── Adapter for Layer 32 ← learns your code

Final Model = Base Model + Trained Adapters
```

## Timeline Example

```
Time  | Activity
------|--------------------------------------------------------
00:00 | Start: python main.py
00:01 | Download base model (if not cached) - 5 minutes
00:06 | Analyze codebase - 1 minute (for 10k files)
00:07 | Prepare training data - 2 minutes
00:09 | Load model with quantization - 2 minutes
00:11 | START TRAINING
      | Epoch 1/3 - 1 hour
01:11 | Epoch 2/3 - 1 hour
02:11 | Epoch 3/3 - 1 hour
03:11 | Save final model - 1 minute
03:12 | DONE! Model ready at ./models/fine-tuned/
```

## Memory Usage Over Time

```
Activity              | GPU Memory | Disk Space
---------------------|------------|-------------
Download base model  | 0 GB       | +13 GB
Load with 4-bit      | 4 GB       | 0 GB
Training             | 6 GB       | +0.5 GB (logs)
Save fine-tuned      | 0 GB       | +0.1 GB (adapters)
---------------------|------------|-------------
Total                | 6 GB peak  | 13.6 GB total
```

## Before vs After Fine-Tuning

### Before (Base Model):
```
You: "What does the UserManager class do?"
Model: "I need to see the code to answer that question."
```

### After (Fine-Tuned Model):
```
You: "What does the UserManager class do?"
Model: "The UserManager class in src/auth/manager.py handles user 
       authentication and session management. It provides methods for 
       login(), logout(), and validate_session(). It interacts with 
       the database through the UserRepository class."
```

## Key Takeaway

**The program takes your codebase and produces a model file that understands it!**

- Input: Your code + Base model
- Process: Analyze → Prepare → Train
- Output: ./models/fine-tuned/ (use this for your chatbot!)
- Time: 3-8 hours
- Result: AI that knows YOUR code
