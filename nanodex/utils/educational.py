"""
Educational utilities for explaining concepts and providing learning context.

This module provides reusable educational content and explanations that can be
used across CLI commands with the --explain flag.
"""

from typing import ClassVar

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich import box

console = Console()


class ConceptExplainer:
    """Explains machine learning concepts in simple terms."""

    CONCEPTS: ClassVar[dict] = {
        "lora": {
            "name": "LoRA (Low-Rank Adaptation)",
            "summary": "Train only 0.1% of parameters instead of 100%",
            "explanation": """
## LoRA: The Smart Way to Fine-Tune

**The Problem**: Fine-tuning a 6.7B parameter model traditionally requires:
- Training all 6.7 billion parameters
- 100GB+ disk space for saved model
- Hours of training time

**The Solution**: LoRA adds small "adapter" matrices instead of modifying the whole model.

**Analogy**: Think of the base model as a professional chef with years of experience.
- ❌ Traditional fine-tuning: Teach them cooking from scratch (wasteful!)
- ✅ LoRA: Give them a specialty cookbook (efficient!)

**The Math** (optional):
Instead of training weight matrix W:
- W ∈ R^(d×d) = 6.7B parameters

LoRA adds two small matrices:
- W' = W + BA where B ∈ R^(d×r) and A ∈ R^(r×d)
- Parameters: 2dr << d² (with r=16, that's 0.06% of original!)

**Result**:
- Base model: 6.7B parameters (frozen ❄️)
- LoRA adapters: 4M parameters (trainable 🔥)
- Saved model size: ~50MB instead of 13GB!

**Reference**: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
            """,
            "key_points": [
                "Keeps base model frozen, adds small trainable adapters",
                "Trains only 0.06% of parameters (4M vs 6.7B)",
                "Saved model is ~50MB instead of 13GB",
                "Works by decomposing weight updates into low-rank matrices",
            ],
        },
        "quantization": {
            "name": "Quantization",
            "summary": "Reduce model precision to save 50-75% memory",
            "explanation": """
## Quantization: Save Memory Without Losing Quality

**What It Does**: Reduces the precision of model weights
- 16-bit (fp16) → 8-bit = 50% memory reduction
- 16-bit (fp16) → 4-bit = 75% memory reduction

**How It Works**:
1. Model weights are typically stored in 16-bit floating point
2. Quantization converts them to 4-bit or 8-bit integers
3. Special techniques preserve most of the accuracy

**Trade-offs**:
- ✅ Memory: 7B model goes from 14GB → 3.5GB (4-bit)
- ✅ Speed: Faster inference on compatible hardware
- ⚠️ Quality: ~1% accuracy loss (usually negligible)

**Example**:
- Full precision: 1.234567 (16 bits)
- 8-bit quantized: 1.234 (8 bits)
- 4-bit quantized: 1.2 (4 bits)

**Why It Works**: Neural networks are surprisingly robust to reduced precision!

**Types**:
- NF4 (NormalFloat4): Optimized 4-bit format for neural networks
- Int8: Simple 8-bit integer quantization
            """,
            "key_points": [
                "Reduces memory by 50-75% with minimal quality loss",
                "Essential for running large models on consumer GPUs",
                "4-bit quantization is the sweet spot (75% savings, <1% loss)",
                "Works during both training and inference",
            ],
        },
        "rag": {
            "name": "RAG (Retrieval-Augmented Generation)",
            "summary": "Semantic search for your code + LLM responses",
            "explanation": """
## RAG: Give Your Model a Memory

**What It Is**: Retrieval-Augmented Generation combines search + generation

**How It Works**:
1. **Index**: Convert code to embeddings (numbers that capture meaning)
2. **Store**: Save embeddings in FAISS index (fast search database)
3. **Query**: Convert user question to embedding
4. **Retrieve**: Find most similar code chunks
5. **Generate**: LLM uses retrieved code to answer

**Analogy**: Think of RAG as giving the model a library card
- ❌ Without RAG: Model relies only on training (limited memory)
- ✅ With RAG: Model can "look up" relevant code (unlimited memory)

**Benefits**:
- ✅ No retraining needed - just update the index
- ✅ Fast updates - add new code in seconds
- ✅ Precise retrieval - gets exactly relevant sections
- ✅ Always current - reflects latest codebase

**Example**:
- Query: "How do we handle authentication?"
- Retrieval: Finds auth-related code chunks (semantic similarity)
- Generation: LLM uses retrieved code to explain auth flow

**Tech Stack**:
- Embeddings: sentence-transformers (convert text → vectors)
- Index: FAISS (Facebook AI Similarity Search)
- Similarity: Cosine similarity between vectors
            """,
            "key_points": [
                "Combines semantic search with LLM generation",
                "No model retraining needed - just update the index",
                "FAISS provides fast similarity search",
                "Embeddings capture the meaning of code",
            ],
        },
        "training": {
            "name": "Fine-Tuning / Training",
            "summary": "Teach the model patterns from your codebase",
            "explanation": """
## Fine-Tuning: Teaching the Model Your Code

**What It Does**: Adapts a pre-trained model to your specific codebase

**The Process**:
1. **Start with base model**: Already knows general coding
2. **Prepare data**: Convert your code to examples
3. **Train**: Model learns your patterns
4. **Save**: Export the adapted model

**Key Concepts**:

**Loss**: Measures prediction error
- Lower is better
- Good: < 1.0
- Great: < 0.5
- Formula: CrossEntropyLoss between predictions and actual

**Epochs**: How many times to see the full dataset
- More epochs = more learning (but risk overfitting)
- Typical: 3-5 epochs

**Learning Rate**: How big are the weight updates?
- Too high: Training unstable
- Too low: Training too slow
- Typical: 2e-5 to 5e-5

**Batch Size**: How many examples at once?
- Larger batches: More stable, slower
- Smaller batches: Faster, less memory
- Typical: 4-16 (with gradient accumulation)

**Gradient Accumulation**: Simulate larger batches
- Train on 4 examples, update every 4 steps
- Effective batch size: 4 × 4 = 16
- Saves memory!
            """,
            "key_points": [
                "Adapts pre-trained model to your code patterns",
                "Loss measures error - lower is better",
                "Learning rate controls update step size",
                "LoRA makes training efficient (0.1% parameters)",
            ],
        },
        "embeddings": {
            "name": "Embeddings",
            "summary": "Convert code to vectors that capture meaning",
            "explanation": """
## Embeddings: Numbers That Understand Code

**What They Are**: Vectors (lists of numbers) that represent code meaning

**Example**:
```python
def login(user, password):
    ...
```
Becomes: `[0.12, -0.34, 0.56, ..., 0.78]` (384 numbers)

**How They Work**:
1. Model trained on millions of code examples
2. Learns that similar code → similar vectors
3. "login", "authenticate", "sign_in" cluster together

**Similarity**:
- Cosine similarity measures angle between vectors
- Similar code has small angle (similarity ≈ 1.0)
- Different code has large angle (similarity ≈ 0.0)

**Visualization** (simplified to 2D):
```
         auth-related
              ↑
    login •  • signin
              |
              |
    calc  •   |  • payment
              → math-related
```

**Use Cases**:
- Semantic search: Find similar code
- Clustering: Group related functions
- Recommendations: Suggest similar examples

**Tech**: Usually sentence-transformers or CodeBERT
            """,
            "key_points": [
                "Vectors (numbers) that capture code meaning",
                "Similar code has similar embeddings",
                "Used for semantic search and clustering",
                "Typical: 384-768 dimensions",
            ],
        },
    }

    @classmethod
    def explain(cls, concept: str, detailed: bool = False):
        """
        Explain a concept to the user.

        Args:
            concept: The concept to explain (lora, quantization, rag, etc.)
            detailed: Show full explanation vs summary
        """
        concept = concept.lower()
        if concept not in cls.CONCEPTS:
            console.print(f"[yellow]Unknown concept: {concept}[/yellow]")
            console.print(f"Available concepts: {', '.join(cls.CONCEPTS.keys())}")
            return

        info = cls.CONCEPTS[concept]

        if detailed:
            # Show full explanation
            console.print(f"\n[bold cyan]{info['name']}[/bold cyan]")
            console.print(Markdown(info["explanation"]))
        else:
            # Show summary
            console.print(
                Panel.fit(
                    f"[bold]{info['name']}[/bold]\n\n"
                    f"{info['summary']}\n\n"
                    f"[dim]Use --explain for full details[/dim]",
                    border_style="cyan",
                )
            )

    @classmethod
    def show_key_points(cls, concept: str):
        """Show key points for a concept."""
        concept = concept.lower()
        if concept not in cls.CONCEPTS:
            return

        info = cls.CONCEPTS[concept]
        console.print(f"\n[bold]Key Points: {info['name']}[/bold]")
        for point in info["key_points"]:
            console.print(f"  • {point}")
        console.print()


class ConfigPresets:
    """Pre-configured settings for different use cases."""

    PRESETS: ClassVar[dict] = {
        "quick": {
            "name": "Quick Test",
            "description": "Fast testing (1 epoch, small LoRA)",
            "config": {
                "training": {
                    "num_epochs": 1,
                    "batch_size": 2,
                    "learning_rate": 2e-5,
                    "lora": {
                        "r": 8,
                        "lora_alpha": 16,
                    },
                }
            },
            "use_case": "Quick iteration, testing configurations",
            "time": "~5-10 minutes",
        },
        "balanced": {
            "name": "Balanced",
            "description": "Good defaults (3 epochs, rank 16)",
            "config": {
                "training": {
                    "num_epochs": 3,
                    "batch_size": 4,
                    "learning_rate": 2e-5,
                    "lora": {
                        "r": 16,
                        "lora_alpha": 32,
                    },
                }
            },
            "use_case": "Most common use case, good quality/speed balance",
            "time": "~30-60 minutes",
        },
        "quality": {
            "name": "High Quality",
            "description": "Best results (5 epochs, rank 32)",
            "config": {
                "training": {
                    "num_epochs": 5,
                    "batch_size": 8,
                    "learning_rate": 1.5e-5,
                    "lora": {
                        "r": 32,
                        "lora_alpha": 64,
                    },
                }
            },
            "use_case": "Production use, maximum quality",
            "time": "~2-4 hours",
        },
    }

    @classmethod
    def get_preset(cls, name: str) -> dict:
        """Get a preset configuration by name."""
        return cls.PRESETS.get(name, {}).get("config", {})

    @classmethod
    def explain_preset(cls, name: str):
        """Explain what a preset does."""
        if name not in cls.PRESETS:
            console.print(f"[yellow]Unknown preset: {name}[/yellow]")
            console.print(f"Available presets: {', '.join(cls.PRESETS.keys())}")
            return

        preset = cls.PRESETS[name]
        console.print(
            Panel.fit(
                f"[bold cyan]{preset['name']}[/bold cyan]\n\n"
                f"{preset['description']}\n\n"
                f"[bold]Use case:[/bold] {preset['use_case']}\n"
                f"[bold]Est. time:[/bold] {preset['time']}\n\n"
                f"[dim]Settings:[/dim]\n"
                f"  Epochs: {preset['config']['training']['num_epochs']}\n"
                f"  LoRA rank: {preset['config']['training']['lora']['r']}\n"
                f"  Batch size: {preset['config']['training']['batch_size']}",
                border_style="cyan",
            )
        )

    @classmethod
    def list_presets(cls):
        """Show all available presets."""
        table = Table(title="Configuration Presets", box=box.ROUNDED)
        table.add_column("Preset", style="cyan")
        table.add_column("Description")
        table.add_column("Use Case")
        table.add_column("Time")

        for name, preset in cls.PRESETS.items():
            table.add_row(name, preset["description"], preset["use_case"], preset["time"])

        console.print(table)
        console.print("\n[dim]Use: nanodex train --preset <name>[/dim]\n")
