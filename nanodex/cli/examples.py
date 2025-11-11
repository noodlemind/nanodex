"""
Examples command showing common workflows.

Provides quick-start examples for new users to understand
how to use nanodex effectively.
"""

import click
from rich.console import Console
from rich.markdown import Markdown

console = Console()

EXAMPLES = """
# nanodex Quick Start Examples

## 1️⃣  First Time Setup
```bash
nanodex init
```
Interactive wizard to configure your project

## 2️⃣  Analyze Your Codebase
```bash
nanodex analyze --config config.yaml
```
Understand code structure and complexity

## 3️⃣  Generate Training Data
```bash
nanodex data generate --mode free
nanodex data preview
```
Extract patterns from your codebase (free mode)

## 4️⃣  Train Your Model
```bash
nanodex train --epochs 3
```
Fine-tune model on your codebase (2-8 hours)

## 5️⃣  Build RAG Index (Optional)
```bash
nanodex rag index
nanodex rag search "authentication logic"
```
Semantic code search

## 6️⃣  Chat with Your Assistant
```bash
nanodex chat --model ./models/final
```
Interactive conversation with fine-tuned model

## 💡 Pro Tips
- Use `nanodex shell` for iterative workflows
- Run `nanodex status` to check pipeline progress (coming soon)
- Add `--help` to any command for details

## 📖 Full Documentation
https://github.com/noodlemind/nanodex/tree/main/docs
"""


@click.command("examples")
def examples_cmd():
    """
    Show common workflow examples

    Display quick-start examples and common usage patterns to help
    new users get started with nanodex quickly.

    \b
    Examples:
        nanodex examples
    """
    console.print()
    console.print(Markdown(EXAMPLES))
    console.print()
