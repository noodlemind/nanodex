.PHONY: help setup clean extract graph-inspect brain brain-embed dataset train-qlora train-lora serve query test lint format type-check

# Default target
help:
	@echo "nanodex - Knowledge Graph Pipeline"
	@echo ""
	@echo "Available targets:"
	@echo "  setup         - Create virtual environment and install dependencies"
	@echo "  clean         - Remove generated files and caches"
	@echo "  extract       - Extract knowledge graph from repository"
	@echo "  graph-inspect - Inspect graph database statistics"
	@echo "  brain         - Classify nodes and generate summaries"
	@echo "  brain-embed   - Generate embeddings for summaries (optional)"
	@echo "  dataset       - Generate training Q&A dataset"
	@echo "  train-qlora   - Train QLoRA adapter (4-bit, 12-24GB VRAM)"
	@echo "  train-lora    - Train LoRA adapter (FP16, 24-40GB VRAM)"
	@echo "  serve         - Print instructions for starting inference server"
	@echo "  query         - Query the inference server (requires server running)"
	@echo "  test          - Run tests with coverage"
	@echo "  lint          - Run linting with ruff"
	@echo "  format        - Format code with black"
	@echo "  type-check    - Run type checking with mypy"
	@echo ""
	@echo "Example usage:"
	@echo "  make setup"
	@echo "  make extract REPO=/path/to/repo CONFIG=config/extract.yaml"
	@echo "  make graph-inspect DB=data/brain/graph.sqlite"
	@echo "  make brain CONFIG=config/brain.yaml"
	@echo "  make brain-embed MODEL=sentence-transformers/all-MiniLM-L6-v2"
	@echo "  make dataset CONFIG=config/dataset.yaml"
	@echo "  make train-qlora CONFIG=config/train_qlora.yaml"
	@echo "  make train-lora CONFIG=config/train_lora.yaml"
	@echo "  make serve CONFIG=config/inference.yaml"
	@echo "  make query QUESTION='How does X work?' ENDPOINT=http://localhost:8000"

# Variables
PYTHON := python3
VENV := .venv
BIN := $(VENV)/bin
REPO ?= .
CONFIG ?= config/extract.yaml
DB ?= data/brain/graph.sqlite

# Detect if uv is available
UV := $(shell command -v uv 2> /dev/null)

# Setup virtual environment (uses uv if available, otherwise pip)
setup:
ifdef UV
	@echo "Using uv (fast mode)"
	uv venv
	uv pip install -r requirements.txt
	uv pip install -r requirements-dev.txt
else
	@echo "Using pip (uv not found - consider installing: https://github.com/astral-sh/uv)"
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt
	$(BIN)/pip install -r requirements-dev.txt
endif
	@echo ""
	@echo "Setup complete! Activate with: source $(VENV)/bin/activate"

# Clean generated files
clean:
	rm -rf data/brain/*.sqlite
	rm -rf data/brain/nodes/
	rm -rf data/dataset/
	rm -rf models/
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete
	find . -type f -name '*~' -delete
	rm -rf .pytest_cache .coverage htmlcov

# Extract knowledge graph from repository
extract:
	$(BIN)/python scripts/extract.py $(REPO) --config $(CONFIG)

# Inspect graph statistics
graph-inspect:
	$(BIN)/python scripts/inspect_graph.py --db $(DB) --check-integrity

# Build brain: classify nodes and generate summaries
brain:
	$(BIN)/python scripts/build_brain.py --config $(CONFIG) --db $(DB)

# Generate embeddings for summaries (optional)
brain-embed:
	$(BIN)/python scripts/embed_summaries.py --config $(CONFIG) $(if $(MODEL),--model $(MODEL),)

# Generate training dataset
dataset:
	$(BIN)/python scripts/generate_dataset.py --config $(CONFIG) --db $(DB)

# Train QLoRA adapter (4-bit quantization, 12-24GB VRAM)
train-qlora:
	$(BIN)/python scripts/train.py --config $(if $(CONFIG),$(CONFIG),config/train_qlora.yaml) $(if $(DATASET),--dataset $(DATASET),) $(if $(EPOCHS),--epochs $(EPOCHS),) $(if $(BATCH_SIZE),--batch-size $(BATCH_SIZE),)

# Train LoRA adapter (FP16, 24-40GB VRAM)
train-lora:
	$(BIN)/python scripts/train.py --config $(if $(CONFIG),$(CONFIG),config/train_lora.yaml) $(if $(DATASET),--dataset $(DATASET),) $(if $(EPOCHS),--epochs $(EPOCHS),) $(if $(BATCH_SIZE),--batch-size $(BATCH_SIZE),)

# Print instructions for starting inference server
serve:
	$(BIN)/python scripts/serve.py --config $(if $(CONFIG),$(CONFIG),config/inference.yaml) $(if $(ADAPTER),--adapter $(ADAPTER),) $(if $(PORT),--port $(PORT),)

# Query inference server
query:
	$(BIN)/python scripts/query.py --endpoint $(if $(ENDPOINT),$(ENDPOINT),http://localhost:8000) $(if $(QUESTION),--question "$(QUESTION)",--interactive) $(if $(MAX_TOKENS),--max-tokens $(MAX_TOKENS),) $(if $(TEMPERATURE),--temperature $(TEMPERATURE),)

# Run tests
test:
	$(BIN)/pytest -v

# Run linting
lint:
	$(BIN)/ruff check nanodex/ tests/ scripts/

# Format code
format:
	$(BIN)/black nanodex/ tests/ scripts/
	$(BIN)/ruff check --fix nanodex/ tests/ scripts/

# Type checking
type-check:
	$(BIN)/mypy nanodex/

# Install package in development mode
install-dev:
	$(BIN)/pip install -e .
