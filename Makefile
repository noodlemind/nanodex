.PHONY: help setup clean extract graph-inspect test lint format type-check

# Default target
help:
	@echo "nanodex - Knowledge Graph Pipeline"
	@echo ""
	@echo "Available targets:"
	@echo "  setup         - Create virtual environment and install dependencies"
	@echo "  clean         - Remove generated files and caches"
	@echo "  extract       - Extract knowledge graph from repository"
	@echo "  graph-inspect - Inspect graph database statistics"
	@echo "  test          - Run tests with coverage"
	@echo "  lint          - Run linting with ruff"
	@echo "  format        - Format code with black"
	@echo "  type-check    - Run type checking with mypy"
	@echo ""
	@echo "Example usage:"
	@echo "  make setup"
	@echo "  make extract REPO=/path/to/repo CONFIG=config/extract.yaml"
	@echo "  make graph-inspect DB=data/brain/graph.sqlite"

# Variables
PYTHON := python3
VENV := .venv
BIN := $(VENV)/bin
REPO ?= .
CONFIG ?= config/extract.yaml
DB ?= data/brain/graph.sqlite

# Setup virtual environment
setup:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt
	$(BIN)/pip install -r requirements-dev.txt
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
