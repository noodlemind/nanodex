# Contributing to nanodex

## Development Setup

### Prerequisites
- Python ≥ 3.10
- Git
- (Optional) CUDA toolkit for training

### Installation

**Recommended: Using uv (10-100x faster)**

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/yourusername/nanodex.git
cd nanodex
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
uv pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Verify installation
make test
```

**Alternative: Using pip**

```bash
git clone https://github.com/yourusername/nanodex.git
cd nanodex
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Verify installation
make test
```

## Code Quality Standards

### Pre-commit Hooks (Recommended)

We use pre-commit hooks to automatically check code quality before each commit.

```bash
# Install hooks (one-time setup)
pre-commit install

# Hooks will run automatically on git commit
# To run manually on all files:
pre-commit run --all-files
```

The pre-commit hooks will automatically:
- Format code with Black
- Lint with Ruff (and auto-fix issues)
- Type-check with MyPy
- Check for trailing whitespace, EOF newlines, YAML syntax, etc.

### Formatting with Black

We use Black for consistent code formatting (line length: 100).

```bash
# Check formatting
make format-check
# or
black --check nanodex/ scripts/ tests/

# Auto-format
make format
# or
black nanodex/ scripts/ tests/
```

### Linting with Ruff

```bash
# Check linting
make lint
# or
ruff check nanodex/ tests/ scripts/

# Auto-fix
ruff check --fix nanodex/ tests/ scripts/
```

### Type Checking with MyPy

All code must have type annotations.

```bash
# Check types
make type-check
# or
mypy nanodex/
```

**Type Hints Required**:
- All function signatures must have type hints
- Use `Optional[T]` for nullable types
- Use `List[T]`, `Dict[K, V]` for collections

### Testing

```bash
# Run all tests
make test

# Run with coverage
pytest -v --cov=nanodex --cov-report=html

# Run specific test
pytest tests/unit/test_qa_generator.py -v

# Run integration tests only
pytest tests/integration/ -v
```

**Test Requirements**:
- All new features must have tests
- Minimum 80% coverage for new code
- Use fixtures from `tests/conftest.py`

## Pull Request Process

### 1. Create Feature Branch

```bash
git checkout -b feat/your-feature-name
# or
git checkout -b fix/bug-description
```

### 2. Make Changes

- Write code following style guidelines
- Add/update tests
- Update documentation if needed

### 3. Run Quality Checks

```bash
# Format code
make format

# Run linting
make lint

# Run type checking
make type-check

# Run tests
make test
```

### 4. Commit

Follow conventional commits format:

```bash
git commit -m "feat: add support for Rust language parsing"
git commit -m "fix: correct SQL query in get_neighbors"
git commit -m "docs: update architecture documentation"
```

**Commit Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `test`: Adding/updating tests
- `refactor`: Code change that neither fixes bug nor adds feature
- `perf`: Performance improvement
- `chore`: Build/tooling changes

### 5. Push and Create PR

```bash
git push origin feat/your-feature-name
```

Then create a pull request on GitHub with:
- Clear description of changes
- Link to related issues
- Screenshots/examples if applicable

## Project Structure

```
nanodex/
├── nanodex/           # Main package
│   ├── extractor/     # Code parsing and graph building
│   ├── brain/         # Node classification and summarization
│   ├── dataset/       # Q&A generation and validation
│   ├── trainer/       # LoRA/QLoRA training
│   └── inference/     # Model serving and querying
├── config/            # YAML configuration templates
├── scripts/           # CLI entry points
├── tests/             # Test suite
│   ├── unit/          # Unit tests
│   └── integration/   # Integration tests
└── docs/              # Documentation
```

## Adding New Features

### Adding a New Language to Extractor

1. Add Tree-sitter grammar dependency
2. Create query file: `nanodex/extractor/queries/{lang}_queries.scm`
3. Update `TreeSitterParser._get_query()`
4. Add language to supported list in config validation
5. Add tests in `tests/unit/test_tree_sitter_parser.py`

### Adding a New Q&A Category

1. Add category to `QA_CATEGORIES` in `qa_generator.py`
2. Implement `generate_{category}_questions()` method
3. Update `generate_all_qa()` to include new category
4. Add config validation in `DatasetConfig`
5. Add tests in `tests/unit/test_qa_generator.py`

### Adding a New Node Type

1. Add type to `NODE_TYPES` in `brain/node_typer.py`
2. Implement classification logic in `_classify_node()`
3. Update config validation in `BrainConfig`
4. Update summary generation templates if needed
5. Add tests

## Common Tasks

### Running End-to-End Pipeline

```bash
# Extract
make extract REPO=/path/to/target

# Build brain
make brain

# Generate dataset
make dataset

# Check results
sqlite3 data/brain/graph.sqlite "SELECT COUNT(*) FROM nodes;"
ls data/brain/nodes/ | wc -l
wc -l data/dataset/train.jsonl
```

### Debugging

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python scripts/extract.py /path/to/repo --config config/extract.yaml

# Inspect graph
make graph-inspect
sqlite3 data/brain/graph.sqlite
```

### Profiling

```bash
# Time extraction
time make extract REPO=/path/to/repo

# Memory profiling
python -m memory_profiler scripts/extract.py /path/to/repo
```

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def extract_symbols(self, code: str, language: str) -> Tuple[List[Node], List[Edge]]:
    """
    Extract symbols and relationships from source code.

    Args:
        code: Source code to parse
        language: Programming language (python, java, etc.)

    Returns:
        Tuple of (nodes, edges) extracted from the code

    Raises:
        ValueError: If language is not supported
    """
```

### Updating Documentation

- `README.md`: User-facing guide and quick start
- `docs/ARCHITECTURE.md`: System design and data flow
- `docs/API_REFERENCE.md`: Module/function documentation
- `CONTRIBUTING.md`: This file

## Getting Help

- **Issues**: Open a GitHub issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Code Review**: Tag maintainers in PRs for review

## License

By contributing, you agree that your contributions will be licensed under Apache-2.0.
