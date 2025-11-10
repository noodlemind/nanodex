# Phase 1: Foundation & Critical Fixes - Detailed Todos

## Overview
**Goal:** Fix critical issues and establish solid foundation
**Estimated Time:** 16-24 hours
**Priority:** 🔴 CRITICAL - Must complete before other phases

---

## 1.1 Fix Critical Bugs (4-6 hours)

### Task 1.1.1: Fix setup.py Entry Point ⭐⭐⭐
**File:** `setup.py`
**Current Issue:** Entry point `nanodex=main:main` is broken
**Fix Required:**
```python
# BEFORE (line 50):
"nanodex=main:main"

# AFTER:
"nanodex=nanodex.__main__:main"
```

**Additional Steps:**
- [ ] Create `nanodex/__main__.py` file
- [ ] Move main() logic from `main.py` or import it
- [ ] Test: `pip install -e .` should work
- [ ] Test: `nanodex --help` should work
- [ ] Update README with installation instructions

**Acceptance Criteria:**
- Package installs without errors
- CLI command works from anywhere
- Help text displays correctly

---

### Task 1.1.2: Add Random Seed for Reproducibility ⭐⭐⭐
**File:** `nanodex/trainers/data_preparer.py`
**Current Issue:** Line 49 - `random.shuffle(training_examples)` has no seed

**Fix Required:**
```python
# At top of DataPreparer.__init__:
self.random_seed = config.get('random_seed', 42)
random.seed(self.random_seed)
np.random.seed(self.random_seed)

# In prepare_data method:
logger.info(f"Using random seed: {self.random_seed}")
random.shuffle(training_examples)  # Now reproducible!
```

**Additional Steps:**
- [ ] Add random_seed to config.yaml (default: 42)
- [ ] Add random_seed to data section in config
- [ ] Document in config file
- [ ] Add to training logs

**Acceptance Criteria:**
- Running twice with same seed produces identical splits
- Seed is logged for debugging
- Can override seed via config

---

### Task 1.1.3: Add Dataset Validation ⭐⭐⭐
**File:** `nanodex/trainers/model_trainer.py`
**Current Issue:** No validation before expensive GPU training

**Fix Required:**
```python
def train(self, train_dataset, val_dataset):
    """Train with validation checks."""

    # Validate datasets
    if len(train_dataset) == 0:
        raise ValueError(
            "Training dataset is empty! No training examples were generated. "
            "Check your repository path and include_extensions configuration."
        )

    if len(train_dataset) < 10:
        logger.warning(
            f"Training dataset is very small ({len(train_dataset)} examples). "
            "This may not be enough for effective fine-tuning. "
            "Consider adding more code files or using synthetic data generation."
        )

    # Log dataset info
    logger.info(f"Training dataset: {len(train_dataset)} examples")
    logger.info(f"Validation dataset: {len(val_dataset)} examples")

    # Continue with training...
```

**Additional Steps:**
- [ ] Add validation in DataPreparer.prepare_data() too
- [ ] Check for duplicate examples
- [ ] Validate example format
- [ ] Add min/max example length checks

**Acceptance Criteria:**
- Clear error if no data
- Warning if insufficient data
- Logs show dataset statistics
- Fails fast before GPU allocation

---

### Task 1.1.4: Fix Path Traversal Security ⭐⭐
**File:** `nanodex/utils/config.py` and `nanodex/analyzers/code_analyzer.py`
**Current Issue:** No validation of repository path

**Fix Required in config.py:**
```python
def get_repository_config(self) -> Dict[str, Any]:
    """Get repository analysis configuration with validation."""
    repo_config = self.get('repository', {})

    # Validate repository path
    repo_path = repo_config.get('path', '.')
    repo_path = Path(repo_path).resolve()  # Resolve to absolute path

    # Security: Ensure path exists and is a directory
    if not repo_path.exists():
        raise ValueError(f"Repository path does not exist: {repo_path}")

    if not repo_path.is_dir():
        raise ValueError(f"Repository path is not a directory: {repo_path}")

    # Update with validated path
    repo_config['path'] = str(repo_path)

    return repo_config
```

**Additional Steps:**
- [ ] Add path validation
- [ ] Prevent access outside allowed directories (optional)
- [ ] Log resolved path
- [ ] Document security considerations

**Acceptance Criteria:**
- Invalid paths raise clear errors
- Paths are resolved to absolute
- Security warning if path is user-controlled

---

## 1.2 Enhanced Code Parsing (8-12 hours)

### Task 1.2.1: Create AST Parser for Python ⭐⭐⭐
**New File:** `nanodex/analyzers/ast_parser.py`

**Implementation:**
```python
"""AST-based code parser for Python."""

import ast
from typing import List, Dict, Optional
from pathlib import Path


class PythonASTParser:
    """Parse Python code using AST."""

    def parse_file(self, file_path: Path, content: str) -> Dict:
        """
        Parse Python file and extract detailed structure.

        Returns:
            {
                'functions': [...],
                'classes': [...],
                'imports': [...],
                'docstrings': [...],
                'complexity': int,
            }
        """
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            return {'error': str(e), 'functions': [], 'classes': []}

        return {
            'functions': self._extract_functions(tree),
            'classes': self._extract_classes(tree),
            'imports': self._extract_imports(tree),
            'docstrings': self._extract_docstrings(tree),
            'global_vars': self._extract_global_vars(tree),
            'complexity': self._calculate_complexity(tree),
        }

    def _extract_functions(self, tree: ast.AST) -> List[Dict]:
        """Extract function definitions."""
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append({
                    'name': node.name,
                    'args': [arg.arg for arg in node.args.args],
                    'returns': ast.unparse(node.returns) if node.returns else None,
                    'docstring': ast.get_docstring(node),
                    'lineno': node.lineno,
                    'end_lineno': node.end_lineno,
                    'body': ast.unparse(node),
                    'decorators': [ast.unparse(d) for d in node.decorator_list],
                    'is_async': isinstance(node, ast.AsyncFunctionDef),
                })
        return functions

    # TODO: Implement other methods
```

**Checklist:**
- [ ] Implement `_extract_functions`
- [ ] Implement `_extract_classes`
- [ ] Implement `_extract_imports`
- [ ] Implement `_extract_docstrings`
- [ ] Implement `_calculate_complexity`
- [ ] Handle syntax errors gracefully
- [ ] Add unit tests
- [ ] Document return format

**Acceptance Criteria:**
- Extracts all function signatures
- Captures docstrings correctly
- Handles decorators and async functions
- Returns structured data
- Doesn't crash on syntax errors

---

### Task 1.2.2: Integrate Tree-Sitter for Multi-Language ⭐⭐
**New File:** `nanodex/analyzers/tree_sitter_parser.py`

**Steps:**
- [ ] Install tree-sitter and language grammars
- [ ] Create TreeSitterParser class
- [ ] Support JavaScript/TypeScript
- [ ] Support Java, C++, Go, Rust
- [ ] Unified output format across languages
- [ ] Fallback to regex if tree-sitter fails

**Dependencies:**
```bash
pip install tree-sitter
pip install tree-sitter-python tree-sitter-javascript
pip install tree-sitter-java tree-sitter-cpp
```

**Acceptance Criteria:**
- Works for at least Python, JavaScript, TypeScript
- Returns same structure as AST parser
- Performance: <100ms per file
- Graceful degradation if language not supported

---

### Task 1.2.3: Build Dependency Graph ⭐⭐
**New File:** `nanodex/analyzers/dependency_graph.py`

**Implementation:**
```python
"""Build dependency graph from import statements."""

from typing import Dict, List, Set
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DependencyGraph:
    """Build and analyze code dependencies."""

    def __init__(self, code_samples: List[Dict]):
        self.code_samples = code_samples
        self.graph = self._build_graph()

    def _build_graph(self) -> Dict[str, Dict]:
        """
        Build dependency graph.

        Returns:
            {
                'file_path': {
                    'imports': ['file1', 'file2'],
                    'imported_by': ['file3', 'file4'],
                    'depth': 2,
                }
            }
        """
        graph = {}

        # First pass: collect all imports
        for sample in self.code_samples:
            file_path = sample['file_path']
            imports = sample.get('imports', [])

            graph[file_path] = {
                'imports': self._resolve_imports(imports),
                'imported_by': [],
                'depth': 0,
            }

        # Second pass: build reverse dependencies
        for file_path, data in graph.items():
            for imported_file in data['imports']:
                if imported_file in graph:
                    graph[imported_file]['imported_by'].append(file_path)

        # Calculate depths
        self._calculate_depths(graph)

        return graph

    def get_dependencies(self, file_path: str, recursive: bool = False) -> List[str]:
        """Get all dependencies of a file."""
        # TODO: Implement
        pass

    def get_dependents(self, file_path: str) -> List[str]:
        """Get all files that depend on this file."""
        # TODO: Implement
        pass

    def concatenate_with_deps(self, file_path: str, max_depth: int = 2) -> str:
        """Concatenate file with its dependencies (DeepSeek approach)."""
        # TODO: Implement
        pass
```

**Checklist:**
- [ ] Build import graph
- [ ] Calculate dependency depths
- [ ] Find circular dependencies
- [ ] Concatenate files with deps
- [ ] Export as JSON
- [ ] Visualize graph (optional)

**Acceptance Criteria:**
- Correctly identifies all dependencies
- Detects circular dependencies
- Can concatenate related files
- Performance: <1s for 1000 files

---

### Task 1.2.4: Enhance CodeAnalyzer ⭐⭐⭐
**File to Modify:** `nanodex/analyzers/code_analyzer.py`

**Changes Required:**
```python
class CodeAnalyzer:
    def __init__(self, config: Dict):
        self.repo_path = Path(config.get('path', '.'))
        self.include_extensions = set(config.get('include_extensions', []))
        self.exclude_dirs = set(config.get('exclude_dirs', []))
        self.max_file_size = config.get('max_file_size', 1048576)

        # NEW: Deep parsing configuration
        self.deep_parsing = config.get('deep_parsing', {})
        if self.deep_parsing.get('enabled', True):
            from .ast_parser import PythonASTParser
            from .tree_sitter_parser import TreeSitterParser

            self.ast_parser = PythonASTParser()
            self.tree_sitter_parser = TreeSitterParser()

    def _extract_code_sample(self, file_path: Path) -> Dict[str, str]:
        """Extract code with deep parsing."""
        # ... existing code to read file ...

        # NEW: Deep parsing
        if self.deep_parsing.get('enabled', True):
            parsed_data = self._parse_structure(file_path, content, language)
        else:
            parsed_data = {}

        return {
            'file_path': str(rel_path),
            'language': language,
            'content': content,
            'lines': len(content.splitlines()),

            # NEW: Structured data
            **parsed_data
        }

    def _parse_structure(self, file_path: Path, content: str, language: str) -> Dict:
        """Parse code structure based on language."""
        if language == 'python':
            return self.ast_parser.parse_file(file_path, content)
        else:
            return self.tree_sitter_parser.parse_file(file_path, content, language)
```

**Checklist:**
- [ ] Integrate AST parser
- [ ] Integrate tree-sitter parser
- [ ] Add configuration for deep parsing
- [ ] Maintain backward compatibility
- [ ] Update tests
- [ ] Log parsing statistics

**Acceptance Criteria:**
- Extracts detailed structure when enabled
- Falls back to basic mode if disabled
- Doesn't break existing functionality
- Performance: <10% slower than current

---

## 1.3 Configuration System Overhaul (4-6 hours)

### Task 1.3.1: Implement Pydantic Validation ⭐⭐⭐
**File to Modify:** `nanodex/utils/config.py`

**Implementation:**
```python
"""Configuration management with Pydantic validation."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, validator
import logging

logger = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    """Model configuration schema."""
    model_name: str = Field(..., min_length=1, description="HuggingFace model name")
    use_4bit: bool = False
    use_8bit: bool = False
    trust_remote_code: bool = False

    @validator('model_name')
    def validate_model_name(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("model_name cannot be empty")
        return v.strip()

    @validator('use_8bit')
    def validate_quantization(cls, v, values):
        if 'use_4bit' in values and values['use_4bit'] and v:
            raise ValueError("Cannot use both 4-bit and 8-bit quantization")
        return v


class RepositoryConfig(BaseModel):
    """Repository analysis configuration."""
    path: str = Field(".", description="Path to codebase")
    include_extensions: List[str] = Field(default_factory=list)
    exclude_dirs: List[str] = Field(default_factory=list)
    exclude_patterns: List[str] = Field(default_factory=list)
    max_file_size: int = Field(1048576, gt=0)
    deep_parsing: Dict[str, Any] = Field(default_factory=dict)


class DataGenerationConfig(BaseModel):
    """Data generation configuration."""
    mode: str = Field("free", regex="^(free|hybrid|synthetic_only)$")
    self_supervised: Dict[str, Any] = Field(default_factory=dict)
    synthetic_data: Dict[str, Any] = Field(default_factory=dict)


class TrainingConfig(BaseModel):
    """Training configuration."""
    output_dir: str = "./models/fine-tuned"
    num_epochs: int = Field(3, gt=0, le=100)
    batch_size: int = Field(4, gt=0)
    learning_rate: float = Field(2e-5, gt=0)
    max_seq_length: int = Field(2048, gt=0, le=16384)
    # ... more fields


class Config:
    """Main configuration class with validation."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.raw_config = self._load_yaml()

        # Validate using Pydantic
        self._validate_config()

    def _validate_config(self):
        """Validate configuration using Pydantic models."""
        try:
            model_source = self.raw_config.get('model_source', 'huggingface')
            model_config_dict = self.raw_config.get('model', {}).get(model_source, {})

            self.model_config = ModelConfig(**model_config_dict)
            self.repository_config = RepositoryConfig(**self.raw_config.get('repository', {}))
            self.data_gen_config = DataGenerationConfig(**self.raw_config.get('data_generation', {}))
            # ... validate other sections

            logger.info("Configuration validated successfully")
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
```

**Checklist:**
- [ ] Create Pydantic models for all config sections
- [ ] Add validators for each field
- [ ] Provide helpful error messages
- [ ] Add default values
- [ ] Document all fields
- [ ] Test validation

**Acceptance Criteria:**
- Invalid config raises clear errors
- All fields have types and validation
- Helpful error messages guide users
- Defaults work out of box

---

### Task 1.3.2: Update config.yaml ⭐⭐
**File to Modify:** `config.yaml`

**Steps:**
- [ ] Add data_generation section
- [ ] Add random_seed to data section
- [ ] Add deep_parsing to repository section
- [ ] Add evaluation section
- [ ] Add rag section
- [ ] Update comments and documentation
- [ ] Test with validation

**Acceptance Criteria:**
- All new fields present
- Well-documented with comments
- Validates successfully
- Backward compatible where possible

---

## Phase 1 Testing Checklist

### Unit Tests
- [ ] Test setup.py entry point
- [ ] Test AST parser on various Python files
- [ ] Test tree-sitter on JS/TS files
- [ ] Test dependency graph building
- [ ] Test configuration validation
- [ ] Test dataset validation

### Integration Tests
- [ ] Test full pipeline with new parsing
- [ ] Test with empty repository
- [ ] Test with minimal repository (1-2 files)
- [ ] Test with large repository (100+ files)
- [ ] Test error handling

### Manual Testing
- [ ] Install package and run CLI
- [ ] Run with free mode config
- [ ] Check logs are helpful
- [ ] Verify reproducibility with seed
- [ ] Test on real codebase

---

## Phase 1 Definition of Done

✅ All critical bugs fixed
✅ Setup.py entry point works
✅ Random seed ensures reproducibility
✅ Dataset validation prevents empty training
✅ Path traversal security addressed
✅ AST parsing extracts functions, classes, docstrings
✅ Tree-sitter supports 3+ languages
✅ Dependency graph builds correctly
✅ Configuration validates with Pydantic
✅ All tests pass
✅ Documentation updated
✅ Code reviewed and committed

---

## Next Steps After Phase 1

Once Phase 1 is complete and tested:
1. Review with user
2. Get approval to proceed to Phase 2
3. Start implementing self-supervised data generators
4. Build on the solid foundation established

**Questions for User:**
1. Should we proceed with Phase 1 implementation?
2. Any specific priorities within Phase 1?
3. Any features to add/remove/modify?
4. Preferred order of tasks?
