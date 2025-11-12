# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Interactive Shell Mode** (`nanodex shell`) - REPL interface with persistent context (#11)
  - Command history with auto-suggestions using `click-repl`
  - Context persists between commands for faster iteration
  - History stored in `.nanodex_history` file
  - Tab completion for commands
- **Examples Command** (`nanodex examples`) - Quick-start guide for new users (#11)
  - Step-by-step workflow examples
  - Pro tips and best practices
  - Common usage patterns
- **Enhanced Chat Interface** - Improved UX with Prompt Toolkit (#11)
  - Persistent chat history in `.nanodex_chat_history`
  - Fish-shell style auto-suggestions
  - Ctrl+R reverse history search
  - Arrow key navigation through history
- **Status Command** (`nanodex status`) - Pipeline progress tracking (#11)
  - Shows completion status of all pipeline steps
  - Displays resource utilization (RAM, disk, GPU if available)
  - Suggests next recommended action
  - Overall progress percentage
- **Pipeline Command** (`nanodex pipeline`) - Workflow guidance (#11)
  - `nanodex pipeline guide` - Complete walkthrough of the pipeline
  - `nanodex pipeline check` - Quick progress check (alias for status)
  - Step-by-step instructions with examples
- **Enhanced Error Utilities** - Better error messages with suggestions (#11)
  - Actionable error messages in `nanodex/utils/errors.py`
  - Helpful suggestions for common mistakes
  - Example commands for resolution
- **Pipeline State Tracking** - Infrastructure for progress monitoring (#11)
  - State persistence in `.nanodex_state.json`
  - Track completion status of each pipeline step
  - Automatic detection of completed steps

### Changed
- Restructured codebase following Python best practices and nanoGPT philosophy
- Moved test scripts to `tests/` directory for better organization
- Moved `demo.py` to `examples/` directory
- Renamed `TurboCodeGPTConfig` to `NanodexConfig` for consistent naming
- Updated chat interface to use Prompt Toolkit for better UX (#11)
- Enhanced CLI help text to include new commands (#11)

### Added (Infrastructure)
- `pyproject.toml` for modern Python packaging (PEP 621)
- `pytest.ini` for test configuration
- `CHANGELOG.md` to track version history
- Tool configuration for Black and mypy in pyproject.toml
- Dependencies: `click-repl>=0.3.0`, `prompt-toolkit>=3.0.0` (#11)

### Fixed
- Updated all references from `turbo_code_gpt` to `nanodex` throughout codebase
- Fixed version inconsistency between `setup.py` (0.2.0) and `__init__.py` (0.1.0)
- Updated configuration file comments to reference correct module paths

### Removed
- Duplicate `docs/index.md` file

## [0.2.0] - 2024

### Added
- Modern CLI interface using Click and Rich
- RAG (Retrieval-Augmented Generation) infrastructure with FAISS indexing
- Interactive chat interface for model interaction
- Semantic code search capabilities
- Multiple data generation modes (free, hybrid, full)
- LoRA-based efficient fine-tuning with 4-bit quantization
- Support for multiple model sources (HuggingFace, Ollama)
- Comprehensive documentation structure

### Changed
- Project renamed from turbo-code-gpt to nanodex
- Improved configuration management with Pydantic validation
- Enhanced code analysis and dependency tracking

## [0.1.0] - Initial Release

### Added
- Basic fine-tuning capabilities for coding models
- Code analysis and AST parsing
- Training data generation
- Model evaluation framework
- Basic CLI interface
- Documentation and examples

[Unreleased]: https://github.com/noodlemind/nanodex/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/noodlemind/nanodex/releases/tag/v0.2.0
[0.1.0]: https://github.com/noodlemind/nanodex/releases/tag/v0.1.0
