# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Restructured codebase following Python best practices and nanoGPT philosophy
- Moved test scripts to `tests/` directory for better organization
- Moved `demo.py` to `examples/` directory
- Renamed `TurboCodeGPTConfig` to `NanodexConfig` for consistent naming

### Added
- `pyproject.toml` for modern Python packaging (PEP 621)
- `pytest.ini` for test configuration
- `CHANGELOG.md` to track version history
- Tool configuration for Black and mypy in pyproject.toml

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
