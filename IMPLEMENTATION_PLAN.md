# Turbo-Code-GPT: Flexible Architecture Implementation Plan

## Executive Summary

Transform Turbo-Code-GPT into a flexible, production-ready system that allows users to choose their training approach based on budget and quality needs:

- **Mode 1: Free** - Self-supervised learning from codebase only ($0 API cost, ~$10-30 GPU)
- **Mode 2: Hybrid** - Self-supervised + paid synthetic data ($20-100 API + GPU)
- **Mode 3: Full Training** - Train from scratch like nanochat ($100-1000+)

## Current State Analysis

### What Works
✅ Basic pipeline (analyze → prepare → train)
✅ LoRA fine-tuning with quantization
✅ Multi-language support
✅ Clean modular structure
✅ Comprehensive documentation

### Critical Issues to Fix
🔴 **Training data is too generic** - Won't teach actual code understanding
🔴 **No code structure parsing** - Just reads raw text
🔴 **Missing RAG infrastructure** - No retrieval at inference time
🔴 **No reproducibility** - No random seed
🔴 **No checkpoint recovery** - Training failures waste GPU time
🔴 **Broken setup.py entry point** - Package installation fails

### Architecture Gaps
⚠️ No pluggable data generation system
⚠️ No evaluation framework
⚠️ No configuration validation
⚠️ Unused dependencies (tree-sitter, gitpython)
⚠️ Missing export functionality

---

## Implementation Phases

## PHASE 1: Foundation & Critical Fixes (Week 1)
**Goal:** Fix critical issues and establish solid foundation

### 1.1 Fix Critical Bugs
- [ ] Fix setup.py entry point (`turbo-code-gpt=main:main` → `turbo_code_gpt.__main__:main`)
- [ ] Add random seed for reproducibility
- [ ] Add validation before training (check non-empty datasets)
- [ ] Fix path traversal security issue

**Files to modify:**
- `setup.py`
- `turbo_code_gpt/trainers/data_preparer.py`
- `turbo_code_gpt/trainers/model_trainer.py`
- `turbo_code_gpt/utils/config.py`

**Estimated time:** 4-6 hours

### 1.2 Enhanced Code Parsing
- [ ] Integrate AST parsing for Python
- [ ] Add tree-sitter for multi-language support
- [ ] Extract: functions, classes, docstrings, imports, type hints
- [ ] Calculate code complexity
- [ ] Build dependency graph

**New files:**
- `turbo_code_gpt/analyzers/ast_parser.py`
- `turbo_code_gpt/analyzers/dependency_graph.py`

**Files to enhance:**
- `turbo_code_gpt/analyzers/code_analyzer.py`

**Estimated time:** 8-12 hours

### 1.3 Configuration System Overhaul
- [ ] Implement Pydantic schema validation
- [ ] Add mode selection (free/hybrid/paid)
- [ ] Add data generation configuration
- [ ] Add RAG configuration
- [ ] Add evaluation configuration

**Files to modify:**
- `turbo_code_gpt/utils/config.py`
- `config.yaml` (update to new format)

**Estimated time:** 4-6 hours

**Phase 1 Total:** 16-24 hours
**Deliverable:** Stable foundation with critical bugs fixed

---

## PHASE 2: Flexible Data Generation (Week 2)
**Goal:** Implement pluggable data generation supporting all modes

### 2.1 Self-Supervised Generator (FREE MODE)
- [ ] Implement docstring-based learning
- [ ] Implement fill-in-the-middle (FIM) tasks
- [ ] Extract from test files
- [ ] Use git history for context
- [ ] Extract type hints
- [ ] Generate dependency relationship examples
- [ ] Run static analysis (mypy/pylint) for error examples

**New files:**
- `turbo_code_gpt/data_generators/self_supervised.py`
- `turbo_code_gpt/data_generators/fim_generator.py`
- `turbo_code_gpt/data_generators/test_extractor.py`
- `turbo_code_gpt/data_generators/git_history.py`

**Estimated time:** 12-16 hours

### 2.2 Synthetic API Generator (PAID MODE)
- [ ] Implement OpenAI integration
- [ ] Implement Anthropic integration
- [ ] Add cost tracking and budget caps
- [ ] Generate diverse Q&A using LLM
- [ ] Template-based prompt generation
- [ ] Few-shot examples for quality

**New files:**
- `turbo_code_gpt/data_generators/synthetic_api.py`
- `turbo_code_gpt/data_generators/prompt_templates.py`

**Estimated time:** 8-12 hours

### 2.3 Data Generation Orchestration
- [ ] Implement FlexibleDataGenerator
- [ ] Cost estimation for all modes
- [ ] Progress tracking and logging
- [ ] Data quality filtering
- [ ] Deduplication

**New files:**
- `turbo_code_gpt/data_generators/orchestrator.py`
- `turbo_code_gpt/data_generators/quality_filter.py`

**Files to modify:**
- `turbo_code_gpt/trainers/data_preparer.py`

**Estimated time:** 6-8 hours

**Phase 2 Total:** 26-36 hours
**Deliverable:** Flexible data generation supporting free, hybrid, and paid modes

---

## PHASE 3: RAG Infrastructure (Week 3)
**Goal:** Build retrieval-augmented generation for inference

### 3.1 Vector Index Building
- [ ] Implement code embedding
- [ ] FAISS index creation
- [ ] Chunk code intelligently (by function/class)
- [ ] Metadata storage
- [ ] Index persistence

**New files:**
- `turbo_code_gpt/rag/__init__.py`
- `turbo_code_gpt/rag/indexer.py`
- `turbo_code_gpt/rag/embedder.py`
- `turbo_code_gpt/rag/chunker.py`

**Estimated time:** 8-10 hours

### 3.2 Retrieval System
- [ ] Semantic search implementation
- [ ] Hybrid search (keyword + semantic)
- [ ] Context assembly
- [ ] Relevance scoring
- [ ] Retrieval optimization

**New files:**
- `turbo_code_gpt/rag/retriever.py`
- `turbo_code_gpt/rag/hybrid_search.py`

**Estimated time:** 6-8 hours

### 3.3 RAG-Augmented Inference
- [ ] Integrate RAG with model inference
- [ ] Context-aware prompting
- [ ] Response generation
- [ ] Update inference examples

**Files to modify:**
- `examples/inference_example.py`

**New files:**
- `turbo_code_gpt/inference/rag_inference.py`

**Estimated time:** 6-8 hours

**Phase 3 Total:** 20-26 hours
**Deliverable:** Working RAG system for code retrieval and inference

---

## PHASE 4: Training Improvements (Week 4)
**Goal:** Production-ready training with proper evaluation

### 4.1 Enhanced Training Pipeline
- [ ] Checkpoint recovery
- [ ] Early stopping
- [ ] Better progress tracking
- [ ] Multi-GPU support (optional)
- [ ] Training resumption
- [ ] Model versioning

**Files to modify:**
- `turbo_code_gpt/trainers/model_trainer.py`
- `main.py`

**Estimated time:** 8-10 hours

### 4.2 Evaluation Framework
- [ ] Code understanding metrics
- [ ] Function identification accuracy
- [ ] Dependency tracking accuracy
- [ ] Error localization accuracy
- [ ] Hold-out test set
- [ ] Report generation

**New files:**
- `turbo_code_gpt/evaluation/__init__.py`
- `turbo_code_gpt/evaluation/evaluator.py`
- `turbo_code_gpt/evaluation/metrics.py`
- `turbo_code_gpt/evaluation/report_generator.py`

**Estimated time:** 10-12 hours

### 4.3 Full Training Mode (Nanochat-style)
- [ ] Pretraining on public corpus
- [ ] Midtraining on codebase
- [ ] Supervised fine-tuning
- [ ] Optional RL training
- [ ] Multi-stage orchestration

**New files:**
- `turbo_code_gpt/trainers/scratch_trainer.py`
- `turbo_code_gpt/trainers/pretraining.py`

**Estimated time:** 12-16 hours (optional, advanced feature)

**Phase 4 Total:** 18-22 hours (without full training) or 30-38 hours (with full training)
**Deliverable:** Production-ready training with evaluation

---

## PHASE 5: Integration & Polish (Week 5)
**Goal:** End-to-end integration and user experience

### 5.1 Main Pipeline Integration
- [ ] Update main.py to support all modes
- [ ] CLI argument improvements
- [ ] Better logging and progress bars
- [ ] Error handling and recovery
- [ ] User-friendly messages

**Files to modify:**
- `main.py`

**New files:**
- `turbo_code_gpt/__main__.py` (for proper package execution)
- `turbo_code_gpt/cli/__init__.py`
- `turbo_code_gpt/cli/commands.py`

**Estimated time:** 8-10 hours

### 5.2 Testing
- [ ] Unit tests for all new modules
- [ ] Integration tests
- [ ] End-to-end tests for each mode
- [ ] Test coverage >70%
- [ ] CI/CD setup (optional)

**New files:**
- `tests/test_data_generators.py`
- `tests/test_rag.py`
- `tests/test_evaluation.py`
- `tests/test_integration.py`

**Estimated time:** 12-16 hours

### 5.3 Documentation Updates
- [ ] Update README with new modes
- [ ] Update GETTING_STARTED guide
- [ ] Document configuration options
- [ ] Add mode selection guide
- [ ] Add troubleshooting for new features
- [ ] API documentation

**Files to update:**
- `README.md`
- `GETTING_STARTED.md`
- `ARCHITECTURE.md`
- `HOW_IT_WORKS.md`

**New files:**
- `docs/MODE_SELECTION_GUIDE.md`
- `docs/API_REFERENCE.md`

**Estimated time:** 6-8 hours

### 5.4 Example Workflows
- [ ] Example: Free mode on small codebase
- [ ] Example: Hybrid mode on medium codebase
- [ ] Example: Full training on large codebase
- [ ] Example: RAG-only (no fine-tuning)
- [ ] Jupyter notebooks for tutorials

**New files:**
- `examples/free_mode_example.py`
- `examples/hybrid_mode_example.py`
- `examples/rag_only_example.py`
- `notebooks/tutorial_free_mode.ipynb`
- `notebooks/tutorial_hybrid_mode.ipynb`

**Estimated time:** 6-8 hours

**Phase 5 Total:** 32-42 hours
**Deliverable:** Polished, documented, tested system ready for users

---

## Total Effort Estimate

| Phase | Time (hours) | Priority |
|-------|-------------|----------|
| Phase 1: Foundation | 16-24 | 🔴 Critical |
| Phase 2: Data Generation | 26-36 | 🔴 Critical |
| Phase 3: RAG Infrastructure | 20-26 | 🟠 High |
| Phase 4: Training Improvements | 18-22 | 🟠 High |
| Phase 5: Integration & Polish | 32-42 | 🟡 Medium |
| **Total (Core)** | **80-108 hours** | **2-3 weeks full-time** |
| **Total (With Full Training)** | **92-120 hours** | **2.5-3.5 weeks** |

---

## Success Criteria

### Phase 1 Success
✅ All critical bugs fixed
✅ Code parsing extracts functions, classes, docstrings
✅ Configuration validates correctly
✅ Tests pass

### Phase 2 Success
✅ Can generate training data in free mode (no API costs)
✅ Can generate training data in hybrid mode (API + self-supervised)
✅ Can track costs accurately
✅ Training data quality >8/10

### Phase 3 Success
✅ RAG index builds successfully
✅ Can retrieve relevant code given query
✅ Inference uses both fine-tuned model + RAG
✅ Quality improvement over model-only

### Phase 4 Success
✅ Training recovers from checkpoints
✅ Evaluation metrics show model understanding
✅ Can train in all modes (free/hybrid/paid)
✅ Training time <6 hours for typical codebase

### Phase 5 Success
✅ Users can run all modes end-to-end
✅ Test coverage >70%
✅ Documentation is complete and clear
✅ Examples work out-of-the-box

---

## Risk Mitigation

### Technical Risks
| Risk | Mitigation |
|------|------------|
| AST parsing fails for complex code | Add fallback to regex-based extraction |
| Static analysis too slow | Make it optional, run in background |
| RAG index too large | Implement chunking and pruning |
| API costs spiral | Hard budget caps, estimation before running |
| Training OOM errors | Better memory management, gradient checkpointing |

### User Experience Risks
| Risk | Mitigation |
|------|------------|
| Configuration too complex | Good defaults, mode presets |
| Unclear error messages | Comprehensive error handling with suggestions |
| Slow first run | Progress bars, estimated time remaining |
| Difficult to choose mode | Decision tree guide, cost calculator |

---

## Dependencies to Add

```bash
# For enhanced code parsing
pip install tree-sitter tree-sitter-python tree-sitter-javascript
pip install radon  # Code complexity

# For RAG
pip install sentence-transformers faiss-cpu chromadb
pip install tiktoken  # Token counting

# For static analysis (training data)
pip install mypy pylint flake8

# For git analysis
pip install gitpython

# For API integrations
pip install openai anthropic together

# For validation
pip install pydantic

# For better UX
pip install rich click tqdm

# For testing
pip install pytest pytest-cov pytest-mock
```

---

## Next Steps

1. **Review this plan** - Get feedback and approval
2. **Create detailed todos** - Break down each phase into actionable tasks
3. **Start Phase 1** - Fix critical issues first
4. **Iterate and test** - Test after each phase
5. **Gather feedback** - Adjust based on usage

---

## Open Questions for User

1. **Priority:** Which mode is most important to you? (Free/Hybrid/Full)
2. **Timeline:** Do we need all phases, or can we ship incrementally?
3. **Dependencies:** Are you okay adding the new dependencies listed above?
4. **Testing:** Do you want comprehensive tests in Phase 5, or as we go?
5. **Full Training:** Is nanochat-style full training a must-have or nice-to-have?
6. **Budget:** What's a reasonable default budget cap for hybrid mode? ($20? $50? $100?)

Please review and let me know if you want to proceed, adjust, or focus on specific phases first!
