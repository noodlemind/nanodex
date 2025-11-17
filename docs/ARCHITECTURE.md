# nanodex Architecture

## System Overview

nanodex transforms codebases into specialized domain experts through a five-stage pipeline:

```
Source Code → Extraction → Brain → Dataset → Training → Inference
     ↓            ↓          ↓        ↓          ↓          ↓
   .java      graph.db   nodes/   train.jsonl  adapter  vLLM API
   .py        17k nodes  17k JSON  2.9k Q&A    10-50MB   queries
```

## Core Components

### 1. Extractor (`nanodex/extractor/`)

**Purpose**: Parse source code into structured knowledge graph

**Key Files**:
- `tree_sitter_parser.py`: AST parsing with Tree-sitter
- `graph_builder.py`: Repository-wide extraction orchestration
- `schema.sql`: SQLite graph schema

**Flow**:
1. Scan repository for supported files (.py, .java, .ts, .cpp)
2. Parse each file with Tree-sitter → extract nodes (classes, functions, variables)
3. Extract relationships (calls, imports, extends, implements)
4. Store in SQLite: `nodes(id, type, name, path)`, `edges(src, dst, relationship)`

**Output**: `data/brain/graph.sqlite` (17k nodes, 16k edges for nanodex itself)

### 2. Brain (`nanodex/brain/`)

**Purpose**: Semantic classification and summary generation

**Key Files**:
- `node_typer.py`: Classify nodes into 5 semantic types
- `summarizer.py`: Generate concise summaries (max 200 tokens)
- `graph_manager.py`: Graph query and traversal utilities

**Node Types**:
- `module`: Files and namespaces
- `capability`: Public functions/methods
- `concept`: Internal classes/functions
- `error`: Exception classes
- `recipe`: Examples, mains, tests

**Flow**:
1. Load graph from SQLite
2. Classify each node based on naming patterns and structure
3. Generate summary using node properties + graph context
4. Save to `data/brain/nodes/{node_id}.json`

**Output**: 17k JSON files with `{id, type, name, summary, context}`

### 3. Dataset (`nanodex/dataset/`)

**Purpose**: Generate Q&A training data from knowledge graph

**Key Files**:
- `qa_generator.py`: Template-based Q&A generation
- `validators.py`: Data quality validation

**Q&A Categories**:
1. **Discovery**: "Does the codebase have X?"
2. **Explain**: "How does X work?"
3. **Howto**: "How do I use X?"
4. **Diagnostics**: "I'm getting error E, what to check?"

**Flow**:
1. Query graph for nodes of each type
2. Generate positive examples using templates + summaries
3. Generate negative examples for contrastive learning (2:1 ratio)
4. Validate: check length, references, duplicates
5. Format as JSONL instruction tuning format

**Output**: `data/dataset/train.jsonl` (2,919 examples)

### 4. Trainer (`nanodex/trainer/`)

**Purpose**: LoRA/QLoRA fine-tuning on instruction data

**Key Files**:
- `data_loader.py`: Load and tokenize JSONL dataset
- `trainer.py`: HuggingFace Trainer wrapper with LoRA/QLoRA

**Training Modes**:
- **QLoRA**: 4-bit quantization (12-24GB VRAM)
- **LoRA**: FP16 full precision (24-40GB VRAM)

**Flow**:
1. Load base model (Qwen2.5-Coder-7B) with optional quantization
2. Apply LoRA adapters to attention layers
3. Tokenize dataset with instruction template
4. Train with gradient accumulation, mixed precision
5. Save LoRA adapter (10-50MB)

**Output**: `models/nanodex-qlora/` (adapter weights)

### 5. Inference (`nanodex/inference/`)

**Purpose**: Serve fine-tuned model via vLLM API

**Key Files**:
- `server.py`: vLLM server wrapper
- `client.py`: Python query client

**Flow**:
1. Start vLLM with base model + LoRA adapter
2. Serve OpenAI-compatible API on port 8000
3. Client sends chat completion requests
4. Model generates responses using specialized knowledge

## Data Flow Diagram

```
┌─────────────┐
│ Source Code │
└──────┬──────┘
       │ Tree-sitter parse
       ▼
┌─────────────┐
│  Graph DB   │ ◄─── GraphManager CRUD operations
│ (SQLite)    │
└──────┬──────┘
       │ Node classification + summarization
       ▼
┌─────────────┐
│ Node JSONs  │ ◄─── Summarizer reads graph context
└──────┬──────┘
       │ Q&A generation from summaries
       ▼
┌─────────────┐
│ JSONL Train │ ◄─── Validators ensure quality
└──────┬──────┘
       │ LoRA/QLoRA fine-tuning
       ▼
┌─────────────┐
│ LoRA Adapter│ ◄─── PEFT applies to attention layers
└──────┬──────┘
       │ vLLM serving
       ▼
┌─────────────┐
│  Inference  │ ◄─── QueryClient makes requests
└─────────────┘
```

## Configuration System

All stages use Pydantic models for validation:

- `ExtractorConfig`: Languages, exclusions, output paths
- `BrainConfig`: Node types, summary style, max tokens
- `DatasetConfig`: Q&A counts per category, negatives ratio
- `TrainingConfig`: LoRA params, batch size, learning rate
- `InferenceConfig`: Server host/port, generation params

Configs loaded from YAML files in `config/`:
- `extract.yaml`
- `brain.yaml`
- `dataset.yaml`
- `train_qlora.yaml` / `train_lora.yaml`
- `inference.yaml`

## Design Decisions

### Why Tree-sitter?
- Language-agnostic AST parsing
- Fast and battle-tested
- No need for language-specific tooling

### Why SQLite?
- Portable single-file database
- Rich query capabilities for graph traversal
- No server setup required

### Why LoRA/QLoRA?
- Efficient: only train 0.1-1% of parameters
- Portable: adapters are 10-50MB vs multi-GB full models
- Composable: can merge multiple adapters

### Why vLLM?
- Fast inference with PagedAttention
- LoRA adapter support
- OpenAI-compatible API

## Testing Strategy

- **Unit tests**: Individual components (parsers, generators, validators)
- **Integration tests**: End-to-end extraction pipeline
- **Coverage**: 66% overall, 89-92% for core logic
- **CI/CD**: All tests run on commit via GitHub Actions (planned)

## Performance Characteristics

### Extraction
- **Speed**: ~100 files/sec on modern hardware
- **Memory**: ~100MB for 1k files
- **Bottleneck**: Tree-sitter parsing

### Brain
- **Speed**: ~1000 summaries/sec
- **Memory**: ~500MB (graph + summaries in RAM)
- **Bottleneck**: SQLite queries

### Training
- **Speed**: ~10-20 examples/sec on A100
- **Memory**: 12-24GB VRAM (QLoRA), 24-40GB (LoRA)
- **Time**: ~1-3 hours for 3 epochs on 3k examples

### Inference
- **Latency**: ~100-500ms per query
- **Throughput**: ~10-50 req/sec (depends on batch size)
- **Memory**: 12-16GB VRAM for 7B model
