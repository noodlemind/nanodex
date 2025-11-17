# nanodex

**Compile any codebase into a portable domain expert.**
nanodex builds a compact **knowledge graph** from source, distills concise **facts/summaries**, and fine‑tunes an **open LLM** with **LoRA/QLoRA** so the resulting model answers capability/architecture/how‑to/diagnostic questions **offline**—without shipping the full repo at inference time.

* **Open stack**: Tree‑sitter + (optional) SCIP → SQLite graph → PEFT (LoRA/QLoRA) on an open code model (e.g., Qwen2.5‑Coder).
* **Artifacts**: `brain/graph.sqlite`, `brain/nodes/*.json`, `models/<proj>-nanodex-lora/`.
* **Outputs**: a local server (vLLM/TGI) for the specialized model + a simple query CLI.

---

## Repository Layout

```
.
├── extractor/          # parse repo → symbols/relations (Tree-sitter, optional SCIP)
├── brain/              # graph (SQLite) + node JSON summaries
├── dataset/            # distilled Q&A / training JSONL
├── trainer/            # LoRA/QLoRA fine-tune scripts (HF Transformers + PEFT)
├── inference/          # serve base+adapter (vLLM) and query CLI
├── config/             # YAML configs for extract/brain/dataset/train
├── scripts/            # thin bash/py wrappers
└── Makefile            # end-to-end targets
```

---

## Quick Start

### 0) Prereqs

* **Python** ≥ 3.10, **CUDA** (for training), **git**, **NodeJS** (Tree‑sitter grammars)
* GPU VRAM guideline

  * LoRA (FP16): 24–40 GB (7–14B base)
  * **QLoRA (4‑bit):** 12–24 GB (7–14B base)
* Optional: SCIP indexers on PATH (language‑specific)

### 1) Install

```bash
git clone <this-repo> nanodex && cd nanodex
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make grammars   # optional: build Tree-sitter grammars
```

### 2) Point to a Codebase

```bash
export TARGET_REPO=/path/to/library-or-sdk
```

### 3) Extract → Graph

```bash
make extract  REPO=$TARGET_REPO CONFIG=config/extract.yaml     # -> brain/graph.sqlite
make graph-inspect                                              # node/edge counts
```

### 4) Build the Brain (Typed Nodes + Summaries)

```bash
make brain     CONFIG=config/brain.yaml
make brain-embed   # optional: embed node summaries for fast NL→node mapping
```

### 5) Generate Training Data (Q&A)

```bash
make dataset   CONFIG=config/dataset.yaml                       # -> dataset/train.jsonl
head -n 3 dataset/train.jsonl | jq .
```

### 6) Fine‑Tune (LoRA / QLoRA)

Pick an open base model (default suggestion: `Qwen/Qwen2.5-Coder-7B`).

```bash
# QLoRA (recommended)
make train-qlora BASE=Qwen/Qwen2.5-Coder-7B \
                 DATASET=dataset/train.jsonl \
                 OUT=models/project-nanodex-lora \
                 CONFIG=config/train_qlora.yaml

# OR LoRA (FP16)
make train-lora  BASE=Qwen/Qwen2.5-Coder-7B \
                 DATASET=dataset/train.jsonl \
                 OUT=models/project-nanodex-lora \
                 CONFIG=config/train_lora.yaml
```

### 7) Serve & Query

```bash
# serve base+adapter with vLLM
make serve BASE=Qwen/Qwen2.5-Coder-7B ADAPTER=models/project-nanodex-lora

# ask nanodex
python inference/query.py \
  --endpoint http://localhost:8000 \
  --question "UnknownTunerMode during channel scan: which modules throw this and what should I check?"
```

---

## What Each Stage Produces

### extractor/

* **Tree‑sitter** parses files → symbols (classes, functions) + relations (calls, imports).
* (Optional) **SCIP** indexers add semantic edges (implements/extends/types).
* Output: `brain/graph.sqlite` with:

  * `nodes(id, type, name, path, lang, meta_json)`
  * `edges(src_id, dst_id, kind)` where `kind ∈ {calls, imports, extends, implements, throws, defined_in, depends_on}`

### brain/

* Build **typed ontology**: `module`, `capability`, `concept`, `error`, `recipe`.
* Generate **tight** summaries per node → `brain/nodes/*.json`.
* Optional: vector map for NL→node lookup.

### dataset/

* Auto‑create high‑signal Q&A:

  * **Discovery** (“Do we support X?”)
  * **Explain** (“How does Y work?”)
  * **How‑to** (recipes)
  * **Diagnostics** (error → modules → fixes)
* Format: JSONL `{id,prompt,response,refs:[node_ids...]}`

### trainer/

* **PEFT** (LoRA/QLoRA) with HF Transformers + bitsandbytes.
* Output: adapter at `models/<proj>-nanodex-lora/` (tens of MB).

### inference/

* vLLM/TGI runner for base+adapter and a small CLI to query.

---

## Config Templates

**config/extract.yaml**

```yaml
languages: [python, java, typescript, cpp]
use_scip: true
exclude: ["**/vendor/**","**/build/**","**/.git/**"]
out_graph: "brain/graph.sqlite"
```

**config/brain.yaml**

```yaml
graph: "brain/graph.sqlite"
node_types: [module, capability, concept, error, recipe]
summary:
  max_tokens: 200
  style: "factual"
out_dir: "brain/nodes"
```

**config/dataset.yaml**

```yaml
graph: "brain/graph.sqlite"
nodes_dir: "brain/nodes"
counts: { discovery: 2000, explain: 2000, howto: 2000, diag: 2000 }
negatives_per_example: 2
out_jsonl: "dataset/train.jsonl"
```

**config/train_qlora.yaml**

```yaml
bits: 4
bnb_4bit_compute_dtype: "bfloat16"
lora_rank: 16
lora_alpha: 32
target_modules: ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
learning_rate: 2e-4
batch_size: 64
max_steps: 3000
max_seq_len: 2048
eval_interval: 500
save_dir: "models/project-nanodex-lora"
```

---

## Make Targets

```
setup           # install Python deps
grammars        # build Tree-sitter grammars
extract         # repo -> brain/graph.sqlite
graph-inspect   # print node/edge counts
brain           # build typed nodes + JSON summaries
brain-embed     # optional: embed summaries
dataset         # graph -> dataset/train.jsonl
train-lora      # FP16 LoRA fine-tune
train-qlora     # 4-bit QLoRA fine-tune
serve           # run vLLM with base+adapter
query           # simple local query
```

---

## Operational Notes

* Prefer **Apache‑2.0** base models (e.g., Qwen2.5‑Coder) for clean redistribution of adapters or merged weights.
* Keep NOTICE/COPYRIGHT files for deps; avoid long verbatim code in summaries/datasets.
* Updates: re‑run `extract → brain → dataset` and do a short adapter refresh.

---

## License

Code in this repo: **Apache‑2.0**.
You are responsible for complying with the licenses of the base model you choose and codebases you process.

---

**nanodex** turns any codebase into a compact brain and a LoRA/QLoRA‑adapted specialist model that runs locally and answers with precision.
