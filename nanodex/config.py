"""Configuration management for nanodex pipeline."""

from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar

import yaml
from pydantic import BaseModel, Field, field_validator

T = TypeVar("T", bound=BaseModel)


class ExtractorConfig(BaseModel):
    """Configuration for the extractor stage."""

    languages: List[str] = Field(
        default=["python", "java", "typescript", "cpp"],
        description="Programming languages to extract from",
    )
    use_scip: bool = Field(default=False, description="Enable SCIP indexers for semantic edges")
    exclude: List[str] = Field(
        default=["**/vendor/**", "**/node_modules/**", "**/.git/**"],
        description="Glob patterns to exclude from extraction",
    )
    out_graph: Path = Field(
        default=Path("data/brain/graph.sqlite"), description="Output path for graph database"
    )
    max_file_size_mb: int = Field(default=5, description="Maximum file size in MB to process", ge=1)
    max_total_files: int = Field(
        default=100000, description="Maximum number of files to process", ge=1
    )
    max_repo_size_mb: int = Field(
        default=10000, description="Maximum total repository size in MB", ge=1
    )
    processing_timeout_seconds: int = Field(
        default=7200, description="Processing timeout in seconds (default: 2 hours)", ge=60
    )

    @field_validator("languages")
    @classmethod
    def validate_languages(cls, v: List[str]) -> List[str]:
        """Validate supported languages."""
        supported = {"python", "java", "typescript", "javascript", "cpp", "c", "rust", "go"}
        for lang in v:
            if lang not in supported:
                raise ValueError(f"Unsupported language: {lang}. Supported: {supported}")
        return v


class BrainConfig(BaseModel):
    """Configuration for the brain stage."""

    node_types: List[str] = Field(
        default=["module", "capability", "concept", "error", "recipe"],
        description="Semantic node types for classification",
    )
    summary_style: str = Field(
        default="concise", description="Summary generation style: concise, detailed, technical"
    )
    summary_max_tokens: int = Field(
        default=200, description="Maximum tokens per summary", ge=50, le=500
    )
    out_dir: Path = Field(
        default=Path("data/brain/nodes"), description="Output directory for node summaries"
    )
    use_embeddings: bool = Field(default=False, description="Generate vector embeddings")
    embedding_model: Optional[str] = Field(
        default=None,
        description="Model for embeddings (e.g., sentence-transformers/all-MiniLM-L6-v2)",
    )

    @field_validator("node_types")
    @classmethod
    def validate_node_types(cls, v: List[str]) -> List[str]:
        """Validate node types."""
        allowed = {"module", "capability", "concept", "error", "recipe"}
        for node_type in v:
            if node_type not in allowed:
                raise ValueError(f"Invalid node type: {node_type}. Allowed: {allowed}")
        return v


class DatasetConfig(BaseModel):
    """Configuration for the dataset generation stage."""

    qa_categories: Dict[str, int] = Field(
        default={
            "discovery": 250,
            "explain": 250,
            "howto": 250,
            "diagnostics": 250,
        },
        description="Number of Q&A pairs per category",
    )
    negatives_per_example: int = Field(
        default=2, description="Number of negative examples per positive", ge=0, le=5
    )
    out_file: Path = Field(
        default=Path("data/dataset/train.jsonl"), description="Output JSONL file path"
    )
    min_response_tokens: int = Field(
        default=50, description="Minimum response length in tokens", ge=10
    )
    max_response_tokens: int = Field(
        default=500, description="Maximum response length in tokens", le=2000
    )

    @field_validator("qa_categories")
    @classmethod
    def validate_categories(cls, v: Dict[str, int]) -> Dict[str, int]:
        """Validate Q&A categories."""
        allowed = {"discovery", "explain", "howto", "diagnostics"}
        for category in v.keys():
            if category not in allowed:
                raise ValueError(f"Invalid category: {category}. Allowed: {allowed}")
        return v


class QuantizationConfig(BaseModel):
    """Quantization configuration for QLoRA."""

    load_in_4bit: bool = Field(default=True, description="Load model in 4-bit")
    bnb_4bit_compute_dtype: str = Field(
        default="bfloat16", description="Compute dtype for 4-bit training"
    )
    bnb_4bit_quant_type: str = Field(default="nf4", description="BitsAndBytes quantization type")
    bnb_4bit_use_double_quant: bool = Field(default=True, description="Use double quantization")


class LoRAConfig(BaseModel):
    """LoRA adapter configuration."""

    r: int = Field(default=16, description="LoRA rank", ge=8, le=64)
    lora_alpha: int = Field(default=32, description="LoRA alpha", ge=16, le=128)
    lora_dropout: float = Field(default=0.05, description="LoRA dropout", ge=0.0, le=0.2)
    target_modules: List[str] = Field(
        default=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        description="Target modules for LoRA",
    )
    bias: str = Field(default="none", description="Bias setting")
    task_type: str = Field(default="CAUSAL_LM", description="Task type")


class TrainingParams(BaseModel):
    """Training hyperparameters."""

    output_dir: str = Field(default="models/nanodex-qlora", description="Output directory")
    num_train_epochs: int = Field(default=3, description="Number of training epochs", ge=1)
    per_device_train_batch_size: int = Field(
        default=4, description="Training batch size", ge=1, le=32
    )
    per_device_eval_batch_size: int = Field(
        default=4, description="Evaluation batch size", ge=1, le=32
    )
    gradient_accumulation_steps: int = Field(
        default=4, description="Gradient accumulation steps", ge=1
    )
    learning_rate: float = Field(default=2e-4, description="Learning rate", ge=1e-5, le=1e-3)
    weight_decay: float = Field(default=0.01, description="Weight decay", ge=0.0, le=0.1)
    warmup_ratio: float = Field(default=0.03, description="Warmup ratio", ge=0.0, le=0.2)
    lr_scheduler_type: str = Field(default="cosine", description="LR scheduler type")
    logging_steps: int = Field(default=10, description="Logging interval", ge=1)
    save_steps: int = Field(default=100, description="Save checkpoint interval", ge=1)
    eval_steps: int = Field(default=100, description="Evaluation interval", ge=1)
    save_total_limit: int = Field(default=3, description="Max checkpoints to keep", ge=1)
    fp16: bool = Field(default=False, description="Use FP16 training")
    bf16: bool = Field(default=True, description="Use BF16 training")
    max_grad_norm: float = Field(default=0.3, description="Max gradient norm", ge=0.0)
    optim: str = Field(default="paged_adamw_32bit", description="Optimizer")
    group_by_length: bool = Field(default=True, description="Group samples by length")
    report_to: str = Field(default="tensorboard", description="Logging destination")


class GenerationConfig(BaseModel):
    """Generation parameters for evaluation."""

    max_new_tokens: int = Field(default=512, description="Max new tokens", ge=64, le=2048)
    temperature: float = Field(default=0.7, description="Sampling temperature", ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, description="Top-p sampling", ge=0.0, le=1.0)
    do_sample: bool = Field(default=True, description="Use sampling")


class TrainingConfig(BaseModel):
    """Configuration for LoRA/QLoRA training."""

    base_model: str = Field(default="Qwen/Qwen2.5-Coder-7B", description="Base model identifier")
    model_max_length: int = Field(
        default=2048, description="Maximum sequence length", ge=512, le=4096
    )
    dataset_path: str = Field(
        default="data/dataset/train.jsonl", description="Training dataset path"
    )
    validation_split: float = Field(
        default=0.1, description="Validation split ratio", ge=0.0, le=0.5
    )
    quantization: Optional[QuantizationConfig] = Field(
        default=None, description="Quantization config (for QLoRA)"
    )
    lora: LoRAConfig = Field(default_factory=LoRAConfig, description="LoRA config")
    training: TrainingParams = Field(default_factory=TrainingParams, description="Training params")
    generation: GenerationConfig = Field(
        default_factory=GenerationConfig, description="Generation config"
    )
    instruction_template: str = Field(
        default="<|im_start|>system\nYou are a helpful code assistant specialized in this codebase.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>",
        description="Instruction formatting template",
    )


class InferenceConfig(BaseModel):
    """Configuration for inference serving."""

    base_model: str = Field(default="Qwen/Qwen2.5-Coder-7B", description="Base model identifier")
    adapter_path: Optional[Path] = Field(
        default=Path("models/project-nanodex-lora"), description="LoRA adapter path"
    )
    max_lora_rank: int = Field(default=32, description="Maximum LoRA rank", ge=8, le=64)
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port", ge=1024, le=65535)
    max_tokens: int = Field(default=512, description="Max tokens per response", ge=64, le=4096)
    temperature: float = Field(default=0.3, description="Sampling temperature", ge=0.0, le=2.0)


def load_config(config_path: Path, config_class: type[T]) -> T:
    """
    Load and validate configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file
        config_class: Pydantic model class for validation

    Returns:
        Validated configuration object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config validation fails
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return config_class(**config_dict)


def save_config(config: BaseModel, output_path: Path) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Pydantic configuration object
        output_path: Path to save YAML file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)
