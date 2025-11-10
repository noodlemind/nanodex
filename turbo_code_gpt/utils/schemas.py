"""
Pydantic schemas for configuration validation.
"""

from typing import List, Dict, Any, Optional, Literal
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, model_validator
import logging

logger = logging.getLogger(__name__)


class HuggingFaceModelConfig(BaseModel):
    """HuggingFace model configuration schema."""

    model_name: str = Field(..., min_length=1, description="HuggingFace model identifier")
    use_4bit: bool = Field(default=True, description="Use 4-bit quantization")
    use_8bit: bool = Field(default=False, description="Use 8-bit quantization")
    trust_remote_code: bool = Field(default=False, description="Trust remote code execution")

    @field_validator('model_name')
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate model name is not empty."""
        if not v or v.strip() == "":
            raise ValueError("model_name cannot be empty")
        return v.strip()

    @model_validator(mode='after')
    def validate_quantization(self) -> 'HuggingFaceModelConfig':
        """Ensure only one quantization method is enabled."""
        if self.use_4bit and self.use_8bit:
            raise ValueError(
                "Cannot use both 4-bit and 8-bit quantization simultaneously. "
                "Please enable only one quantization method."
            )
        return self


class OllamaModelConfig(BaseModel):
    """Ollama model configuration schema."""

    model_name: str = Field(..., min_length=1, description="Ollama model name")
    base_url: str = Field(default="http://localhost:11434", description="Ollama API base URL")

    @field_validator('model_name')
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate model name is not empty."""
        if not v or v.strip() == "":
            raise ValueError("model_name cannot be empty")
        return v.strip()

    @field_validator('base_url')
    @classmethod
    def validate_base_url(cls, v: str) -> str:
        """Validate base URL format."""
        if not v.startswith(('http://', 'https://')):
            raise ValueError(
                f"Invalid base_url: {v}. Must start with http:// or https://"
            )
        return v.strip()


class ModelConfigSection(BaseModel):
    """Model configuration section."""

    huggingface: HuggingFaceModelConfig = Field(..., description="HuggingFace model config")
    ollama: OllamaModelConfig = Field(..., description="Ollama model config")


class DeepParsingConfig(BaseModel):
    """Deep parsing configuration schema."""

    enabled: bool = Field(default=True, description="Enable deep code parsing")
    extract_functions: bool = Field(default=True, description="Extract function definitions")
    extract_classes: bool = Field(default=True, description="Extract class definitions")
    extract_imports: bool = Field(default=True, description="Extract import statements")
    extract_docstrings: bool = Field(default=True, description="Extract docstrings")
    calculate_complexity: bool = Field(default=True, description="Calculate code complexity")


class RepositoryConfig(BaseModel):
    """Repository analysis configuration schema."""

    path: str = Field(default=".", description="Path to code repository")
    include_extensions: List[str] = Field(
        default_factory=lambda: [".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rs"],
        min_length=1,
        description="File extensions to include"
    )
    exclude_dirs: List[str] = Field(
        default_factory=lambda: ["node_modules", "venv", ".git", "__pycache__", "build", "dist"],
        description="Directories to exclude"
    )
    max_file_size: int = Field(
        default=1048576,
        gt=0,
        description="Maximum file size in bytes (default: 1MB)"
    )
    deep_parsing: DeepParsingConfig = Field(
        default_factory=DeepParsingConfig,
        description="Deep parsing configuration"
    )

    @field_validator('include_extensions')
    @classmethod
    def validate_extensions(cls, v: List[str]) -> List[str]:
        """Validate file extensions format."""
        if not v:
            raise ValueError("include_extensions cannot be empty. Specify at least one file extension.")

        validated = []
        for ext in v:
            ext = ext.strip()
            if not ext:
                continue
            if not ext.startswith('.'):
                ext = f'.{ext}'
            validated.append(ext.lower())

        if not validated:
            raise ValueError("No valid file extensions provided")

        return validated

    @field_validator('path')
    @classmethod
    def validate_path(cls, v: str) -> str:
        """Validate repository path exists."""
        if not v or v.strip() == "":
            raise ValueError("Repository path cannot be empty")

        path = Path(v).resolve()

        if not path.exists():
            raise ValueError(
                f"Repository path does not exist: {path}\n"
                "Please provide a valid path to your codebase."
            )

        if not path.is_dir():
            raise ValueError(
                f"Repository path is not a directory: {path}\n"
                "Please provide a path to a directory, not a file."
            )

        return str(path)

    @field_validator('max_file_size')
    @classmethod
    def validate_max_file_size(cls, v: int) -> int:
        """Validate max file size is reasonable."""
        if v < 1024:  # Less than 1KB
            raise ValueError(
                f"max_file_size too small: {v} bytes. "
                "Minimum recommended: 1024 bytes (1KB)"
            )

        if v > 100 * 1024 * 1024:  # More than 100MB
            logger.warning(
                f"max_file_size is very large: {v} bytes ({v / (1024*1024):.1f}MB). "
                "This may cause memory issues. Recommended maximum: 10MB"
            )

        return v


class LoRAConfig(BaseModel):
    """LoRA (Low-Rank Adaptation) configuration schema."""

    r: int = Field(default=16, gt=0, le=256, description="LoRA rank")
    lora_alpha: int = Field(default=32, gt=0, description="LoRA alpha parameter")
    lora_dropout: float = Field(default=0.05, ge=0.0, le=0.5, description="LoRA dropout rate")
    target_modules: List[str] = Field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"],
        min_length=1,
        description="Target modules for LoRA"
    )

    @field_validator('target_modules')
    @classmethod
    def validate_target_modules(cls, v: List[str]) -> List[str]:
        """Validate target modules are not empty."""
        if not v:
            raise ValueError("target_modules cannot be empty. Specify at least one module.")

        validated = [m.strip() for m in v if m.strip()]
        if not validated:
            raise ValueError("No valid target modules provided")

        return validated


class TrainingConfig(BaseModel):
    """Training configuration schema."""

    output_dir: str = Field(default="./models/fine-tuned", description="Training output directory")
    num_epochs: int = Field(default=3, gt=0, le=100, description="Number of training epochs")
    batch_size: int = Field(default=4, gt=0, le=128, description="Training batch size")
    learning_rate: float = Field(default=2.0e-5, gt=0, le=1.0, description="Learning rate")
    max_seq_length: int = Field(default=2048, gt=0, description="Maximum sequence length")
    gradient_accumulation_steps: int = Field(default=4, gt=0, description="Gradient accumulation steps")
    warmup_steps: int = Field(default=100, ge=0, description="Warmup steps")
    logging_steps: int = Field(default=10, gt=0, description="Logging frequency")
    save_steps: int = Field(default=500, gt=0, description="Checkpoint save frequency")
    lora: LoRAConfig = Field(default_factory=LoRAConfig, description="LoRA configuration")

    @field_validator('learning_rate')
    @classmethod
    def validate_learning_rate(cls, v: float) -> float:
        """Validate learning rate is in reasonable range."""
        if v < 1e-6:
            raise ValueError(
                f"learning_rate too small: {v}. "
                "Minimum recommended: 1e-6"
            )

        if v > 1e-3:
            logger.warning(
                f"learning_rate is very high: {v}. "
                "This may cause training instability. Recommended: 1e-5 to 5e-5"
            )

        return v

    @field_validator('batch_size')
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        """Validate batch size is power of 2 (recommended)."""
        if v not in [1, 2, 4, 8, 16, 32, 64, 128]:
            logger.warning(
                f"batch_size {v} is not a power of 2. "
                "For optimal GPU performance, use powers of 2: 1, 2, 4, 8, 16, 32, etc."
            )
        return v

    @model_validator(mode='after')
    def validate_steps_relationship(self) -> 'TrainingConfig':
        """Validate relationship between different step parameters."""
        if self.logging_steps > self.save_steps:
            logger.warning(
                f"logging_steps ({self.logging_steps}) > save_steps ({self.save_steps}). "
                "You may miss logging between checkpoints."
            )

        if self.warmup_steps > self.num_epochs * 1000:
            logger.warning(
                f"warmup_steps ({self.warmup_steps}) seems too high for {self.num_epochs} epochs. "
                "Warmup may take too long."
            )

        return self


class DataConfig(BaseModel):
    """Data preparation configuration schema."""

    output_dir: str = Field(default="./data/processed", description="Data output directory")
    train_split: float = Field(default=0.9, gt=0.0, lt=1.0, description="Training data split ratio")
    validation_split: float = Field(default=0.1, gt=0.0, lt=1.0, description="Validation data split ratio")
    context_window: int = Field(default=4096, gt=0, description="Context window size")
    random_seed: int = Field(default=42, ge=0, description="Random seed for reproducibility")

    @model_validator(mode='after')
    def validate_splits(self) -> 'DataConfig':
        """Validate train and validation splits sum to 1.0."""
        total = self.train_split + self.validation_split
        if not (0.99 <= total <= 1.01):  # Allow small floating point errors
            raise ValueError(
                f"train_split ({self.train_split}) + validation_split ({self.validation_split}) "
                f"must equal 1.0, got {total:.4f}"
            )
        return self


class ExportConfig(BaseModel):
    """Model export configuration schema."""

    output_dir: str = Field(default="./models/exported", description="Export output directory")
    format: Literal["gguf", "onnx", "pytorch"] = Field(
        default="gguf",
        description="Export format"
    )
    quantization: str = Field(
        default="q4_k_m",
        description="Quantization method for GGUF format"
    )

    @field_validator('quantization')
    @classmethod
    def validate_quantization(cls, v: str) -> str:
        """Validate quantization method."""
        valid_methods = [
            "q4_0", "q4_1", "q4_k_m", "q4_k_s",
            "q5_0", "q5_1", "q5_k_m", "q5_k_s",
            "q8_0", "f16", "f32"
        ]

        if v not in valid_methods:
            logger.warning(
                f"Unknown quantization method: {v}. "
                f"Valid options: {', '.join(valid_methods)}"
            )

        return v


class TurboCodeGPTConfig(BaseModel):
    """Complete Turbo Code GPT configuration schema."""

    model_source: Literal["huggingface", "ollama"] = Field(
        default="huggingface",
        description="Model source: huggingface or ollama"
    )
    model: ModelConfigSection = Field(..., description="Model configuration")
    repository: RepositoryConfig = Field(..., description="Repository configuration")
    training: TrainingConfig = Field(default_factory=TrainingConfig, description="Training configuration")
    data: DataConfig = Field(default_factory=DataConfig, description="Data configuration")
    export: ExportConfig = Field(default_factory=ExportConfig, description="Export configuration")

    @model_validator(mode='after')
    def validate_output_dirs(self) -> 'TurboCodeGPTConfig':
        """Create output directories if they don't exist."""
        dirs_to_create = [
            self.training.output_dir,
            self.data.output_dir,
            self.export.output_dir
        ]

        for dir_path in dirs_to_create:
            path = Path(dir_path)
            if not path.exists():
                logger.info(f"Creating output directory: {path}")
                path.mkdir(parents=True, exist_ok=True)

        return self

    class Config:
        """Pydantic configuration."""
        extra = "forbid"  # Disallow extra fields not in schema
        validate_assignment = True  # Validate on assignment after creation
