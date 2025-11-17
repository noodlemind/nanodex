"""Unit tests for training configuration."""

from pathlib import Path

import pytest

from nanodex.config import (
    GenerationConfig,
    LoRAConfig,
    QuantizationConfig,
    TrainingConfig,
    TrainingParams,
    load_config,
)


def test_quantization_config():
    """Test QuantizationConfig initialization."""
    config = QuantizationConfig()

    assert config.load_in_4bit is True
    assert config.bnb_4bit_compute_dtype == "bfloat16"
    assert config.bnb_4bit_quant_type == "nf4"
    assert config.bnb_4bit_use_double_quant is True


def test_lora_config():
    """Test LoRAConfig initialization."""
    config = LoRAConfig()

    assert config.r == 16
    assert config.lora_alpha == 32
    assert config.lora_dropout == 0.05
    assert "q_proj" in config.target_modules
    assert config.bias == "none"
    assert config.task_type == "CAUSAL_LM"


def test_training_params():
    """Test TrainingParams initialization."""
    config = TrainingParams()

    assert config.output_dir == "models/nanodex-qlora"
    assert config.num_train_epochs == 3
    assert config.per_device_train_batch_size == 4
    assert config.learning_rate == 2e-4
    assert config.optim == "paged_adamw_32bit"


def test_generation_config():
    """Test GenerationConfig initialization."""
    config = GenerationConfig()

    assert config.max_new_tokens == 512
    assert config.temperature == 0.7
    assert config.top_p == 0.9
    assert config.do_sample is True


def test_training_config():
    """Test TrainingConfig initialization."""
    config = TrainingConfig()

    assert config.base_model == "Qwen/Qwen2.5-Coder-7B"
    assert config.model_max_length == 2048
    assert config.validation_split == 0.1
    assert config.lora is not None
    assert config.training is not None
    assert config.generation is not None


def test_training_config_custom():
    """Test TrainingConfig with custom values."""
    config = TrainingConfig(
        base_model="custom/model",
        model_max_length=4096,
        validation_split=0.2,
    )

    assert config.base_model == "custom/model"
    assert config.model_max_length == 4096
    assert config.validation_split == 0.2


def test_load_training_config(temp_dir):
    """Test loading training config from YAML."""
    # Create minimal config YAML
    config_file = temp_dir / "train.yaml"
    config_file.write_text(
        """
base_model: "test/model"
model_max_length: 1024
dataset_path: "data/test.jsonl"
validation_split: 0.15
lora:
  r: 8
  lora_alpha: 16
  lora_dropout: 0.1
  target_modules:
    - q_proj
    - v_proj
  bias: "none"
  task_type: "CAUSAL_LM"
training:
  output_dir: "models/test"
  num_train_epochs: 2
  per_device_train_batch_size: 2
  per_device_eval_batch_size: 2
  gradient_accumulation_steps: 2
  learning_rate: 1.0e-4
  weight_decay: 0.01
  warmup_ratio: 0.05
  lr_scheduler_type: "linear"
  logging_steps: 5
  save_steps: 50
  eval_steps: 50
  save_total_limit: 2
  fp16: false
  bf16: true
  max_grad_norm: 1.0
  optim: "adamw_torch"
  group_by_length: true
  report_to: "none"
generation:
  max_new_tokens: 256
  temperature: 0.5
  top_p: 0.95
  do_sample: true
instruction_template: "test template"
"""
    )

    config = load_config(config_file, TrainingConfig)

    assert config.base_model == "test/model"
    assert config.model_max_length == 1024
    assert config.validation_split == 0.15
    assert config.lora.r == 8
    assert config.lora.lora_alpha == 16
    assert config.training.num_train_epochs == 2
    assert config.training.learning_rate == 1.0e-4
    assert config.generation.max_new_tokens == 256


def test_load_qlora_config():
    """Test loading QLoRA config from YAML."""
    config_path = Path("config/train_qlora.yaml")

    if not config_path.exists():
        pytest.skip("train_qlora.yaml not found")

    config = load_config(config_path, TrainingConfig)

    assert config.base_model == "Qwen/Qwen2.5-Coder-7B"
    assert config.quantization is not None
    assert config.quantization.load_in_4bit is True
    assert config.lora.r == 16
    assert config.training.bf16 is True


def test_load_lora_config():
    """Test loading LoRA config from YAML."""
    config_path = Path("config/train_lora.yaml")

    if not config_path.exists():
        pytest.skip("train_lora.yaml not found")

    config = load_config(config_path, TrainingConfig)

    assert config.base_model == "Qwen/Qwen2.5-Coder-7B"
    assert config.quantization is None  # No quantization for full precision
    assert config.lora.r == 32  # Higher rank
    assert config.training.fp16 is True
