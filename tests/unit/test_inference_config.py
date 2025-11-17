"""Unit tests for inference configuration."""

from pathlib import Path

import pytest

from nanodex.config import InferenceConfig, load_config


def test_inference_config_defaults():
    """Test InferenceConfig default values."""
    config = InferenceConfig()

    assert config.base_model == "Qwen/Qwen2.5-Coder-7B"
    assert config.host == "0.0.0.0"
    assert config.port == 8000
    assert config.max_lora_rank == 32
    assert config.max_tokens == 512
    assert config.temperature == 0.3


def test_inference_config_custom():
    """Test InferenceConfig with custom values."""
    config = InferenceConfig(
        base_model="custom/model",
        adapter_path=Path("custom/adapter"),
        host="127.0.0.1",
        port=9000,
        max_tokens=1024,
        temperature=0.7,
    )

    assert config.base_model == "custom/model"
    assert config.adapter_path == Path("custom/adapter")
    assert config.host == "127.0.0.1"
    assert config.port == 9000
    assert config.max_tokens == 1024
    assert config.temperature == 0.7


def test_load_inference_config(temp_dir):
    """Test loading inference config from YAML."""
    config_file = temp_dir / "inference.yaml"
    config_file.write_text(
        """
base_model: "test/model"
adapter_path: "test/adapter"
host: "localhost"
port: 7000
max_lora_rank: 16
max_tokens: 256
temperature: 0.5
"""
    )

    config = load_config(config_file, InferenceConfig)

    assert config.base_model == "test/model"
    assert str(config.adapter_path) == "test/adapter"
    assert config.host == "localhost"
    assert config.port == 7000
    assert config.max_lora_rank == 16
    assert config.max_tokens == 256
    assert config.temperature == 0.5


def test_load_inference_config_file():
    """Test loading actual inference config from file."""
    config_path = Path("config/inference.yaml")

    if not config_path.exists():
        pytest.skip("inference.yaml not found")

    config = load_config(config_path, InferenceConfig)

    assert config.base_model == "Qwen/Qwen2.5-Coder-7B"
    assert config.port == 8000
    assert config.max_lora_rank == 32
