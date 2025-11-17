"""Unit tests for instruction dataset loader."""

import json
from pathlib import Path

import pytest

# Only run these tests if transformers is available
pytest.importorskip("transformers")

from nanodex.trainer.data_loader import InstructionDataset


def test_format_messages():
    """Test message formatting."""
    from transformers import AutoTokenizer

    # Use a small tokenizer for testing
    tokenizer = AutoTokenizer.from_pretrained("gpt2", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create dummy dataset
    dataset_file = Path("/tmp/test_dataset.jsonl")
    with open(dataset_file, "w") as f:
        example = {
            "id": "test_001",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a programming language."},
            ],
        }
        f.write(json.dumps(example) + "\n")

    # Test dataset loading
    dataset = InstructionDataset(
        dataset_path=dataset_file,
        tokenizer=tokenizer,
        max_length=512,
        validation_split=0.0,
    )

    # Check dataset was created
    assert dataset.train_dataset is not None
    assert len(dataset.train_dataset) == 1

    # Clean up
    dataset_file.unlink()


def test_dataset_split():
    """Test train/validation split."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create dataset with multiple examples
    dataset_file = Path("/tmp/test_split_dataset.jsonl")
    with open(dataset_file, "w") as f:
        for i in range(10):
            example = {
                "id": f"test_{i:03d}",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Question {i}?"},
                    {"role": "assistant", "content": f"Answer {i}."},
                ],
            }
            f.write(json.dumps(example) + "\n")

    # Test with validation split
    dataset = InstructionDataset(
        dataset_path=dataset_file,
        tokenizer=tokenizer,
        max_length=512,
        validation_split=0.2,  # 20% validation
    )

    # Check split
    assert dataset.train_dataset is not None
    assert dataset.val_dataset is not None
    assert len(dataset.train_dataset) == 8  # 80% of 10
    assert len(dataset.val_dataset) == 2  # 20% of 10

    # Clean up
    dataset_file.unlink()


def test_format_messages_helper():
    """Test _format_messages helper method."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset_file = Path("/tmp/test_format.jsonl")
    with open(dataset_file, "w") as f:
        f.write(json.dumps({"id": "test", "messages": []}) + "\n")

    dataset = InstructionDataset(
        dataset_path=dataset_file,
        tokenizer=tokenizer,
        max_length=512,
        validation_split=0.0,
    )

    messages = [
        {"role": "system", "content": "System message"},
        {"role": "user", "content": "User message"},
        {"role": "assistant", "content": "Assistant message"},
    ]

    formatted = dataset._format_messages(messages)

    assert "<|im_start|>system" in formatted
    assert "System message" in formatted
    assert "<|im_start|>user" in formatted
    assert "User message" in formatted
    assert "<|im_start|>assistant" in formatted
    assert "Assistant message" in formatted

    # Clean up
    dataset_file.unlink()
