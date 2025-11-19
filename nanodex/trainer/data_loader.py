"""Data loader for instruction tuning datasets."""

import logging
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class InstructionDataset:
    """Dataset loader for instruction tuning format."""

    def __init__(
        self,
        dataset_path: Path,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        validation_split: float = 0.1,
    ):
        """
        Initialize instruction dataset.

        Args:
            dataset_path: Path to JSONL dataset file
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
            validation_split: Fraction of data for validation
        """
        self.dataset_path = Path(dataset_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.validation_split = validation_split

        # Load and process dataset
        self.train_dataset, self.val_dataset = self._load_and_split()

    def _load_and_split(self) -> tuple[Dataset, Dataset]:
        """Load JSONL dataset and split into train/val."""
        logger.info(f"Loading dataset from {self.dataset_path}")

        # Load JSONL file
        dataset = load_dataset("json", data_files=str(self.dataset_path), split="train")
        logger.info(f"Loaded {len(dataset)} examples")

        # Split into train/val
        if self.validation_split > 0:
            split = dataset.train_test_split(test_size=self.validation_split, seed=42)
            train_dataset = split["train"]
            val_dataset = split["test"]
            logger.info(f"Split into {len(train_dataset)} train, {len(val_dataset)} validation")
        else:
            train_dataset = dataset
            val_dataset = None
            logger.info("No validation split")

        # Tokenize datasets
        train_dataset = self._tokenize_dataset(train_dataset)
        if val_dataset:
            val_dataset = self._tokenize_dataset(val_dataset)

        return train_dataset, val_dataset

    def _tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """Tokenize dataset examples."""
        logger.info("Tokenizing dataset...")

        def tokenize_function(examples: dict) -> dict[str, Any]:
            """Tokenize a batch of examples."""
            # Format instruction examples
            texts = []
            for messages in examples["messages"]:
                # Format: system + user + assistant
                text = self._format_messages(messages)
                texts.append(text)

            # Tokenize
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors=None,  # Return lists, not tensors
            )

            # Create labels (same as input_ids for causal LM)
            tokenized["labels"] = tokenized["input_ids"].copy()

            return tokenized  # type: ignore[no-any-return]

        # Apply tokenization
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing",
        )

        return tokenized_dataset

    def _format_messages(self, messages: list[dict[str, str]]) -> str:
        """
        Format messages into instruction template.

        Args:
            messages: List of message dicts with role and content

        Returns:
            Formatted instruction text
        """
        formatted_parts = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                formatted_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                formatted_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                formatted_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")

        return "\n".join(formatted_parts)

    def get_train_dataset(self) -> Dataset:
        """Get training dataset."""
        return self.train_dataset

    def get_val_dataset(self) -> Dataset | None:
        """Get validation dataset."""
        return self.val_dataset


def create_dataloaders(
    dataset: InstructionDataset,
    batch_size: int = 4,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader | None]:
    """
    Create train and validation dataloaders.

    Args:
        dataset: InstructionDataset instance
        batch_size: Batch size for training
        num_workers: Number of data loading workers

    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    train_dataset = dataset.get_train_dataset()
    val_dataset = dataset.get_val_dataset()

    # Create train dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Create validation dataloader
    val_dataloader = None
    if val_dataset:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    logger.info(f"Created dataloaders with batch_size={batch_size}")
    return train_dataloader, val_dataloader
