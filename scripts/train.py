#!/usr/bin/env python3
"""Train LoRA/QLoRA adapter on instruction tuning dataset."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nanodex.config import TrainingConfig, load_config
from nanodex.trainer import LoRATrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train LoRA/QLoRA adapter")
    parser.add_argument("--config", type=Path, required=True, help="Path to training config YAML")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override output directory from config",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Override dataset path from config",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override per-device batch size",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("nanodex - LoRA/QLoRA Training")
    logger.info("=" * 60)

    # Load configuration
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config, TrainingConfig)

    # Apply CLI overrides
    if args.output_dir:
        config.training.output_dir = str(args.output_dir)
        logger.info(f"Override output_dir: {args.output_dir}")

    if args.dataset:
        config.dataset_path = str(args.dataset)
        logger.info(f"Override dataset: {args.dataset}")

    if args.epochs:
        config.training.num_train_epochs = args.epochs
        logger.info(f"Override epochs: {args.epochs}")

    if args.batch_size:
        config.training.per_device_train_batch_size = args.batch_size
        config.training.per_device_eval_batch_size = args.batch_size
        logger.info(f"Override batch_size: {args.batch_size}")

    # Create trainer
    logger.info("Initializing trainer...")
    trainer = LoRATrainer(config)

    # Setup (load model, tokenizer, dataset)
    logger.info("Setting up trainer (this may take a few minutes)...")
    trainer.setup()

    # Train
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)
    trainer.train()

    # Save adapter
    logger.info("=" * 60)
    logger.info("Saving trained adapter...")
    logger.info("=" * 60)
    trainer.save()

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Adapter saved to: {config.training.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
