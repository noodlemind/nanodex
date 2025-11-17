#!/usr/bin/env python3
"""Quick training test - verifies setup works on current hardware."""

import logging
from pathlib import Path

from nanodex.config import TrainingConfig, load_config
from nanodex.trainer.trainer import LoRATrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    """Test training initialization."""
    logger.info("=" * 60)
    logger.info("NanoDex Training Test (NanoChat-inspired)")
    logger.info("=" * 60)

    # Load config
    config_path = Path("config/train_mps.yaml")
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        logger.info("Using default QLoRA config as fallback")
        config_path = Path("config/train_qlora.yaml")

    logger.info(f"Loading config: {config_path}")
    config = load_config(config_path, TrainingConfig)

    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = LoRATrainer(config)

    # Test device detection
    logger.info(f"✓ Device: {trainer.device}")
    logger.info(f"✓ Quantization: {trainer.can_use_quantization}")

    logger.info("=" * 60)
    logger.info("Training test PASSED - setup is ready!")
    logger.info("=" * 60)

    # Print next steps
    if trainer.device == "cuda":
        logger.info("🚀 You can run full training with:")
        logger.info("   make train-qlora CONFIG=config/train_qlora.yaml")
    elif trainer.device == "mps":
        logger.info("🍎 You can run training on Apple Silicon with:")
        logger.info("   .venv/bin/python scripts/train.py --config config/train_mps.yaml")
    else:
        logger.info("💻 Training will be slow on CPU, but you can test with:")
        logger.info("   .venv/bin/python scripts/train.py --config config/train_mps.yaml")


if __name__ == "__main__":
    main()
