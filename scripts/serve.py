#!/usr/bin/env python3
"""Print instructions for starting the inference server."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nanodex.config import InferenceConfig, load_config
from nanodex.inference.server import print_server_instructions

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for serving."""
    parser = argparse.ArgumentParser(description="nanodex inference server instructions")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/inference.yaml"),
        help="Path to inference config YAML",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        help="Override base model",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        help="Override adapter path",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Override server port",
    )

    args = parser.parse_args()

    # Load configuration
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config, InferenceConfig)

    # Apply CLI overrides
    if args.base_model:
        config.base_model = args.base_model
        logger.info(f"Override base_model: {args.base_model}")

    if args.adapter:
        config.adapter_path = args.adapter
        logger.info(f"Override adapter: {args.adapter}")

    if args.port:
        config.port = args.port
        logger.info(f"Override port: {args.port}")

    # Print server instructions
    print_server_instructions(config)


if __name__ == "__main__":
    main()
