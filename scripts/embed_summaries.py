#!/usr/bin/env python3
"""Generate embeddings for node summaries."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nanodex.brain.embedder import Embedder
from nanodex.config import BrainConfig, load_config


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate embeddings for node summaries",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/brain.yaml"),
        help="Path to brain configuration file",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Embedding model name (overrides config)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)

    try:
        # Load configuration
        logger.info(f"Loading configuration from: {args.config}")
        config = load_config(args.config, BrainConfig)

        # Determine model to use
        model_name = args.model or config.embedding_model
        if not model_name:
            logger.error("No embedding model specified in config or arguments")
            logger.info("Example: --model sentence-transformers/all-MiniLM-L6-v2")
            return 1

        # Check summary directory exists
        summary_dir = Path(config.out_dir)
        if not summary_dir.exists():
            logger.error(f"Summary directory not found: {summary_dir}")
            logger.info("Run 'make brain' first to generate summaries")
            return 1

        # Generate embeddings
        logger.info(f"Generating embeddings using model: {model_name}")
        embedder = Embedder(model_name)
        count = embedder.embed_summaries(summary_dir)

        logger.info(f"Embedding generation complete: {count} embeddings")

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Embedding generation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
