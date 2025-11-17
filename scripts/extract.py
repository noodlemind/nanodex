#!/usr/bin/env python3
"""Extract knowledge graph from a code repository."""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanodex.config import ExtractorConfig, load_config
from nanodex.extractor.graph_builder import GraphBuilder


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
        description="Extract knowledge graph from code repository",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "repo",
        type=Path,
        help="Path to repository to extract",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/extract.yaml"),
        help="Path to extractor configuration file",
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
        config = load_config(args.config, ExtractorConfig)

        # Build graph
        logger.info(f"Extracting symbols from: {args.repo}")
        builder = GraphBuilder(config)
        builder.build_graph(args.repo)

        logger.info(f"Knowledge graph saved to: {config.out_graph}")
        logger.info("Extraction complete!")

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Extraction failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
