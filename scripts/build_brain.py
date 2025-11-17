#!/usr/bin/env python3
"""Build brain: classify nodes and generate summaries."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nanodex.brain.graph_manager import GraphManager
from nanodex.brain.node_typer import NodeTyper
from nanodex.brain.summarizer import Summarizer
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
        description="Build brain: classify nodes and generate summaries",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/brain.yaml"),
        help="Path to brain configuration file",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/brain/graph.sqlite"),
        help="Path to graph database",
    )
    parser.add_argument(
        "--skip-classification",
        action="store_true",
        help="Skip node type classification",
    )
    parser.add_argument(
        "--skip-summaries",
        action="store_true",
        help="Skip summary generation",
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

        # Check database exists
        if not args.db.exists():
            logger.error(f"Graph database not found: {args.db}")
            logger.info("Run 'make extract' first to create the graph database")
            return 1

        # Classify nodes
        if not args.skip_classification:
            logger.info("Step 1: Classifying nodes into semantic types")
            with GraphManager(args.db) as gm:
                typer = NodeTyper(gm)
                type_counts = typer.classify_all_nodes()
                logger.info(f"Classification complete: {type_counts}")

        # Generate summaries
        if not args.skip_summaries:
            logger.info("Step 2: Generating node summaries")
            with GraphManager(args.db) as gm:
                summarizer = Summarizer(gm, config)
                count = summarizer.generate_all_summaries()
                logger.info(f"Generated {count} new summaries")

                # Get summary stats
                stats = summarizer.get_summary_stats()
                logger.info(f"Summary statistics: {stats}")

        logger.info("Brain build complete!")
        logger.info(f"Summaries saved to: {config.out_dir}")

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Brain build failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
