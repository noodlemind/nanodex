#!/usr/bin/env python3
"""Generate training dataset from knowledge graph."""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nanodex.brain.graph_manager import GraphManager
from nanodex.config import BrainConfig, DatasetConfig, load_config
from nanodex.dataset.qa_generator import QAGenerator
from nanodex.dataset.validators import DatasetValidator


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
        description="Generate training dataset from knowledge graph",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/dataset.yaml"),
        help="Path to dataset configuration file",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/brain/graph.sqlite"),
        help="Path to graph database",
    )
    parser.add_argument(
        "--summaries",
        type=Path,
        default=Path("data/brain/nodes"),
        help="Path to node summaries directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSONL file path (overrides config)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip dataset validation",
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
        config = load_config(args.config, DatasetConfig)

        # Check inputs exist
        if not args.db.exists():
            logger.error(f"Graph database not found: {args.db}")
            logger.info("Run 'make brain' first to build the graph and summaries")
            return 1

        if not args.summaries.exists():
            logger.error(f"Summaries directory not found: {args.summaries}")
            logger.info("Run 'make brain' first to generate summaries")
            return 1

        # Determine output path
        output_path = args.output or config.out_file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate Q&A pairs
        logger.info("Generating Q&A pairs from knowledge graph")
        with GraphManager(args.db) as gm:
            generator = QAGenerator(gm, args.summaries)

            qa_pairs = generator.generate_all_qa(
                counts=config.qa_categories,
                negatives_per_positive=config.negatives_per_example,
            )

            logger.info(f"Generated {len(qa_pairs)} Q&A pairs")

        # Validate dataset
        if not args.skip_validation:
            logger.info("Validating dataset quality")
            with GraphManager(args.db) as gm:
                validator = DatasetValidator(
                    gm,
                    min_response_tokens=config.min_response_tokens,
                    max_response_tokens=config.max_response_tokens,
                )

                is_valid, report = validator.validate_dataset(qa_pairs)

                if not is_valid:
                    logger.warning("Dataset validation found issues:")
                    for issue in report["issues"][:20]:
                        logger.warning(f"  - {issue}")

                    if report["invalid_examples"] > 0:
                        logger.error(f"Dataset has {report['invalid_examples']} invalid examples")
                        logger.info("Fix issues or use --skip-validation to proceed anyway")
                        return 1

        # Convert to instruction format and save
        logger.info(f"Saving dataset to: {output_path}")
        with open(output_path, "w") as f:
            for qa in qa_pairs:
                # Convert to instruction format
                instruction_example = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a domain expert on this codebase. "
                            "Answer questions accurately based on the code structure and documentation.",
                        },
                        {"role": "user", "content": qa["prompt"]},
                        {"role": "assistant", "content": qa["response"]},
                    ],
                    "metadata": {
                        "id": qa["id"],
                        "type": qa["type"],
                        "refs": qa["refs"],
                        **qa.get("metadata", {}),
                    },
                }

                f.write(json.dumps(instruction_example) + "\n")

        logger.info("Dataset generation complete!")
        logger.info(f"Total examples: {len(qa_pairs)}")
        logger.info(f"Output: {output_path}")

        # Print summary statistics
        type_counts = {}
        for qa in qa_pairs:
            qa_type = qa["type"]
            type_counts[qa_type] = type_counts.get(qa_type, 0) + 1

        logger.info("Type distribution:")
        for qa_type, count in sorted(type_counts.items()):
            percentage = (count / len(qa_pairs)) * 100
            logger.info(f"  {qa_type}: {count} ({percentage:.1f}%)")

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Dataset generation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
