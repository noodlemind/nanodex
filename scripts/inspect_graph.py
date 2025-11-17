#!/usr/bin/env python3
"""Inspect knowledge graph statistics and integrity."""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanodex.brain.graph_manager import GraphManager


def print_stats(stats: dict) -> None:
    """Pretty print graph statistics."""
    print("\n" + "=" * 70)
    print("KNOWLEDGE GRAPH STATISTICS")
    print("=" * 70)
    print(f"\nTotal Nodes:      {stats['total_nodes']:,}")
    print(f"Total Edges:      {stats['total_edges']:,}")
    print(f"Unique Node Types: {stats['unique_node_types']}")
    print(f"Unique Edge Types: {stats['unique_edge_types']}")

    print("\n" + "-" * 70)
    print("NODE TYPE DISTRIBUTION")
    print("-" * 70)
    print(f"{'Type':<20} {'Count':>10} {'Percentage':>12}")
    print("-" * 70)
    for dist in stats["node_distribution"]:
        print(f"{dist['type']:<20} {dist['count']:>10,} {dist['percentage']:>11.2f}%")

    print("\n" + "-" * 70)
    print("EDGE RELATIONSHIP DISTRIBUTION")
    print("-" * 70)
    print(f"{'Relationship':<20} {'Count':>10} {'Percentage':>12}")
    print("-" * 70)
    for dist in stats["edge_distribution"]:
        print(f"{dist['relationship']:<20} {dist['count']:>10,} {dist['percentage']:>11.2f}%")

    print("=" * 70 + "\n")


def print_integrity(integrity: dict) -> None:
    """Pretty print integrity check results."""
    print("\n" + "=" * 70)
    print("INTEGRITY CHECK")
    print("=" * 70)

    if integrity["valid"]:
        print("\n✓ Graph integrity: PASSED")
        print("\nNo issues found.")
    else:
        print("\n✗ Graph integrity: FAILED")
        print("\nIssues found:")
        for i, issue in enumerate(integrity["issues"], 1):
            print(f"  {i}. {issue}")

    print("=" * 70 + "\n")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Inspect knowledge graph database",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/brain/graph.sqlite"),
        help="Path to graph database",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of formatted text",
    )
    parser.add_argument(
        "--check-integrity",
        action="store_true",
        help="Run integrity validation checks",
    )

    args = parser.parse_args()

    if not args.db.exists():
        print(f"Error: Database not found: {args.db}", file=sys.stderr)
        return 1

    try:
        with GraphManager(args.db) as gm:
            stats = gm.get_stats()

            if args.json:
                output = {"stats": stats}
                if args.check_integrity:
                    output["integrity"] = gm.validate_integrity()
                print(json.dumps(output, indent=2))
            else:
                print_stats(stats)
                if args.check_integrity:
                    integrity = gm.validate_integrity()
                    print_integrity(integrity)

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
