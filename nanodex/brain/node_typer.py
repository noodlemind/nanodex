"""Node type classifier for semantic categorization."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

from nanodex.brain.graph_manager import GraphManager

logger = logging.getLogger(__name__)

# Semantic node types
SEMANTIC_TYPES = {"module", "capability", "concept", "error", "recipe"}


class NodeTyper:
    """Classify nodes into semantic types based on heuristics."""

    def __init__(self, graph_manager: GraphManager):
        """
        Initialize node typer.

        Args:
            graph_manager: Graph manager instance
        """
        self.gm = graph_manager

    def classify_all_nodes(self) -> Dict[str, int]:
        """
        Classify all nodes in the graph into semantic types.

        Returns:
            Dictionary mapping semantic types to counts
        """
        logger.info("Starting node type classification")

        if not self.gm.conn:
            raise RuntimeError("Graph manager not connected")

        # Get all nodes
        cursor = self.gm.conn.execute("SELECT id, type, name, properties FROM nodes")
        nodes = cursor.fetchall()

        type_counts: Dict[str, int] = {t: 0 for t in SEMANTIC_TYPES}
        classified = 0

        for row in nodes:
            node_id = row["id"]
            current_type = row["type"]
            name = row["name"]
            properties = json.loads(row["properties"]) if row["properties"] else {}

            # Classify based on current type and heuristics
            semantic_type = self._infer_semantic_type(current_type, name, properties)

            if semantic_type != current_type:
                # Update node type
                self.gm.conn.execute(
                    "UPDATE nodes SET type = ? WHERE id = ?", (semantic_type, node_id)
                )
                classified += 1

            type_counts[semantic_type] = type_counts.get(semantic_type, 0) + 1

        self.gm.conn.commit()

        logger.info(f"Classified {classified} nodes")
        logger.info(f"Type distribution: {type_counts}")

        return type_counts

    def _infer_semantic_type(
        self, current_type: str, name: str, properties: Dict
    ) -> str:
        """
        Infer semantic type based on node characteristics.

        Args:
            current_type: Current node type (file, function, class, etc.)
            name: Node name
            properties: Node properties

        Returns:
            Semantic type (module, capability, concept, error, recipe)
        """
        name_lower = name.lower()

        # Files are modules
        if current_type == "file":
            return "module"

        # Error/Exception classes
        if current_type == "class" and self._is_error_class(name):
            return "error"

        # Recipe: main functions, examples, demos
        if current_type == "function" and self._is_recipe_function(name_lower):
            return "recipe"

        # Capability: public functions (not starting with _)
        if current_type == "function" and self._is_capability_function(name):
            return "capability"

        # Concept: classes that are not errors
        if current_type == "class":
            return "concept"

        # Default: internal functions and variables are concepts
        if current_type in ("function", "variable"):
            return "concept"

        # Keep external and other types as-is
        return current_type

    def _is_error_class(self, name: str) -> bool:
        """Check if a class name indicates an error/exception."""
        error_patterns = [
            "error",
            "exception",
            "fault",
            "failure",
            "warning",
        ]
        name_lower = name.lower()
        return any(pattern in name_lower for pattern in error_patterns)

    def _is_recipe_function(self, name_lower: str) -> bool:
        """Check if a function is a recipe (main, example, demo)."""
        recipe_patterns = [
            "main",
            "example",
            "demo",
            "test_",  # Test functions can be examples
            "run_",
            "execute_",
        ]
        return any(name_lower.startswith(pattern) or name_lower == pattern.rstrip("_")
                   for pattern in recipe_patterns)

    def _is_capability_function(self, name: str) -> bool:
        """Check if a function is a public capability."""
        # Public functions don't start with underscore
        if name.startswith("_"):
            return False

        # Special methods are not capabilities
        if name.startswith("__") and name.endswith("__"):
            return False

        return True

    def get_classification_stats(self) -> Dict[str, any]:
        """
        Get detailed classification statistics.

        Returns:
            Dictionary with classification breakdown
        """
        if not self.gm.conn:
            raise RuntimeError("Graph manager not connected")

        stats = {}

        # Count by semantic type
        cursor = self.gm.conn.execute(
            """
            SELECT type, COUNT(*) as count
            FROM nodes
            WHERE type IN ({})
            GROUP BY type
            """.format(",".join("?" * len(SEMANTIC_TYPES))),
            tuple(SEMANTIC_TYPES),
        )

        type_counts = {row["type"]: row["count"] for row in cursor.fetchall()}
        stats["semantic_types"] = type_counts

        # Count original types (from properties if stored)
        cursor = self.gm.conn.execute(
            """
            SELECT type, COUNT(*) as count
            FROM nodes
            GROUP BY type
            """
        )

        all_types = {row["type"]: row["count"] for row in cursor.fetchall()}
        stats["all_types"] = all_types

        return stats


def classify_graph_nodes(db_path: Path) -> Dict[str, int]:
    """
    Classify nodes in a graph database.

    Args:
        db_path: Path to graph database

    Returns:
        Type counts dictionary
    """
    with GraphManager(db_path) as gm:
        typer = NodeTyper(gm)
        return typer.classify_all_nodes()
