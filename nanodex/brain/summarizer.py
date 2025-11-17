"""Summary generator for knowledge graph nodes."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from nanodex.brain.graph_manager import GraphManager
from nanodex.config import BrainConfig

logger = logging.getLogger(__name__)


class Summarizer:
    """Generate summaries for knowledge graph nodes."""

    def __init__(self, graph_manager: GraphManager, config: BrainConfig):
        """
        Initialize summarizer.

        Args:
            graph_manager: Graph manager instance
            config: Brain configuration
        """
        self.gm = graph_manager
        self.config = config
        self.output_dir = Path(config.out_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all_summaries(self) -> int:
        """
        Generate summaries for all nodes.

        Returns:
            Number of summaries generated
        """
        logger.info("Generating node summaries")

        if not self.gm.conn:
            raise RuntimeError("Graph manager not connected")

        # Get all nodes
        cursor = self.gm.conn.execute("SELECT id, type, name, path, lang, properties FROM nodes")
        nodes = cursor.fetchall()

        generated = 0
        skipped = 0

        for row in nodes:
            node_id = row["id"]
            summary_path = self.output_dir / f"{node_id}.json"

            # Skip if summary already exists (caching)
            if summary_path.exists():
                skipped += 1
                if generated % 100 == 0 and generated > 0:
                    logger.info(f"Progress: {generated} generated, {skipped} cached")
                continue

            # Generate summary
            try:
                summary_data = self._generate_node_summary(row)
                self._save_summary(summary_path, summary_data)
                generated += 1

                if generated % 100 == 0:
                    logger.info(f"Progress: {generated} generated, {skipped} cached")

            except Exception as e:
                logger.error(f"Failed to generate summary for {node_id}: {e}")

        logger.info(f"Summary generation complete: {generated} new, {skipped} cached")
        return generated

    def _generate_node_summary(self, node_row: Any) -> Dict[str, Any]:
        """
        Generate summary for a single node.

        Args:
            node_row: Database row with node data

        Returns:
            Summary data dictionary
        """
        node_id = node_row["id"]
        node_type = node_row["type"]
        name = node_row["name"]
        path = node_row["path"]
        lang = node_row["lang"]
        properties = json.loads(node_row["properties"]) if node_row["properties"] else {}

        # Get neighbors for context
        neighbors = self.gm.get_neighbors(node_id, direction="both")
        neighbor_names = [n["name"] for n in neighbors[:5]]  # Limit to 5 for brevity

        # Build context
        context = {
            "file": path,
            "language": lang,
            "neighbors": neighbor_names,
        }

        # Add type-specific context
        if "start_line" in properties:
            context["line"] = properties["start_line"]
        if "end_line" in properties:
            context["lines"] = f"{properties['start_line']}-{properties['end_line']}"

        # Generate summary based on node type and style
        summary = self._build_summary_text(node_type, name, context, properties)

        return {
            "id": node_id,
            "type": node_type,
            "name": name,
            "summary": summary,
            "context": context,
        }

    def _build_summary_text(
        self, node_type: str, name: str, context: Dict, properties: Dict
    ) -> str:
        """
        Build summary text based on node type and style.

        Args:
            node_type: Type of node
            name: Node name
            context: Context dictionary
            properties: Node properties

        Returns:
            Summary text
        """
        if self.config.summary_style == "concise":
            return self._build_concise_summary(node_type, name, context, properties)
        elif self.config.summary_style == "detailed":
            return self._build_detailed_summary(node_type, name, context, properties)
        else:  # technical
            return self._build_technical_summary(node_type, name, context, properties)

    def _build_concise_summary(
        self, node_type: str, name: str, context: Dict, properties: Dict
    ) -> str:
        """Build concise summary."""
        summaries = {
            "module": f"Module '{name}' in {context.get('file', 'unknown file')}.",
            "capability": f"Function '{name}' provides functionality. "
                         f"Located in {context.get('file', 'unknown')}.",
            "concept": f"'{name}' is a {node_type} representing a core concept. "
                      f"Defined in {context.get('file', 'unknown')}.",
            "error": f"Error type '{name}' for handling exceptional conditions. "
                    f"Defined in {context.get('file', 'unknown')}.",
            "recipe": f"'{name}' demonstrates usage or provides an entry point. "
                     f"Located in {context.get('file', 'unknown')}.",
            "function": f"Function '{name}' in {context.get('file', 'unknown')}.",
            "class": f"Class '{name}' in {context.get('file', 'unknown')}.",
            "file": f"Source file '{name}' in {context.get('language', 'unknown')}.",
        }

        base_summary = summaries.get(node_type, f"'{name}' is a {node_type}.")

        # Add neighbor context if available
        if context.get("neighbors"):
            neighbor_str = ", ".join(context["neighbors"][:3])
            base_summary += f" Related to: {neighbor_str}."

        return base_summary

    def _build_detailed_summary(
        self, node_type: str, name: str, context: Dict, properties: Dict
    ) -> str:
        """Build detailed summary."""
        parts = [self._build_concise_summary(node_type, name, context, properties)]

        # Add location details
        if context.get("lines"):
            parts.append(f"Spans lines {context['lines']}.")
        elif context.get("line"):
            parts.append(f"Located at line {context['line']}.")

        # Add language info
        if context.get("language"):
            parts.append(f"Written in {context['language']}.")

        # Add neighbor information
        if context.get("neighbors") and len(context["neighbors"]) > 0:
            parts.append(f"Connected to: {', '.join(context['neighbors'])}.")

        # Add properties
        if properties.get("doc"):
            parts.append(f"Documentation: {properties['doc'][:100]}...")

        return " ".join(parts)

    def _build_technical_summary(
        self, node_type: str, name: str, context: Dict, properties: Dict
    ) -> str:
        """Build technical summary."""
        parts = [
            f"Node ID: {name}",
            f"Type: {node_type}",
            f"Location: {context.get('file', 'N/A')}",
        ]

        if context.get("lines"):
            parts.append(f"Lines: {context['lines']}")

        if context.get("language"):
            parts.append(f"Language: {context['language']}")

        if context.get("neighbors"):
            parts.append(f"Dependencies: {len(context['neighbors'])}")

        for key, value in properties.items():
            if isinstance(value, (str, int, float, bool)):
                parts.append(f"{key}: {value}")

        return " | ".join(parts)

    def _save_summary(self, path: Path, summary_data: Dict[str, Any]) -> None:
        """
        Save summary to JSON file.

        Args:
            path: Output file path
            summary_data: Summary dictionary
        """
        with open(path, "w") as f:
            json.dump(summary_data, f, indent=2)

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get statistics about generated summaries.

        Returns:
            Statistics dictionary
        """
        summary_files = list(self.output_dir.glob("*.json"))

        stats = {
            "total_summaries": len(summary_files),
            "output_dir": str(self.output_dir),
        }

        if summary_files:
            # Sample a few summaries for length statistics
            lengths = []
            for summary_file in summary_files[:100]:
                try:
                    with open(summary_file) as f:
                        data = json.load(f)
                        summary_text = data.get("summary", "")
                        # Rough token count (words * 1.3)
                        tokens = len(summary_text.split()) * 1.3
                        lengths.append(tokens)
                except Exception:
                    continue

            if lengths:
                stats["avg_summary_tokens"] = int(sum(lengths) / len(lengths))
                stats["min_summary_tokens"] = int(min(lengths))
                stats["max_summary_tokens"] = int(max(lengths))

        return stats


def generate_summaries(db_path: Path, config: BrainConfig) -> int:
    """
    Generate summaries for all nodes in graph.

    Args:
        db_path: Path to graph database
        config: Brain configuration

    Returns:
        Number of summaries generated
    """
    with GraphManager(db_path) as gm:
        summarizer = Summarizer(gm, config)
        return summarizer.generate_all_summaries()
