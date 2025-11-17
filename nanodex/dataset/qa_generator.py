"""Q&A generator for training data from knowledge graph."""

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from nanodex.brain.graph_manager import GraphManager

logger = logging.getLogger(__name__)

# Q&A categories
QA_CATEGORIES = {"discovery", "explain", "howto", "diagnostics"}


class QAGenerator:
    """Generate Q&A pairs from knowledge graph."""

    def __init__(self, graph_manager: GraphManager, summary_dir: Path):
        """
        Initialize Q&A generator.

        Args:
            graph_manager: Graph manager instance
            summary_dir: Directory containing node summaries
        """
        self.gm = graph_manager
        self.summary_dir = Path(summary_dir)
        self.summaries_cache: Dict[str, Dict] = {}
        self._load_summaries()

    def _load_summaries(self) -> None:
        """Load all summaries into memory for faster access."""
        logger.info(f"Loading summaries from {self.summary_dir}")
        count = 0
        for summary_file in self.summary_dir.glob("*.json"):
            try:
                with open(summary_file) as f:
                    data = json.load(f)
                    self.summaries_cache[data["id"]] = data
                    count += 1
            except Exception as e:
                logger.warning(f"Failed to load {summary_file}: {e}")

        logger.info(f"Loaded {count} summaries")

    def generate_discovery_questions(self, count: int) -> List[Dict[str, Any]]:
        """
        Generate discovery questions: "Does X support Y?", "What capabilities are in X?"

        Args:
            count: Number of questions to generate

        Returns:
            List of Q&A dictionaries
        """
        logger.info(f"Generating {count} discovery questions")
        qa_pairs: List[Dict[str, Any]] = []

        # Get all capability nodes
        if not self.gm.conn:
            raise RuntimeError("Graph manager not connected")

        cursor = self.gm.conn.execute(
            "SELECT id, name, path, lang FROM nodes WHERE type = 'capability' LIMIT ?",
            (count * 2,),
        )
        capabilities = cursor.fetchall()

        for cap in capabilities[:count]:
            node_id = cap["id"]
            name = cap["name"]
            path = cap["path"]
            lang = cap["lang"]

            summary = self.summaries_cache.get(node_id, {})
            summary_text = summary.get("summary", "")

            if len(summary_text) < 20:
                continue

            # Generate discovery question
            question_variants = [
                f"Does the codebase have a '{name}' function?",
                f"What does the '{name}' capability do?",
                f"Is there a '{name}' function available?",
                f"Can I use '{name}' in this codebase?",
            ]

            question = random.choice(question_variants)

            # Generate answer using summary
            answer = f"Yes, there is a '{name}' function. {summary_text}"

            qa_pairs.append(
                {
                    "id": f"discovery_{len(qa_pairs):04d}",
                    "type": "discovery",
                    "prompt": question,
                    "response": answer,
                    "refs": [node_id],
                    "metadata": {
                        "node_name": name,
                        "node_type": "capability",
                        "file": path,
                        "language": lang,
                    },
                }
            )

        logger.info(f"Generated {len(qa_pairs)} discovery questions")
        return qa_pairs

    def generate_explain_questions(self, count: int) -> List[Dict[str, Any]]:
        """
        Generate explain questions: "How does X work?", "What is the purpose of Y?"

        Args:
            count: Number of questions to generate

        Returns:
            List of Q&A dictionaries
        """
        logger.info(f"Generating {count} explain questions")
        qa_pairs: List[Dict[str, Any]] = []

        # Get high-value nodes (concepts, capabilities with many connections)
        if not self.gm.conn:
            raise RuntimeError("Graph manager not connected")

        cursor = self.gm.conn.execute(
            """
            SELECT n.id, n.name, n.type, n.path, n.lang, COUNT(e.target_id) as edge_count
            FROM nodes n
            LEFT JOIN edges e ON n.id = e.source_id
            WHERE n.type IN ('capability', 'concept', 'class')
            GROUP BY n.id
            ORDER BY edge_count DESC
            LIMIT ?
            """,
            (count * 2,),
        )
        nodes = cursor.fetchall()

        for node in nodes[:count]:
            node_id = node["id"]
            name = node["name"]
            node_type = node["type"]
            path = node["path"]

            summary = self.summaries_cache.get(node_id, {})
            summary_text = summary.get("summary", "")
            context = summary.get("context", {})

            if len(summary_text) < 20:
                continue

            # Generate explain question
            question_variants = [
                f"How does '{name}' work?",
                f"What is the purpose of '{name}'?",
                f"Explain the '{name}' {node_type}.",
                f"What does '{name}' do?",
            ]

            question = random.choice(question_variants)

            # Generate detailed answer
            answer_parts = [summary_text]

            # Add neighbor context
            neighbors = context.get("neighbors", [])
            if neighbors:
                answer_parts.append(f"It interacts with: {', '.join(neighbors[:3])}.")

            # Add location context
            if context.get("lines"):
                answer_parts.append(f"Defined at lines {context['lines']} in {path}.")

            answer = " ".join(answer_parts)

            qa_pairs.append(
                {
                    "id": f"explain_{len(qa_pairs):04d}",
                    "type": "explain",
                    "prompt": question,
                    "response": answer,
                    "refs": [node_id],
                    "metadata": {
                        "node_name": name,
                        "node_type": node_type,
                        "file": path,
                    },
                }
            )

        logger.info(f"Generated {len(qa_pairs)} explain questions")
        return qa_pairs

    def generate_howto_questions(self, count: int) -> List[Dict[str, Any]]:
        """
        Generate howto questions: "How do I accomplish Z?", "Show example of W"

        Args:
            count: Number of questions to generate

        Returns:
            List of Q&A dictionaries
        """
        logger.info(f"Generating {count} howto questions")
        qa_pairs: List[Dict[str, Any]] = []

        # Get recipe nodes (examples, mains, tests)
        if not self.gm.conn:
            raise RuntimeError("Graph manager not connected")

        cursor = self.gm.conn.execute(
            "SELECT id, name, path, lang FROM nodes WHERE type = 'recipe' LIMIT ?",
            (count * 2,),
        )
        recipes = cursor.fetchall()

        # Also get capabilities for howto questions
        cursor = self.gm.conn.execute(
            "SELECT id, name, path, lang FROM nodes WHERE type = 'capability' LIMIT ?",
            (count,),
        )
        capabilities = cursor.fetchall()

        all_nodes = list(recipes) + list(capabilities)
        random.shuffle(all_nodes)

        for node in all_nodes[:count]:
            node_id = node["id"]
            name = node["name"]
            path = node["path"]

            summary = self.summaries_cache.get(node_id, {})
            summary_text = summary.get("summary", "")
            context = summary.get("context", {})

            if len(summary_text) < 20:
                continue

            # Generate howto question
            question_variants = [
                f"How do I use '{name}'?",
                f"Show me an example of using '{name}'.",
                f"How can I call '{name}'?",
                f"What's the correct way to use '{name}'?",
            ]

            question = random.choice(question_variants)

            # Generate answer with usage guidance
            answer_parts = [summary_text]
            answer_parts.append(f"You can find it in {path}.")

            neighbors = context.get("neighbors", [])
            if neighbors:
                answer_parts.append(f"It works with: {', '.join(neighbors[:2])}.")

            answer = " ".join(answer_parts)

            qa_pairs.append(
                {
                    "id": f"howto_{len(qa_pairs):04d}",
                    "type": "howto",
                    "prompt": question,
                    "response": answer,
                    "refs": [node_id],
                    "metadata": {
                        "node_name": name,
                        "file": path,
                    },
                }
            )

        logger.info(f"Generated {len(qa_pairs)} howto questions")
        return qa_pairs

    def generate_diagnostic_questions(self, count: int) -> List[Dict[str, Any]]:
        """
        Generate diagnostic questions: "I'm getting error E, what to check?"

        Args:
            count: Number of questions to generate

        Returns:
            List of Q&A dictionaries
        """
        logger.info(f"Generating {count} diagnostic questions")
        qa_pairs: List[Dict[str, Any]] = []

        # Get error nodes
        if not self.gm.conn:
            raise RuntimeError("Graph manager not connected")

        cursor = self.gm.conn.execute(
            "SELECT id, name, path, lang FROM nodes WHERE type = 'error' LIMIT ?",
            (count * 2,),
        )
        errors = cursor.fetchall()

        for error in errors[:count]:
            node_id = error["id"]
            name = error["name"]
            path = error["path"]

            summary = self.summaries_cache.get(node_id, {})
            summary_text = summary.get("summary", "")

            if len(summary_text) < 20:
                continue

            # Find nodes that might throw this error
            cursor = self.gm.conn.execute(
                """
                SELECT n.name
                FROM edges e
                JOIN nodes n ON e.source_id = n.id
                WHERE e.target_id = ? AND e.relationship = 'throws'
                LIMIT 3
                """,
                (node_id,),
            )
            throwers = [row["name"] for row in cursor.fetchall()]

            # Generate diagnostic question
            question_variants = [
                f"I'm getting a {name} error. What should I check?",
                f"What causes {name}?",
                f"How do I fix {name}?",
                f"When does {name} occur?",
            ]

            question = random.choice(question_variants)

            # Generate diagnostic answer
            answer_parts = [summary_text]

            if throwers:
                answer_parts.append(f"This error can be raised by: {', '.join(throwers)}.")

            answer_parts.append(f"Check the implementation in {path}.")

            answer = " ".join(answer_parts)

            qa_pairs.append(
                {
                    "id": f"diagnostic_{len(qa_pairs):04d}",
                    "type": "diagnostics",
                    "prompt": question,
                    "response": answer,
                    "refs": [node_id],
                    "metadata": {
                        "error_name": name,
                        "file": path,
                    },
                }
            )

        logger.info(f"Generated {len(qa_pairs)} diagnostic questions")
        return qa_pairs

    def generate_negative_examples(
        self, positive_examples: List[Dict[str, Any]], negatives_per_positive: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Generate negative examples for contrastive learning.

        Args:
            positive_examples: List of positive Q&A pairs
            negatives_per_positive: Number of negatives per positive

        Returns:
            List of negative Q&A dictionaries
        """
        logger.info(f"Generating negative examples ({negatives_per_positive} per positive)")
        negatives: List[Dict[str, Any]] = []

        # Get all nodes for creating false answers
        if not self.gm.conn:
            raise RuntimeError("Graph manager not connected")

        cursor = self.gm.conn.execute("SELECT id, name, type, path FROM nodes LIMIT 1000")
        all_nodes = cursor.fetchall()

        for example in positive_examples:
            for i in range(negatives_per_positive):
                # Create negative by referencing wrong node
                wrong_node = random.choice(all_nodes)
                summary = self.summaries_cache.get(wrong_node["id"], {})
                wrong_summary = summary.get("summary", "")

                # Create more detailed negative response
                negative_responses = [
                    f"No, the codebase does not have that specific functionality. "
                    f"However, you might be interested in '{wrong_node['name']}' which is a {wrong_node['type']} "
                    f"located in {wrong_node['path']}. {wrong_summary[:100] if wrong_summary else ''}",
                    f"That functionality is not available in the current codebase. "
                    f"The closest match is '{wrong_node['name']}' in {wrong_node['path']}, "
                    f"but it serves a different purpose. {wrong_summary[:100] if wrong_summary else ''} "
                    f"Please check the documentation for alternative approaches.",
                    f"I could not find that feature in the codebase. "
                    f"You may want to look at '{wrong_node['name']}' instead, which is available "
                    f"in {wrong_node['path']}. {wrong_summary[:100] if wrong_summary else ''} "
                    f"Consider reviewing the API documentation for similar capabilities.",
                ]

                negative = {
                    "id": f"{example['id']}_neg_{i}_{len(negatives)}",
                    "type": example["type"],
                    "prompt": example["prompt"],
                    "response": random.choice(negative_responses),
                    "refs": [wrong_node["id"]],
                    "metadata": {
                        **example["metadata"],
                        "is_negative": True,
                        "original_id": example["id"],
                    },
                }

                negatives.append(negative)

        logger.info(f"Generated {len(negatives)} negative examples")
        return negatives

    def generate_all_qa(
        self, counts: Dict[str, int], negatives_per_positive: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Generate all Q&A pairs across all categories.

        Args:
            counts: Dictionary mapping category to count
            negatives_per_positive: Number of negative examples per positive

        Returns:
            List of all Q&A pairs
        """
        all_qa = []

        # Generate positives for each category
        if counts.get("discovery", 0) > 0:
            all_qa.extend(self.generate_discovery_questions(counts["discovery"]))

        if counts.get("explain", 0) > 0:
            all_qa.extend(self.generate_explain_questions(counts["explain"]))

        if counts.get("howto", 0) > 0:
            all_qa.extend(self.generate_howto_questions(counts["howto"]))

        if counts.get("diagnostics", 0) > 0:
            all_qa.extend(self.generate_diagnostic_questions(counts["diagnostics"]))

        # Generate negatives
        if negatives_per_positive > 0:
            negatives = self.generate_negative_examples(all_qa, negatives_per_positive)
            all_qa.extend(negatives)

        # Shuffle to mix categories
        random.shuffle(all_qa)

        logger.info(f"Generated {len(all_qa)} total Q&A pairs")
        return all_qa
