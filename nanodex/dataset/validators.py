"""Data quality validators for training datasets."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from nanodex.brain.graph_manager import GraphManager

logger = logging.getLogger(__name__)


class DatasetValidator:
    """Validate training dataset quality."""

    def __init__(
        self,
        graph_manager: GraphManager,
        min_response_tokens: int = 50,
        max_response_tokens: int = 500,
    ):
        """
        Initialize validator.

        Args:
            graph_manager: Graph manager for reference validation
            min_response_tokens: Minimum response length in tokens
            max_response_tokens: Maximum response length in tokens
        """
        self.gm = graph_manager
        self.min_response_tokens = min_response_tokens
        self.max_response_tokens = max_response_tokens

    def validate_example(self, example: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a single Q&A example.

        Args:
            example: Q&A dictionary

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # Check required fields
        required_fields = ["id", "type", "prompt", "response", "refs"]
        for field in required_fields:
            if field not in example:
                issues.append(f"Missing required field: {field}")

        if issues:
            return False, issues

        # Check prompt is non-empty
        if not example["prompt"].strip():
            issues.append("Empty prompt")

        # Check response is non-empty
        if not example["response"].strip():
            issues.append("Empty response")

        # Check response length (rough token count)
        response_tokens = len(example["response"].split()) * 1.3
        if response_tokens < self.min_response_tokens:
            issues.append(
                f"Response too short: {response_tokens:.0f} tokens "
                f"(min: {self.min_response_tokens})"
            )
        if response_tokens > self.max_response_tokens:
            issues.append(
                f"Response too long: {response_tokens:.0f} tokens "
                f"(max: {self.max_response_tokens})"
            )

        # Check for placeholder text
        placeholders = ["{}", "[]", "[TODO]", "TODO", "FIXME", "XXX", "[...]"]
        response_text = example["response"]
        for placeholder in placeholders:
            if placeholder in response_text:
                issues.append(f"Contains placeholder: {placeholder}")

        # Check refs is a list
        if not isinstance(example["refs"], list):
            issues.append("refs must be a list")
        elif len(example["refs"]) == 0:
            issues.append("refs list is empty")

        # Check type is valid
        valid_types = {"discovery", "explain", "howto", "diagnostics"}
        if example["type"] not in valid_types:
            issues.append(f"Invalid type: {example['type']}")

        return len(issues) == 0, issues

    def validate_node_references(self, examples: List[Dict[str, Any]]) -> Tuple[int, List[str]]:
        """
        Validate that all node references exist in the graph.

        Args:
            examples: List of Q&A examples

        Returns:
            Tuple of (valid_count, list_of_invalid_refs)
        """
        logger.info("Validating node references")

        if not self.gm.conn:
            raise RuntimeError("Graph manager not connected")

        # Get all valid node IDs
        cursor = self.gm.conn.execute("SELECT id FROM nodes")
        valid_node_ids = {row["id"] for row in cursor.fetchall()}

        invalid_refs = []
        valid_count = 0

        for example in examples:
            refs = example.get("refs", [])
            for ref in refs:
                if ref not in valid_node_ids:
                    invalid_refs.append(f"{example['id']}: {ref}")
                else:
                    valid_count += 1

        logger.info(f"Validated {valid_count} references, {len(invalid_refs)} invalid")
        return valid_count, invalid_refs

    def check_duplicates(self, examples: List[Dict[str, Any]]) -> List[str]:
        """
        Check for duplicate examples.

        Args:
            examples: List of Q&A examples

        Returns:
            List of duplicate IDs
        """
        logger.info("Checking for duplicates")
        seen_ids: Set[str] = set()
        seen_prompts: Set[str] = set()
        duplicates = []

        for example in examples:
            example_id = example["id"]
            prompt = example["prompt"].strip().lower()

            if example_id in seen_ids:
                duplicates.append(f"Duplicate ID: {example_id}")

            if prompt in seen_prompts:
                duplicates.append(f"Duplicate prompt: {example_id}")

            seen_ids.add(example_id)
            seen_prompts.add(prompt)

        logger.info(f"Found {len(duplicates)} duplicates")
        return duplicates

    def get_distribution(self, examples: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Get distribution of examples by type.

        Args:
            examples: List of Q&A examples

        Returns:
            Dictionary mapping type to count
        """
        distribution: Dict[str, int] = {}
        for example in examples:
            example_type = example.get("type", "unknown")
            distribution[example_type] = distribution.get(example_type, 0) + 1

        return distribution

    def validate_dataset(self, examples: List[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate entire dataset.

        Args:
            examples: List of Q&A examples

        Returns:
            Tuple of (is_valid, validation_report)
        """
        logger.info(f"Validating dataset with {len(examples)} examples")

        report: Dict[str, Any] = {
            "total_examples": len(examples),
            "valid_examples": 0,
            "invalid_examples": 0,
            "issues": [],
        }

        # Validate each example
        for i, example in enumerate(examples):
            is_valid, issues = self.validate_example(example)
            if is_valid:
                report["valid_examples"] += 1
            else:
                report["invalid_examples"] += 1
                report["issues"].extend(
                    [f"Example {i} ({example.get('id', 'unknown')}): {issue}" for issue in issues]
                )

        # Validate node references
        valid_refs, invalid_refs = self.validate_node_references(examples)
        report["valid_references"] = valid_refs
        report["invalid_references"] = len(invalid_refs)
        if invalid_refs:
            report["issues"].extend([f"Invalid node reference: {ref}" for ref in invalid_refs[:10]])

        # Check duplicates
        duplicates = self.check_duplicates(examples)
        report["duplicates"] = len(duplicates)
        if duplicates:
            report["issues"].extend(duplicates[:10])

        # Get distribution
        distribution = self.get_distribution(examples)
        report["type_distribution"] = distribution

        # Overall validation
        is_valid = (
            report["invalid_examples"] == 0
            and report["invalid_references"] == 0
            and report["duplicates"] == 0
        )

        report["is_valid"] = is_valid

        # Log summary
        logger.info("=" * 60)
        logger.info("VALIDATION REPORT")
        logger.info("=" * 60)
        logger.info(f"Total examples: {report['total_examples']}")
        logger.info(f"Valid examples: {report['valid_examples']}")
        logger.info(f"Invalid examples: {report['invalid_examples']}")
        logger.info(f"Valid references: {report['valid_references']}")
        logger.info(f"Invalid references: {report['invalid_references']}")
        logger.info(f"Duplicates: {report['duplicates']}")
        logger.info(f"Type distribution: {distribution}")
        logger.info(f"Overall valid: {is_valid}")
        logger.info("=" * 60)

        return is_valid, report


def validate_jsonl_file(
    jsonl_path: Path, db_path: Path, min_tokens: int = 50, max_tokens: int = 500
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate a JSONL dataset file.

    Args:
        jsonl_path: Path to JSONL file
        db_path: Path to graph database
        min_tokens: Minimum response tokens
        max_tokens: Maximum response tokens

    Returns:
        Tuple of (is_valid, report)
    """
    # Load examples
    examples = []
    with open(jsonl_path) as f:
        for line in f:
            examples.append(json.loads(line))

    # Validate
    with GraphManager(db_path) as gm:
        validator = DatasetValidator(gm, min_tokens, max_tokens)
        return validator.validate_dataset(examples)
