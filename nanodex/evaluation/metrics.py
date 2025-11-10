"""
Metrics for evaluating code understanding models.
"""

from typing import List, Dict, Any, Tuple
import logging
from collections import Counter
import difflib

logger = logging.getLogger(__name__)


class CodeMetrics:
    """
    Metrics for code understanding evaluation.

    Includes:
    - Exact match accuracy
    - BLEU score for code generation
    - Token-level F1
    - Functional correctness (if possible)
    """

    @staticmethod
    def exact_match(predictions: List[str], references: List[str]) -> float:
        """
        Calculate exact match accuracy.

        Args:
            predictions: List of predicted outputs
            references: List of reference outputs

        Returns:
            Exact match accuracy (0-1)
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")

        if len(predictions) == 0:
            return 0.0

        matches = sum(1 for pred, ref in zip(predictions, references)
                     if pred.strip() == ref.strip())

        return matches / len(predictions)

    @staticmethod
    def token_level_f1(predictions: List[str], references: List[str]) -> Tuple[float, float, float]:
        """
        Calculate token-level precision, recall, and F1.

        Args:
            predictions: List of predicted outputs
            references: List of reference outputs

        Returns:
            Tuple of (precision, recall, f1)
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")

        if len(predictions) == 0:
            return 0.0, 0.0, 0.0

        total_precision = 0.0
        total_recall = 0.0

        for pred, ref in zip(predictions, references):
            # Tokenize (simple whitespace split)
            pred_tokens = set(pred.split())
            ref_tokens = set(ref.split())

            if len(pred_tokens) == 0 and len(ref_tokens) == 0:
                precision = 1.0
                recall = 1.0
            elif len(pred_tokens) == 0:
                precision = 0.0
                recall = 0.0
            elif len(ref_tokens) == 0:
                precision = 0.0
                recall = 0.0
            else:
                overlap = len(pred_tokens & ref_tokens)
                precision = overlap / len(pred_tokens)
                recall = overlap / len(ref_tokens)

            total_precision += precision
            total_recall += recall

        avg_precision = total_precision / len(predictions)
        avg_recall = total_recall / len(predictions)

        if avg_precision + avg_recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)

        return avg_precision, avg_recall, f1

    @staticmethod
    def bleu_score(predictions: List[str], references: List[str], n: int = 4) -> float:
        """
        Calculate BLEU score (simplified implementation).

        Args:
            predictions: List of predicted outputs
            references: List of reference outputs
            n: Maximum n-gram size (default: 4)

        Returns:
            BLEU score (0-1)
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")

        if len(predictions) == 0:
            return 0.0

        scores = []

        for pred, ref in zip(predictions, references):
            pred_tokens = pred.split()
            ref_tokens = ref.split()

            # Calculate n-gram precisions
            precisions = []
            for i in range(1, n + 1):
                pred_ngrams = Counter(CodeMetrics._get_ngrams(pred_tokens, i))
                ref_ngrams = Counter(CodeMetrics._get_ngrams(ref_tokens, i))

                overlap = sum((pred_ngrams & ref_ngrams).values())
                total = sum(pred_ngrams.values())

                if total == 0:
                    precision = 0.0
                else:
                    precision = overlap / total

                precisions.append(precision)

            # Geometric mean of precisions
            if all(p > 0 for p in precisions):
                score = sum(precisions) / len(precisions)
            else:
                score = 0.0

            scores.append(score)

        return sum(scores) / len(scores)

    @staticmethod
    def _get_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        """Get n-grams from token list."""
        if n > len(tokens):
            return []
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    @staticmethod
    def edit_distance(predictions: List[str], references: List[str]) -> float:
        """
        Calculate average normalized edit distance (Levenshtein).

        Args:
            predictions: List of predicted outputs
            references: List of reference outputs

        Returns:
            Average normalized edit distance (0-1, lower is better)
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")

        if len(predictions) == 0:
            return 0.0

        distances = []

        for pred, ref in zip(predictions, references):
            # Use SequenceMatcher for edit distance
            ratio = difflib.SequenceMatcher(None, pred, ref).ratio()
            distance = 1.0 - ratio  # Convert similarity to distance
            distances.append(distance)

        return sum(distances) / len(distances)

    @staticmethod
    def code_similarity(pred_code: str, ref_code: str) -> float:
        """
        Calculate code similarity using SequenceMatcher.

        Args:
            pred_code: Predicted code
            ref_code: Reference code

        Returns:
            Similarity score (0-1)
        """
        return difflib.SequenceMatcher(None, pred_code, ref_code).ratio()

    @staticmethod
    def function_identification_accuracy(
        predictions: List[Dict[str, Any]],
        references: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate function identification metrics.

        Args:
            predictions: List of predicted function dicts with 'name', 'args', etc.
            references: List of reference function dicts

        Returns:
            Dict with accuracy metrics
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")

        if len(predictions) == 0:
            return {
                'name_accuracy': 0.0,
                'args_accuracy': 0.0,
                'returns_accuracy': 0.0,
            }

        name_matches = 0
        args_matches = 0
        returns_matches = 0

        for pred, ref in zip(predictions, references):
            # Check function name
            if pred.get('name') == ref.get('name'):
                name_matches += 1

            # Check arguments
            pred_args = set(pred.get('args', []))
            ref_args = set(ref.get('args', []))
            if pred_args == ref_args:
                args_matches += 1

            # Check return type
            if pred.get('returns') == ref.get('returns'):
                returns_matches += 1

        return {
            'name_accuracy': name_matches / len(predictions),
            'args_accuracy': args_matches / len(predictions),
            'returns_accuracy': returns_matches / len(predictions),
        }

    @staticmethod
    def aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate multiple evaluation results.

        Args:
            results: List of evaluation result dicts

        Returns:
            Aggregated metrics
        """
        if not results:
            return {}

        # Collect all metric names
        all_metrics = set()
        for result in results:
            all_metrics.update(result.keys())

        # Aggregate
        aggregated = {}
        for metric in all_metrics:
            values = [r.get(metric, 0) for r in results if metric in r]
            if values:
                aggregated[metric] = {
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values),
                }

        return aggregated
