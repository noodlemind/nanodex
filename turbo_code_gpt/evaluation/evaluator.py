"""
Model evaluator for code understanding tasks.
"""

from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
from datasets import Dataset
from tqdm import tqdm

from .metrics import CodeMetrics

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluate model performance on code understanding tasks.
    """

    def __init__(self, model=None, tokenizer=None):
        """
        Initialize evaluator.

        Args:
            model: Model to evaluate (optional)
            tokenizer: Tokenizer (optional)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.metrics = CodeMetrics()

    def evaluate_dataset(
        self,
        dataset: Dataset,
        max_samples: Optional[int] = None,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate model on a dataset.

        Args:
            dataset: Dataset with 'instruction', 'input', 'output' fields
            max_samples: Maximum number of samples to evaluate (None = all)
            show_progress: Show progress bar

        Returns:
            Dict with evaluation metrics
        """
        logger.info(f"Evaluating on {len(dataset)} samples...")

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        predictions = []
        references = []

        iterator = tqdm(dataset, desc="Evaluating") if show_progress else dataset

        for example in iterator:
            # Generate prediction
            if self.model and self.tokenizer:
                pred = self._generate_prediction(example)
            else:
                pred = ""  # No model, just compute reference metrics

            predictions.append(pred)
            references.append(example.get('output', ''))

        # Calculate metrics
        results = self._calculate_metrics(predictions, references)

        logger.info(f"Evaluation complete. Results: {results}")
        return results

    def _generate_prediction(self, example: Dict[str, Any]) -> str:
        """
        Generate prediction for a single example.

        Args:
            example: Example with 'instruction' and 'input'

        Returns:
            Generated output
        """
        # Format input
        instruction = example.get('instruction', '')
        input_text = example.get('input', '')

        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )

        # Move to device
        if hasattr(self.model, 'device'):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        with logger.debug("Generating..."):
            outputs = self.model.generate(
                **inputs,
                max_length=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove prompt
        if response.startswith(prompt):
            response = response[len(prompt):].strip()

        return response

    def _calculate_metrics(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Calculate all evaluation metrics.

        Args:
            predictions: List of predictions
            references: List of references

        Returns:
            Dict with metrics
        """
        results = {}

        # Exact match
        results['exact_match'] = self.metrics.exact_match(predictions, references)

        # Token-level F1
        precision, recall, f1 = self.metrics.token_level_f1(predictions, references)
        results['token_precision'] = precision
        results['token_recall'] = recall
        results['token_f1'] = f1

        # BLEU
        results['bleu'] = self.metrics.bleu_score(predictions, references)

        # Edit distance
        results['edit_distance'] = self.metrics.edit_distance(predictions, references)

        # Code similarity (average)
        similarities = [
            self.metrics.code_similarity(pred, ref)
            for pred, ref in zip(predictions, references)
        ]
        results['avg_code_similarity'] = sum(similarities) / len(similarities) if similarities else 0.0

        return results

    def evaluate_on_test_set(
        self,
        test_file: str,
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate on a test set file.

        Args:
            test_file: Path to test JSON file
            output_file: Optional path to save results

        Returns:
            Evaluation results
        """
        import json

        # Load test set
        with open(test_file, 'r') as f:
            test_data = json.load(f)

        # Convert to dataset
        from datasets import Dataset
        dataset = Dataset.from_list(test_data)

        # Evaluate
        results = self.evaluate_dataset(dataset)

        # Save results if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_file}")

        return results

    def compare_checkpoints(
        self,
        checkpoint_dirs: List[str],
        test_dataset: Dataset
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple checkpoints on the same test set.

        Args:
            checkpoint_dirs: List of checkpoint directories
            test_dataset: Test dataset

        Returns:
            Dict mapping checkpoint paths to results
        """
        results = {}

        for checkpoint_dir in checkpoint_dirs:
            logger.info(f"Evaluating checkpoint: {checkpoint_dir}")

            # Load model (simplified - actual loading would use ModelLoader)
            # For now, just evaluate with current model
            checkpoint_results = self.evaluate_dataset(test_dataset, max_samples=100)
            results[checkpoint_dir] = checkpoint_results

        return results
