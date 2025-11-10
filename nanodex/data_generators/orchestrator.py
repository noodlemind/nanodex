"""
Data generation orchestrator.

Coordinates multiple generators and handles:
- Mode selection (free/hybrid/full)
- Quality filtering
- Deduplication
- Progress tracking
- Cost estimation and management
"""

from typing import List, Dict, Optional
import logging
from pathlib import Path
from tqdm import tqdm
import hashlib
import json

from .self_supervised import SelfSupervisedGenerator
from .synthetic_api import SyntheticAPIGenerator

logger = logging.getLogger(__name__)


class DataGenerationOrchestrator:
    """
    Orchestrates data generation across multiple generators.

    Supports three modes:
    - free: Only self-supervised (zero cost)
    - hybrid: Self-supervised + optional synthetic (budget-limited)
    - full: Maximum quality with synthetic data (higher cost)
    """

    MODES = ['free', 'hybrid', 'full']

    def __init__(self, config: Dict):
        """
        Initialize orchestrator.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.mode = config.get('mode', 'free')

        if self.mode not in self.MODES:
            raise ValueError(
                f"Invalid mode: {self.mode}. "
                f"Must be one of: {', '.join(self.MODES)}"
            )

        # Initialize generators based on mode
        self.generators = self._initialize_generators()

        # Quality filtering
        self.min_example_length = config.get('min_example_length', 50)
        self.max_example_length = config.get('max_example_length', 4096)

        # Deduplication
        self.seen_hashes = set()
        self.dedup_enabled = config.get('enable_deduplication', True)

        # Statistics
        self.total_generated = 0
        self.total_filtered = 0
        self.total_duplicates = 0
        self.total_cost = 0.0

        logger.info(f"Initialized orchestrator in '{self.mode}' mode with {len(self.generators)} generators")

    def _initialize_generators(self) -> List:
        """Initialize generators based on selected mode."""
        generators = []

        # All modes get self-supervised generation (it's free!)
        generators.append(SelfSupervisedGenerator(self.config))
        logger.info("Added SelfSupervisedGenerator (free)")

        # Hybrid and full modes get synthetic generation
        if self.mode in ['hybrid', 'full']:
            api_config = self.config.get('synthetic_api', {})
            if api_config.get('api_key'):
                generators.append(SyntheticAPIGenerator(self.config))
                logger.info(f"Added SyntheticAPIGenerator ({self.mode} mode)")
            else:
                logger.warning(
                    f"Mode is '{self.mode}' but no API key provided. "
                    "Falling back to free mode (self-supervised only)"
                )

        return generators

    def generate_from_codebase(
        self,
        code_samples: List[Dict],
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Generate training examples from entire codebase.

        Args:
            code_samples: List of code samples with parsed structure
            show_progress: Whether to show progress bar

        Returns:
            List of all training examples
        """
        all_examples = []

        # Create progress bar
        iterator = tqdm(code_samples, desc="Generating training data") if show_progress else code_samples

        for code_sample in iterator:
            try:
                examples = self.generate_from_sample(code_sample)
                all_examples.extend(examples)

                if show_progress:
                    iterator.set_postfix({
                        'examples': len(all_examples),
                        'filtered': self.total_filtered,
                        'duplicates': self.total_duplicates,
                        'cost': f'${self.total_cost:.2f}'
                    })

            except Exception as e:
                logger.error(
                    f"Error generating from {code_sample.get('file_path')}: {e}",
                    exc_info=True
                )

        logger.info(
            f"Generation complete: {len(all_examples)} examples "
            f"({self.total_filtered} filtered, {self.total_duplicates} duplicates, "
            f"${self.total_cost:.2f} cost)"
        )

        return all_examples

    def generate_from_sample(self, code_sample: Dict) -> List[Dict]:
        """
        Generate training examples from a single code sample.

        Args:
            code_sample: Code sample with parsed structure

        Returns:
            List of filtered, deduplicated training examples
        """
        raw_examples = []

        # Run all generators
        for generator in self.generators:
            try:
                examples = generator.generate(code_sample)
                raw_examples.extend(examples)

                # Track cost
                if hasattr(generator, 'total_cost'):
                    self.total_cost += generator.total_cost

            except Exception as e:
                logger.error(
                    f"Error in {generator.get_name()}: {e}",
                    exc_info=True
                )

        # Filter and deduplicate
        filtered_examples = []
        for example in raw_examples:
            if self._should_keep_example(example):
                filtered_examples.append(example)
            else:
                self.total_filtered += 1

        self.total_generated += len(filtered_examples)
        return filtered_examples

    def _should_keep_example(self, example: Dict) -> bool:
        """
        Determine if example should be kept.

        Args:
            example: Training example

        Returns:
            True if example should be kept, False if filtered out
        """
        # Quality filtering
        if not self._passes_quality_filter(example):
            return False

        # Deduplication
        if self.dedup_enabled and not self._is_unique(example):
            self.total_duplicates += 1
            return False

        return True

    def _passes_quality_filter(self, example: Dict) -> bool:
        """Check if example meets quality standards."""
        # Check required fields
        if not example.get('input') or not example.get('output'):
            return False

        # Check length constraints
        input_len = len(example['input'])
        output_len = len(example['output'])

        if input_len < self.min_example_length or input_len > self.max_example_length:
            return False

        if output_len < self.min_example_length or output_len > self.max_example_length:
            return False

        # Check for code quality signals
        # (can be extended with more sophisticated checks)

        return True

    def _is_unique(self, example: Dict) -> bool:
        """
        Check if example is unique (not a duplicate).

        Args:
            example: Training example

        Returns:
            True if unique, False if duplicate
        """
        # Create hash of example content
        content = json.dumps({
            'input': example.get('input', ''),
            'output': example.get('output', ''),
        }, sort_keys=True)

        example_hash = hashlib.md5(content.encode()).hexdigest()

        if example_hash in self.seen_hashes:
            return False

        self.seen_hashes.add(example_hash)
        return True

    def estimate_total_cost(self, num_files: int) -> Dict[str, float]:
        """
        Estimate total cost for processing codebase.

        Args:
            num_files: Number of files in codebase

        Returns:
            Dictionary with cost estimates per generator
        """
        estimates = {}

        for generator in self.generators:
            cost = generator.estimate_cost(num_files)
            estimates[generator.get_name()] = cost

        estimates['total'] = sum(estimates.values())

        return estimates

    def get_stats(self) -> Dict:
        """Get generation statistics."""
        stats = {
            'mode': self.mode,
            'total_generated': self.total_generated,
            'total_filtered': self.total_filtered,
            'total_duplicates': self.total_duplicates,
            'total_cost_usd': self.total_cost,
            'generators': []
        }

        for generator in self.generators:
            stats['generators'].append(generator.get_stats())

        return stats

    def save_examples(self, examples: List[Dict], output_path: str):
        """
        Save generated examples to file.

        Args:
            examples: List of training examples
            output_path: Path to save JSON file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(examples, f, indent=2)

        logger.info(f"Saved {len(examples)} examples to {output_file}")

    def load_examples(self, input_path: str) -> List[Dict]:
        """
        Load examples from file.

        Args:
            input_path: Path to JSON file

        Returns:
            List of training examples
        """
        with open(input_path, 'r') as f:
            examples = json.load(f)

        logger.info(f"Loaded {len(examples)} examples from {input_path}")
        return examples


class QualityFilter:
    """
    Advanced quality filtering for training examples.

    Can be extended with more sophisticated checks:
    - Syntax validation
    - Semantic similarity
    - Diversity metrics
    - Domain-specific rules
    """

    def __init__(self, config: Dict):
        """
        Initialize quality filter.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.min_instruction_length = config.get('min_instruction_length', 10)
        self.min_code_lines = config.get('min_code_lines', 3)

    def filter_batch(self, examples: List[Dict]) -> List[Dict]:
        """
        Filter a batch of examples.

        Args:
            examples: List of training examples

        Returns:
            Filtered list
        """
        filtered = []

        for example in examples:
            if self.is_high_quality(example):
                filtered.append(example)

        logger.info(
            f"Quality filter: kept {len(filtered)}/{len(examples)} examples "
            f"({len(filtered)/len(examples)*100:.1f}%)"
        )

        return filtered

    def is_high_quality(self, example: Dict) -> bool:
        """
        Check if example is high quality.

        Args:
            example: Training example

        Returns:
            True if high quality, False otherwise
        """
        # Check instruction quality
        instruction = example.get('instruction', '')
        if len(instruction) < self.min_instruction_length:
            return False

        # Check code quality (for code examples)
        output = example.get('output', '')
        if output.strip():
            lines = [line for line in output.split('\n') if line.strip()]
            if len(lines) < self.min_code_lines:
                return False

        # Check for common issues
        if self._has_common_issues(example):
            return False

        return True

    def _has_common_issues(self, example: Dict) -> bool:
        """Check for common quality issues."""
        output = example.get('output', '').lower()

        # Check for placeholder text
        placeholders = [
            'todo', 'fixme', 'xxx', 'placeholder',
            'implement this', 'not implemented',
            '...', 'pass  # implementation'
        ]

        for placeholder in placeholders:
            if placeholder in output:
                return True

        # Check for error messages in output
        error_indicators = [
            'error:', 'exception:', 'traceback',
            'failed', 'cannot', "doesn't work"
        ]

        for indicator in error_indicators:
            if indicator in output[:200]:  # Check beginning of output
                return True

        return False
