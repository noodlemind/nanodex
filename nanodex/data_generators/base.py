"""Base class for data generation plugins."""

from abc import ABC, abstractmethod
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class DataGeneratorPlugin(ABC):
    """
    Abstract base class for data generation strategies.

    This allows users to choose between:
    - Free (self-supervised from code only)
    - Paid (synthetic data from GPT-4/Claude)
    - Hybrid (combination of both)
    """

    def __init__(self, config: Dict):
        """
        Initialize the data generator.

        Args:
            config: Configuration dictionary for this generator
        """
        self.config = config
        self.total_cost = 0.0
        self.examples_generated = 0

    @abstractmethod
    def generate(self, code_sample: Dict) -> List[Dict]:
        """
        Generate training examples from a code sample.

        Args:
            code_sample: Code sample with metadata (functions, classes, etc.)

        Returns:
            List of training examples in format:
            {
                'instruction': str,
                'input': str,
                'output': str,
                'metadata': dict (optional)
            }
        """
        pass

    @abstractmethod
    def estimate_cost(self, num_files: int) -> float:
        """
        Estimate cost in USD for processing given number of files.

        Args:
            num_files: Number of code files to process

        Returns:
            Estimated cost in USD
        """
        pass

    def get_name(self) -> str:
        """Get human-readable name of this generator."""
        return self.__class__.__name__

    def get_stats(self) -> Dict:
        """Get statistics about generation."""
        return {
            'generator': self.get_name(),
            'examples_generated': self.examples_generated,
            'total_cost_usd': self.total_cost
        }

    def log_progress(self, message: str):
        """Log progress with generator name."""
        logger.info(f"[{self.get_name()}] {message}")
