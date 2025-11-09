"""
Data preparation module for creating training datasets.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
import logging
from datasets import Dataset
import random

logger = logging.getLogger(__name__)


class DataPreparer:
    """Prepares training data from code samples."""
    
    def __init__(self, config: Dict):
        """
        Initialize data preparer.
        
        Args:
            config: Data configuration dictionary
        """
        self.output_dir = Path(config.get('output_dir', './data/processed'))
        self.train_split = config.get('train_split', 0.9)
        self.validation_split = config.get('validation_split', 0.1)
        self.context_window = config.get('context_window', 4096)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def prepare_data(self, code_samples: List[Dict[str, str]]) -> Tuple[Dataset, Dataset]:
        """
        Prepare training and validation datasets.
        
        Args:
            code_samples: List of code samples from the analyzer
            
        Returns:
            Tuple of (train_dataset, validation_dataset)
        """
        logger.info(f"Preparing data from {len(code_samples)} code samples")
        
        # Create training examples
        training_examples = self._create_training_examples(code_samples)
        
        # Split into train and validation
        random.shuffle(training_examples)
        split_idx = int(len(training_examples) * self.train_split)
        
        train_examples = training_examples[:split_idx]
        val_examples = training_examples[split_idx:]
        
        logger.info(f"Created {len(train_examples)} training examples")
        logger.info(f"Created {len(val_examples)} validation examples")
        
        # Create datasets
        train_dataset = Dataset.from_list(train_examples)
        val_dataset = Dataset.from_list(val_examples)
        
        # Save datasets
        self._save_datasets(train_dataset, val_dataset)
        
        return train_dataset, val_dataset
    
    def _create_training_examples(self, code_samples: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Create training examples from code samples.
        
        This creates instruction-following examples where the model learns to:
        - Explain code
        - Answer questions about code
        - Understand code structure
        
        Args:
            code_samples: List of code samples
            
        Returns:
            List of training examples
        """
        examples = []
        
        for sample in code_samples:
            # Create different types of training examples
            
            # 1. Code explanation example
            examples.append({
                'instruction': f"Explain the following {sample['language']} code from {sample['file_path']}:",
                'input': sample['content'],
                'output': self._generate_code_explanation(sample)
            })
            
            # 2. Code understanding example
            examples.append({
                'instruction': f"What does this {sample['language']} code do?",
                'input': f"File: {sample['file_path']}\n\n{sample['content']}",
                'output': f"This code from {sample['file_path']} is a {sample['language']} implementation."
            })
            
            # 3. Code structure example
            if sample['lines'] > 10:
                examples.append({
                    'instruction': f"Describe the structure of this {sample['language']} file:",
                    'input': sample['content'],
                    'output': f"This {sample['language']} file ({sample['file_path']}) contains {sample['lines']} lines of code."
                })
        
        return examples
    
    def _generate_code_explanation(self, sample: Dict[str, str]) -> str:
        """
        Generate a basic explanation for code sample.
        
        Args:
            sample: Code sample dictionary
            
        Returns:
            Explanation string
        """
        return (
            f"This is a {sample['language']} file located at {sample['file_path']}. "
            f"It contains {sample['lines']} lines of code that implements functionality "
            f"specific to this codebase."
        )
    
    def _save_datasets(self, train_dataset: Dataset, val_dataset: Dataset):
        """
        Save datasets to disk.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
        """
        train_path = self.output_dir / "train"
        val_path = self.output_dir / "validation"
        
        train_dataset.save_to_disk(str(train_path))
        val_dataset.save_to_disk(str(val_path))
        
        logger.info(f"Saved training dataset to {train_path}")
        logger.info(f"Saved validation dataset to {val_path}")
        
        # Also save as JSON for inspection
        train_json_path = self.output_dir / "train.json"
        val_json_path = self.output_dir / "validation.json"
        
        with open(train_json_path, 'w') as f:
            json.dump(train_dataset.to_list(), f, indent=2)
        
        with open(val_json_path, 'w') as f:
            json.dump(val_dataset.to_list(), f, indent=2)
        
        logger.info(f"Saved JSON versions for inspection")
    
    def format_instruction(self, example: Dict[str, str]) -> str:
        """
        Format an example into instruction format.
        
        Args:
            example: Training example dictionary
            
        Returns:
            Formatted instruction string
        """
        if example.get('input'):
            return (
                f"### Instruction:\n{example['instruction']}\n\n"
                f"### Input:\n{example['input']}\n\n"
                f"### Response:\n{example['output']}"
            )
        else:
            return (
                f"### Instruction:\n{example['instruction']}\n\n"
                f"### Response:\n{example['output']}"
            )
