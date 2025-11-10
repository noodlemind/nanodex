"""
Model trainer for fine-tuning code models.
"""

import logging
from pathlib import Path
from typing import Optional
import torch
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Fine-tunes models on code data."""
    
    def __init__(self, model, tokenizer, config: dict):
        """
        Initialize model trainer.
        
        Args:
            model: The model to train
            tokenizer: The tokenizer
            config: Training configuration dictionary
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
    
    def train(self, train_dataset: Dataset, val_dataset: Optional[Dataset] = None):
        """
        Train the model on the provided datasets.

        Args:
            train_dataset: Training dataset
            val_dataset: Optional validation dataset

        Raises:
            ValueError: If datasets are invalid or empty
        """
        logger.info("Starting model training")

        # Validate datasets before expensive operations
        self._validate_datasets(train_dataset, val_dataset)

        # Prepare datasets
        train_dataset = self._prepare_dataset(train_dataset)
        if val_dataset:
            val_dataset = self._prepare_dataset(val_dataset)
        
        # Create training arguments
        training_args = self._create_training_args()
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        
        # Train
        logger.info("Beginning training...")
        trainer.train()
        
        # Save model
        output_dir = Path(self.config.get('output_dir', './models/fine-tuned'))
        trainer.save_model(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))
        
        logger.info(f"Model saved to {output_dir}")

        return trainer

    def _validate_datasets(self, train_dataset: Dataset, val_dataset: Optional[Dataset]):
        """
        Validate datasets before training.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)

        Raises:
            ValueError: If validation fails
        """
        # Check if training dataset is empty
        if len(train_dataset) == 0:
            raise ValueError(
                "Training dataset is empty! No training examples were generated. "
                "Check your repository path and include_extensions configuration. "
                "Make sure your codebase has files matching the configured extensions."
            )

        # Warn if training dataset is very small
        if len(train_dataset) < 10:
            logger.warning(
                f"Training dataset is very small ({len(train_dataset)} examples). "
                "This may not be enough for effective fine-tuning. "
                "Consider adding more code files or using synthetic data generation. "
                "Recommended minimum: 100 examples for basic fine-tuning."
            )

        # Warn if no validation dataset
        if val_dataset is None:
            logger.warning(
                "No validation dataset provided. "
                "Training will proceed without validation monitoring. "
                "Consider setting validation_split > 0 in your config."
            )
        elif len(val_dataset) == 0:
            logger.warning("Validation dataset is empty.")

        # Log dataset sizes
        logger.info(f"Dataset validation passed:")
        logger.info(f"  Training examples: {len(train_dataset)}")
        if val_dataset:
            logger.info(f"  Validation examples: {len(val_dataset)}")
            logger.info(f"  Split ratio: {len(train_dataset)/(len(train_dataset)+len(val_dataset)):.1%} train")

    def _prepare_dataset(self, dataset: Dataset) -> Dataset:
        """
        Prepare dataset for training by tokenizing.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Tokenized dataset
        """
        max_length = self.config.get('max_seq_length', 2048)
        
        def tokenize_function(examples):
            # Format the examples
            texts = []
            for i in range(len(examples['instruction'])):
                if examples.get('input') and examples['input'][i]:
                    text = (
                        f"### Instruction:\n{examples['instruction'][i]}\n\n"
                        f"### Input:\n{examples['input'][i]}\n\n"
                        f"### Response:\n{examples['output'][i]}"
                    )
                else:
                    text = (
                        f"### Instruction:\n{examples['instruction'][i]}\n\n"
                        f"### Response:\n{examples['output'][i]}"
                    )
                texts.append(text)
            
            # Tokenize
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors=None
            )
            
            # For causal LM, labels are the same as input_ids
            tokenized['labels'] = tokenized['input_ids'].copy()
            
            return tokenized
        
        logger.info("Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing"
        )
        
        return tokenized_dataset
    
    def _create_training_args(self) -> TrainingArguments:
        """
        Create training arguments.
        
        Returns:
            TrainingArguments object
        """
        output_dir = self.config.get('output_dir', './models/fine-tuned')
        
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.get('num_epochs', 3),
            per_device_train_batch_size=self.config.get('batch_size', 4),
            per_device_eval_batch_size=self.config.get('batch_size', 4),
            gradient_accumulation_steps=self.config.get('gradient_accumulation_steps', 4),
            learning_rate=self.config.get('learning_rate', 2e-5),
            warmup_steps=self.config.get('warmup_steps', 100),
            logging_steps=self.config.get('logging_steps', 10),
            save_steps=self.config.get('save_steps', 500),
            eval_strategy="steps" if self.config.get('eval_steps') else "no",
            eval_steps=self.config.get('eval_steps', 500),
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
            optim="paged_adamw_8bit",
            lr_scheduler_type="cosine",
            report_to="none",
        )
