"""
Model trainer for fine-tuning code models with enhanced features.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, ClassVar
import torch
import json
from datetime import datetime
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    EarlyStoppingCallback,
)
from datasets import Dataset

# Import bitsandbytes availability flag for optimizer selection
try:
    from ..models.model_loader import HAS_BITSANDBYTES
except ImportError:
    HAS_BITSANDBYTES = False

logger = logging.getLogger(__name__)


class ProgressCallback(TrainerCallback):
    """Custom callback for enhanced progress tracking."""

    def __init__(self):
        super().__init__()
        self.start_time = None
        self.best_loss = float("inf")

    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training."""
        self.start_time = datetime.now()
        logger.info(f"Training started at {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called after logging."""
        if logs:
            # Log training progress
            if "loss" in logs:
                current_loss = logs["loss"]
                if current_loss < self.best_loss:
                    self.best_loss = current_loss
                    logger.info(f"✓ New best training loss: {current_loss:.4f}")

            if "eval_loss" in logs:
                logger.info(f"Evaluation loss: {logs['eval_loss']:.4f}")

    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        if self.start_time:
            duration = datetime.now() - self.start_time
            logger.info(f"Training completed in {duration}")
            logger.info(f"Best training loss: {self.best_loss:.4f}")


class TipCallback(TrainerCallback):
    """
    Educational callback that shows tips during training.

    Shows helpful information during long training runs to help users
    understand what's happening and learn about the training process.
    """

    # Educational tips shown during training
    TIPS: ClassVar[list] = [
        "💡 LoRA trains only 0.1% of parameters - that's why it's so fast!",
        "💡 Loss measures prediction error. Good: <1.0, Great: <0.5",
        "💡 Your fine-tuned model is just ~50MB of LoRA adapters",
        "💡 4-bit quantization reduces memory by 75% with minimal accuracy loss",
        "💡 Gradient accumulation lets you use larger effective batch sizes",
        "💡 Learning rate warmup prevents unstable training at the start",
        "💡 The model learns patterns from your codebase, not memorizes it",
        "💡 Lower training loss = better fit to your code's patterns",
        "💡 Validation loss shows generalization to unseen code samples",
        "💡 Save checkpoints frequently - training can be resumed if interrupted",
    ]

    def __init__(self):
        super().__init__()
        self.tip_index = 0
        self.last_tip_step = 0

    def on_log(self, _args, state, _control, _logs=None, **_kwargs):
        """
        Show educational tip every N steps during training.

        Tips are shown periodically to educate without overwhelming the log output.
        """
        # Show tip every 50 steps (configurable)
        if (
            state.global_step > 0
            and state.global_step % 50 == 0
            and state.global_step != self.last_tip_step
        ):
            tip = self.TIPS[self.tip_index % len(self.TIPS)]
            logger.info(f"\n{tip}\n")
            self.tip_index += 1
            self.last_tip_step = state.global_step


class ModelTrainer:
    """Enhanced model trainer with checkpoint recovery and early stopping."""

    def __init__(self, model, tokenizer, config: dict):
        """
        Initialize model trainer.

        Args:
            model: The model to train
            tokenizer: The tokenizer
            config: Training configuration dictionary with enhanced options:
                - enable_early_stopping: Enable early stopping (default: False)
                - early_stopping_patience: Patience for early stopping (default: 3)
                - early_stopping_threshold: Threshold for early stopping (default: 0.0)
                - save_best_model: Save best model based on eval loss (default: True)
                - resume_from_checkpoint: Path to checkpoint to resume from (default: None)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.training_metadata = {}

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        resume_from_checkpoint: Optional[str] = None,
    ):
        """
        Train the model on the provided datasets with enhanced features.

        Args:
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
            resume_from_checkpoint: Optional path to checkpoint to resume from

        Returns:
            Trainer object with training history

        Raises:
            ValueError: If datasets are invalid or empty
        """
        logger.info("=" * 60)
        logger.info("Starting Enhanced Model Training")
        logger.info("=" * 60)

        # Validate datasets before expensive operations
        self._validate_datasets(train_dataset, val_dataset)

        # Prepare datasets
        logger.info("Preparing datasets...")
        train_dataset = self._prepare_dataset(train_dataset)
        if val_dataset:
            val_dataset = self._prepare_dataset(val_dataset)

        # Create training arguments
        training_args = self._create_training_args()

        # Create data collator
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        # Create callbacks
        #
        # CALLBACKS: Monitor and enhance training
        # - ProgressCallback: Track training progress and best metrics
        # - TipCallback: Show educational tips during training
        # - EarlyStoppingCallback: Stop if validation loss stops improving (optional)
        #
        callbacks = [ProgressCallback(), TipCallback()]

        # Add early stopping if enabled
        if self.config.get("enable_early_stopping", False) and val_dataset:
            patience = self.config.get("early_stopping_patience", 3)
            threshold = self.config.get("early_stopping_threshold", 0.0)
            logger.info(f"Early stopping enabled: patience={patience}, threshold={threshold}")
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=patience, early_stopping_threshold=threshold
                )
            )

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            callbacks=callbacks,
        )

        # Train with checkpoint recovery
        logger.info("Beginning training...")

        # Check for checkpoint to resume from
        checkpoint_path = resume_from_checkpoint or self.config.get("resume_from_checkpoint")
        if checkpoint_path:
            checkpoint_path = str(Path(checkpoint_path).resolve())
            if Path(checkpoint_path).exists():
                logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
            else:
                logger.warning(f"Checkpoint not found: {checkpoint_path}. Starting from scratch.")
                checkpoint_path = None

        # Train
        train_result = trainer.train(resume_from_checkpoint=checkpoint_path)

        # Save training metadata
        self._save_training_metadata(trainer, train_result)

        # Save model
        output_dir = Path(self.config.get("output_dir", "./models/fine-tuned"))
        logger.info(f"Saving model to {output_dir}")
        trainer.save_model(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))

        # Save training history
        self._save_training_history(trainer, output_dir)

        logger.info("=" * 60)
        logger.info(f"Training completed! Model saved to {output_dir}")
        logger.info("=" * 60)

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
            logger.info(
                f"  Split ratio: {len(train_dataset)/(len(train_dataset)+len(val_dataset)):.1%} train"
            )

    def _prepare_dataset(self, dataset: Dataset) -> Dataset:
        """
        Prepare dataset for training by tokenizing.

        Tokenization converts text into numbers (tokens) that the model can process.
        We use a structured format with Instruction/Input/Response sections to teach
        the model how to understand and respond to code-related questions.

        Args:
            dataset: Input dataset

        Returns:
            Tokenized dataset
        """
        max_length = self.config.get("max_seq_length", 2048)

        def tokenize_function(examples):
            #
            # TOKENIZATION: Convert text to numbers
            # - Each word/subword gets a unique number (token ID)
            # - Example: "def hello()" → [1234, 5678, 90, 91]
            # - Model learns patterns in these number sequences
            #
            # FORMAT: Instruction-Input-Response
            # This structured format helps the model learn:
            # 1. What task to perform (Instruction)
            # 2. What code to analyze (Input)
            # 3. What output to generate (Response)
            #
            texts = []
            for i in range(len(examples["instruction"])):
                if examples.get("input") and examples["input"][i]:
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
                padding="max_length",
                return_tensors=None,
            )

            # For causal LM, labels are the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].copy()

            return tokenized

        logger.info("Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            tokenize_function, batched=True, remove_columns=dataset.column_names, desc="Tokenizing"
        )

        return tokenized_dataset

    def _create_training_args(self) -> TrainingArguments:
        """
        Create enhanced training arguments.

        Returns:
            TrainingArguments object with best practices
        """
        output_dir = self.config.get("output_dir", "./models/fine-tuned")

        # Determine evaluation strategy
        eval_strategy = "no"
        eval_steps = None
        if self.config.get("enable_evaluation", True):
            eval_strategy = "steps"
            eval_steps = self.config.get("eval_steps", 500)

        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.get("num_epochs", 3),
            per_device_train_batch_size=self.config.get("batch_size", 4),
            per_device_eval_batch_size=self.config.get("batch_size", 4),
            gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 4),
            learning_rate=self.config.get("learning_rate", 2e-5),
            warmup_steps=self.config.get("warmup_steps", 100),
            logging_steps=self.config.get("logging_steps", 10),
            save_steps=self.config.get("save_steps", 500),
            eval_strategy=eval_strategy,
            eval_steps=eval_steps,
            save_total_limit=self.config.get("save_total_limit", 3),
            load_best_model_at_end=self.config.get("save_best_model", True)
            and eval_strategy != "no",
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=torch.cuda.is_available(),
            optim=(
                "paged_adamw_8bit"
                if (HAS_BITSANDBYTES and torch.cuda.is_available())
                else "adamw_torch"
            ),
            lr_scheduler_type=self.config.get("lr_scheduler_type", "cosine"),
            report_to="none",
            logging_dir=f"{output_dir}/logs",
            save_safetensors=True,
            dataloader_num_workers=self.config.get("num_workers", 0),
        )

    def _save_training_metadata(self, trainer: Trainer, train_result):
        """
        Save training metadata for reproducibility.

        Args:
            trainer: Trainer object
            train_result: Training result object
        """
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
            "train_samples": train_result.metrics.get("train_samples", 0),
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "train_steps_per_second": train_result.metrics.get("train_steps_per_second", 0),
            "total_flos": train_result.metrics.get("total_flos", 0),
            "train_loss": train_result.metrics.get("train_loss", 0),
            "epoch": train_result.metrics.get("epoch", 0),
        }

        # Save to output directory
        output_dir = Path(self.config.get("output_dir", "./models/fine-tuned"))
        metadata_file = output_dir / "training_metadata.json"

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Training metadata saved to {metadata_file}")

    def _save_training_history(self, trainer: Trainer, output_dir: Path):
        """
        Save training history (loss, eval metrics over time).

        Args:
            trainer: Trainer object
            output_dir: Output directory
        """
        if hasattr(trainer.state, "log_history"):
            history_file = output_dir / "training_history.json"
            with open(history_file, "w") as f:
                json.dump(trainer.state.log_history, f, indent=2)
            logger.info(f"Training history saved to {history_file}")

    def get_latest_checkpoint(self, output_dir: Optional[str] = None) -> Optional[str]:
        """
        Get the path to the latest checkpoint.

        Args:
            output_dir: Output directory (default: from config)

        Returns:
            Path to latest checkpoint or None
        """
        if output_dir is None:
            output_dir = self.config.get("output_dir", "./models/fine-tuned")

        output_path = Path(output_dir)

        if not output_path.exists():
            return None

        # Find checkpoint directories
        checkpoints = list(output_path.glob("checkpoint-*"))

        if not checkpoints:
            return None

        # Sort by step number
        checkpoints.sort(key=lambda x: int(x.name.split("-")[1]))

        latest = checkpoints[-1]
        logger.info(f"Found latest checkpoint: {latest}")

        return str(latest)
