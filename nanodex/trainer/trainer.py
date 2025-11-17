"""LoRA/QLoRA trainer implementation."""

import logging
from pathlib import Path
from typing import Any, Optional

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from nanodex.config import TrainingConfig
from nanodex.trainer.data_loader import InstructionDataset, create_dataloaders

logger = logging.getLogger(__name__)


class LoRATrainer:
    """Trainer for LoRA/QLoRA fine-tuning."""

    def __init__(self, config: TrainingConfig):
        """
        Initialize LoRA trainer.

        Args:
            config: Training configuration
        """
        self.config = config
        self.model: Any = None
        self.tokenizer: Any = None
        self.dataset: Any = None

    def setup(self) -> None:
        """Setup model, tokenizer, and datasets."""
        logger.info("Setting up trainer...")

        # Load tokenizer
        self._load_tokenizer()

        # Load model with quantization (if configured)
        self._load_model()

        # Prepare model for LoRA
        self._setup_lora()

        # Load dataset
        self._load_dataset()

        logger.info("Trainer setup complete")

    def _load_tokenizer(self) -> None:
        """Load and configure tokenizer."""
        logger.info(f"Loading tokenizer: {self.config.base_model}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
        )

        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Set model max length
        self.tokenizer.model_max_length = self.config.model_max_length

        logger.info(f"Tokenizer loaded: {len(self.tokenizer)} tokens")

    def _load_model(self) -> None:
        """Load base model with optional quantization."""
        logger.info(f"Loading model: {self.config.base_model}")

        # Configure quantization for QLoRA
        bnb_config = None
        if self.config.quantization:
            logger.info("Configuring 4-bit quantization")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=self.config.quantization.load_in_4bit,
                bnb_4bit_quant_type=self.config.quantization.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=self._get_compute_dtype(
                    self.config.quantization.bnb_4bit_compute_dtype
                ),
                bnb_4bit_use_double_quant=self.config.quantization.bnb_4bit_use_double_quant,
            )

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=self._get_compute_dtype(
                self.config.quantization.bnb_4bit_compute_dtype
                if self.config.quantization
                else "bfloat16"
            ),
        )

        # Prepare model for k-bit training (for QLoRA)
        if self.config.quantization:
            self.model = prepare_model_for_kbit_training(self.model)

        logger.info(f"Model loaded: {self.model.config.model_type}")

    def _get_compute_dtype(self, dtype_str: str) -> torch.dtype:
        """Convert dtype string to torch dtype."""
        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        return dtype_map.get(dtype_str, torch.bfloat16)

    def _setup_lora(self) -> None:
        """Configure and apply LoRA to model."""
        logger.info("Setting up LoRA...")

        # Create LoRA config
        lora_config = LoraConfig(
            r=self.config.lora.r,
            lora_alpha=self.config.lora.lora_alpha,
            target_modules=self.config.lora.target_modules,
            lora_dropout=self.config.lora.lora_dropout,
            bias=self.config.lora.bias,
            task_type=self.config.lora.task_type,
        )

        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)

        # Print trainable parameters
        self.model.print_trainable_parameters()

    def _load_dataset(self) -> None:
        """Load and prepare dataset."""
        logger.info(f"Loading dataset: {self.config.dataset_path}")

        self.dataset = InstructionDataset(
            dataset_path=Path(self.config.dataset_path),
            tokenizer=self.tokenizer,
            max_length=self.config.model_max_length,
            validation_split=self.config.validation_split,
        )

        logger.info("Dataset loaded")

    def train(self) -> None:
        """Run training loop."""
        if self.model is None or self.tokenizer is None or self.dataset is None:
            raise RuntimeError("Trainer not setup. Call setup() first.")

        logger.info("Starting training...")

        # Create training arguments
        training_args = TrainingArguments(
            output_dir=self.config.training.output_dir,
            num_train_epochs=self.config.training.num_train_epochs,
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.training.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            warmup_ratio=self.config.training.warmup_ratio,
            lr_scheduler_type=self.config.training.lr_scheduler_type,
            logging_steps=self.config.training.logging_steps,
            save_steps=self.config.training.save_steps,
            eval_steps=self.config.training.eval_steps,
            save_total_limit=self.config.training.save_total_limit,
            fp16=self.config.training.fp16,
            bf16=self.config.training.bf16,
            max_grad_norm=self.config.training.max_grad_norm,
            optim=self.config.training.optim,
            group_by_length=self.config.training.group_by_length,
            report_to=self.config.training.report_to,
            evaluation_strategy="steps" if self.dataset.get_val_dataset() else "no",
            load_best_model_at_end=True if self.dataset.get_val_dataset() else False,
            metric_for_best_model="eval_loss" if self.dataset.get_val_dataset() else None,
        )

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset.get_train_dataset(),
            eval_dataset=self.dataset.get_val_dataset(),
            tokenizer=self.tokenizer,
        )

        # Train
        logger.info("Training started...")
        trainer.train()

        logger.info("Training complete!")

    def save(self, output_dir: Optional[Path] = None) -> None:
        """
        Save trained adapter.

        Args:
            output_dir: Output directory (defaults to config output_dir)
        """
        if self.model is None:
            raise RuntimeError("No model to save")

        save_dir = output_dir or Path(self.config.training.output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving adapter to {save_dir}")

        # Save LoRA adapter
        self.model.save_pretrained(save_dir)

        # Save tokenizer
        if self.tokenizer:
            self.tokenizer.save_pretrained(save_dir)

        logger.info(f"Adapter saved to {save_dir}")
