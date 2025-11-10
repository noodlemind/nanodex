"""
Model loader for loading and configuring HuggingFace models.
"""

import logging
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

logger = logging.getLogger(__name__)


class ModelLoader:
    """Load and configure models for fine-tuning."""
    
    def __init__(self, model_config: dict, training_config: dict):
        """
        Initialize model loader.
        
        Args:
            model_config: Model configuration dictionary
            training_config: Training configuration dictionary
        """
        self.model_config = model_config
        self.training_config = training_config
        self.model_name = model_config.get('model_name')
        self.use_4bit = model_config.get('use_4bit', False)
        self.use_8bit = model_config.get('use_8bit', False)
    
    def load_huggingface_model(self):
        """
        Load a HuggingFace model with optional quantization.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading model: {self.model_name}")
        
        # Configure quantization if requested
        quantization_config = None
        if self.use_4bit:
            logger.info("Using 4-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif self.use_8bit:
            logger.info("Using 8-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Load model
        logger.info("Loading model weights...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16 if quantization_config else torch.float32,
        )
        
        # Prepare model for k-bit training if using quantization
        if self.use_4bit or self.use_8bit:
            logger.info("Preparing model for k-bit training...")
            model = prepare_model_for_kbit_training(model)
        
        # Enable gradient checkpointing for memory efficiency
        model.config.use_cache = False
        if hasattr(model, 'enable_input_require_grads'):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        
        logger.info("Model loaded successfully")
        return model, tokenizer
    
    def apply_lora(self, model):
        """
        Apply LoRA (Low-Rank Adaptation) to the model.
        
        Args:
            model: The model to apply LoRA to
            
        Returns:
            Model with LoRA applied
        """
        logger.info("Applying LoRA configuration...")
        
        lora_config_dict = self.training_config.get('lora', {})
        
        # Create LoRA configuration
        lora_config = LoraConfig(
            r=lora_config_dict.get('r', 16),
            lora_alpha=lora_config_dict.get('lora_alpha', 32),
            lora_dropout=lora_config_dict.get('lora_dropout', 0.05),
            target_modules=lora_config_dict.get('target_modules', [
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
            ]),
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_percent = 100 * trainable_params / total_params
        
        logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_percent:.2f}%)")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info("LoRA applied successfully")
        
        return model
