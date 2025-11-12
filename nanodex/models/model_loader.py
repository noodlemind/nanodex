"""
Model loader for loading and configuring HuggingFace models.
"""

import logging
from typing import Any, Tuple
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig, get_peft_model

# Optional bitsandbytes support for quantization
try:
    from transformers import BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training

    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False
    BitsAndBytesConfig = None
    prepare_model_for_kbit_training = None

logger = logging.getLogger(__name__)


class ModelLoader:
    """Load and configure models for fine-tuning."""

    def __init__(
        self,
        model_config: dict[str, Any],
        training_config: dict[str, Any],
        trust_remote_code: bool = False,
    ) -> None:
        """
        Initialize model loader.

        Args:
            model_config: Model configuration dictionary
            training_config: Training configuration dictionary
            trust_remote_code: Whether to trust remote code execution (default: False).
                              WARNING: Setting this to True allows arbitrary code execution
                              from model repositories, which is a security risk.

        Raises:
            ValueError: If model_name is missing or both use_4bit and use_8bit are True
        """
        self.model_config = model_config
        self.training_config = training_config
        self.model_name = model_config.get("model_name")
        self.use_4bit = model_config.get("use_4bit", False)
        self.use_8bit = model_config.get("use_8bit", False)
        self.trust_remote_code = trust_remote_code

        # Validate model_name
        if not self.model_name:
            raise ValueError(
                "model_name is required but was not found in model_config. "
                "Please specify a valid model name (e.g., 'deepseek-ai/deepseek-coder-6.7b-base')."
            )

        # Validate quantization settings
        if self.use_4bit and self.use_8bit:
            raise ValueError(
                "Both use_4bit and use_8bit are set to True. "
                "Please enable only one quantization method or neither."
            )

        # Security warning for trust_remote_code
        if self.trust_remote_code:
            logger.warning(
                "trust_remote_code is enabled. This allows arbitrary code execution "
                "from model repositories, which poses a security risk. Only use this "
                "with models from trusted sources."
            )

    def load_huggingface_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load a HuggingFace model with optional quantization.

        Returns:
            Tuple of (model, tokenizer)

        Raises:
            RuntimeError: If model/tokenizer loading fails
        """
        try:
            logger.info(f"Loading model: {self.model_name}")

            # Check if quantization is available
            if (self.use_4bit or self.use_8bit) and not HAS_BITSANDBYTES:
                logger.warning(
                    "Quantization requested but bitsandbytes is not installed. "
                    "Install with: pip install 'nanodex[gpu]' for GPU acceleration. "
                    "Falling back to fp16 precision."
                )
                self.use_4bit = False
                self.use_8bit = False

            # Configure quantization if requested and available
            #
            # QUANTIZATION: Reduces model precision to save memory
            # - 16-bit (fp16) → 4-bit = 75% memory reduction
            # - Trade-off: ~1% accuracy loss for massive memory savings
            # - Example: 7B model: 14GB (16-bit) → 3.5GB (4-bit)
            #
            # Why it works: Most model weights have redundant precision.
            # Neural networks are surprisingly robust to reduced precision!
            #
            quantization_config = None
            if self.use_4bit:
                logger.info("Using 4-bit quantization")
                # NF4 (NormalFloat4): Optimized 4-bit format for neural networks
                # Double quantization: Quantize the quantization constants (meta-quantization)
                # Result: Best compression with minimal quality loss
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,  # Use 4-bit precision for weights
                    bnb_4bit_compute_dtype=torch.float16,  # Compute activations in fp16
                    bnb_4bit_quant_type="nf4",  # Use NormalFloat4 quantization
                    bnb_4bit_use_double_quant=True,  # Quantize quantization constants too
                )
            elif self.use_8bit:
                logger.info("Using 8-bit quantization")
                # 8-bit quantization: Less aggressive than 4-bit
                # Better quality preservation, still 50% memory reduction
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )

            # Load tokenizer
            logger.info("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=self.trust_remote_code
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
                trust_remote_code=self.trust_remote_code,
                torch_dtype=torch.float16,  # Always use fp16 for efficiency
            )

            # Prepare model for k-bit training if using quantization
            #
            # K-bit training preparation:
            # - Enables gradient computation through quantized weights
            # - Casts layer norms to fp32 for stability
            # - Prepares input embeddings for backpropagation
            #
            if (self.use_4bit or self.use_8bit) and HAS_BITSANDBYTES:
                logger.info("Preparing model for k-bit training...")
                model = prepare_model_for_kbit_training(model)

            # Enable gradient checkpointing for memory efficiency
            #
            # GRADIENT CHECKPOINTING: Trade computation for memory
            # - Don't store all activations during forward pass
            # - Recompute them during backward pass when needed
            # - Result: 2x slower training, but uses ~50% less memory
            # - Essential for training large models on consumer GPUs
            #
            model.config.use_cache = False  # Disable KV-cache (incompatible with training)
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:
                # Fallback for older transformers versions
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

            logger.info("Model loaded successfully")
            return model, tokenizer

        except Exception as e:
            logger.error(f"Failed to load model '{self.model_name}': {e}", exc_info=True)
            raise RuntimeError(
                f"Failed to load model '{self.model_name}'. "
                f"This could be due to network issues, authentication problems, "
                f"insufficient memory, or an invalid model name. "
                f"Original error: {e}"
            ) from e

    def apply_lora(self, model: AutoModelForCausalLM) -> AutoModelForCausalLM:
        """
        Apply LoRA (Low-Rank Adaptation) to the model.

        This method applies LoRA adapters to specific modules in the model for efficient
        fine-tuning. The target_modules parameter specifies which model components to adapt.

        Common target_modules for transformer architectures:
        - Standard attention: ["q_proj", "v_proj", "k_proj", "o_proj"]
        - All linear layers: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        - Specific to model architecture (check model documentation)

        Args:
            model: The model to apply LoRA to

        Returns:
            Model with LoRA applied

        Raises:
            RuntimeError: If LoRA application fails or no modules are targeted
        """
        logger.info("Applying LoRA configuration...")

        #
        # LORA (Low-Rank Adaptation): Train only 0.1% of parameters
        #
        # How it works:
        # - Instead of updating weight matrix W directly
        # - Decompose update into: W + B×A (two small matrices)
        # - Train only B and A, keep W frozen
        #
        # Math example for 7B model:
        # - Full training: 6.7 billion parameters
        # - LoRA with r=16: ~4 million trainable parameters
        # - That's 0.06% of the full model!
        #
        # Result:
        # - 6.7B params → 4M trainable (~50MB saved model vs 13GB full model)
        # - Much faster training (only updating 0.1% of weights)
        # - Lower memory requirements (no optimizer states for frozen weights)
        # - Can train on consumer GPUs instead of requiring datacenter hardware
        #
        lora_config_dict = self.training_config.get("lora", {})

        # Create LoRA configuration
        target_modules = lora_config_dict.get(
            "target_modules",
            [
                "q_proj",  # Query projection in attention
                "v_proj",  # Value projection in attention
                "k_proj",  # Key projection in attention
                "o_proj",  # Output projection in attention
            ],
        )

        logger.info(f"Target modules for LoRA: {target_modules}")

        lora_config = LoraConfig(
            r=lora_config_dict.get("r", 16),  # Rank: controls adapter size (higher = more capacity)
            lora_alpha=lora_config_dict.get("lora_alpha", 32),  # Scaling factor (typically 2× rank)
            lora_dropout=lora_config_dict.get("lora_dropout", 0.05),  # Dropout for regularization
            target_modules=target_modules,  # Which model layers to adapt
            bias="none",  # Don't train bias terms
            task_type="CAUSAL_LM",  # Causal language modeling (predict next token)
        )

        # Apply LoRA with error handling
        try:
            model = get_peft_model(model, lora_config)
        except Exception as e:
            logger.error(
                f"Failed to apply LoRA with target_modules={target_modules}: {e}", exc_info=True
            )
            raise RuntimeError(
                f"Failed to apply LoRA adapters. This may be due to invalid target_modules "
                f"that don't exist in the model architecture. Target modules: {target_modules}. "
                f"Original error: {e}"
            ) from e

        # Inspect which modules were actually wrapped
        lora_modules = []
        for name, module in model.named_modules():
            if hasattr(module, "lora_A") or "lora" in name.lower():
                lora_modules.append(name)

        if not lora_modules:
            error_msg = (
                f"No LoRA modules were applied to the model. "
                f"The specified target_modules {target_modules} may not match "
                f"any modules in the model architecture. Please check the model "
                f"structure and update target_modules accordingly."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        logger.info(
            f"LoRA applied to {len(lora_modules)} module(s): {lora_modules[:5]}{'...' if len(lora_modules) > 5 else ''}"
        )

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_percent = 100 * trainable_params / total_params

        logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_percent:.2f}%)")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info("LoRA applied successfully")

        return model
