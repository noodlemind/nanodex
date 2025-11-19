#!/usr/bin/env python3
"""Local inference - run model directly without vLLM server."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def detect_device() -> str:
    """Detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    else:
        return "cpu"


def load_model(model_name: str, adapter_path: Path | None, device: str):
    """Load model and tokenizer."""
    logger.info(f"Loading model: {model_name}")
    logger.info(f"Device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32 if device == "mps" else torch.float16,
        low_cpu_mem_usage=True,
    )

    # Load LoRA adapter if provided
    if adapter_path and adapter_path.exists():
        logger.info(f"Loading LoRA adapter: {adapter_path}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_path))
        logger.info("LoRA adapter loaded")
    else:
        logger.info("Using base model (no adapter)")

    # Move to device
    model = model.to(device)
    model.eval()

    logger.info("Model loaded successfully")
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    question: str,
    system_prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: str = "cpu",
) -> str:
    """Generate response to question."""

    # Format prompt
    prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate
    logger.info("Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Local inference (no server required)")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-Coder-1.5B",
        help="Base model name",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        help="Path to LoRA adapter (optional)",
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Question to ask",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a helpful code assistant specialized in this codebase.",
        help="System prompt",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("NanoDex Local Inference")
    logger.info("=" * 60)

    # Detect device
    device = detect_device()
    logger.info(f"Using device: {device}")

    # Load model
    model, tokenizer = load_model(args.model, args.adapter, device)

    # Interactive mode
    if args.interactive:
        logger.info("\nInteractive mode - type 'exit' to quit")
        logger.info("=" * 60)

        while True:
            try:
                question = input("\nQuestion: ").strip()
                if question.lower() in ["exit", "quit", "q"]:
                    break

                if not question:
                    continue

                response = generate_response(
                    model,
                    tokenizer,
                    question,
                    args.system_prompt,
                    args.max_tokens,
                    args.temperature,
                    device=device,
                )

                print(f"\nAnswer: {response}\n")
                print("-" * 60)

            except KeyboardInterrupt:
                print("\nExiting...")
                break

    # Single question mode
    elif args.question:
        logger.info(f"\nQuestion: {args.question}")

        response = generate_response(
            model,
            tokenizer,
            args.question,
            args.system_prompt,
            args.max_tokens,
            args.temperature,
            device=device,
        )

        logger.info("=" * 60)
        print(f"\nAnswer: {response}\n")
        logger.info("=" * 60)

    else:
        logger.error("Please provide --question or use --interactive mode")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
