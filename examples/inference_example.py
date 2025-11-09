"""
Example script for using the fine-tuned model for inference.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse


def load_fine_tuned_model(model_path: str):
    """
    Load the fine-tuned model.
    
    Args:
        model_path: Path to the fine-tuned model
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    return model, tokenizer


def generate_response(model, tokenizer, instruction: str, input_text: str = "", max_length: int = 512):
    """
    Generate a response from the model.
    
    Args:
        model: The model
        tokenizer: The tokenizer
        instruction: Instruction for the model
        input_text: Optional input context
        max_length: Maximum length of generated text
        
    Returns:
        Generated response
    """
    # Format the prompt
    if input_text:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n"
        )
    else:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n"
        )
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the response part
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
    
    return response


def main():
    """Main function for inference example."""
    parser = argparse.ArgumentParser(description='Test fine-tuned model')
    parser.add_argument(
        '--model-path',
        default='./models/fine-tuned',
        help='Path to fine-tuned model'
    )
    parser.add_argument(
        '--instruction',
        default='Explain what this codebase does',
        help='Instruction for the model'
    )
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_fine_tuned_model(args.model_path)
    
    print("\nModel loaded successfully!")
    print("\nExample queries:\n")
    
    # Example 1: General question about codebase
    print("=" * 60)
    print("Question: What does this codebase do?")
    print("=" * 60)
    response = generate_response(
        model, tokenizer,
        "Explain what this codebase does and its main purpose."
    )
    print(response)
    
    # Example 2: Ask about specific functionality
    print("\n" + "=" * 60)
    print("Question: What are the main components?")
    print("=" * 60)
    response = generate_response(
        model, tokenizer,
        "List and describe the main components of this codebase."
    )
    print(response)
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("Interactive Mode (type 'quit' to exit)")
    print("=" * 60)
    
    while True:
        instruction = input("\nYour question: ")
        if instruction.lower() in ['quit', 'exit', 'q']:
            break
        
        response = generate_response(model, tokenizer, instruction)
        print(f"\nResponse: {response}")


if __name__ == '__main__':
    main()
