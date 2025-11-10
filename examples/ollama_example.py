"""
Example script for using Ollama models.
"""

import requests
import json


class OllamaClient:
    """Client for interacting with Ollama models."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama client.
        
        Args:
            base_url: Base URL for Ollama API
        """
        self.base_url = base_url
    
    def list_models(self):
        """List available models in Ollama."""
        response = requests.get(f"{self.base_url}/api/tags")
        return response.json()
    
    def create_modelfile(self, base_model: str, training_data_path: str) -> str:
        """
        Create a Modelfile for fine-tuning.
        
        Args:
            base_model: Base model name (e.g., 'deepseek-coder:6.7b')
            training_data_path: Path to training data
            
        Returns:
            Modelfile content
        """
        modelfile = f"""FROM {base_model}

# Set the temperature to control creativity
PARAMETER temperature 0.7

# Set context window
PARAMETER num_ctx 4096

# System message for code understanding
SYSTEM You are an expert on this codebase. You have been trained on the complete source code and can answer questions about it, explain functionality, and help developers understand the code structure.
"""
        return modelfile
    
    def generate(self, model: str, prompt: str, stream: bool = False):
        """
        Generate response from model.
        
        Args:
            model: Model name
            prompt: Input prompt
            stream: Whether to stream the response
            
        Returns:
            Model response
        """
        data = {
            "model": model,
            "prompt": prompt,
            "stream": stream
        }
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=data,
            stream=stream
        )
        
        if stream:
            for line in response.iter_lines():
                if line:
                    yield json.loads(line)
        else:
            return response.json()


def main():
    """Example usage of Ollama client."""
    client = OllamaClient()
    
    # List available models
    print("Available models:")
    models = client.list_models()
    for model in models.get('models', []):
        print(f"  - {model['name']}")
    
    # Create a Modelfile
    modelfile = client.create_modelfile(
        base_model="deepseek-coder:6.7b",
        training_data_path="./data/processed/train.json"
    )
    
    print("\nGenerated Modelfile:")
    print(modelfile)
    
    print("\nTo create a custom model with Ollama:")
    print("1. Save the Modelfile to a file (e.g., 'Modelfile')")
    print("2. Run: ollama create my-code-expert -f Modelfile")
    print("3. Use the model: ollama run my-code-expert")


if __name__ == '__main__':
    main()
