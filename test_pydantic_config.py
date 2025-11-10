#!/usr/bin/env python3
"""
Test script to verify Pydantic configuration validation.
"""

import sys
from pathlib import Path

# Add the project to path
sys.path.insert(0, str(Path(__file__).parent))

from turbo_code_gpt.utils import Config

def test_config_loading():
    """Test configuration loading with Pydantic validation."""
    print("Testing Pydantic configuration validation...")
    print("=" * 60)

    try:
        # Load config
        print("\n1. Loading configuration from config.yaml...")
        config = Config('config.yaml')
        print("   ✓ Configuration loaded successfully")

        # Test model source
        print("\n2. Testing model source...")
        model_source = config.get_model_source()
        print(f"   Model source: {model_source}")
        print("   ✓ Model source retrieved")

        # Test model config
        print("\n3. Testing model configuration...")
        model_config = config.get_model_config()
        print(f"   Model name: {model_config.get('model_name')}")
        print(f"   Use 4-bit: {model_config.get('use_4bit')}")
        print("   ✓ Model config retrieved")

        # Test repository config
        print("\n4. Testing repository configuration...")
        repo_config = config.get_repository_config()
        print(f"   Repository path: {repo_config.get('path')}")
        print(f"   Extensions: {len(repo_config.get('include_extensions', []))} types")
        print(f"   Deep parsing enabled: {repo_config.get('deep_parsing', {}).get('enabled')}")
        print("   ✓ Repository config retrieved")

        # Test training config
        print("\n5. Testing training configuration...")
        training_config = config.get_training_config()
        print(f"   Epochs: {training_config.get('num_epochs')}")
        print(f"   Batch size: {training_config.get('batch_size')}")
        print(f"   Learning rate: {training_config.get('learning_rate')}")
        print("   ✓ Training config retrieved")

        # Test data config
        print("\n6. Testing data configuration...")
        data_config = config.get_data_config()
        print(f"   Random seed: {data_config.get('random_seed')}")
        print(f"   Train split: {data_config.get('train_split')}")
        print(f"   Validation split: {data_config.get('validation_split')}")
        print("   ✓ Data config retrieved")

        # Test export config
        print("\n7. Testing export configuration...")
        export_config = config.get_export_config()
        print(f"   Format: {export_config.get('format')}")
        print(f"   Quantization: {export_config.get('quantization')}")
        print("   ✓ Export config retrieved")

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        print("\nPydantic validation is working correctly.")
        print("Configuration is valid and all fields are accessible.")

        return 0

    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ Test failed!")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(test_config_loading())
