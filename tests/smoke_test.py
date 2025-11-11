#!/usr/bin/env python3
"""
Quick smoke test to verify the project structure works.
This doesn't require heavy ML dependencies.
"""

import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nanodex.utils import Config
from nanodex.analyzers import CodeAnalyzer


def test_analyzer():
    """Test the code analyzer."""
    print("Testing Code Analyzer...")
    
    # Create temporary directory with sample files
    temp_dir = tempfile.mkdtemp(prefix="test_")
    
    try:
        # Create test files
        (Path(temp_dir) / "test.py").write_text("def hello():\n    print('Hello')")
        (Path(temp_dir) / "test.js").write_text("function hello() { console.log('Hello'); }")
        
        # Configure and run analyzer
        config = {
            'path': temp_dir,
            'include_extensions': ['.py', '.js'],
            'exclude_dirs': [],
            'max_file_size': 1048576
        }
        
        analyzer = CodeAnalyzer(config)
        code_samples = analyzer.analyze()
        
        assert len(code_samples) == 2, f"Expected 2 files, got {len(code_samples)}"
        
        stats = analyzer.get_statistics(code_samples)
        assert stats['total_files'] == 2
        assert 'python' in stats['languages']
        assert 'javascript' in stats['languages']
        
        print("  ✅ Code analyzer works correctly")
        return True
        
    finally:
        shutil.rmtree(temp_dir)


def test_config():
    """Test configuration loading."""
    print("Testing Configuration...")
    
    temp_dir = tempfile.mkdtemp(prefix="test_")
    
    try:
        import yaml
        
        config_path = Path(temp_dir) / "test_config.yaml"
        config_data = {
            'model_source': 'huggingface',
            'model': {'huggingface': {'model_name': 'test'}},
            'repository': {'path': '.'}
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        config = Config(str(config_path))
        assert config.get_model_source() == 'huggingface'
        assert config.get_model_config()['model_name'] == 'test'
        
        print("  ✅ Configuration system works correctly")
        return True
        
    finally:
        shutil.rmtree(temp_dir)


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("nanodex - Smoke Test")
    print("=" * 60)
    print()
    
    tests = [
        test_config,
        test_analyzer,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  ❌ {test.__name__} failed: {e}")
            failed += 1
    
    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n✅ All smoke tests passed!")
        print("\nThe core functionality is working.")
        print("Install full dependencies to use training features:")
        print("  pip install -r requirements.txt")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
