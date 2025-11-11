#!/usr/bin/env python3
"""
Validation script to check the project structure and basic imports.
This can run without installing heavy dependencies.
"""

import sys
from pathlib import Path

def check_structure():
    """Check that all expected files and directories exist."""
    print("Checking project structure...")
    
    required_files = [
        "main.py",
        "config.yaml",
        "requirements.txt",
        "setup.py",
        "README.md",
        "docs/README.md",
        "docs/getting-started.md",
        "LICENSE",
        "nanodex/__init__.py",
        "nanodex/utils/__init__.py",
        "nanodex/utils/config.py",
        "nanodex/analyzers/__init__.py",
        "nanodex/analyzers/code_analyzer.py",
        "nanodex/models/__init__.py",
        "nanodex/models/model_loader.py",
        "nanodex/trainers/__init__.py",
        "nanodex/trainers/data_preparer.py",
        "nanodex/trainers/model_trainer.py",
        "examples/__init__.py",
        "examples/inference_example.py",
        "examples/ollama_example.py",
        "tests/__init__.py",
        "tests/test_basic.py",
    ]
    
    missing = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing.append(file_path)
    
    if missing:
        print("❌ Missing files:")
        for f in missing:
            print(f"  - {f}")
        return False
    else:
        print("✅ All required files exist")
        return True


def check_config():
    """Check that config.yaml is valid."""
    print("\nChecking configuration...")
    
    try:
        import yaml
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        required_keys = ['model_source', 'model', 'repository', 'training', 'data', 'export']
        missing = [k for k in required_keys if k not in config]
        
        if missing:
            print(f"❌ Missing config keys: {missing}")
            return False
        else:
            print("✅ Configuration is valid")
            print(f"   Model source: {config['model_source']}")
            return True
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return False


def check_imports():
    """Check that basic imports work."""
    print("\nChecking basic imports (without heavy dependencies)...")
    
    try:
        # These should work without installing dependencies
        import nanodex
        from nanodex.utils import Config
        from nanodex.analyzers import CodeAnalyzer
        
        print("✅ Basic imports successful")
        print(f"   Package version: {nanodex.__version__}")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False


def check_readme():
    """Check that README has proper content."""
    print("\nChecking documentation...")
    
    try:
        with open('README.md', 'r') as f:
            content = f.read()
        
        required_sections = [
            'Features',
            'Quick Start',
            'Documentation',
        ]
        
        missing = [s for s in required_sections if s not in content]
        
        if missing:
            print(f"❌ Missing README sections: {missing}")
            return False
        else:
            print("✅ README is comprehensive")
            return True
    except Exception as e:
        print(f"❌ Error reading README: {e}")
        return False


def main():
    """Run all validation checks."""
    print("=" * 60)
    print("nanodex - Project Validation")
    print("=" * 60)
    
    checks = [
        check_structure(),
        check_config(),
        check_imports(),
        check_readme(),
    ]
    
    print("\n" + "=" * 60)
    if all(checks):
        print("✅ All validation checks passed!")
        print("\nProject is ready to use.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Configure your project: edit config.yaml")
        print("3. Run the analyzer: python main.py --analyze-only")
        print("=" * 60)
        return 0
    else:
        print("❌ Some validation checks failed")
        print("=" * 60)
        return 1


if __name__ == '__main__':
    sys.exit(main())
