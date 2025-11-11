#!/usr/bin/env python3
"""
Demo script showing how to use nanodex on a small example codebase.
This creates a minimal example and runs through the analysis phase.

Note: This demo only shows the analysis phase. Full training requires
installing all dependencies: pip install -r requirements.txt
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


def create_sample_codebase(base_dir: Path):
    """Create a sample codebase for demonstration."""
    print("Creating sample codebase...")
    
    # Create a simple Python module
    (base_dir / "utils.py").write_text("""
\"\"\"Utility functions for data processing.\"\"\"

def clean_data(data):
    \"\"\"Clean and normalize data.\"\"\"
    return [item.strip().lower() for item in data if item]

def validate_email(email):
    \"\"\"Simple email validation.\"\"\"
    return '@' in email and '.' in email
""")
    
    # Create a main application file
    (base_dir / "app.py").write_text("""
\"\"\"Main application entry point.\"\"\"

from utils import clean_data, validate_email

def process_user_input(users):
    \"\"\"Process user input data.\"\"\"
    cleaned = clean_data(users)
    valid_emails = [u for u in cleaned if validate_email(u)]
    return valid_emails

if __name__ == '__main__':
    users = ['user@example.com', 'invalid', 'another@test.org']
    result = process_user_input(users)
    print(f"Valid emails: {result}")
""")
    
    # Create a JavaScript file
    (base_dir / "helpers.js").write_text("""
/**
 * Helper functions for the frontend.
 */

function formatDate(date) {
    return new Date(date).toLocaleDateString();
}

function capitalizeString(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}

module.exports = { formatDate, capitalizeString };
""")
    
    print(f"✅ Created sample codebase in {base_dir}")


def run_demo():
    """Run the demonstration."""
    print("=" * 70)
    print("nanodex - Demo")
    print("=" * 70)
    print()
    print("NOTE: This demo shows code analysis only.")
    print("For full training, install: pip install -r requirements.txt")
    print()
    
    # Create temporary directory for demo
    temp_dir = tempfile.mkdtemp(prefix="nanodex_demo_")
    
    try:
        # Create sample codebase
        create_sample_codebase(Path(temp_dir))
        
        # Configure analyzer
        print("\n" + "-" * 70)
        print("Step 1: Analyzing Code Repository")
        print("-" * 70)
        
        config = {
            'path': temp_dir,
            'include_extensions': ['.py', '.js', '.ts'],
            'exclude_dirs': ['node_modules', '__pycache__'],
            'max_file_size': 1048576
        }
        
        analyzer = CodeAnalyzer(config)
        code_samples = analyzer.analyze()
        
        print(f"\n📊 Analysis Results:")
        print(f"   Found {len(code_samples)} code files")
        
        # Show statistics
        stats = analyzer.get_statistics(code_samples)
        print(f"\n   Total lines of code: {stats['total_lines']}")
        print(f"\n   Languages:")
        for lang, lang_stats in stats['languages'].items():
            print(f"      - {lang}: {lang_stats['files']} files, {lang_stats['lines']} lines")
        
        # Show sample files
        print(f"\n   Files analyzed:")
        for sample in code_samples:
            print(f"      - {sample['file_path']} ({sample['language']}, {sample['lines']} lines)")
        
        # Show sample content
        print(f"\n   Sample file content (app.py):")
        for sample in code_samples:
            if sample['file_path'] == 'app.py':
                lines = sample['content'].split('\n')[:10]
                for i, line in enumerate(lines, 1):
                    print(f"      {i:2}: {line}")
                break
        
        # Summary
        print("\n" + "=" * 70)
        print("✅ Demo Complete!")
        print("=" * 70)
        print("\nWhat was demonstrated:")
        print("  1. ✅ Code repository analysis")
        print("  2. ✅ Multi-language support (Python, JavaScript)")
        print("  3. ✅ File discovery and filtering")
        print("  4. ✅ Statistics generation")
        
        print("\nNext steps for full training:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Point to your codebase in config.yaml")
        print("  3. Run: python main.py")
        print("  4. Wait for training to complete")
        print("  5. Use the fine-tuned model for your chatbot")
        
        print("\n" + "=" * 70)
        
    finally:
        # Cleanup
        print(f"\nCleaning up temporary files...")
        shutil.rmtree(temp_dir)
        print("✅ Cleanup complete")


if __name__ == '__main__':
    try:
        run_demo()
    except Exception as e:
        print(f"\n❌ Error running demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
