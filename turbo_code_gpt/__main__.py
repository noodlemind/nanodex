"""
Main entry point for Turbo Code GPT CLI.

This module makes turbo_code_gpt runnable as a module:
    python -m turbo_code_gpt

And provides the entry point for the installed command:
    turbo-code-gpt
"""

import sys
from pathlib import Path

# Add the parent directory to the path to ensure imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the modern Click-based CLI
from turbo_code_gpt.cli import cli


def main():
    """Main entry point for CLI."""
    return cli()


if __name__ == '__main__':
    sys.exit(main())
