"""
Main entry point for nanodex CLI.

This module makes nanodex runnable as a module:
    python -m nanodex

And provides the entry point for the installed command:
    nanodex
"""

import sys
from pathlib import Path

# Add the parent directory to the path to ensure imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the modern Click-based CLI
from nanodex.cli import cli


def main():
    """Main entry point for CLI."""
    return cli()


if __name__ == '__main__':
    sys.exit(main())
