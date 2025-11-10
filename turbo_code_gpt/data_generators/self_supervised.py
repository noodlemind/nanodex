"""
Self-supervised data generation from code (zero API cost).

Generates training examples from code structure without requiring external APIs.
"""

from typing import List, Dict, Optional
import random
import re
import logging
from pathlib import Path

from .base import DataGeneratorPlugin

logger = logging.getLogger(__name__)


class DocstringGenerator(DataGeneratorPlugin):
    """
    Generate training pairs from docstrings and their associated code.

    Creates instruction-following examples like:
    - Docstring -> Function implementation
    - Function signature + docstring -> Body
    - Class docstring -> Class structure
    """

    def generate(self, code_sample: Dict) -> List[Dict]:
        """
        Generate docstring-to-code training examples.

        Args:
            code_sample: Code sample with parsed structure (functions, classes, docstrings)

        Returns:
            List of training examples
        """
        examples = []
        parsed = code_sample.get('parsed', {})

        # Generate from functions
        functions = parsed.get('functions', [])
        for func in functions:
            docstring = func.get('docstring')
            if not docstring or len(docstring.strip()) < 10:
                continue

            # Example 1: Docstring -> Full function
            examples.append({
                'instruction': 'Implement the following function based on its docstring:',
                'input': f'```python\n{docstring}\n```',
                'output': func.get('body', ''),
                'metadata': {
                    'type': 'docstring_to_function',
                    'function_name': func.get('name'),
                    'file_path': code_sample.get('file_path'),
                }
            })

            # Example 2: Signature + docstring -> Body
            signature = self._extract_signature(func)
            if signature:
                examples.append({
                    'instruction': 'Complete this function implementation:',
                    'input': f'```python\n{signature}\n    """{docstring}"""\n```',
                    'output': func.get('body', ''),
                    'metadata': {
                        'type': 'signature_to_body',
                        'function_name': func.get('name'),
                        'file_path': code_sample.get('file_path'),
                    }
                })

        # Generate from classes
        classes = parsed.get('classes', [])
        for cls in classes:
            docstring = cls.get('docstring')
            if not docstring or len(docstring.strip()) < 10:
                continue

            # Class docstring -> Class implementation
            examples.append({
                'instruction': 'Implement the following class based on its docstring:',
                'input': f'Class: {cls.get("name")}\n\nDocstring:\n{docstring}',
                'output': cls.get('body', ''),
                'metadata': {
                    'type': 'docstring_to_class',
                    'class_name': cls.get('name'),
                    'file_path': code_sample.get('file_path'),
                }
            })

        self.examples_generated += len(examples)
        return examples

    def _extract_signature(self, func: Dict) -> Optional[str]:
        """Extract function signature from function dict."""
        name = func.get('name', '')
        args = func.get('args', [])
        returns = func.get('returns')

        if not name:
            return None

        # Build signature
        args_str = ', '.join(args)
        signature = f"def {name}({args_str})"

        if returns:
            signature += f" -> {returns}"

        signature += ":"

        # Add decorators if any
        decorators = func.get('decorators', [])
        if decorators:
            decorator_str = '\n'.join(decorators)
            signature = f"{decorator_str}\n{signature}"

        return signature

    def estimate_cost(self, num_files: int) -> float:
        """Free - no API costs."""
        return 0.0


class FIMGenerator(DataGeneratorPlugin):
    """
    Fill-in-the-Middle (FIM) generator for code completion training.

    Inspired by DeepSeek-Coder approach:
    - Randomly split code into prefix, middle, suffix
    - Train model to predict middle from context
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.min_middle_lines = config.get('fim_min_middle_lines', 1)
        self.max_middle_lines = config.get('fim_max_middle_lines', 10)
        self.examples_per_file = config.get('fim_examples_per_file', 3)

    def generate(self, code_sample: Dict) -> List[Dict]:
        """
        Generate FIM training examples.

        Args:
            code_sample: Code sample with content

        Returns:
            List of FIM training examples
        """
        examples = []
        content = code_sample.get('content', '')

        if not content:
            return examples

        lines = content.split('\n')
        if len(lines) < 5:
            return examples

        # Generate multiple FIM examples per file
        for _ in range(self.examples_per_file):
            example = self._create_fim_example(lines, code_sample.get('file_path', ''))
            if example:
                examples.append(example)

        self.examples_generated += len(examples)
        return examples

    def _create_fim_example(self, lines: List[str], file_path: str) -> Optional[Dict]:
        """Create a single FIM example by splitting code."""
        if len(lines) < 5:
            return None

        # Randomly choose where to split
        middle_size = random.randint(self.min_middle_lines, self.max_middle_lines)
        middle_size = min(middle_size, len(lines) - 2)  # Leave room for prefix and suffix

        # Random start position for middle section
        max_start = len(lines) - middle_size - 1
        middle_start = random.randint(1, max_start) if max_start > 1 else 1

        middle_end = middle_start + middle_size

        # Extract sections
        prefix = '\n'.join(lines[:middle_start])
        middle = '\n'.join(lines[middle_start:middle_end])
        suffix = '\n'.join(lines[middle_end:])

        # Create FIM training example
        # Format: <PRE> prefix <SUF> suffix <MID> middle
        return {
            'instruction': 'Fill in the middle section of this code:',
            'input': f'<PRE>\n{prefix}\n<SUF>\n{suffix}',
            'output': middle,
            'metadata': {
                'type': 'fill_in_middle',
                'file_path': file_path,
                'lines': f'{middle_start}-{middle_end}'
            }
        }

    def estimate_cost(self, num_files: int) -> float:
        """Free - no API costs."""
        return 0.0


class TestExtractor(DataGeneratorPlugin):
    """
    Extract test-to-implementation pairs for training.

    Learns from existing tests to understand:
    - How to implement functions that pass tests
    - How to write tests for functions
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.test_patterns = [
            r'test_.*\.py$',
            r'.*_test\.py$',
            r'.*/tests/.*\.py$',
            r'.*/test/.*\.py$',
        ]

    def generate(self, code_sample: Dict) -> List[Dict]:
        """
        Generate test-related training examples.

        Args:
            code_sample: Code sample with parsed structure

        Returns:
            List of training examples
        """
        examples = []
        file_path = code_sample.get('file_path', '')

        # Check if this is a test file
        if not self._is_test_file(file_path):
            return examples

        parsed = code_sample.get('parsed', {})
        functions = parsed.get('functions', [])

        for func in functions:
            name = func.get('name', '')

            # Skip if not a test function
            if not name.startswith('test_'):
                continue

            # Extract what's being tested
            tested_function = self._extract_tested_function(name)
            if not tested_function:
                continue

            body = func.get('body', '')
            docstring = func.get('docstring', '')

            # Example 1: Test -> Implementation requirement
            examples.append({
                'instruction': f'Implement the `{tested_function}` function to pass this test:',
                'input': body,
                'output': f'# Implementation of {tested_function} would go here\n# Based on test requirements',
                'metadata': {
                    'type': 'test_to_implementation',
                    'test_function': name,
                    'target_function': tested_function,
                    'file_path': file_path,
                }
            })

            # Example 2: Function name -> Test
            if docstring:
                examples.append({
                    'instruction': f'Write a test for a function named `{tested_function}`:',
                    'input': docstring if docstring else f'Function: {tested_function}',
                    'output': body,
                    'metadata': {
                        'type': 'function_to_test',
                        'test_function': name,
                        'target_function': tested_function,
                        'file_path': file_path,
                    }
                })

        self.examples_generated += len(examples)
        return examples

    def _is_test_file(self, file_path: str) -> bool:
        """Check if file is a test file."""
        for pattern in self.test_patterns:
            if re.search(pattern, file_path):
                return True
        return False

    def _extract_tested_function(self, test_name: str) -> Optional[str]:
        """Extract the name of the function being tested."""
        # Remove 'test_' prefix
        if test_name.startswith('test_'):
            return test_name[5:]
        return None

    def estimate_cost(self, num_files: int) -> float:
        """Free - no API costs."""
        return 0.0


class GitHistoryGenerator(DataGeneratorPlugin):
    """
    Generate training examples from git history.

    Learns from:
    - Bug fixes (before/after)
    - Feature additions (what changed)
    - Refactorings (how code improved)
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.max_commits = config.get('git_max_commits', 100)
        self.min_diff_lines = config.get('git_min_diff_lines', 5)
        self.max_diff_lines = config.get('git_max_diff_lines', 200)

    def generate(self, code_sample: Dict) -> List[Dict]:
        """
        Generate git history training examples.

        Note: This generator needs git integration.
        For now, returns empty list. Will be enhanced when git integration is added.

        Args:
            code_sample: Code sample (not used for git-based generation)

        Returns:
            List of training examples from git history
        """
        examples = []

        # TODO: Implement git history extraction
        # This requires:
        # 1. git.Repo integration
        # 2. Iterating through commits
        # 3. Extracting diffs
        # 4. Filtering useful changes
        # 5. Creating before/after examples

        # Placeholder for future implementation
        logger.debug("GitHistoryGenerator requires git integration - not yet implemented")

        return examples

    def estimate_cost(self, num_files: int) -> float:
        """Free - no API costs."""
        return 0.0


class SelfSupervisedGenerator(DataGeneratorPlugin):
    """
    Orchestrator for all self-supervised generators.

    Combines multiple free generators:
    - Docstring -> Code
    - Fill-in-Middle (FIM)
    - Test extraction
    - Git history
    """

    def __init__(self, config: Dict):
        super().__init__(config)

        # Initialize all sub-generators
        self.generators = []

        # Enable/disable individual generators
        generator_config = config.get('self_supervised', {})

        if generator_config.get('enable_docstring', True):
            self.generators.append(DocstringGenerator(config))

        if generator_config.get('enable_fim', True):
            self.generators.append(FIMGenerator(config))

        if generator_config.get('enable_tests', True):
            self.generators.append(TestExtractor(config))

        if generator_config.get('enable_git', False):  # Disabled by default until implemented
            self.generators.append(GitHistoryGenerator(config))

        logger.info(f"Initialized {len(self.generators)} self-supervised generators")

    def generate(self, code_sample: Dict) -> List[Dict]:
        """
        Generate training examples using all enabled generators.

        Args:
            code_sample: Code sample with parsed structure

        Returns:
            Combined list of training examples from all generators
        """
        all_examples = []

        for generator in self.generators:
            try:
                examples = generator.generate(code_sample)
                all_examples.extend(examples)
                self.log_progress(
                    f"{generator.get_name()} generated {len(examples)} examples"
                )
            except Exception as e:
                logger.error(
                    f"Error in {generator.get_name()}: {e}",
                    exc_info=True
                )

        self.examples_generated = len(all_examples)
        return all_examples

    def estimate_cost(self, num_files: int) -> float:
        """Free - all self-supervised generators have zero API cost."""
        return 0.0

    def get_stats(self) -> Dict:
        """Get combined statistics from all generators."""
        stats = super().get_stats()
        stats['sub_generators'] = [
            gen.get_stats() for gen in self.generators
        ]
        return stats
