"""
AST-based code parser for Python.

This module provides detailed code structure extraction using Python's built-in AST module.
"""

import ast
import logging
from typing import List, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class PythonASTParser:
    """Parse Python code using Abstract Syntax Tree."""

    def parse_file(self, file_path: Path, content: str) -> Dict:
        """
        Parse Python file and extract detailed structure.

        Args:
            file_path: Path to the file being parsed
            content: File content as string

        Returns:
            Dictionary containing:
            {
                'functions': List of function definitions,
                'classes': List of class definitions,
                'imports': List of import statements,
                'docstrings': List of docstrings found,
                'global_vars': List of global variables,
                'complexity': Estimated code complexity,
                'error': Error message if parsing failed
            }
        """
        try:
            tree = ast.parse(content, filename=str(file_path))
        except SyntaxError as e:
            logger.warning(f"Syntax error parsing {file_path}: {e}")
            return {
                'error': str(e),
                'functions': [],
                'classes': [],
                'imports': [],
                'docstrings': [],
                'global_vars': [],
                'complexity': 0,
            }
        except Exception as e:
            logger.warning(f"Error parsing {file_path}: {e}")
            return {
                'error': str(e),
                'functions': [],
                'classes': [],
                'imports': [],
                'docstrings': [],
                'global_vars': [],
                'complexity': 0,
            }

        return {
            'functions': self._extract_functions(tree),
            'classes': self._extract_classes(tree),
            'imports': self._extract_imports(tree),
            'docstrings': self._extract_docstrings(tree),
            'global_vars': self._extract_global_vars(tree),
            'complexity': self._calculate_complexity(tree),
        }

    def _extract_functions(self, tree: ast.AST) -> List[Dict]:
        """
        Extract function definitions from AST.

        Args:
            tree: AST tree

        Returns:
            List of function information dictionaries
        """
        functions = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                try:
                    # Extract function information
                    func_info = {
                        'name': node.name,
                        'lineno': node.lineno,
                        'end_lineno': getattr(node, 'end_lineno', node.lineno),
                        'is_async': isinstance(node, ast.AsyncFunctionDef),
                        'docstring': ast.get_docstring(node),
                        'decorators': [ast.unparse(d) for d in node.decorator_list],
                    }

                    # Extract arguments
                    args = []
                    for arg in node.args.args:
                        arg_info = {'name': arg.arg}
                        if arg.annotation:
                            arg_info['type'] = ast.unparse(arg.annotation)
                        args.append(arg_info)
                    func_info['args'] = args

                    # Extract return type
                    if node.returns:
                        func_info['returns'] = ast.unparse(node.returns)
                    else:
                        func_info['returns'] = None

                    # Get function body (limited to avoid huge strings)
                    try:
                        body = ast.unparse(node)
                        # Limit body size to 2000 characters
                        if len(body) > 2000:
                            body = body[:2000] + "..."
                        func_info['body'] = body
                    except Exception:
                        func_info['body'] = ""

                    # Calculate function complexity (count control flow statements)
                    func_info['complexity'] = self._count_control_flow(node)

                    functions.append(func_info)

                except Exception as e:
                    logger.debug(f"Error extracting function {getattr(node, 'name', 'unknown')}: {e}")
                    continue

        return functions

    def _extract_classes(self, tree: ast.AST) -> List[Dict]:
        """
        Extract class definitions from AST.

        Args:
            tree: AST tree

        Returns:
            List of class information dictionaries
        """
        classes = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                try:
                    class_info = {
                        'name': node.name,
                        'lineno': node.lineno,
                        'end_lineno': getattr(node, 'end_lineno', node.lineno),
                        'docstring': ast.get_docstring(node),
                        'decorators': [ast.unparse(d) for d in node.decorator_list],
                        'bases': [ast.unparse(b) for b in node.bases],
                    }

                    # Extract methods
                    methods = []
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            methods.append({
                                'name': item.name,
                                'lineno': item.lineno,
                                'is_async': isinstance(item, ast.AsyncFunctionDef),
                            })
                    class_info['methods'] = methods
                    class_info['method_count'] = len(methods)

                    classes.append(class_info)

                except Exception as e:
                    logger.debug(f"Error extracting class {getattr(node, 'name', 'unknown')}: {e}")
                    continue

        return classes

    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """
        Extract import statements from AST.

        Args:
            tree: AST tree

        Returns:
            List of import strings
        """
        imports = []

        for node in ast.walk(tree):
            try:
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.asname:
                            imports.append(f"import {alias.name} as {alias.asname}")
                        else:
                            imports.append(f"import {alias.name}")

                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        if alias.name == '*':
                            imports.append(f"from {module} import *")
                        elif alias.asname:
                            imports.append(f"from {module} import {alias.name} as {alias.asname}")
                        else:
                            imports.append(f"from {module} import {alias.name}")

            except Exception as e:
                logger.debug(f"Error extracting import: {e}")
                continue

        return imports

    def _extract_docstrings(self, tree: ast.AST) -> List[Dict]:
        """
        Extract all docstrings from the AST.

        Args:
            tree: AST tree

        Returns:
            List of docstring dictionaries with location info
        """
        docstrings = []

        # Module docstring
        module_doc = ast.get_docstring(tree)
        if module_doc:
            docstrings.append({
                'type': 'module',
                'content': module_doc,
                'lineno': 1,
            })

        # Function and class docstrings
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node)
                if doc:
                    docstrings.append({
                        'type': 'function',
                        'name': node.name,
                        'content': doc,
                        'lineno': node.lineno,
                    })

            elif isinstance(node, ast.ClassDef):
                doc = ast.get_docstring(node)
                if doc:
                    docstrings.append({
                        'type': 'class',
                        'name': node.name,
                        'content': doc,
                        'lineno': node.lineno,
                    })

        return docstrings

    def _extract_global_vars(self, tree: ast.AST) -> List[Dict]:
        """
        Extract global variable assignments.

        Args:
            tree: AST tree

        Returns:
            List of global variable information
        """
        global_vars = []

        # Only look at module-level assignments
        if not isinstance(tree, ast.Module):
            return global_vars

        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        try:
                            var_info = {
                                'name': target.id,
                                'lineno': node.lineno,
                            }
                            # Try to get the value (as string)
                            try:
                                var_info['value'] = ast.unparse(node.value)[:100]  # Limit length
                            except Exception:
                                var_info['value'] = "<complex>"

                            global_vars.append(var_info)
                        except Exception:
                            continue

        return global_vars

    def _calculate_complexity(self, tree: ast.AST) -> int:
        """
        Calculate cyclomatic complexity of the code.

        This is a simplified complexity metric based on control flow statements.

        Args:
            tree: AST tree

        Returns:
            Complexity score
        """
        complexity = 1  # Base complexity

        for node in ast.walk(tree):
            # Count control flow statements
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, (ast.And, ast.Or)):
                complexity += 1
            elif isinstance(node, ast.Lambda):
                complexity += 1

        return complexity

    def _count_control_flow(self, node: ast.AST) -> int:
        """
        Count control flow statements in a node (for function complexity).

        Args:
            node: AST node

        Returns:
            Count of control flow statements
        """
        count = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                count += 1
            elif isinstance(child, ast.ExceptHandler):
                count += 1
            elif isinstance(child, (ast.And, ast.Or)):
                count += 1

        return count
