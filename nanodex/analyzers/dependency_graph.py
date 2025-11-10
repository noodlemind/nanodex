"""
Dependency graph builder for code analysis.

This module builds a graph of file dependencies based on import statements.
"""

import logging
from typing import Dict, List, Set
from pathlib import Path
import re

logger = logging.getLogger(__name__)


class DependencyGraph:
    """Build and analyze code dependencies."""

    def __init__(self, code_samples: List[Dict]):
        """
        Initialize dependency graph.

        Args:
            code_samples: List of code samples with import information
        """
        self.code_samples = code_samples
        self.file_map = {s['file_path']: s for s in code_samples}
        self.graph = self._build_graph()

    def _build_graph(self) -> Dict[str, Dict]:
        """
        Build dependency graph from code samples.

        Returns:
            Dictionary mapping file paths to dependency information:
            {
                'file_path': {
                    'imports': ['file1', 'file2'],  # Files this file imports
                    'imported_by': ['file3'],        # Files that import this file
                    'depth': 2,                       # Dependency depth
                }
            }
        """
        graph = {}

        # First pass: Initialize graph with all files
        for sample in self.code_samples:
            file_path = sample['file_path']
            graph[file_path] = {
                'imports': [],
                'imported_by': [],
                'depth': 0,
                'is_entry_point': False,
            }

        # Second pass: Build import relationships
        for sample in self.code_samples:
            file_path = sample['file_path']
            imports = sample.get('imports', [])

            # Resolve each import to actual file paths
            for import_stmt in imports:
                resolved_files = self._resolve_import(import_stmt, file_path)
                for resolved_file in resolved_files:
                    if resolved_file in graph:
                        graph[file_path]['imports'].append(resolved_file)

        # Third pass: Build reverse dependencies
        for file_path, data in graph.items():
            for imported_file in data['imports']:
                if imported_file in graph:
                    graph[imported_file]['imported_by'].append(file_path)

        # Fourth pass: Calculate depths and identify entry points
        self._calculate_depths(graph)
        self._identify_entry_points(graph)

        return graph

    def _resolve_import(self, import_stmt: str, current_file: str) -> List[str]:
        """
        Resolve an import statement to actual file paths.

        Args:
            import_stmt: Import statement (e.g., "from foo import bar")
            current_file: Current file path

        Returns:
            List of resolved file paths
        """
        resolved = []

        # Extract module name from import statement
        # Handle: "import foo", "from foo import bar", "import foo.bar"
        match = re.match(r'(?:from\s+)?([^\s]+)', import_stmt)
        if not match:
            return resolved

        module_path = match.group(1)

        # Convert module path to potential file paths
        # e.g., "foo.bar" -> ["foo/bar.py", "foo/bar/__init__.py"]
        potential_paths = [
            module_path.replace('.', '/') + '.py',
            module_path.replace('.', '/') + '/__init__.py',
        ]

        # Check if any of these paths exist in our codebase
        for potential_path in potential_paths:
            # Try to find matching file
            for file_path in self.file_map.keys():
                if file_path.endswith(potential_path) or potential_path in file_path:
                    resolved.append(file_path)
                    break

        return resolved

    def _calculate_depths(self, graph: Dict[str, Dict]):
        """
        Calculate dependency depth for each file.

        Depth is the maximum distance from entry points (files with no imports).

        Args:
            graph: Dependency graph to update
        """
        # Find files with no imports (entry points / leaves)
        no_imports = [f for f, data in graph.items() if not data['imports']]

        # BFS to calculate depths
        visited = set()
        queue = [(f, 0) for f in no_imports]

        while queue:
            file_path, depth = queue.pop(0)

            if file_path in visited:
                continue

            visited.add(file_path)
            graph[file_path]['depth'] = max(graph[file_path]['depth'], depth)

            # Add files that import this file
            for dependent in graph[file_path]['imported_by']:
                if dependent not in visited:
                    queue.append((dependent, depth + 1))

    def _identify_entry_points(self, graph: Dict[str, Dict]):
        """
        Identify entry point files.

        Entry points are files that are not imported by any other file
        in the codebase (typically main.py, __main__.py, etc.).

        Args:
            graph: Dependency graph to update
        """
        for file_path, data in graph.items():
            if not data['imported_by']:
                data['is_entry_point'] = True

                # Additional heuristics for entry points
                filename = Path(file_path).name
                if filename in ['main.py', '__main__.py', 'app.py', 'run.py', 'cli.py']:
                    logger.debug(f"Identified entry point: {file_path}")

    def get_dependencies(self, file_path: str, recursive: bool = False) -> List[str]:
        """
        Get all dependencies of a file.

        Args:
            file_path: File to get dependencies for
            recursive: If True, get transitive dependencies

        Returns:
            List of file paths this file depends on
        """
        if file_path not in self.graph:
            return []

        if not recursive:
            return self.graph[file_path]['imports'].copy()

        # BFS for transitive dependencies
        dependencies = set()
        queue = [file_path]
        visited = set()

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue

            visited.add(current)

            if current in self.graph:
                for dep in self.graph[current]['imports']:
                    dependencies.add(dep)
                    if dep not in visited:
                        queue.append(dep)

        return list(dependencies)

    def get_dependents(self, file_path: str, recursive: bool = False) -> List[str]:
        """
        Get all files that depend on this file.

        Args:
            file_path: File to get dependents for
            recursive: If True, get transitive dependents

        Returns:
            List of file paths that depend on this file
        """
        if file_path not in self.graph:
            return []

        if not recursive:
            return self.graph[file_path]['imported_by'].copy()

        # BFS for transitive dependents
        dependents = set()
        queue = [file_path]
        visited = set()

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue

            visited.add(current)

            if current in self.graph:
                for dep in self.graph[current]['imported_by']:
                    dependents.add(dep)
                    if dep not in visited:
                        queue.append(dep)

        return list(dependents)

    def get_entry_points(self) -> List[str]:
        """
        Get all entry point files.

        Returns:
            List of entry point file paths
        """
        return [f for f, data in self.graph.items() if data['is_entry_point']]

    def has_circular_dependencies(self) -> bool:
        """
        Check if there are circular dependencies.

        Returns:
            True if circular dependencies exist
        """
        for file_path in self.graph:
            if self._has_cycle(file_path, set(), set()):
                return True
        return False

    def _has_cycle(self, node: str, visited: Set[str], rec_stack: Set[str]) -> bool:
        """
        Check for cycles using DFS.

        Args:
            node: Current node
            visited: Set of visited nodes
            rec_stack: Recursion stack for cycle detection

        Returns:
            True if cycle detected
        """
        visited.add(node)
        rec_stack.add(node)

        if node in self.graph:
            for neighbor in self.graph[node]['imports']:
                if neighbor not in visited:
                    if self._has_cycle(neighbor, visited, rec_stack):
                        return True
                elif neighbor in rec_stack:
                    return True

        rec_stack.remove(node)
        return False

    def get_hub_files(self, top_n: int = 10) -> List[tuple]:
        """
        Get the most imported files (hub files).

        Args:
            top_n: Number of top files to return

        Returns:
            List of (file_path, import_count) tuples
        """
        import_counts = [
            (f, len(data['imported_by']))
            for f, data in self.graph.items()
        ]
        import_counts.sort(key=lambda x: x[1], reverse=True)
        return import_counts[:top_n]

    def concatenate_with_deps(self, file_path: str, max_depth: int = 2) -> str:
        """
        Concatenate a file with its dependencies (DeepSeek approach).

        This creates a training example that includes the file and its
        immediate dependencies, helping the model understand relationships.

        Args:
            file_path: Main file to concatenate
            max_depth: Maximum dependency depth to include

        Returns:
            Concatenated file content
        """
        if file_path not in self.file_map:
            return ""

        # Get dependencies up to max_depth
        deps = self._get_deps_by_depth(file_path, max_depth)

        # Sort by depth (dependencies first)
        deps_with_depth = [
            (dep, self.graph.get(dep, {}).get('depth', 0))
            for dep in deps
        ]
        deps_with_depth.sort(key=lambda x: x[1])

        # Concatenate
        parts = []

        # Add dependencies first
        for dep, _ in deps_with_depth:
            if dep != file_path and dep in self.file_map:
                sample = self.file_map[dep]
                parts.append(f"# File: {dep}\n")
                parts.append(sample.get('content', ''))
                parts.append("\n\n")

        # Add main file last
        main_sample = self.file_map[file_path]
        parts.append(f"# File: {file_path}\n")
        parts.append(main_sample.get('content', ''))

        return ''.join(parts)

    def _get_deps_by_depth(self, file_path: str, max_depth: int) -> List[str]:
        """
        Get dependencies up to a certain depth.

        Args:
            file_path: Starting file
            max_depth: Maximum depth to traverse

        Returns:
            List of file paths
        """
        deps = set([file_path])
        queue = [(file_path, 0)]
        visited = set()

        while queue:
            current, depth = queue.pop(0)

            if current in visited or depth > max_depth:
                continue

            visited.add(current)

            if current in self.graph:
                for dep in self.graph[current]['imports']:
                    deps.add(dep)
                    if depth < max_depth:
                        queue.append((dep, depth + 1))

        return list(deps)

    def to_dict(self) -> Dict:
        """
        Export graph as dictionary for JSON serialization.

        Returns:
            Dictionary representation of the graph
        """
        return {
            'files': list(self.graph.keys()),
            'dependencies': self.graph,
            'entry_points': self.get_entry_points(),
            'hub_files': self.get_hub_files(5),
            'has_cycles': self.has_circular_dependencies(),
        }
