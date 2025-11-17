"""Graph builder for extracting symbols from repositories."""

import logging
from pathlib import Path
from typing import List, Optional, Set

from nanodex.brain.graph_manager import GraphManager
from nanodex.config import ExtractorConfig
from nanodex.extractor.tree_sitter_parser import TreeSitterParser, get_file_language

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Build knowledge graph from source code repository."""

    def __init__(self, config: ExtractorConfig):
        """
        Initialize graph builder.

        Args:
            config: Extractor configuration
        """
        self.config = config
        self.graph_manager = GraphManager(config.out_graph)
        self.parsers: dict[str, TreeSitterParser] = {}
        self.processed_files = 0
        self.skipped_files = 0
        self.total_nodes = 0
        self.total_edges = 0

    def build_graph(self, repo_path: Path) -> None:
        """
        Build knowledge graph from repository.

        Args:
            repo_path: Path to repository root

        Raises:
            FileNotFoundError: If repo_path doesn't exist
        """
        if not repo_path.exists():
            raise FileNotFoundError(f"Repository path not found: {repo_path}")

        if not repo_path.is_dir():
            raise ValueError(f"Repository path is not a directory: {repo_path}")

        logger.info(f"Building knowledge graph from: {repo_path}")

        with self.graph_manager:
            self._process_repository(repo_path)
            self._log_statistics()

        logger.info(
            f"Graph build complete: {self.total_nodes} nodes, "
            f"{self.total_edges} edges from {self.processed_files} files"
        )

    def _process_repository(self, repo_path: Path) -> None:
        """Recursively process all files in repository."""
        for file_path in self._find_source_files(repo_path):
            try:
                self._process_file(file_path, repo_path)
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                self.skipped_files += 1

    def _find_source_files(self, repo_path: Path) -> List[Path]:
        """
        Find all source files in repository.

        Args:
            repo_path: Repository root path

        Returns:
            List of source file paths
        """
        source_files = []
        exclude_patterns = self._compile_exclude_patterns()

        for lang in self.config.languages:
            extensions = self._get_language_extensions(lang)
            for ext in extensions:
                for file_path in repo_path.rglob(f"*{ext}"):
                    if not self._is_excluded(file_path, repo_path, exclude_patterns):
                        source_files.append(file_path)

        logger.info(f"Found {len(source_files)} source files to process")
        return source_files

    def _compile_exclude_patterns(self) -> Set[str]:
        """Compile exclude patterns into a set for faster lookups."""
        patterns = set()
        for pattern in self.config.exclude:
            # Remove glob wildcards for simple path matching
            clean_pattern = pattern.replace("**/", "").replace("/**", "")
            patterns.add(clean_pattern)
        return patterns

    def _is_excluded(self, file_path: Path, repo_path: Path, patterns: Set[str]) -> bool:
        """
        Check if file should be excluded.

        Args:
            file_path: File to check
            repo_path: Repository root
            patterns: Set of exclude patterns

        Returns:
            True if file should be excluded
        """
        relative_path = file_path.relative_to(repo_path)
        path_str = str(relative_path)

        # Check each exclude pattern
        for pattern in patterns:
            if pattern in path_str:
                return True

        # Check file size
        try:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            if size_mb > self.config.max_file_size_mb:
                logger.warning(
                    f"Skipping {file_path}: size {size_mb:.2f}MB exceeds limit "
                    f"{self.config.max_file_size_mb}MB"
                )
                return True
        except Exception as e:
            logger.warning(f"Failed to check file size for {file_path}: {e}")
            return True

        return False

    def _get_language_extensions(self, language: str) -> List[str]:
        """Get file extensions for a language."""
        extension_map = {
            "python": [".py"],
            "java": [".java"],
            "typescript": [".ts", ".tsx"],
            "javascript": [".js", ".jsx"],
            "cpp": [".cpp", ".cc", ".cxx", ".hpp", ".h"],
            "c": [".c", ".h"],
            "rust": [".rs"],
            "go": [".go"],
        }
        return extension_map.get(language, [])

    def _process_file(self, file_path: Path, repo_path: Path) -> None:
        """
        Process a single source file.

        Args:
            file_path: Path to source file
            repo_path: Repository root path
        """
        language = get_file_language(file_path)
        if not language or language not in self.config.languages:
            return

        # Get or create parser for this language
        if language not in self.parsers:
            try:
                self.parsers[language] = TreeSitterParser(language)
            except Exception as e:
                logger.error(f"Failed to create parser for {language}: {e}")
                return

        parser = self.parsers[language]

        # Read file content
        try:
            with open(file_path, "rb") as f:
                source_code = f.read()
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            return

        # Extract symbols
        try:
            nodes, edges = parser.extract_symbols(source_code, file_path.relative_to(repo_path))
        except Exception as e:
            logger.error(f"Failed to extract symbols from {file_path}: {e}")
            return

        # Add to graph
        for node in nodes:
            try:
                self.graph_manager.add_node(
                    node_id=node["id"],
                    node_type=node["type"],
                    name=node["name"],
                    path=node.get("path"),
                    lang=node.get("lang"),
                    properties=node.get("properties"),
                )
                self.total_nodes += 1
            except Exception as e:
                logger.warning(f"Failed to add node {node['id']}: {e}")

        for edge in edges:
            try:
                # Ensure target node exists (create placeholder if needed)
                target_node = self.graph_manager.get_node(edge["target"])
                if not target_node:
                    # Create placeholder node for external reference
                    self.graph_manager.add_node(
                        node_id=edge["target"],
                        node_type="external",
                        name=edge["target"],
                        properties={"placeholder": True},
                    )
                    self.total_nodes += 1

                self.graph_manager.add_edge(
                    source_id=edge["source"],
                    target_id=edge["target"],
                    relationship=edge["relationship"],
                    properties=edge.get("properties"),
                )
                self.total_edges += 1
            except Exception as e:
                logger.warning(f"Failed to add edge {edge['source']}->{edge['target']}: {e}")

        self.processed_files += 1
        if self.processed_files % 10 == 0:
            logger.info(
                f"Progress: {self.processed_files} files processed, "
                f"{self.total_nodes} nodes, {self.total_edges} edges"
            )

    def _log_statistics(self) -> None:
        """Log graph statistics."""
        try:
            stats = self.graph_manager.get_stats()
            logger.info("=" * 60)
            logger.info("Graph Statistics:")
            logger.info(f"  Total nodes: {stats['total_nodes']}")
            logger.info(f"  Total edges: {stats['total_edges']}")
            logger.info(f"  Unique node types: {stats['unique_node_types']}")
            logger.info(f"  Unique edge types: {stats['unique_edge_types']}")
            logger.info("")
            logger.info("Node type distribution:")
            for dist in stats["node_distribution"]:
                logger.info(f"  {dist['type']}: {dist['count']} ({dist['percentage']}%)")
            logger.info("")
            logger.info("Edge relationship distribution:")
            for dist in stats["edge_distribution"]:
                logger.info(f"  {dist['relationship']}: {dist['count']} ({dist['percentage']}%)")
            logger.info("=" * 60)
        except Exception as e:
            logger.error(f"Failed to log statistics: {e}")


def build_graph_from_config(config_path: Path, repo_path: Path) -> None:
    """
    Build knowledge graph from configuration file.

    Args:
        config_path: Path to extractor configuration YAML
        repo_path: Path to repository to extract
    """
    from nanodex.config import load_config, ExtractorConfig

    config = load_config(config_path, ExtractorConfig)
    builder = GraphBuilder(config)
    builder.build_graph(repo_path)
