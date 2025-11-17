"""Tree-sitter based code parser for multi-language symbol extraction."""

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import tree_sitter_languages as tsl
from tree_sitter import Language, Parser, Node

logger = logging.getLogger(__name__)

# Language mapping to Tree-sitter language names
LANGUAGE_MAP = {
    "python": "python",
    "java": "java",
    "typescript": "typescript",
    "javascript": "javascript",
    "cpp": "cpp",
    "c": "c",
    "rust": "rust",
    "go": "go",
}


class TreeSitterParser:
    """Parse source code using Tree-sitter."""

    def __init__(self, language: str):
        """
        Initialize parser for a specific language.

        Args:
            language: Programming language (e.g., 'python', 'java')

        Raises:
            ValueError: If language is not supported
        """
        if language not in LANGUAGE_MAP:
            raise ValueError(
                f"Unsupported language: {language}. Supported: {list(LANGUAGE_MAP.keys())}"
            )

        self.language = language
        self.lang_name = LANGUAGE_MAP[language]

        try:
            self.parser = Parser()
            self.ts_language = tsl.get_language(self.lang_name)
            self.parser.set_language(self.ts_language)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize parser for {language}: {e}") from e

        self.query_path = self._get_query_path()
        self.query = self._load_query() if self.query_path.exists() else None

    def _get_query_path(self) -> Path:
        """Get path to language-specific query file."""
        return Path(__file__).parent / "queries" / f"{self.language}_queries.scm"

    def _load_query(self) -> Optional[Any]:
        """Load Tree-sitter query for symbol extraction."""
        try:
            with open(self.query_path, "r") as f:
                query_source = f.read()
            return self.ts_language.query(query_source)
        except Exception as e:
            logger.warning(f"Failed to load query for {self.language}: {e}")
            return None

    def parse(self, source_code: bytes, file_path: Optional[Path] = None) -> Optional[Node]:
        """
        Parse source code into an AST.

        Args:
            source_code: Source code as bytes
            file_path: Optional file path for error reporting

        Returns:
            Root node of the AST, or None if parsing failed
        """
        try:
            tree = self.parser.parse(source_code)
            if tree.root_node.has_error:
                logger.warning(
                    f"Parse errors in {file_path or 'source'} for language {self.language}"
                )
            return tree.root_node
        except Exception as e:
            logger.error(f"Failed to parse {file_path or 'source'}: {e}")
            return None

    def extract_symbols(
        self, source_code: bytes, file_path: Path
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Extract symbols and relationships from source code.

        Args:
            source_code: Source code as bytes
            file_path: File path for context

        Returns:
            Tuple of (nodes, edges) where:
                - nodes: List of symbol nodes (functions, classes, etc.)
                - edges: List of relationships (calls, imports, etc.)
        """
        root = self.parse(source_code, file_path)
        if not root:
            return [], []

        nodes = []
        edges = []

        # Add file node
        file_id = self._generate_id(str(file_path))
        nodes.append(
            {
                "id": file_id,
                "type": "file",
                "name": file_path.name,
                "path": str(file_path),
                "lang": self.language,
                "properties": {"lines": source_code.count(b"\n") + 1},
            }
        )

        if not self.query:
            logger.warning(f"No query available for {self.language}, skipping symbol extraction")
            return nodes, edges

        # Extract symbols using queries
        try:
            captures = self.query.captures(root)
            self._process_captures(captures, source_code, file_path, file_id, nodes, edges)
        except Exception as e:
            logger.error(f"Failed to extract symbols from {file_path}: {e}")

        return nodes, edges

    def _process_captures(
        self,
        captures: List[Tuple[Node, str]],
        source_code: bytes,
        file_path: Path,
        file_id: str,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
    ) -> None:
        """Process Tree-sitter query captures into nodes and edges."""
        processed_nodes: Dict[str, Dict[str, Any]] = {}

        for node, capture_name in captures:
            text = node.text.decode("utf-8", errors="ignore")
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1

            # Process function definitions
            if capture_name == "func.def":
                func_id = self._generate_id(f"{file_path}:{start_line}:function:{text[:50]}")
                if func_id not in processed_nodes:
                    processed_nodes[func_id] = {
                        "id": func_id,
                        "type": "function",
                        "name": self._extract_name(node, "func.name", captures),
                        "path": str(file_path),
                        "lang": self.language,
                        "properties": {"start_line": start_line, "end_line": end_line},
                    }
                    edges.append(
                        {
                            "source": func_id,
                            "target": file_id,
                            "relationship": "defined_in",
                        }
                    )

            # Process class definitions
            elif capture_name == "class.def":
                class_id = self._generate_id(f"{file_path}:{start_line}:class:{text[:50]}")
                if class_id not in processed_nodes:
                    processed_nodes[class_id] = {
                        "id": class_id,
                        "type": "class",
                        "name": self._extract_name(node, "class.name", captures),
                        "path": str(file_path),
                        "lang": self.language,
                        "properties": {"start_line": start_line, "end_line": end_line},
                    }
                    edges.append(
                        {
                            "source": class_id,
                            "target": file_id,
                            "relationship": "defined_in",
                        }
                    )

            # Process import statements
            elif capture_name in ("import.stmt", "import.from"):
                import_name = text.strip()
                import_id = self._generate_id(f"import:{import_name}")
                edges.append(
                    {
                        "source": file_id,
                        "target": import_id,
                        "relationship": "imports",
                        "properties": {"import_line": start_line},
                    }
                )

            # Process function calls
            elif capture_name == "call.expr":
                target_name = self._extract_name(node, "call.target", captures)
                if target_name:
                    # Create edge from containing function to called function
                    caller_id = self._find_containing_function(node, processed_nodes)
                    if caller_id:
                        callee_id = self._generate_id(f"call:{target_name}")
                        edges.append(
                            {
                                "source": caller_id,
                                "target": callee_id,
                                "relationship": "calls",
                                "properties": {"call_line": start_line},
                            }
                        )

        # Add all processed nodes
        nodes.extend(processed_nodes.values())

    def _extract_name(self, node: Node, capture_type: str, captures: List[Tuple[Node, str]]) -> str:
        """Extract name from a specific capture type."""
        for child_node, child_capture in captures:
            if child_capture == capture_type and self._is_descendant(child_node, node):
                return child_node.text.decode("utf-8", errors="ignore")
        return "unknown"

    def _is_descendant(self, potential_child: Node, potential_parent: Node) -> bool:
        """Check if a node is a descendant of another node."""
        current = potential_child
        while current:
            if current == potential_parent:
                return True
            current = current.parent
        return False

    def _find_containing_function(
        self, node: Node, processed_nodes: Dict[str, Dict[str, Any]]
    ) -> Optional[str]:
        """Find the ID of the function containing this node."""
        current = node.parent
        while current:
            if current.type in ("function_definition", "function_declaration", "method_definition"):
                start_line = current.start_point[0] + 1
                # Look for matching function node
                for node_id, node_data in processed_nodes.items():
                    if (
                        node_data["type"] == "function"
                        and node_data["properties"]["start_line"] == start_line
                    ):
                        return node_id
            current = current.parent
        return None

    def _generate_id(self, identifier: str) -> str:
        """Generate a stable ID from an identifier string."""
        return hashlib.sha256(identifier.encode()).hexdigest()[:16]


def get_file_language(file_path: Path) -> Optional[str]:
    """
    Determine programming language from file extension.

    Args:
        file_path: Path to source file

    Returns:
        Language identifier, or None if not supported
    """
    extension_map = {
        ".py": "python",
        ".java": "java",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".js": "javascript",
        ".jsx": "javascript",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".hpp": "cpp",
        ".h": "cpp",
        ".c": "c",
        ".rs": "rust",
        ".go": "go",
    }

    return extension_map.get(file_path.suffix.lower())
