"""Graph database manager for nanodex knowledge graph."""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Allowed edge relationship types
EDGE_RELATIONSHIPS = {
    "calls",
    "imports",
    "extends",
    "implements",
    "throws",
    "defined_in",
    "depends_on",
}

# Allowed node types
NODE_TYPES = {
    "module",
    "capability",
    "concept",
    "error",
    "recipe",
    "file",
    "class",
    "function",
    "variable",
    "external",  # Placeholder for unresolved references
}


class GraphManager:
    """Manages the SQLite knowledge graph database."""

    def __init__(self, db_path: Path):
        """
        Initialize graph manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn: Optional[sqlite3.Connection] = None

    def connect(self) -> None:
        """Establish database connection and initialize schema."""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self) -> "GraphManager":
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()

    def _init_schema(self) -> None:
        """Initialize database schema from SQL file."""
        if not self.conn:
            raise RuntimeError("Database not connected")

        schema_path = Path(__file__).parent.parent / "extractor" / "schema.sql"
        with open(schema_path, "r") as f:
            schema_sql = f.read()

        self.conn.executescript(schema_sql)
        self.conn.commit()

    def add_node(
        self,
        node_id: str,
        node_type: str,
        name: str,
        path: Optional[str] = None,
        lang: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a node to the graph.

        Args:
            node_id: Unique node identifier
            node_type: Type of node (e.g., 'function', 'class', 'module')
            name: Human-readable name
            path: File path (for code entities)
            lang: Programming language
            properties: Additional metadata as dict

        Raises:
            ValueError: If node_type is invalid
        """
        if not self.conn:
            raise RuntimeError("Database not connected")

        if node_type not in NODE_TYPES:
            raise ValueError(f"Invalid node type: {node_type}. Allowed: {NODE_TYPES}")

        props_json = json.dumps(properties or {})

        self.conn.execute(
            """
            INSERT OR REPLACE INTO nodes (id, type, name, path, lang, properties)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (node_id, node_type, name, path, lang, props_json),
        )
        self.conn.commit()

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relationship: str,
        weight: float = 1.0,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add an edge between two nodes.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            relationship: Type of relationship (e.g., 'calls', 'imports')
            weight: Edge weight (default: 1.0)
            properties: Additional metadata

        Raises:
            ValueError: If relationship is invalid or nodes don't exist
        """
        if not self.conn:
            raise RuntimeError("Database not connected")

        if relationship not in EDGE_RELATIONSHIPS:
            raise ValueError(f"Invalid relationship: {relationship}. Allowed: {EDGE_RELATIONSHIPS}")

        # Verify nodes exist
        cursor = self.conn.execute(
            "SELECT id FROM nodes WHERE id IN (?, ?)", (source_id, target_id)
        )
        existing_ids = {row["id"] for row in cursor.fetchall()}

        if source_id not in existing_ids:
            raise ValueError(f"Source node not found: {source_id}")
        if target_id not in existing_ids:
            raise ValueError(f"Target node not found: {target_id}")

        props_json = json.dumps(properties or {})

        self.conn.execute(
            """
            INSERT OR REPLACE INTO edges (source_id, target_id, relationship, weight, properties)
            VALUES (?, ?, ?, ?, ?)
            """,
            (source_id, target_id, relationship, weight, props_json),
        )
        self.conn.commit()

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a node by ID.

        Args:
            node_id: Node identifier

        Returns:
            Node data as dict, or None if not found
        """
        if not self.conn:
            raise RuntimeError("Database not connected")

        cursor = self.conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,))
        row = cursor.fetchone()

        if not row:
            return None

        return {
            "id": row["id"],
            "type": row["type"],
            "name": row["name"],
            "path": row["path"],
            "lang": row["lang"],
            "properties": json.loads(row["properties"]),
        }

    def get_neighbors(
        self, node_id: str, relationship: Optional[str] = None, direction: str = "outgoing"
    ) -> List[Dict[str, Any]]:
        """
        Get neighboring nodes.

        Args:
            node_id: Source node ID
            relationship: Filter by relationship type (optional)
            direction: 'outgoing', 'incoming', or 'both'

        Returns:
            List of neighbor nodes with edge data
        """
        if not self.conn:
            raise RuntimeError("Database not connected")

        if direction not in {"outgoing", "incoming", "both"}:
            raise ValueError(f"Invalid direction: {direction}")

        # Build base query params for CASE statements (used in SELECT and JOIN)
        case_params = [node_id, node_id]

        # Build WHERE clause conditions and params
        where_conditions = []
        where_params: List[Any] = []

        if direction == "outgoing":
            where_conditions.append("e.source_id = ?")
            where_params.append(node_id)
        elif direction == "incoming":
            where_conditions.append("e.target_id = ?")
            where_params.append(node_id)
        elif direction == "both":
            where_conditions.append("(e.source_id = ? OR e.target_id = ?)")
            where_params.extend([node_id, node_id])

        if relationship:
            where_conditions.append("e.relationship = ?")
            where_params.append(relationship)

        where_clause = " AND ".join(where_conditions)

        query = f"""
        SELECT
            n.*,
            e.relationship,
            e.weight,
            e.properties AS edge_properties,
            CASE
                WHEN e.source_id = ? THEN 'outgoing'
                ELSE 'incoming'
            END AS edge_direction
        FROM edges e
        JOIN nodes n ON (
            CASE
                WHEN e.source_id = ? THEN n.id = e.target_id
                ELSE n.id = e.source_id
            END
        )
        WHERE {where_clause}
        """

        # Combine params: first case_params, then where_params
        all_params = case_params + where_params
        cursor = self.conn.execute(query, all_params)

        results = []
        for row in cursor.fetchall():
            results.append(
                {
                    "id": row["id"],
                    "type": row["type"],
                    "name": row["name"],
                    "path": row["path"],
                    "lang": row["lang"],
                    "properties": json.loads(row["properties"]),
                    "edge": {
                        "relationship": row["relationship"],
                        "weight": row["weight"],
                        "properties": (
                            json.loads(row["edge_properties"]) if row["edge_properties"] else {}
                        ),
                        "direction": row["edge_direction"],
                    },
                }
            )

        return results

    def traverse_path(
        self, start_id: str, relationship: str, max_depth: int = 3
    ) -> List[List[str]]:
        """
        Traverse graph paths from a starting node.

        Args:
            start_id: Starting node ID
            relationship: Relationship type to follow
            max_depth: Maximum traversal depth

        Returns:
            List of paths (each path is a list of node IDs)
        """
        if not self.conn:
            raise RuntimeError("Database not connected")

        if relationship not in EDGE_RELATIONSHIPS:
            raise ValueError(f"Invalid relationship: {relationship}")

        paths: List[List[str]] = []
        visited: Set[str] = set()

        def dfs(node_id: str, path: List[str], depth: int) -> None:
            if depth > max_depth or node_id in visited:
                return

            visited.add(node_id)
            path.append(node_id)

            neighbors = self.get_neighbors(node_id, relationship=relationship, direction="outgoing")

            if not neighbors:
                paths.append(path.copy())
            else:
                for neighbor in neighbors:
                    dfs(neighbor["id"], path.copy(), depth + 1)

            visited.remove(node_id)

        dfs(start_id, [], 0)
        return paths

    def get_stats(self) -> Dict[str, Any]:
        """
        Get graph statistics.

        Returns:
            Dictionary with node counts, edge counts, and distributions
        """
        if not self.conn:
            raise RuntimeError("Database not connected")

        cursor = self.conn.execute("SELECT * FROM graph_stats")
        stats_row = cursor.fetchone()

        cursor = self.conn.execute("SELECT * FROM node_type_distribution")
        node_dist = [
            {"type": row["type"], "count": row["count"], "percentage": row["percentage"]}
            for row in cursor.fetchall()
        ]

        cursor = self.conn.execute("SELECT * FROM edge_relationship_distribution")
        edge_dist = [
            {
                "relationship": row["relationship"],
                "count": row["count"],
                "percentage": row["percentage"],
            }
            for row in cursor.fetchall()
        ]

        return {
            "total_nodes": stats_row["total_nodes"],
            "total_edges": stats_row["total_edges"],
            "unique_node_types": stats_row["unique_node_types"],
            "unique_edge_types": stats_row["unique_edge_types"],
            "node_distribution": node_dist,
            "edge_distribution": edge_dist,
        }

    def validate_integrity(self) -> Dict[str, Any]:
        """
        Validate graph integrity.

        Returns:
            Dictionary with validation results
        """
        if not self.conn:
            raise RuntimeError("Database not connected")

        issues = []

        # Check for invalid node types
        cursor = self.conn.execute(
            "SELECT DISTINCT type FROM nodes WHERE type NOT IN ({})".format(
                ",".join("?" * len(NODE_TYPES))
            ),
            tuple(NODE_TYPES),
        )
        invalid_node_types = [row["type"] for row in cursor.fetchall()]
        if invalid_node_types:
            issues.append(f"Invalid node types found: {invalid_node_types}")

        # Check for invalid edge relationships
        cursor = self.conn.execute(
            "SELECT DISTINCT relationship FROM edges WHERE relationship NOT IN ({})".format(
                ",".join("?" * len(EDGE_RELATIONSHIPS))
            ),
            tuple(EDGE_RELATIONSHIPS),
        )
        invalid_relationships = [row["relationship"] for row in cursor.fetchall()]
        if invalid_relationships:
            issues.append(f"Invalid edge relationships found: {invalid_relationships}")

        # Check for orphaned edges (should not happen with foreign keys, but good to verify)
        cursor = self.conn.execute(
            """
            SELECT COUNT(*) as orphaned_count
            FROM edges e
            WHERE NOT EXISTS (SELECT 1 FROM nodes WHERE id = e.source_id)
               OR NOT EXISTS (SELECT 1 FROM nodes WHERE id = e.target_id)
        """
        )
        orphaned_count = cursor.fetchone()["orphaned_count"]
        if orphaned_count > 0:
            issues.append(f"Orphaned edges found: {orphaned_count}")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
        }
