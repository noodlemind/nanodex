"""Unit tests for GraphManager."""

import pytest

from nanodex.brain.graph_manager import GraphManager


def test_graph_manager_init(temp_dir):
    """Test GraphManager initialization."""
    db_path = temp_dir / "test.sqlite"
    gm = GraphManager(db_path)

    assert gm.db_path == db_path
    assert gm.conn is None


def test_graph_manager_connect(temp_dir):
    """Test database connection."""
    db_path = temp_dir / "test.sqlite"
    gm = GraphManager(db_path)
    gm.connect()

    assert gm.conn is not None
    assert db_path.exists()

    gm.close()
    assert gm.conn is None


def test_graph_manager_context_manager(temp_dir):
    """Test using GraphManager as context manager."""
    db_path = temp_dir / "test.sqlite"

    with GraphManager(db_path) as gm:
        assert gm.conn is not None

    # Connection should be closed after context
    assert gm.conn is None


def test_add_node(sample_graph_db):
    """Test adding nodes to the graph."""
    gm = sample_graph_db

    gm.add_node(
        node_id="test_node",
        node_type="function",
        name="test_function",
        path="/test.py",
        lang="python",
        properties={"doc": "Test function"},
    )

    node = gm.get_node("test_node")
    assert node is not None
    assert node["id"] == "test_node"
    assert node["type"] == "function"
    assert node["name"] == "test_function"
    assert node["path"] == "/test.py"
    assert node["lang"] == "python"
    assert node["properties"]["doc"] == "Test function"


def test_add_node_invalid_type(sample_graph_db):
    """Test adding node with invalid type."""
    gm = sample_graph_db

    with pytest.raises(ValueError, match="Invalid node type"):
        gm.add_node("bad_node", "invalid_type", "bad")


def test_get_node_not_found(sample_graph_db):
    """Test getting non-existent node."""
    gm = sample_graph_db
    node = gm.get_node("nonexistent")
    assert node is None


def test_add_edge(sample_graph_db):
    """Test adding edges between nodes."""
    gm = sample_graph_db

    # Add edge between existing nodes
    gm.add_edge("node1", "node2", "calls", weight=2.0, properties={"line": 42})

    neighbors = gm.get_neighbors("node1", relationship="calls", direction="outgoing")
    assert len(neighbors) == 1
    assert neighbors[0]["id"] == "node2"
    assert neighbors[0]["edge"]["relationship"] == "calls"
    assert neighbors[0]["edge"]["weight"] == 2.0
    assert neighbors[0]["edge"]["properties"]["line"] == 42


def test_add_edge_invalid_relationship(sample_graph_db):
    """Test adding edge with invalid relationship."""
    gm = sample_graph_db

    with pytest.raises(ValueError, match="Invalid relationship"):
        gm.add_edge("node1", "node2", "invalid_rel")


def test_add_edge_missing_node(sample_graph_db):
    """Test adding edge with missing source/target."""
    gm = sample_graph_db

    with pytest.raises(ValueError, match="Source node not found"):
        gm.add_edge("missing_node", "node1", "calls")

    with pytest.raises(ValueError, match="Target node not found"):
        gm.add_edge("node1", "missing_node", "calls")


def test_get_neighbors_outgoing(sample_graph_db):
    """Test getting outgoing neighbors."""
    gm = sample_graph_db

    neighbors = gm.get_neighbors("node1", direction="outgoing")
    assert len(neighbors) == 1
    assert neighbors[0]["id"] == "node3"
    assert neighbors[0]["edge"]["direction"] == "outgoing"


def test_get_neighbors_incoming(sample_graph_db):
    """Test getting incoming neighbors."""
    gm = sample_graph_db

    neighbors = gm.get_neighbors("node3", direction="incoming")
    assert len(neighbors) == 2
    node_ids = {n["id"] for n in neighbors}
    assert "node1" in node_ids
    assert "node2" in node_ids


def test_get_neighbors_both_directions(sample_graph_db):
    """Test getting neighbors in both directions."""
    gm = sample_graph_db

    # Add bidirectional edge
    gm.add_edge("node2", "node1", "calls")

    neighbors = gm.get_neighbors("node1", direction="both")
    assert len(neighbors) >= 2


def test_get_neighbors_with_relationship_filter(sample_graph_db):
    """Test filtering neighbors by relationship."""
    gm = sample_graph_db

    # Add another edge with different relationship
    gm.add_node("node4", "function", "other_func", path="/test.py", lang="python")
    gm.add_edge("node1", "node4", "calls")

    neighbors = gm.get_neighbors("node1", relationship="calls", direction="outgoing")
    assert len(neighbors) == 1
    assert neighbors[0]["id"] == "node4"


def test_get_stats(sample_graph_db):
    """Test getting graph statistics."""
    gm = sample_graph_db

    stats = gm.get_stats()

    assert stats["total_nodes"] >= 3
    assert stats["total_edges"] >= 2
    assert "node_distribution" in stats
    assert "edge_distribution" in stats
    assert len(stats["node_distribution"]) > 0
    assert len(stats["edge_distribution"]) > 0


def test_validate_integrity_valid_graph(sample_graph_db):
    """Test integrity validation on valid graph."""
    gm = sample_graph_db

    integrity = gm.validate_integrity()

    assert integrity["valid"] is True
    assert len(integrity["issues"]) == 0


def test_validate_integrity_invalid_node_type(sample_graph_db):
    """Test integrity validation with invalid node type."""
    gm = sample_graph_db

    # Manually insert invalid node type
    gm.conn.execute(
        "INSERT INTO nodes (id, type, name, properties) VALUES (?, ?, ?, ?)",
        ("bad_node", "invalid_type", "bad", "{}"),
    )
    gm.conn.commit()

    integrity = gm.validate_integrity()

    assert integrity["valid"] is False
    assert len(integrity["issues"]) > 0
    assert any("invalid_type" in str(issue).lower() for issue in integrity["issues"])


def test_node_replace(sample_graph_db):
    """Test replacing existing node."""
    gm = sample_graph_db

    # Add node
    gm.add_node("replace_test", "function", "original_name", path="/test.py")

    # Replace with new data
    gm.add_node("replace_test", "function", "new_name", path="/test.py")

    node = gm.get_node("replace_test")
    assert node["name"] == "new_name"


def test_edge_replace(sample_graph_db):
    """Test replacing existing edge."""
    gm = sample_graph_db

    # Add edge
    gm.add_edge("node1", "node2", "calls", weight=1.0)

    # Replace with new weight
    gm.add_edge("node1", "node2", "calls", weight=5.0)

    neighbors = gm.get_neighbors("node1", relationship="calls")
    assert len(neighbors) == 1
    assert neighbors[0]["edge"]["weight"] == 5.0
