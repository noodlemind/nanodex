"""Unit tests for Summarizer."""

import json
from pathlib import Path

import pytest

from nanodex.brain.summarizer import Summarizer
from nanodex.config import BrainConfig


@pytest.fixture
def brain_config(temp_dir):
    """Create test brain configuration."""
    return BrainConfig(
        node_types=["module", "capability", "concept", "error", "recipe"],
        summary_style="concise",
        summary_max_tokens=200,
        out_dir=temp_dir / "summaries",
        use_embeddings=False,
    )


def test_summarizer_init(sample_graph_db, brain_config):
    """Test Summarizer initialization."""
    summarizer = Summarizer(sample_graph_db, brain_config)

    assert summarizer.gm == sample_graph_db
    assert summarizer.config == brain_config
    assert summarizer.output_dir.exists()


def test_build_concise_summary(sample_graph_db, brain_config):
    """Test building concise summaries."""
    summarizer = Summarizer(sample_graph_db, brain_config)

    context = {"file": "/test.py", "language": "python", "neighbors": ["func2", "func3"]}

    summary = summarizer._build_concise_summary("capability", "process_data", context, {})

    assert "process_data" in summary
    assert "capability" in summary or "function" in summary.lower()
    assert "/test.py" in summary


def test_build_detailed_summary(sample_graph_db, brain_config):
    """Test building detailed summaries."""
    brain_config.summary_style = "detailed"
    summarizer = Summarizer(sample_graph_db, brain_config)

    context = {
        "file": "/test.py",
        "language": "python",
        "line": 10,
        "neighbors": ["func2", "func3"],
    }
    properties = {"doc": "This is a test function"}

    summary = summarizer._build_detailed_summary("capability", "process_data", context, properties)

    assert "process_data" in summary
    assert "line 10" in summary
    assert "python" in summary.lower()
    assert "func2" in summary


def test_build_technical_summary(sample_graph_db, brain_config):
    """Test building technical summaries."""
    brain_config.summary_style = "technical"
    summarizer = Summarizer(sample_graph_db, brain_config)

    context = {
        "file": "/test.py",
        "language": "python",
        "lines": "10-20",
    }
    properties = {"complexity": 5}

    summary = summarizer._build_technical_summary("capability", "process_data", context, properties)

    assert "process_data" in summary
    assert "capability" in summary
    assert "/test.py" in summary
    assert "python" in summary


def test_generate_node_summary(sample_graph_db, brain_config):
    """Test generating summary for a single node."""
    gm = sample_graph_db
    summarizer = Summarizer(gm, brain_config)

    # Add a test node
    gm.add_node(
        "test_func",
        "function",
        "process_data",
        path="/test.py",
        lang="python",
        properties={"start_line": 10, "end_line": 20},
    )

    # Get the node
    cursor = gm.conn.execute("SELECT * FROM nodes WHERE id = ?", ("test_func",))
    node_row = cursor.fetchone()

    # Generate summary
    summary_data = summarizer._generate_node_summary(node_row)

    assert summary_data["id"] == "test_func"
    assert summary_data["type"] == "function"
    assert summary_data["name"] == "process_data"
    assert "summary" in summary_data
    assert "context" in summary_data
    assert isinstance(summary_data["summary"], str)
    assert len(summary_data["summary"]) > 0


def test_save_summary(sample_graph_db, brain_config, temp_dir):
    """Test saving summary to JSON file."""
    summarizer = Summarizer(sample_graph_db, brain_config)

    summary_data = {
        "id": "test_node",
        "type": "capability",
        "name": "test_function",
        "summary": "This is a test summary.",
        "context": {"file": "/test.py"},
    }

    output_path = temp_dir / "test_summary.json"
    summarizer._save_summary(output_path, summary_data)

    assert output_path.exists()

    with open(output_path) as f:
        loaded_data = json.load(f)

    assert loaded_data == summary_data


def test_generate_all_summaries(sample_graph_db, brain_config):
    """Test generating summaries for all nodes."""
    gm = sample_graph_db
    summarizer = Summarizer(gm, brain_config)

    # Add some test nodes
    gm.add_node("func1", "function", "main", path="/test.py", lang="python")
    gm.add_node("func2", "function", "helper", path="/test.py", lang="python")
    gm.add_node("class1", "class", "DataModel", path="/test.py", lang="python")

    # Generate all summaries
    count = summarizer.generate_all_summaries()

    # Should have generated summaries for all nodes (including fixture nodes)
    assert count >= 3

    # Check that summary files were created
    summary_files = list(brain_config.out_dir.glob("*.json"))
    assert len(summary_files) >= 3


def test_summary_caching(sample_graph_db, brain_config):
    """Test that summaries are cached."""
    gm = sample_graph_db
    summarizer = Summarizer(gm, brain_config)

    # Add a test node
    gm.add_node("cached_node", "function", "cached_func", path="/test.py")

    # Generate summaries first time
    count1 = summarizer.generate_all_summaries()

    # Generate again - should use cache
    count2 = summarizer.generate_all_summaries()

    # Second run should generate 0 new summaries (all cached)
    assert count2 == 0


def test_get_summary_stats(sample_graph_db, brain_config):
    """Test getting summary statistics."""
    gm = sample_graph_db
    summarizer = Summarizer(gm, brain_config)

    # Add nodes and generate summaries
    gm.add_node("func1", "function", "test_func", path="/test.py")
    summarizer.generate_all_summaries()

    stats = summarizer.get_summary_stats()

    assert "total_summaries" in stats
    assert "output_dir" in stats
    assert stats["total_summaries"] > 0


def test_summary_different_styles(sample_graph_db, temp_dir):
    """Test that different summary styles produce different outputs."""
    gm = sample_graph_db

    # Add a test node
    gm.add_node(
        "style_test",
        "function",
        "test_func",
        path="/test.py",
        lang="python",
        properties={"start_line": 10},
    )

    # Test concise style
    config_concise = BrainConfig(summary_style="concise", out_dir=temp_dir / "concise")
    summarizer_concise = Summarizer(gm, config_concise)

    cursor = gm.conn.execute("SELECT * FROM nodes WHERE id = ?", ("style_test",))
    node_row = cursor.fetchone()

    summary_concise = summarizer_concise._generate_node_summary(node_row)

    # Test detailed style
    config_detailed = BrainConfig(summary_style="detailed", out_dir=temp_dir / "detailed")
    summarizer_detailed = Summarizer(gm, config_detailed)
    summary_detailed = summarizer_detailed._generate_node_summary(node_row)

    # Detailed should be longer
    assert len(summary_detailed["summary"]) >= len(summary_concise["summary"])
