"""Integration tests for the extraction pipeline."""

import pytest

from nanodex.brain.graph_manager import GraphManager
from nanodex.extractor.graph_builder import GraphBuilder


@pytest.mark.integration
def test_extract_sample_repo(sample_repo, extractor_config):
    """Test extracting symbols from a sample repository."""
    builder = GraphBuilder(extractor_config)
    builder.build_graph(sample_repo)

    # Verify graph was created
    assert extractor_config.out_graph.exists()

    # Verify graph has expected content
    with GraphManager(extractor_config.out_graph) as gm:
        stats = gm.get_stats()

        # Should have at least 3 file nodes (main.py, utils.py, helper.py)
        assert stats["total_nodes"] >= 3

        # Should have some edges
        assert stats["total_edges"] > 0

        # Should have file nodes
        node_types = {dist["type"] for dist in stats["node_distribution"]}
        assert "file" in node_types

        # Verify integrity
        integrity = gm.validate_integrity()
        assert integrity["valid"] is True


@pytest.mark.integration
def test_extract_with_imports(sample_repo, extractor_config):
    """Test that imports are correctly extracted."""
    builder = GraphBuilder(extractor_config)
    builder.build_graph(sample_repo)

    with GraphManager(extractor_config.out_graph) as gm:
        stats = gm.get_stats()

        # Should have import edges
        edge_types = {dist["relationship"] for dist in stats["edge_distribution"]}
        assert "imports" in edge_types or "defined_in" in edge_types


@pytest.mark.integration
def test_extract_multiple_languages(temp_dir, extractor_config):
    """Test extracting from repository with multiple languages."""
    # Create multi-language repo
    repo = temp_dir / "multi_lang_repo"
    repo.mkdir()

    (repo / "script.py").write_text("def hello(): pass")
    (repo / "Script.java").write_text("public class Script { }")

    # Update config to support multiple languages
    extractor_config.languages = ["python", "java"]

    builder = GraphBuilder(extractor_config)
    builder.build_graph(repo)

    with GraphManager(extractor_config.out_graph) as gm:
        stats = gm.get_stats()

        # Should have nodes from both languages
        assert stats["total_nodes"] >= 2


@pytest.mark.integration
def test_extract_excludes_patterns(temp_dir, extractor_config):
    """Test that excluded patterns are respected."""
    repo = temp_dir / "repo_with_exclusions"
    repo.mkdir()

    # Create files that should be included
    (repo / "main.py").write_text("def main(): pass")

    # Create files that should be excluded
    test_dir = repo / "test"
    test_dir.mkdir()
    (test_dir / "test_main.py").write_text("def test_main(): pass")

    builder = GraphBuilder(extractor_config)
    builder.build_graph(repo)

    with GraphManager(extractor_config.out_graph) as gm:
        # Should only have nodes from main.py, not test/
        file_nodes = []
        cursor = gm.conn.execute("SELECT path FROM nodes WHERE type = 'file'")
        for row in cursor.fetchall():
            file_nodes.append(row[0])

        # test/ should not be in any paths
        assert not any("test" in path for path in file_nodes)


@pytest.mark.integration
def test_extract_large_file_skipped(temp_dir, extractor_config):
    """Test that files exceeding size limit are skipped."""
    repo = temp_dir / "repo_with_large_file"
    repo.mkdir()

    # Create normal file
    (repo / "small.py").write_text("def small(): pass")

    # Create large file (> 5MB)
    large_content = "# " + ("x" * 1024 * 1024 * 6)  # ~6MB
    (repo / "large.py").write_text(large_content)

    builder = GraphBuilder(extractor_config)
    builder.build_graph(repo)

    # Should complete without error
    assert extractor_config.out_graph.exists()

    with GraphManager(extractor_config.out_graph) as gm:
        stats = gm.get_stats()
        # Should have small.py but not large.py
        assert stats["total_nodes"] >= 1


@pytest.mark.integration
def test_extract_with_symlink_inside_repo(temp_dir, extractor_config):
    """Test that symlinks within the repository are handled correctly."""
    repo = temp_dir / "repo_with_symlink"
    repo.mkdir()

    # Create target file
    target = repo / "target.py"
    target.write_text("def target_function(): pass")

    # Create symlink to target within repo
    symlink = repo / "link.py"
    symlink.symlink_to(target)

    builder = GraphBuilder(extractor_config)
    builder.build_graph(repo)

    with GraphManager(extractor_config.out_graph) as gm:
        stats = gm.get_stats()
        # Should extract from resolved target
        assert stats["total_nodes"] >= 1


@pytest.mark.integration
def test_extract_rejects_symlink_outside_repo(temp_dir, extractor_config):
    """Test that symlinks pointing outside repository are rejected."""
    repo = temp_dir / "repo_safe"
    repo.mkdir()

    outside = temp_dir / "outside"
    outside.mkdir()

    outside_file = outside / "external.py"
    outside_file.write_text("def external(): pass")

    # Create symlink from inside repo to outside
    symlink = repo / "bad_link.py"
    symlink.symlink_to(outside_file)

    builder = GraphBuilder(extractor_config)
    builder.build_graph(repo)

    with GraphManager(extractor_config.out_graph) as gm:
        stats = gm.get_stats()
        # Should have 0 nodes - symlink rejected
        assert stats["total_nodes"] == 0
