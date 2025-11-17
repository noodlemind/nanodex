"""Unit tests for NodeTyper."""

import pytest

from nanodex.brain.node_typer import NodeTyper


def test_node_typer_init(sample_graph_db):
    """Test NodeTyper initialization."""
    typer = NodeTyper(sample_graph_db)
    assert typer.gm == sample_graph_db


def test_classify_file_nodes(sample_graph_db):
    """Test that file nodes are classified as modules."""
    typer = NodeTyper(sample_graph_db)

    # node3 is a file node
    semantic_type = typer._infer_semantic_type("file", "test.py", {})
    assert semantic_type == "module"


def test_classify_error_class(sample_graph_db):
    """Test that error classes are classified correctly."""
    typer = NodeTyper(sample_graph_db)

    # Classes with "Error" in name should be error type
    assert typer._infer_semantic_type("class", "ValueError", {}) == "error"
    assert typer._infer_semantic_type("class", "CustomException", {}) == "error"
    assert typer._infer_semantic_type("class", "NetworkError", {}) == "error"


def test_classify_recipe_function(sample_graph_db):
    """Test that recipe functions are classified correctly."""
    typer = NodeTyper(sample_graph_db)

    # Functions with special names are recipes
    assert typer._infer_semantic_type("function", "main", {}) == "recipe"
    assert typer._infer_semantic_type("function", "example_usage", {}) == "recipe"
    assert typer._infer_semantic_type("function", "demo", {}) == "recipe"
    assert typer._infer_semantic_type("function", "test_something", {}) == "recipe"


def test_classify_capability_function(sample_graph_db):
    """Test that public functions are classified as capabilities."""
    typer = NodeTyper(sample_graph_db)

    # Public functions are capabilities
    assert typer._infer_semantic_type("function", "process_data", {}) == "capability"
    assert typer._infer_semantic_type("function", "calculate", {}) == "capability"


def test_classify_internal_function(sample_graph_db):
    """Test that internal functions are classified as concepts."""
    typer = NodeTyper(sample_graph_db)

    # Private functions (starting with _) are concepts
    assert typer._infer_semantic_type("function", "_helper", {}) == "concept"
    assert typer._infer_semantic_type("function", "_internal_process", {}) == "concept"

    # Special methods are concepts
    assert typer._infer_semantic_type("function", "__init__", {}) == "concept"
    assert typer._infer_semantic_type("function", "__str__", {}) == "concept"


def test_classify_regular_class(sample_graph_db):
    """Test that regular classes are classified as concepts."""
    typer = NodeTyper(sample_graph_db)

    # Regular classes (no "Error" in name) are concepts
    assert typer._infer_semantic_type("class", "DataProcessor", {}) == "concept"
    assert typer._infer_semantic_type("class", "UserModel", {}) == "concept"


def test_is_error_class(sample_graph_db):
    """Test error class detection."""
    typer = NodeTyper(sample_graph_db)

    assert typer._is_error_class("ValueError") is True
    assert typer._is_error_class("CustomException") is True
    assert typer._is_error_class("NetworkFailure") is True
    assert typer._is_error_class("ErrorHandler") is True
    assert typer._is_error_class("DataProcessor") is False
    assert typer._is_error_class("UserModel") is False


def test_is_recipe_function(sample_graph_db):
    """Test recipe function detection."""
    typer = NodeTyper(sample_graph_db)

    assert typer._is_recipe_function("main") is True
    assert typer._is_recipe_function("example") is True
    assert typer._is_recipe_function("demo") is True
    assert typer._is_recipe_function("test_function") is True
    assert typer._is_recipe_function("run_example") is True
    assert typer._is_recipe_function("process_data") is False
    assert typer._is_recipe_function("helper") is False


def test_is_capability_function(sample_graph_db):
    """Test capability function detection."""
    typer = NodeTyper(sample_graph_db)

    assert typer._is_capability_function("process_data") is True
    assert typer._is_capability_function("calculate") is True
    assert typer._is_capability_function("_private") is False
    assert typer._is_capability_function("__init__") is False
    assert typer._is_capability_function("__str__") is False


def test_classify_all_nodes(sample_graph_db):
    """Test classifying all nodes in the graph."""
    gm = sample_graph_db
    typer = NodeTyper(gm)

    # Add some test nodes with different types
    gm.add_node("func1", "function", "main", path="/test.py", lang="python")
    gm.add_node("func2", "function", "process_data", path="/test.py", lang="python")
    gm.add_node("func3", "function", "_helper", path="/test.py", lang="python")
    gm.add_node("class1", "class", "ValueError", path="/test.py", lang="python")
    gm.add_node("class2", "class", "DataModel", path="/test.py", lang="python")
    gm.add_node("file1", "file", "module.py", path="/module.py", lang="python")

    # Classify all nodes
    type_counts = typer.classify_all_nodes()

    # Verify classifications
    assert type_counts["recipe"] >= 1  # main function
    assert type_counts["capability"] >= 1  # process_data
    assert type_counts["concept"] >= 2  # _helper, DataModel, private functions
    assert type_counts["error"] >= 1  # ValueError
    assert type_counts["module"] >= 2  # file nodes


def test_get_classification_stats(sample_graph_db):
    """Test getting classification statistics."""
    gm = sample_graph_db
    typer = NodeTyper(gm)

    # Add and classify some nodes
    gm.add_node("func1", "function", "main", path="/test.py")
    typer.classify_all_nodes()

    stats = typer.get_classification_stats()

    assert "semantic_types" in stats
    assert "all_types" in stats
    assert isinstance(stats["semantic_types"], dict)
    assert isinstance(stats["all_types"], dict)
