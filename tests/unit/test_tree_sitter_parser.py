"""Unit tests for TreeSitterParser."""

from pathlib import Path

import pytest

from nanodex.extractor.tree_sitter_parser import TreeSitterParser, get_file_language


def test_parser_init_python():
    """Test initializing Python parser."""
    parser = TreeSitterParser("python")
    assert parser.language == "python"
    assert parser.parser is not None


def test_parser_init_java():
    """Test initializing Java parser."""
    parser = TreeSitterParser("java")
    assert parser.language == "java"
    assert parser.parser is not None


def test_parser_init_invalid_language():
    """Test initializing parser with invalid language."""
    with pytest.raises(ValueError, match="Unsupported language"):
        TreeSitterParser("invalid_lang")


def test_parse_python(sample_python_code):
    """Test parsing Python code."""
    parser = TreeSitterParser("python")
    root = parser.parse(sample_python_code)

    assert root is not None
    assert root.type == "module"


def test_parse_java(sample_java_code):
    """Test parsing Java code."""
    parser = TreeSitterParser("java")
    root = parser.parse(sample_java_code)

    assert root is not None
    assert root.type == "program"


def test_extract_symbols_python(sample_python_code, temp_dir):
    """Test extracting symbols from Python code."""
    parser = TreeSitterParser("python")
    file_path = temp_dir / "test.py"

    nodes, edges = parser.extract_symbols(sample_python_code, file_path)

    # Should have file node
    file_nodes = [n for n in nodes if n["type"] == "file"]
    assert len(file_nodes) == 1
    assert file_nodes[0]["name"] == "test.py"

    # Should have function nodes
    func_nodes = [n for n in nodes if n["type"] == "function"]
    assert len(func_nodes) >= 2  # hello_world, main, and Calculator methods

    # Should have class node
    class_nodes = [n for n in nodes if n["type"] == "class"]
    assert len(class_nodes) >= 1

    # Should have defined_in edges
    defined_in_edges = [e for e in edges if e["relationship"] == "defined_in"]
    assert len(defined_in_edges) > 0


def test_extract_symbols_java(sample_java_code, temp_dir):
    """Test extracting symbols from Java code."""
    parser = TreeSitterParser("java")
    file_path = temp_dir / "HelloWorld.java"

    nodes, edges = parser.extract_symbols(sample_java_code, file_path)

    # Should have file node
    file_nodes = [n for n in nodes if n["type"] == "file"]
    assert len(file_nodes) == 1

    # Should have class node
    class_nodes = [n for n in nodes if n["type"] == "class"]
    assert len(class_nodes) >= 1

    # Should have defined_in edges
    defined_in_edges = [e for e in edges if e["relationship"] == "defined_in"]
    assert len(defined_in_edges) > 0


def test_get_file_language():
    """Test determining language from file extension."""
    assert get_file_language(Path("test.py")) == "python"
    assert get_file_language(Path("Test.java")) == "java"
    assert get_file_language(Path("app.ts")) == "typescript"
    assert get_file_language(Path("app.tsx")) == "typescript"
    assert get_file_language(Path("script.js")) == "javascript"
    assert get_file_language(Path("main.cpp")) == "cpp"
    assert get_file_language(Path("main.c")) == "c"
    assert get_file_language(Path("main.rs")) == "rust"
    assert get_file_language(Path("main.go")) == "go"
    assert get_file_language(Path("unknown.txt")) is None


def test_parse_invalid_code():
    """Test parsing invalid code."""
    parser = TreeSitterParser("python")
    invalid_code = b"def incomplete_function(:"

    root = parser.parse(invalid_code)

    # Should still return a tree, but with errors
    assert root is not None
    # Parser may or may not detect errors depending on the input


def test_extract_symbols_empty_file(temp_dir):
    """Test extracting symbols from empty file."""
    parser = TreeSitterParser("python")
    file_path = temp_dir / "empty.py"

    nodes, edges = parser.extract_symbols(b"", file_path)

    # Should at least have file node
    file_nodes = [n for n in nodes if n["type"] == "file"]
    assert len(file_nodes) == 1
