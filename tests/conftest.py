"""Pytest configuration and fixtures."""

import tempfile
from pathlib import Path
from typing import Generator

import pytest

from nanodex.brain.graph_manager import GraphManager
from nanodex.config import ExtractorConfig


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_graph_db(temp_dir: Path) -> Generator[GraphManager, None, None]:
    """Create a sample graph database for testing."""
    db_path = temp_dir / "test_graph.sqlite"
    gm = GraphManager(db_path)
    gm.connect()

    # Add sample nodes
    gm.add_node("node1", "function", "test_func", path="/test.py", lang="python")
    gm.add_node("node2", "class", "TestClass", path="/test.py", lang="python")
    gm.add_node("node3", "file", "test.py", path="/test.py", lang="python")

    # Add sample edges
    gm.add_edge("node1", "node3", "defined_in")
    gm.add_edge("node2", "node3", "defined_in")

    yield gm

    gm.close()


@pytest.fixture
def sample_python_code() -> bytes:
    """Sample Python code for testing."""
    return b"""
def hello_world(name):
    \"\"\"Greet the user.\"\"\"
    print(f"Hello, {name}!")
    return name

class Calculator:
    \"\"\"Simple calculator.\"\"\"

    def add(self, a, b):
        return a + b

    def multiply(self, a, b):
        result = self.add(a, 0)
        for _ in range(b - 1):
            result = self.add(result, a)
        return result

def main():
    calc = Calculator()
    result = calc.multiply(5, 3)
    hello_world("World")
"""


@pytest.fixture
def sample_java_code() -> bytes:
    """Sample Java code for testing."""
    return b"""
public class HelloWorld {
    private String message;

    public HelloWorld(String message) {
        this.message = message;
    }

    public void greet() {
        System.out.println(message);
    }

    public static void main(String[] args) {
        HelloWorld hw = new HelloWorld("Hello, Java!");
        hw.greet();
    }
}
"""


@pytest.fixture
def sample_code_file(temp_dir: Path, sample_python_code: bytes) -> Path:
    """Create a sample code file for testing."""
    file_path = temp_dir / "sample.py"
    file_path.write_bytes(sample_python_code)
    return file_path


@pytest.fixture
def sample_repo(temp_dir: Path) -> Path:
    """Create a sample repository structure for testing."""
    repo_path = temp_dir / "sample_repo"
    repo_path.mkdir()

    # Create some Python files
    (repo_path / "main.py").write_text(
        """
def main():
    print("Hello from main!")

if __name__ == "__main__":
    main()
"""
    )

    (repo_path / "utils.py").write_text(
        """
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b
"""
    )

    # Create a subdirectory
    (repo_path / "lib").mkdir()
    (repo_path / "lib" / "helper.py").write_text(
        """
from utils import add

def helper_add(x, y):
    return add(x, y)
"""
    )

    return repo_path


@pytest.fixture
def extractor_config(temp_dir: Path) -> ExtractorConfig:
    """Create a test extractor configuration."""
    return ExtractorConfig(
        languages=["python"],
        use_scip=False,
        exclude=["**/test/**", "**/__pycache__/**"],
        out_graph=temp_dir / "graph.sqlite",
        max_file_size_mb=5,
    )
