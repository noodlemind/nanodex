"""Unit tests for DatasetValidator."""

import pytest

from nanodex.dataset.validators import DatasetValidator


def test_validator_init(sample_graph_db):
    """Test validator initialization."""
    validator = DatasetValidator(sample_graph_db, min_response_tokens=50, max_response_tokens=500)

    assert validator.gm == sample_graph_db
    assert validator.min_response_tokens == 50
    assert validator.max_response_tokens == 500


def test_validate_example_valid(sample_graph_db):
    """Test validating a valid example."""
    validator = DatasetValidator(sample_graph_db)

    example = {
        "id": "test_001",
        "type": "discovery",
        "prompt": "Does the codebase have a process_data function?",
        "response": "Yes, there is a process_data function that processes input data and returns results. "
        "It is located in /test.py and works with validate_input and other functions. "
        "This function handles data transformation, validation, and processing workflows. "
        "You can use it by passing your data as the first argument and it will return processed output.",
        "refs": ["node1"],
        "metadata": {"node_name": "process_data"},
    }

    is_valid, issues = validator.validate_example(example)

    assert is_valid is True
    assert len(issues) == 0


def test_validate_example_missing_field(sample_graph_db):
    """Test validating example with missing field."""
    validator = DatasetValidator(sample_graph_db)

    example = {
        "id": "test_001",
        "type": "discovery",
        "prompt": "Test question?",
        # Missing "response" field
        "refs": ["node1"],
    }

    is_valid, issues = validator.validate_example(example)

    assert is_valid is False
    assert any("Missing required field: response" in issue for issue in issues)


def test_validate_example_empty_prompt(sample_graph_db):
    """Test validating example with empty prompt."""
    validator = DatasetValidator(sample_graph_db)

    example = {
        "id": "test_001",
        "type": "discovery",
        "prompt": "   ",  # Empty/whitespace
        "response": "This is a response with enough tokens to pass the minimum length requirement.",
        "refs": ["node1"],
    }

    is_valid, issues = validator.validate_example(example)

    assert is_valid is False
    assert any("Empty prompt" in issue for issue in issues)


def test_validate_example_short_response(sample_graph_db):
    """Test validating example with too short response."""
    validator = DatasetValidator(sample_graph_db, min_response_tokens=50)

    example = {
        "id": "test_001",
        "type": "discovery",
        "prompt": "Test question?",
        "response": "Short answer.",  # Too short
        "refs": ["node1"],
    }

    is_valid, issues = validator.validate_example(example)

    assert is_valid is False
    assert any("too short" in issue.lower() for issue in issues)


def test_validate_example_placeholder(sample_graph_db):
    """Test validating example with placeholder text."""
    validator = DatasetValidator(sample_graph_db)

    example = {
        "id": "test_001",
        "type": "discovery",
        "prompt": "Test question?",
        "response": "This is a response with [TODO] placeholder that should fail validation. "
        "Adding more text to meet minimum token requirement for the test to work properly.",
        "refs": ["node1"],
    }

    is_valid, issues = validator.validate_example(example)

    assert is_valid is False
    assert any("placeholder" in issue.lower() for issue in issues)


def test_validate_example_empty_refs(sample_graph_db):
    """Test validating example with empty refs."""
    validator = DatasetValidator(sample_graph_db)

    example = {
        "id": "test_001",
        "type": "discovery",
        "prompt": "Test question?",
        "response": "This is a valid response with enough tokens to meet the minimum requirement.",
        "refs": [],  # Empty refs
    }

    is_valid, issues = validator.validate_example(example)

    assert is_valid is False
    assert any("refs list is empty" in issue for issue in issues)


def test_validate_example_invalid_type(sample_graph_db):
    """Test validating example with invalid type."""
    validator = DatasetValidator(sample_graph_db)

    example = {
        "id": "test_001",
        "type": "invalid_type",  # Invalid
        "prompt": "Test question?",
        "response": "This is a valid response with enough tokens to meet minimum requirements.",
        "refs": ["node1"],
    }

    is_valid, issues = validator.validate_example(example)

    assert is_valid is False
    assert any("Invalid type" in issue for issue in issues)


def test_validate_node_references(sample_graph_db):
    """Test validating node references."""
    gm = sample_graph_db
    validator = DatasetValidator(gm)

    # node1, node2, node3 exist from fixture
    examples = [
        {"id": "ex1", "refs": ["node1", "node2"]},
        {"id": "ex2", "refs": ["node3"]},
        {"id": "ex3", "refs": ["nonexistent"]},  # Invalid
    ]

    valid_count, invalid_refs = validator.validate_node_references(examples)

    assert valid_count == 3  # node1, node2, node3
    assert len(invalid_refs) == 1
    assert "nonexistent" in invalid_refs[0]


def test_check_duplicates(sample_graph_db):
    """Test checking for duplicates."""
    validator = DatasetValidator(sample_graph_db)

    examples = [
        {
            "id": "test_001",
            "prompt": "Question 1?",
            "response": "Answer 1",
            "refs": ["node1"],
            "type": "discovery",
        },
        {
            "id": "test_002",
            "prompt": "Question 2?",
            "response": "Answer 2",
            "refs": ["node2"],
            "type": "discovery",
        },
        {
            "id": "test_001",  # Duplicate ID
            "prompt": "Question 3?",
            "response": "Answer 3",
            "refs": ["node3"],
            "type": "discovery",
        },
        {
            "id": "test_003",
            "prompt": "Question 1?",  # Duplicate prompt
            "response": "Answer 4",
            "refs": ["node1"],
            "type": "discovery",
        },
    ]

    duplicates = validator.check_duplicates(examples)

    assert len(duplicates) >= 2  # At least duplicate ID and prompt


def test_get_distribution(sample_graph_db):
    """Test getting type distribution."""
    validator = DatasetValidator(sample_graph_db)

    examples = [
        {"id": "1", "type": "discovery", "prompt": "Q", "response": "A", "refs": ["n1"]},
        {"id": "2", "type": "discovery", "prompt": "Q", "response": "A", "refs": ["n1"]},
        {"id": "3", "type": "explain", "prompt": "Q", "response": "A", "refs": ["n1"]},
        {"id": "4", "type": "howto", "prompt": "Q", "response": "A", "refs": ["n1"]},
    ]

    distribution = validator.get_distribution(examples)

    assert distribution["discovery"] == 2
    assert distribution["explain"] == 1
    assert distribution["howto"] == 1


def test_validate_dataset(sample_graph_db):
    """Test validating entire dataset."""
    gm = sample_graph_db
    validator = DatasetValidator(gm, min_response_tokens=10, max_response_tokens=500)

    examples = [
        {
            "id": "valid_001",
            "type": "discovery",
            "prompt": "Valid question?",
            "response": "Valid response with enough tokens to pass validation checks.",
            "refs": ["node1"],
        },
        {
            "id": "invalid_001",
            "type": "invalid_type",
            "prompt": "Invalid type question?",
            "response": "Response with enough tokens to pass length check.",
            "refs": ["node1"],
        },
    ]

    is_valid, report = validator.validate_dataset(examples)

    assert report["total_examples"] == 2
    assert report["valid_examples"] == 1
    assert report["invalid_examples"] == 1
    assert is_valid is False  # Because of invalid example
    assert len(report["issues"]) > 0
