"""Unit tests for QAGenerator."""

import json
from pathlib import Path

import pytest

from nanodex.dataset.qa_generator import QAGenerator


@pytest.fixture
def qa_generator(sample_graph_db, temp_dir):
    """Create QA generator with sample data."""
    gm = sample_graph_db

    # Add more test nodes
    gm.add_node("cap1", "capability", "process_data", path="/test.py", lang="python")
    gm.add_node("cap2", "capability", "validate_input", path="/test.py", lang="python")
    gm.add_node("concept1", "concept", "DataModel", path="/test.py", lang="python")
    gm.add_node("error1", "error", "ValueError", path="/test.py", lang="python")
    gm.add_node("recipe1", "recipe", "main", path="/test.py", lang="python")

    # Create summaries
    summary_dir = temp_dir / "summaries"
    summary_dir.mkdir()

    summaries = [
        {
            "id": "cap1",
            "type": "capability",
            "name": "process_data",
            "summary": "Function 'process_data' processes input data and returns results.",
            "context": {"file": "/test.py", "neighbors": ["validate_input"]},
        },
        {
            "id": "cap2",
            "type": "capability",
            "name": "validate_input",
            "summary": "Function 'validate_input' validates user input.",
            "context": {"file": "/test.py", "neighbors": ["process_data"]},
        },
        {
            "id": "concept1",
            "type": "concept",
            "name": "DataModel",
            "summary": "Class 'DataModel' represents data structure.",
            "context": {"file": "/test.py", "neighbors": []},
        },
        {
            "id": "error1",
            "type": "error",
            "name": "ValueError",
            "summary": "Error type 'ValueError' for invalid values.",
            "context": {"file": "/test.py", "neighbors": []},
        },
        {
            "id": "recipe1",
            "type": "recipe",
            "name": "main",
            "summary": "Function 'main' demonstrates usage.",
            "context": {"file": "/test.py", "neighbors": ["process_data"]},
        },
    ]

    for summary in summaries:
        with open(summary_dir / f"{summary['id']}.json", "w") as f:
            json.dump(summary, f)

    return QAGenerator(gm, summary_dir)


def test_qa_generator_init(qa_generator):
    """Test QA generator initialization."""
    assert qa_generator.gm is not None
    assert len(qa_generator.summaries_cache) >= 5


def test_generate_discovery_questions(qa_generator):
    """Test generating discovery questions."""
    qa_pairs = qa_generator.generate_discovery_questions(count=2)

    assert len(qa_pairs) >= 1
    for qa in qa_pairs:
        assert qa["type"] == "discovery"
        assert "prompt" in qa
        assert "response" in qa
        assert "refs" in qa
        assert len(qa["refs"]) > 0
        assert qa["id"].startswith("discovery_")


def test_generate_explain_questions(qa_generator):
    """Test generating explain questions."""
    qa_pairs = qa_generator.generate_explain_questions(count=2)

    assert len(qa_pairs) >= 1
    for qa in qa_pairs:
        assert qa["type"] == "explain"
        assert "prompt" in qa
        assert "response" in qa
        assert "refs" in qa
        assert "How" in qa["prompt"] or "What" in qa["prompt"]


def test_generate_howto_questions(qa_generator):
    """Test generating howto questions."""
    qa_pairs = qa_generator.generate_howto_questions(count=2)

    assert len(qa_pairs) >= 1
    for qa in qa_pairs:
        assert qa["type"] == "howto"
        assert "prompt" in qa
        assert "response" in qa
        prompt_lower = qa["prompt"].lower()
        assert "how" in prompt_lower or "use" in prompt_lower or "example" in prompt_lower


def test_generate_diagnostic_questions(qa_generator):
    """Test generating diagnostic questions."""
    qa_pairs = qa_generator.generate_diagnostic_questions(count=1)

    assert len(qa_pairs) >= 1
    for qa in qa_pairs:
        assert qa["type"] == "diagnostics"
        assert "prompt" in qa
        assert "response" in qa
        assert "error" in qa["prompt"].lower() or "Error" in qa["prompt"]


def test_generate_negative_examples(qa_generator):
    """Test generating negative examples."""
    positive_examples = [
        {
            "id": "test_001",
            "type": "discovery",
            "prompt": "Does X exist?",
            "response": "Yes, X exists.",
            "refs": ["cap1"],
            "metadata": {},
        }
    ]

    negatives = qa_generator.generate_negative_examples(
        positive_examples, negatives_per_positive=2
    )

    assert len(negatives) == 2
    for neg in negatives:
        assert "_neg_" in neg["id"]
        assert neg["type"] == "discovery"
        assert neg["prompt"] == "Does X exist?"
        assert "No" in neg["response"] or "not" in neg["response"].lower()
        assert neg["metadata"]["is_negative"] is True


def test_generate_all_qa(qa_generator):
    """Test generating all Q&A categories."""
    counts = {
        "discovery": 2,
        "explain": 2,
        "howto": 1,
        "diagnostics": 1,
    }

    all_qa = qa_generator.generate_all_qa(counts, negatives_per_positive=1)

    # Should have positives + negatives
    # Total positives = 6, negatives = 6, total = 12
    assert len(all_qa) >= 6  # At least the positives

    # Check type distribution
    types = [qa["type"] for qa in all_qa]
    assert "discovery" in types
    assert "explain" in types
    assert "howto" in types
    assert "diagnostics" in types


def test_qa_metadata(qa_generator):
    """Test that Q&A pairs have proper metadata."""
    qa_pairs = qa_generator.generate_discovery_questions(count=1)

    assert len(qa_pairs) > 0
    qa = qa_pairs[0]

    assert "metadata" in qa
    assert "node_name" in qa["metadata"]
    assert "node_type" in qa["metadata"]
    assert "file" in qa["metadata"]


def test_qa_refs_valid(qa_generator):
    """Test that Q&A refs point to valid nodes."""
    qa_pairs = qa_generator.generate_discovery_questions(count=2)

    for qa in qa_pairs:
        for ref in qa["refs"]:
            # Should be able to get this node
            node = qa_generator.gm.get_node(ref)
            assert node is not None or ref in qa_generator.summaries_cache
