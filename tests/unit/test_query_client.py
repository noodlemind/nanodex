"""Unit tests for query client."""

import pytest

from nanodex.inference.client import QueryClient


def test_query_client_init():
    """Test QueryClient initialization."""
    client = QueryClient(endpoint="http://localhost:8000")

    assert client.endpoint == "http://localhost:8000"
    assert client.system_prompt is not None
    assert "helpful" in client.system_prompt.lower()


def test_query_client_custom_prompt():
    """Test QueryClient with custom system prompt."""
    custom_prompt = "Custom system prompt"
    client = QueryClient(endpoint="http://localhost:9000", system_prompt=custom_prompt)

    assert client.endpoint == "http://localhost:9000"
    assert client.system_prompt == custom_prompt


def test_query_client_endpoint_normalization():
    """Test endpoint URL normalization."""
    client = QueryClient(endpoint="http://localhost:8000/")

    # Should strip trailing slash
    assert client.endpoint == "http://localhost:8000"


def test_query_client_batch():
    """Test batch querying (without actual server)."""
    client = QueryClient()

    questions = ["Question 1?", "Question 2?"]

    # This will fail without a server, but tests the method exists
    # and accepts the right parameters
    assert hasattr(client, "query_batch")
    assert callable(client.query_batch)
