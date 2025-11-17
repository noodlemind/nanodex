"""Query client for inference server."""

import json
import logging
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class QueryClient:
    """Client for querying vLLM inference server."""

    def __init__(
        self,
        endpoint: str = "http://localhost:8000",
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize query client.

        Args:
            endpoint: Server endpoint URL
            system_prompt: System prompt for context
        """
        self.endpoint = endpoint.rstrip("/")
        self.system_prompt = (
            system_prompt or "You are a helpful code assistant specialized in this codebase."
        )

    def query(
        self,
        question: str,
        max_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.9,
    ) -> Dict:
        """
        Query the inference server.

        Args:
            question: User question
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            top_p: Top-p sampling

        Returns:
            Response dictionary with answer and metadata
        """
        logger.info(f"Querying: {question}")

        # Build chat messages
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]

        # Build request payload
        payload = {
            "model": "nanodex",  # LoRA adapter name
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        try:
            # Send request to vLLM OpenAI-compatible API
            response = requests.post(
                f"{self.endpoint}/v1/chat/completions",
                json=payload,
                timeout=60,
            )

            response.raise_for_status()
            result = response.json()

            # Extract answer
            answer = result["choices"][0]["message"]["content"]

            logger.info(f"Answer: {answer[:100]}...")

            return {
                "question": question,
                "answer": answer,
                "usage": result.get("usage", {}),
                "model": result.get("model", ""),
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"Query failed: {e}")
            return {
                "question": question,
                "answer": None,
                "error": str(e),
            }

    def query_batch(
        self,
        questions: List[str],
        max_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.9,
    ) -> List[Dict]:
        """
        Query multiple questions.

        Args:
            questions: List of questions
            max_tokens: Maximum tokens per response
            temperature: Sampling temperature
            top_p: Top-p sampling

        Returns:
            List of response dictionaries
        """
        results = []

        for question in questions:
            result = self.query(
                question=question,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            results.append(result)

        return results

    def health_check(self) -> bool:
        """
        Check if server is healthy.

        Returns:
            True if server is reachable and healthy
        """
        try:
            response = requests.get(f"{self.endpoint}/health", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def get_models(self) -> List[str]:
        """
        Get available models from server.

        Returns:
            List of model names
        """
        try:
            response = requests.get(f"{self.endpoint}/v1/models", timeout=5)
            response.raise_for_status()
            result = response.json()
            return [model["id"] for model in result.get("data", [])]
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get models: {e}")
            return []
