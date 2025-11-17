"""Optional embedding generator for semantic search."""

import json
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class Embedder:
    """Generate embeddings for node summaries (optional feature)."""

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize embedder.

        Args:
            model_name: Name of embedding model (e.g., sentence-transformers model)
        """
        self.model_name = model_name
        self.model = None

        if model_name:
            try:
                # Lazy import to make this optional
                from sentence_transformers import SentenceTransformer

                logger.info(f"Loading embedding model: {model_name}")
                self.model = SentenceTransformer(model_name)
                logger.info("Embedding model loaded successfully")
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed. Install with: "
                    "pip install sentence-transformers"
                )
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")

    def embed_summaries(self, summary_dir: Path) -> int:
        """
        Generate embeddings for all summaries.

        Args:
            summary_dir: Directory containing summary JSON files

        Returns:
            Number of embeddings generated
        """
        if not self.model:
            logger.warning("No embedding model loaded, skipping embedding generation")
            return 0

        summary_files = list(summary_dir.glob("*.json"))
        logger.info(f"Generating embeddings for {len(summary_files)} summaries")

        embedded = 0

        # Process in batches for efficiency
        batch_size = 32
        for i in range(0, len(summary_files), batch_size):
            batch_files = summary_files[i : i + batch_size]
            batch_texts = []
            batch_data = []

            for summary_file in batch_files:
                try:
                    with open(summary_file) as f:
                        data = json.load(f)

                    # Skip if already has embedding
                    if "embedding" in data:
                        continue

                    summary_text = data.get("summary", "")
                    batch_texts.append(summary_text)
                    batch_data.append((summary_file, data))

                except Exception as e:
                    logger.error(f"Failed to read {summary_file}: {e}")

            if not batch_texts:
                continue

            # Generate embeddings for batch
            try:
                embeddings = self.model.encode(batch_texts, show_progress_bar=False)

                # Save embeddings back to files
                for (summary_file, data), embedding in zip(batch_data, embeddings):
                    data["embedding"] = embedding.tolist()

                    with open(summary_file, "w") as f:
                        json.dump(data, f, indent=2)

                    embedded += 1

                if embedded % 100 == 0:
                    logger.info(f"Progress: {embedded} embeddings generated")

            except Exception as e:
                logger.error(f"Failed to generate embeddings for batch: {e}")

        logger.info(f"Embedding generation complete: {embedded} embeddings")
        return embedded

    def search_similar(
        self, query: str, summary_dir: Path, top_k: int = 5
    ) -> List[tuple[str, float]]:
        """
        Search for similar nodes using embedding similarity.

        Args:
            query: Query text
            summary_dir: Directory containing summary JSON files
            top_k: Number of results to return

        Returns:
            List of (node_id, similarity_score) tuples
        """
        if not self.model:
            logger.error("No embedding model loaded")
            return []

        # Get query embedding
        query_embedding = self.model.encode([query])[0]

        # Load all embeddings
        similarities = []
        for summary_file in summary_dir.glob("*.json"):
            try:
                with open(summary_file) as f:
                    data = json.load(f)

                if "embedding" not in data:
                    continue

                node_embedding = data["embedding"]

                # Compute cosine similarity
                similarity = self._cosine_similarity(query_embedding, node_embedding)
                similarities.append((data["id"], similarity))

            except Exception as e:
                logger.error(f"Failed to load {summary_file}: {e}")

        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import numpy as np

        a = np.array(a)
        b = np.array(b)

        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def embed_all_summaries(summary_dir: Path, model_name: str) -> int:
    """
    Generate embeddings for all summaries.

    Args:
        summary_dir: Directory with summary JSON files
        model_name: Embedding model name

    Returns:
        Number of embeddings generated
    """
    embedder = Embedder(model_name)
    return embedder.embed_summaries(summary_dir)
