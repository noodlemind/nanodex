"""
Code embedding using sentence transformers.

Converts code snippets into dense vector representations for semantic search.
"""

from typing import List, Dict, Optional, Union
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class CodeEmbedder:
    """
    Embed code using sentence transformers.

    Supports multiple embedding models optimized for code:
    - microsoft/codebert-base
    - sentence-transformers/all-MiniLM-L6-v2
    - Custom models
    """

    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', device: str = 'cpu'):
        """
        Initialize code embedder.

        Args:
            model_name: HuggingFace model name for embedding
            device: Device to run on ('cpu', 'cuda', 'mps')
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.embedding_dim = None

        logger.info(f"Initializing CodeEmbedder with model: {model_name}")

    def _load_model(self):
        """Lazy load the embedding model."""
        if self.model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for RAG functionality. "
                "Install it with: pip install sentence-transformers"
            )

        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")

    def embed_code(self, code: Union[str, List[str]]) -> np.ndarray:
        """
        Embed code snippet(s) into vector representation(s).

        Args:
            code: Single code string or list of code strings

        Returns:
            Numpy array of embeddings, shape (n, embedding_dim)
        """
        self._load_model()

        if isinstance(code, str):
            code = [code]

        # Preprocess code for better embedding
        processed_code = [self._preprocess_code(c) for c in code]

        # Generate embeddings
        embeddings = self.model.encode(
            processed_code,
            convert_to_numpy=True,
            show_progress_bar=len(processed_code) > 100
        )

        return embeddings

    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Embed a list of code chunks with metadata.

        Args:
            chunks: List of chunk dicts with 'content', 'type', etc.

        Returns:
            Chunks with added 'embedding' field
        """
        self._load_model()

        # Extract code content
        code_texts = [chunk['content'] for chunk in chunks]

        # Generate embeddings
        embeddings = self.embed_code(code_texts)

        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding

        logger.info(f"Embedded {len(chunks)} chunks")
        return chunks

    def _preprocess_code(self, code: str) -> str:
        """
        Preprocess code for better embedding.

        Removes excessive whitespace while preserving structure.

        Args:
            code: Raw code string

        Returns:
            Preprocessed code
        """
        # Remove leading/trailing whitespace
        code = code.strip()

        # Limit length (embedding models have max length)
        max_length = 512  # tokens, approximately
        if len(code) > max_length * 4:  # rough char estimate
            code = code[:max_length * 4]
            logger.debug("Truncated long code snippet for embedding")

        return code

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a natural language query for code search.

        Args:
            query: Natural language query

        Returns:
            Query embedding
        """
        self._load_model()

        embedding = self.model.encode([query], convert_to_numpy=True)
        return embedding[0]

    def get_embedding_dim(self) -> int:
        """Get the embedding dimension."""
        if self.embedding_dim is None:
            self._load_model()
        return self.embedding_dim

    def batch_embed(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Embed texts in batches for efficiency.

        Args:
            texts: List of text strings to embed
            batch_size: Batch size for encoding
            show_progress: Show progress bar

        Returns:
            Numpy array of embeddings
        """
        self._load_model()

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=show_progress
        )

        return embeddings

    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Cosine similarity score (0-1)
        """
        # Normalize
        norm1 = embedding1 / np.linalg.norm(embedding1)
        norm2 = embedding2 / np.linalg.norm(embedding2)

        # Cosine similarity
        similarity = np.dot(norm1, norm2)

        return float(similarity)

    def save_model(self, path: str):
        """Save the embedding model to disk."""
        self._load_model()
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save(str(save_path))
        logger.info(f"Saved embedding model to {path}")

    def load_model(self, path: str):
        """Load embedding model from disk."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. "
                "Install it with: pip install sentence-transformers"
            )

        logger.info(f"Loading embedding model from {path}")
        self.model = SentenceTransformer(path, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded from {path}")
