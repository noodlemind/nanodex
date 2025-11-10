"""
Vector indexing using FAISS for efficient similarity search.
"""

from typing import List, Dict, Optional, Tuple
import logging
import numpy as np
from pathlib import Path
import json
import pickle

logger = logging.getLogger(__name__)


class VectorIndexer:
    """
    FAISS-based vector indexer for code chunks.

    Supports:
    - Flat index (exact search, good for small datasets)
    - IVF index (approximate search, good for large datasets)
    - Index persistence
    - Metadata storage
    """

    def __init__(
        self,
        embedding_dim: int,
        index_type: str = 'flat',
        metric: str = 'cosine'
    ):
        """
        Initialize vector indexer.

        Args:
            embedding_dim: Dimension of embeddings
            index_type: Type of index ('flat', 'ivf')
            metric: Distance metric ('cosine', 'l2')
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.metric = metric

        self.index = None
        self.chunks = []  # Store chunk metadata
        self.id_to_chunk = {}  # Map index ID to chunk

        logger.info(
            f"Initializing VectorIndexer: "
            f"dim={embedding_dim}, type={index_type}, metric={metric}"
        )

    def _create_index(self, num_vectors: Optional[int] = None):
        """Create FAISS index."""
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu is required for RAG functionality. "
                "Install it with: pip install faiss-cpu"
            )

        if self.index_type == 'flat':
            # Exact search - good for small datasets (<100k vectors)
            if self.metric == 'cosine':
                # Normalize vectors and use L2 distance for cosine similarity
                self.index = faiss.IndexFlatL2(self.embedding_dim)
                self.normalize_embeddings = True
            else:
                self.index = faiss.IndexFlatL2(self.embedding_dim)
                self.normalize_embeddings = False

            logger.info("Created Flat index for exact search")

        elif self.index_type == 'ivf':
            # Approximate search - good for large datasets
            if num_vectors is None:
                raise ValueError("num_vectors required for IVF index")

            # Choose number of clusters
            nlist = min(100, max(10, num_vectors // 100))

            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)

            if self.metric == 'cosine':
                self.normalize_embeddings = True
            else:
                self.normalize_embeddings = False

            logger.info(f"Created IVF index with {nlist} clusters")

        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

    def add_chunks(self, chunks: List[Dict]):
        """
        Add chunks to the index.

        Args:
            chunks: List of chunks with 'embedding' field
        """
        if not chunks:
            logger.warning("No chunks to add")
            return

        # Create index if needed
        if self.index is None:
            self._create_index(num_vectors=len(chunks))

        # Extract embeddings
        embeddings = np.array([chunk['embedding'] for chunk in chunks], dtype=np.float32)

        # Normalize if using cosine similarity
        if self.normalize_embeddings:
            faiss.normalize_L2(embeddings)

        # Train index if needed (for IVF)
        if self.index_type == 'ivf' and not self.index.is_trained:
            logger.info("Training IVF index...")
            self.index.train(embeddings)

        # Add to index
        start_id = len(self.chunks)
        self.index.add(embeddings)

        # Store metadata
        for i, chunk in enumerate(chunks):
            chunk_id = start_id + i
            # Remove embedding from stored chunk to save memory
            stored_chunk = {k: v for k, v in chunk.items() if k != 'embedding'}
            self.chunks.append(stored_chunk)
            self.id_to_chunk[chunk_id] = stored_chunk

        logger.info(f"Added {len(chunks)} chunks to index (total: {len(self.chunks)})")

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        return_scores: bool = True
    ) -> List[Dict]:
        """
        Search for similar chunks.

        Args:
            query_embedding: Query vector
            k: Number of results to return
            return_scores: Include similarity scores

        Returns:
            List of chunks with optional scores
        """
        if self.index is None or len(self.chunks) == 0:
            logger.warning("Index is empty")
            return []

        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_embedding = query_embedding.astype(np.float32)

        # Normalize if using cosine
        if self.normalize_embeddings:
            import faiss
            faiss.normalize_L2(query_embedding)

        # Search
        k = min(k, len(self.chunks))  # Don't search for more than we have
        distances, indices = self.index.search(query_embedding, k)

        # Convert to results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for invalid results
                continue

            chunk = self.id_to_chunk.get(int(idx))
            if chunk is None:
                continue

            result = chunk.copy()

            if return_scores:
                # Convert distance to similarity score
                if self.normalize_embeddings:
                    # L2 distance on normalized vectors -> cosine similarity
                    similarity = 1 - (dist / 2)
                else:
                    # L2 distance -> inverse for score
                    similarity = 1 / (1 + dist)

                result['score'] = float(similarity)
                result['rank'] = i + 1

            results.append(result)

        return results

    def save(self, path: str):
        """
        Save index and metadata to disk.

        Args:
            path: Directory path to save to
        """
        if self.index is None:
            logger.warning("No index to save")
            return

        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        try:
            import faiss
            index_file = save_path / 'index.faiss'
            faiss.write_index(self.index, str(index_file))
            logger.info(f"Saved FAISS index to {index_file}")
        except ImportError:
            logger.error("faiss-cpu not installed, cannot save index")
            return

        # Save metadata
        metadata_file = save_path / 'metadata.json'
        metadata = {
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type,
            'metric': self.metric,
            'num_chunks': len(self.chunks),
            'normalize_embeddings': self.normalize_embeddings,
        }
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save chunks
        chunks_file = save_path / 'chunks.pkl'
        with open(chunks_file, 'wb') as f:
            pickle.dump(self.chunks, f)

        logger.info(f"Saved index metadata and {len(self.chunks)} chunks")

    def load(self, path: str):
        """
        Load index and metadata from disk.

        Args:
            path: Directory path to load from
        """
        load_path = Path(path)

        if not load_path.exists():
            raise FileNotFoundError(f"Index path does not exist: {path}")

        # Load metadata
        metadata_file = load_path / 'metadata.json'
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        self.embedding_dim = metadata['embedding_dim']
        self.index_type = metadata['index_type']
        self.metric = metadata['metric']
        self.normalize_embeddings = metadata.get('normalize_embeddings', False)

        # Load FAISS index
        try:
            import faiss
            index_file = load_path / 'index.faiss'
            self.index = faiss.read_index(str(index_file))
            logger.info(f"Loaded FAISS index from {index_file}")
        except ImportError:
            raise ImportError("faiss-cpu not installed, cannot load index")

        # Load chunks
        chunks_file = load_path / 'chunks.pkl'
        with open(chunks_file, 'rb') as f:
            self.chunks = pickle.load(f)

        # Rebuild id_to_chunk mapping
        self.id_to_chunk = {i: chunk for i, chunk in enumerate(self.chunks)}

        logger.info(f"Loaded index with {len(self.chunks)} chunks")

    def get_stats(self) -> Dict:
        """Get index statistics."""
        if self.index is None:
            return {
                'is_empty': True,
                'num_chunks': 0,
            }

        return {
            'is_empty': False,
            'num_chunks': len(self.chunks),
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type,
            'metric': self.metric,
            'is_trained': self.index.is_trained if hasattr(self.index, 'is_trained') else True,
        }

    def clear(self):
        """Clear the index and all chunks."""
        self.index = None
        self.chunks = []
        self.id_to_chunk = {}
        logger.info("Cleared index")

    def remove_chunks_by_file(self, file_path: str):
        """
        Remove all chunks from a specific file.

        Note: This rebuilds the entire index, which can be slow.

        Args:
            file_path: Path of file to remove chunks from
        """
        # Find chunks to keep
        chunks_to_keep = [
            chunk for chunk in self.chunks
            if chunk.get('file_path') != file_path
        ]

        if len(chunks_to_keep) == len(self.chunks):
            logger.info(f"No chunks found for {file_path}")
            return

        # Rebuild index
        logger.info(f"Rebuilding index: removing {len(self.chunks) - len(chunks_to_keep)} chunks")

        # Clear current index
        self.clear()

        # Re-add chunks (they need embeddings, so this is best done externally)
        logger.warning(
            "Chunks were removed but need to be re-embedded and re-added. "
            "Consider rebuilding the entire index."
        )

        self.chunks = chunks_to_keep
        self.id_to_chunk = {i: chunk for i, chunk in enumerate(chunks_to_keep)}
