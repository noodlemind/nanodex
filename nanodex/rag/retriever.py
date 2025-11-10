"""
Semantic retrieval for code search.

Combines embedding, indexing, and search into a unified interface.
"""

from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path

from .embedder import CodeEmbedder
from .chunker import CodeChunker
from .indexer import VectorIndexer

logger = logging.getLogger(__name__)


class SemanticRetriever:
    """
    High-level interface for semantic code retrieval.

    Handles:
    - Chunking code
    - Embedding chunks
    - Indexing vectors
    - Semantic search
    - Context assembly
    """

    def __init__(
        self,
        embedder: Optional[CodeEmbedder] = None,
        chunker: Optional[CodeChunker] = None,
        indexer: Optional[VectorIndexer] = None,
        embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
        chunk_strategy: str = 'hybrid'
    ):
        """
        Initialize semantic retriever.

        Args:
            embedder: CodeEmbedder instance (created if None)
            chunker: CodeChunker instance (created if None)
            indexer: VectorIndexer instance (created if None)
            embedding_model: Model name for embedder
            chunk_strategy: Strategy for chunker
        """
        # Initialize embedder
        if embedder is None:
            self.embedder = CodeEmbedder(model_name=embedding_model)
        else:
            self.embedder = embedder

        # Initialize chunker
        if chunker is None:
            self.chunker = CodeChunker(strategy=chunk_strategy)
        else:
            self.chunker = chunker

        # Initialize indexer (lazy - needs embedding dim)
        if indexer is None:
            embedding_dim = self.embedder.get_embedding_dim()
            self.indexer = VectorIndexer(embedding_dim=embedding_dim)
        else:
            self.indexer = indexer

        logger.info("Initialized SemanticRetriever")

    def index_codebase(
        self,
        code_samples: List[Dict],
        show_progress: bool = True
    ) -> Dict:
        """
        Index an entire codebase.

        Args:
            code_samples: List of code samples from CodeAnalyzer
            show_progress: Show progress during indexing

        Returns:
            Statistics about indexing
        """
        logger.info(f"Indexing {len(code_samples)} code samples...")

        # Step 1: Chunk code
        logger.info("Step 1: Chunking code...")
        chunks = self.chunker.chunk_code_samples(code_samples)
        chunk_stats = self.chunker.get_chunk_summary(chunks)

        if not chunks:
            logger.warning("No chunks created!")
            return {
                'success': False,
                'error': 'No chunks created',
                'chunk_stats': chunk_stats,
            }

        logger.info(f"Created {len(chunks)} chunks")

        # Step 2: Embed chunks
        logger.info("Step 2: Embedding chunks...")
        chunks = self.embedder.embed_chunks(chunks)

        # Step 3: Index chunks
        logger.info("Step 3: Adding to vector index...")
        self.indexer.add_chunks(chunks)

        # Get stats
        index_stats = self.indexer.get_stats()

        stats = {
            'success': True,
            'num_samples': len(code_samples),
            'num_chunks': len(chunks),
            'chunk_stats': chunk_stats,
            'index_stats': index_stats,
        }

        logger.info(f"Indexing complete: {len(chunks)} chunks indexed")
        return stats

    def search(
        self,
        query: str,
        k: int = 5,
        filter_type: Optional[str] = None,
        filter_language: Optional[str] = None
    ) -> List[Dict]:
        """
        Search for relevant code chunks.

        Args:
            query: Natural language query
            k: Number of results
            filter_type: Filter by chunk type ('function', 'class', etc.)
            filter_language: Filter by language ('python', 'javascript', etc.)

        Returns:
            List of relevant chunks with scores
        """
        logger.info(f"Searching for: '{query}' (k={k})")

        # Embed query
        query_embedding = self.embedder.embed_query(query)

        # Search index (get more than k for filtering)
        search_k = k * 3 if (filter_type or filter_language) else k
        results = self.indexer.search(query_embedding, k=search_k)

        # Apply filters
        if filter_type:
            results = [r for r in results if r.get('type') == filter_type]

        if filter_language:
            results = [r for r in results if r.get('language') == filter_language]

        # Limit to k results
        results = results[:k]

        logger.info(f"Found {len(results)} results")
        return results

    def search_similar_code(
        self,
        code: str,
        k: int = 5,
        exclude_exact: bool = True
    ) -> List[Dict]:
        """
        Find code similar to given code snippet.

        Args:
            code: Code snippet to find similar code for
            k: Number of results
            exclude_exact: Exclude exact matches

        Returns:
            List of similar code chunks
        """
        logger.info(f"Searching for similar code (k={k})")

        # Embed code
        code_embedding = self.embedder.embed_code(code)[0]

        # Search
        results = self.indexer.search(code_embedding, k=k+1 if exclude_exact else k)

        # Exclude exact matches if requested
        if exclude_exact:
            results = [r for r in results if r.get('content', '').strip() != code.strip()]

        # Limit to k
        results = results[:k]

        logger.info(f"Found {len(results)} similar code chunks")
        return results

    def get_context_for_query(
        self,
        query: str,
        k: int = 3,
        max_context_length: int = 2000
    ) -> str:
        """
        Get assembled context for a query (for RAG).

        Args:
            query: Query string
            k: Number of chunks to retrieve
            max_context_length: Maximum context length in characters

        Returns:
            Assembled context string
        """
        # Search for relevant chunks
        results = self.search(query, k=k)

        if not results:
            return ""

        # Assemble context
        context_parts = []
        current_length = 0

        for i, result in enumerate(results):
            # Format chunk
            chunk_text = self._format_chunk_for_context(result, index=i+1)
            chunk_length = len(chunk_text)

            # Check if adding this chunk exceeds limit
            if current_length + chunk_length > max_context_length:
                # Truncate this chunk to fit
                remaining = max_context_length - current_length
                if remaining > 100:  # Only add if meaningful space left
                    chunk_text = chunk_text[:remaining] + "\n..."
                    context_parts.append(chunk_text)
                break

            context_parts.append(chunk_text)
            current_length += chunk_length

        context = "\n\n".join(context_parts)
        logger.info(f"Assembled context: {len(context)} chars from {len(context_parts)} chunks")

        return context

    def _format_chunk_for_context(self, chunk: Dict, index: int) -> str:
        """Format a chunk for inclusion in RAG context."""
        parts = []

        # Header
        chunk_type = chunk.get('type', 'code')
        name = chunk.get('name', 'unknown')
        file_path = chunk.get('file_path', '')

        parts.append(f"[{index}] {chunk_type.upper()}: {name}")
        if file_path:
            parts.append(f"File: {file_path}")

        # Score if available
        if 'score' in chunk:
            parts.append(f"Relevance: {chunk['score']:.2f}")

        # Content
        parts.append("\n```")
        parts.append(chunk.get('content', ''))
        parts.append("```")

        return "\n".join(parts)

    def save(self, path: str):
        """
        Save retriever state to disk.

        Args:
            path: Directory to save to
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save indexer (includes chunks and metadata)
        self.indexer.save(str(save_path / 'index'))

        # Save embedder model (optional - can be large)
        # Uncomment if you want to save the embedding model
        # self.embedder.save_model(str(save_path / 'embedder'))

        logger.info(f"Saved retriever to {path}")

    def load(self, path: str):
        """
        Load retriever state from disk.

        Args:
            path: Directory to load from
        """
        load_path = Path(path)

        if not load_path.exists():
            raise FileNotFoundError(f"Retriever path does not exist: {path}")

        # Load indexer
        self.indexer.load(str(load_path / 'index'))

        # Load embedder if saved
        embedder_path = load_path / 'embedder'
        if embedder_path.exists():
            self.embedder.load_model(str(embedder_path))

        logger.info(f"Loaded retriever from {path}")

    def get_stats(self) -> Dict:
        """Get retriever statistics."""
        return {
            'indexer': self.indexer.get_stats(),
            'embedder': {
                'model': self.embedder.model_name,
                'embedding_dim': self.embedder.embedding_dim,
            },
            'chunker': {
                'strategy': self.chunker.strategy,
                'max_chunk_size': self.chunker.max_chunk_size,
            }
        }

    def clear(self):
        """Clear all indexed data."""
        self.indexer.clear()
        logger.info("Cleared retriever")


class HybridRetriever:
    """
    Hybrid retrieval combining semantic and keyword search.

    More advanced retrieval using both vector similarity and keyword matching.
    """

    def __init__(
        self,
        semantic_retriever: SemanticRetriever,
        keyword_weight: float = 0.3,
        semantic_weight: float = 0.7
    ):
        """
        Initialize hybrid retriever.

        Args:
            semantic_retriever: SemanticRetriever instance
            keyword_weight: Weight for keyword search (0-1)
            semantic_weight: Weight for semantic search (0-1)
        """
        self.semantic_retriever = semantic_retriever
        self.keyword_weight = keyword_weight
        self.semantic_weight = semantic_weight

        # Normalize weights
        total = keyword_weight + semantic_weight
        self.keyword_weight /= total
        self.semantic_weight /= total

        logger.info(
            f"Initialized HybridRetriever: "
            f"keyword={self.keyword_weight:.2f}, semantic={self.semantic_weight:.2f}"
        )

    def search(
        self,
        query: str,
        k: int = 5
    ) -> List[Dict]:
        """
        Hybrid search combining semantic and keyword matching.

        Args:
            query: Search query
            k: Number of results

        Returns:
            Ranked list of results
        """
        # Get semantic results
        semantic_results = self.semantic_retriever.search(query, k=k*2)

        # Get keyword results (simple implementation)
        keyword_results = self._keyword_search(query, k=k*2)

        # Combine and re-rank
        combined = self._combine_results(semantic_results, keyword_results)

        # Sort by combined score
        combined.sort(key=lambda x: x['combined_score'], reverse=True)

        # Return top k
        return combined[:k]

    def _keyword_search(self, query: str, k: int) -> List[Dict]:
        """
        Simple keyword search.

        Searches through indexed chunks for keyword matches.
        """
        query_lower = query.lower()
        keywords = set(query_lower.split())

        results = []

        for i, chunk in enumerate(self.semantic_retriever.indexer.chunks):
            content_lower = chunk.get('content', '').lower()

            # Count keyword matches
            matches = sum(1 for kw in keywords if kw in content_lower)

            if matches > 0:
                score = matches / len(keywords)  # Normalized score
                result = chunk.copy()
                result['keyword_score'] = score
                results.append(result)

        # Sort by score
        results.sort(key=lambda x: x['keyword_score'], reverse=True)

        return results[:k]

    def _combine_results(
        self,
        semantic_results: List[Dict],
        keyword_results: List[Dict]
    ) -> List[Dict]:
        """
        Combine and re-rank results.

        Uses weighted combination of semantic and keyword scores.
        """
        # Create mapping by file_path + name for deduplication
        result_map = {}

        # Add semantic results
        for result in semantic_results:
            key = (result.get('file_path', ''), result.get('name', ''))
            if key not in result_map:
                result_map[key] = result.copy()
                result_map[key]['semantic_score'] = result.get('score', 0)
                result_map[key]['keyword_score'] = 0

        # Add/update with keyword results
        for result in keyword_results:
            key = (result.get('file_path', ''), result.get('name', ''))
            if key in result_map:
                result_map[key]['keyword_score'] = result.get('keyword_score', 0)
            else:
                result_map[key] = result.copy()
                result_map[key]['semantic_score'] = 0
                result_map[key]['keyword_score'] = result.get('keyword_score', 0)

        # Calculate combined scores
        for result in result_map.values():
            semantic = result.get('semantic_score', 0)
            keyword = result.get('keyword_score', 0)
            result['combined_score'] = (
                self.semantic_weight * semantic +
                self.keyword_weight * keyword
            )

        return list(result_map.values())
