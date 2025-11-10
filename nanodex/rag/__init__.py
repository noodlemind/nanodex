"""
RAG (Retrieval-Augmented Generation) infrastructure for code understanding.

This module provides:
- Code embedding using sentence transformers
- Vector indexing with FAISS
- Intelligent code chunking (by function/class)
- Semantic search and retrieval
- Hybrid search (keyword + semantic)
- Context assembly for RAG-augmented inference
"""

from .embedder import CodeEmbedder
from .chunker import CodeChunker
from .indexer import VectorIndexer
from .retriever import SemanticRetriever

__all__ = [
    'CodeEmbedder',
    'CodeChunker',
    'VectorIndexer',
    'SemanticRetriever',
]
