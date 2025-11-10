"""
Intelligent code chunking for RAG.

Chunks code based on semantic boundaries (functions, classes) rather than
arbitrary character limits.
"""

from typing import List, Dict, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class CodeChunker:
    """
    Intelligently chunk code for RAG indexing.

    Strategies:
    - Function-level: Each function is a chunk
    - Class-level: Each class is a chunk
    - File-level: Entire file is a chunk (for small files)
    - Hybrid: Mix of strategies based on code structure
    """

    def __init__(
        self,
        strategy: str = 'hybrid',
        max_chunk_size: int = 1000,
        min_chunk_size: int = 50,
        include_context: bool = True
    ):
        """
        Initialize chunker.

        Args:
            strategy: Chunking strategy ('function', 'class', 'file', 'hybrid')
            max_chunk_size: Maximum chunk size in characters
            min_chunk_size: Minimum chunk size in characters
            include_context: Include surrounding context in chunks
        """
        self.strategy = strategy
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.include_context = include_context

        logger.info(f"Initialized CodeChunker with strategy: {strategy}")

    def chunk_code_samples(self, code_samples: List[Dict]) -> List[Dict]:
        """
        Chunk a list of code samples.

        Args:
            code_samples: List of code samples with parsed structure

        Returns:
            List of code chunks with metadata
        """
        all_chunks = []

        for sample in code_samples:
            chunks = self.chunk_sample(sample)
            all_chunks.extend(chunks)

        logger.info(f"Created {len(all_chunks)} chunks from {len(code_samples)} samples")
        return all_chunks

    def chunk_sample(self, code_sample: Dict) -> List[Dict]:
        """
        Chunk a single code sample.

        Args:
            code_sample: Code sample with parsed structure

        Returns:
            List of chunks
        """
        if self.strategy == 'function':
            return self._chunk_by_function(code_sample)
        elif self.strategy == 'class':
            return self._chunk_by_class(code_sample)
        elif self.strategy == 'file':
            return self._chunk_by_file(code_sample)
        elif self.strategy == 'hybrid':
            return self._chunk_hybrid(code_sample)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _chunk_by_function(self, code_sample: Dict) -> List[Dict]:
        """Chunk by functions."""
        chunks = []
        parsed = code_sample.get('parsed', {})
        functions = parsed.get('functions', [])

        for func in functions:
            body = func.get('body', '')

            # Skip if too small or too large
            if len(body) < self.min_chunk_size:
                continue
            if len(body) > self.max_chunk_size:
                body = body[:self.max_chunk_size]

            # Create chunk
            chunk = {
                'content': body,
                'type': 'function',
                'name': func.get('name', ''),
                'file_path': code_sample.get('file_path', ''),
                'language': code_sample.get('language', ''),
                'metadata': {
                    'docstring': func.get('docstring', ''),
                    'args': func.get('args', []),
                    'returns': func.get('returns'),
                    'lineno': func.get('lineno'),
                    'complexity': parsed.get('complexity', 1),
                }
            }

            # Add context if requested
            if self.include_context:
                chunk['metadata']['file_context'] = code_sample.get('content', '')[:500]

            chunks.append(chunk)

        return chunks

    def _chunk_by_class(self, code_sample: Dict) -> List[Dict]:
        """Chunk by classes."""
        chunks = []
        parsed = code_sample.get('parsed', {})
        classes = parsed.get('classes', [])

        for cls in classes:
            body = cls.get('body', '')

            # Skip if too small or too large
            if len(body) < self.min_chunk_size:
                continue
            if len(body) > self.max_chunk_size:
                body = body[:self.max_chunk_size]

            # Create chunk
            chunk = {
                'content': body,
                'type': 'class',
                'name': cls.get('name', ''),
                'file_path': code_sample.get('file_path', ''),
                'language': code_sample.get('language', ''),
                'metadata': {
                    'docstring': cls.get('docstring', ''),
                    'methods': [m.get('name') for m in cls.get('methods', [])],
                    'bases': cls.get('bases', []),
                    'lineno': cls.get('lineno'),
                }
            }

            # Add context
            if self.include_context:
                chunk['metadata']['file_context'] = code_sample.get('content', '')[:500]

            chunks.append(chunk)

        return chunks

    def _chunk_by_file(self, code_sample: Dict) -> List[Dict]:
        """Chunk by entire file (for small files)."""
        content = code_sample.get('content', '')

        # If file is too large, skip or split
        if len(content) > self.max_chunk_size:
            # Split into multiple chunks
            return self._split_large_content(content, code_sample)

        # Create single chunk for file
        chunk = {
            'content': content,
            'type': 'file',
            'name': Path(code_sample.get('file_path', '')).name,
            'file_path': code_sample.get('file_path', ''),
            'language': code_sample.get('language', ''),
            'metadata': {
                'total_lines': len(content.split('\n')),
                'num_functions': len(code_sample.get('parsed', {}).get('functions', [])),
                'num_classes': len(code_sample.get('parsed', {}).get('classes', [])),
            }
        }

        return [chunk]

    def _chunk_hybrid(self, code_sample: Dict) -> List[Dict]:
        """
        Hybrid chunking strategy.

        - If file is small: chunk by file
        - If file has functions/classes: chunk by function/class
        - Otherwise: split into fixed-size chunks
        """
        content = code_sample.get('content', '')
        parsed = code_sample.get('parsed', {})

        # Small file: use file-level
        if len(content) < self.max_chunk_size:
            return self._chunk_by_file(code_sample)

        # Has functions: use function-level
        functions = parsed.get('functions', [])
        if functions:
            chunks = self._chunk_by_function(code_sample)
            if chunks:
                return chunks

        # Has classes: use class-level
        classes = parsed.get('classes', [])
        if classes:
            chunks = self._chunk_by_class(code_sample)
            if chunks:
                return chunks

        # Fallback: split into fixed-size chunks
        return self._split_large_content(content, code_sample)

    def _split_large_content(self, content: str, code_sample: Dict) -> List[Dict]:
        """
        Split large content into fixed-size chunks.

        Tries to split on logical boundaries (newlines).
        """
        chunks = []
        lines = content.split('\n')

        current_chunk = []
        current_size = 0

        for line in lines:
            line_size = len(line) + 1  # +1 for newline

            if current_size + line_size > self.max_chunk_size and current_chunk:
                # Save current chunk
                chunk_content = '\n'.join(current_chunk)
                if len(chunk_content) >= self.min_chunk_size:
                    chunks.append({
                        'content': chunk_content,
                        'type': 'chunk',
                        'name': f"{Path(code_sample.get('file_path', '')).name}_chunk_{len(chunks)}",
                        'file_path': code_sample.get('file_path', ''),
                        'language': code_sample.get('language', ''),
                        'metadata': {
                            'chunk_index': len(chunks),
                            'is_partial': True,
                        }
                    })

                # Start new chunk
                current_chunk = [line]
                current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size

        # Add final chunk
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            if len(chunk_content) >= self.min_chunk_size:
                chunks.append({
                    'content': chunk_content,
                    'type': 'chunk',
                    'name': f"{Path(code_sample.get('file_path', '')).name}_chunk_{len(chunks)}",
                    'file_path': code_sample.get('file_path', ''),
                    'language': code_sample.get('language', ''),
                    'metadata': {
                        'chunk_index': len(chunks),
                        'is_partial': True,
                    }
                })

        return chunks

    def get_chunk_summary(self, chunks: List[Dict]) -> Dict:
        """
        Get summary statistics about chunks.

        Args:
            chunks: List of chunks

        Returns:
            Summary dict
        """
        if not chunks:
            return {
                'total_chunks': 0,
                'avg_chunk_size': 0,
                'types': {},
            }

        sizes = [len(chunk['content']) for chunk in chunks]
        types = {}

        for chunk in chunks:
            chunk_type = chunk['type']
            types[chunk_type] = types.get(chunk_type, 0) + 1

        return {
            'total_chunks': len(chunks),
            'avg_chunk_size': sum(sizes) / len(sizes),
            'min_chunk_size': min(sizes),
            'max_chunk_size': max(sizes),
            'types': types,
        }
