"""
Code repository analyzer for extracting code context.
"""

import os
from pathlib import Path
from typing import List, Dict, Set
import logging

logger = logging.getLogger(__name__)


class CodeAnalyzer:
    """Analyzes code repositories to extract training data."""
    
    def __init__(self, config: Dict):
        """
        Initialize the code analyzer.
        
        Args:
            config: Repository configuration dictionary
        """
        self.repo_path = Path(config.get('path', '.'))
        self.include_extensions = set(config.get('include_extensions', []))
        self.exclude_dirs = set(config.get('exclude_dirs', []))
        self.max_file_size = config.get('max_file_size', 1048576)
        
    def analyze(self) -> List[Dict[str, str]]:
        """
        Analyze the code repository and extract code samples.
        
        Returns:
            List of dictionaries containing code samples with metadata
        """
        code_samples = []
        
        logger.info(f"Analyzing repository at: {self.repo_path}")
        
        for file_path in self._walk_repository():
            try:
                code_sample = self._extract_code_sample(file_path)
                if code_sample:
                    code_samples.append(code_sample)
            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")
        
        logger.info(f"Extracted {len(code_samples)} code samples")
        return code_samples
    
    def _walk_repository(self):
        """Walk through the repository and yield file paths."""
        for root, dirs, files in os.walk(self.repo_path):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs]
            
            for file in files:
                file_path = Path(root) / file
                
                # Check if file extension is included
                if file_path.suffix in self.include_extensions:
                    # Check file size
                    if file_path.stat().st_size <= self.max_file_size:
                        yield file_path
    
    def _extract_code_sample(self, file_path: Path) -> Dict[str, str]:
        """
        Extract code sample from a file.
        
        Args:
            file_path: Path to the code file
            
        Returns:
            Dictionary containing code sample and metadata
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            if not content.strip():
                return None
            
            # Get relative path from repository root
            rel_path = file_path.relative_to(self.repo_path)
            
            return {
                'file_path': str(rel_path),
                'language': self._detect_language(file_path.suffix),
                'content': content,
                'lines': len(content.splitlines())
            }
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return None
    
    def _detect_language(self, extension: str) -> str:
        """
        Detect programming language from file extension.
        
        Args:
            extension: File extension
            
        Returns:
            Language name
        """
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.cc': 'cpp',
            '.h': 'c',
            '.hpp': 'cpp',
        }
        return language_map.get(extension.lower(), 'unknown')
    
    def get_statistics(self, code_samples: List[Dict[str, str]]) -> Dict:
        """
        Get statistics about the analyzed code.
        
        Args:
            code_samples: List of code samples
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_files': len(code_samples),
            'total_lines': sum(s['lines'] for s in code_samples),
            'languages': {},
        }
        
        for sample in code_samples:
            lang = sample['language']
            if lang not in stats['languages']:
                stats['languages'][lang] = {'files': 0, 'lines': 0}
            stats['languages'][lang]['files'] += 1
            stats['languages'][lang]['lines'] += sample['lines']
        
        return stats
