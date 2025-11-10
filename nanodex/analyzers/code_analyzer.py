"""
Code repository analyzer for extracting code context.
"""

import os
from pathlib import Path
from typing import List, Dict, Set, Optional
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

        # Deep parsing configuration
        self.deep_parsing_config = config.get('deep_parsing', {})
        self.enable_deep_parsing = self.deep_parsing_config.get('enabled', True)

        # Initialize parsers if deep parsing is enabled
        self.ast_parser = None
        if self.enable_deep_parsing:
            try:
                from .ast_parser import PythonASTParser
                self.ast_parser = PythonASTParser()
                logger.info("Deep parsing enabled with AST parser")
            except ImportError as e:
                logger.warning(f"Could not enable deep parsing: {e}")
                self.enable_deep_parsing = False
        
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
    
    def _extract_code_sample(self, file_path: Path) -> Optional[Dict]:
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
            language = self._detect_language(file_path.suffix)

            # Basic metadata
            sample = {
                'file_path': str(rel_path),
                'language': language,
                'content': content,
                'lines': len(content.splitlines())
            }

            # Add deep parsing if enabled
            if self.enable_deep_parsing:
                parsed_data = self._parse_code_structure(file_path, content, language)
                sample.update(parsed_data)

            return sample

        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return None

    def _parse_code_structure(self, file_path: Path, content: str, language: str) -> Dict:
        """
        Parse code structure based on language.

        Args:
            file_path: Path to the file
            content: File content
            language: Programming language

        Returns:
            Dictionary with parsed structure information
        """
        parsed_data = {}

        # Parse Python files with AST
        if language == 'python' and self.ast_parser:
            try:
                ast_result = self.ast_parser.parse_file(file_path, content)
                parsed_data.update({
                    'functions': ast_result.get('functions', []),
                    'classes': ast_result.get('classes', []),
                    'imports': ast_result.get('imports', []),
                    'docstrings': ast_result.get('docstrings', []),
                    'global_vars': ast_result.get('global_vars', []),
                    'complexity': ast_result.get('complexity', 1),
                })
                if 'error' in ast_result:
                    parsed_data['parse_error'] = ast_result['error']

            except Exception as e:
                logger.debug(f"Error parsing {file_path}: {e}")
                parsed_data['parse_error'] = str(e)

        # For other languages, we could add tree-sitter support here
        # For now, just return empty structures
        else:
            parsed_data.update({
                'functions': [],
                'classes': [],
                'imports': [],
                'docstrings': [],
                'global_vars': [],
                'complexity': 1,
            })

        return parsed_data
    
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
