"""
Tests for nanodex modules.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import yaml
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nanodex.utils import Config
from nanodex.analyzers import CodeAnalyzer


class TestConfig(unittest.TestCase):
    """Test configuration loading."""
    
    def setUp(self):
        """Create a temporary config file."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.yaml"
        
        config_data = {
            'model_source': 'huggingface',
            'model': {
                'huggingface': {
                    'model_name': 'test-model',
                    'use_4bit': True
                }
            },
            'repository': {
                'path': '.',
                'include_extensions': ['.py', '.js']
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config_data, f)
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def test_load_config(self):
        """Test loading configuration."""
        config = Config(str(self.config_path))
        self.assertEqual(config.get_model_source(), 'huggingface')
    
    def test_get_model_config(self):
        """Test getting model configuration."""
        config = Config(str(self.config_path))
        model_config = config.get_model_config()
        self.assertEqual(model_config['model_name'], 'test-model')
        self.assertTrue(model_config['use_4bit'])
    
    def test_get_repository_config(self):
        """Test getting repository configuration."""
        config = Config(str(self.config_path))
        repo_config = config.get_repository_config()
        self.assertEqual(repo_config['path'], '.')
        self.assertIn('.py', repo_config['include_extensions'])


class TestCodeAnalyzer(unittest.TestCase):
    """Test code analyzer."""
    
    def setUp(self):
        """Create a temporary repository with test files."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test Python file
        test_py = Path(self.temp_dir) / "test.py"
        test_py.write_text("def hello():\n    print('Hello, world!')\n")
        
        # Create test JavaScript file
        test_js = Path(self.temp_dir) / "test.js"
        test_js.write_text("function hello() {\n    console.log('Hello, world!');\n}\n")
        
        # Create excluded directory
        excluded = Path(self.temp_dir) / "node_modules"
        excluded.mkdir()
        (excluded / "package.js").write_text("// Should be excluded")
        
        self.config = {
            'path': str(self.temp_dir),
            'include_extensions': ['.py', '.js'],
            'exclude_dirs': ['node_modules'],
            'max_file_size': 1048576
        }
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def test_analyze_repository(self):
        """Test analyzing a repository."""
        analyzer = CodeAnalyzer(self.config)
        code_samples = analyzer.analyze()
        
        # Should find 2 files (test.py and test.js)
        self.assertEqual(len(code_samples), 2)
        
        # Check that both files are found
        file_paths = [s['file_path'] for s in code_samples]
        self.assertIn('test.py', file_paths)
        self.assertIn('test.js', file_paths)
    
    def test_language_detection(self):
        """Test language detection."""
        analyzer = CodeAnalyzer(self.config)
        code_samples = analyzer.analyze()
        
        for sample in code_samples:
            if sample['file_path'] == 'test.py':
                self.assertEqual(sample['language'], 'python')
            elif sample['file_path'] == 'test.js':
                self.assertEqual(sample['language'], 'javascript')
    
    def test_exclude_directories(self):
        """Test that excluded directories are ignored."""
        analyzer = CodeAnalyzer(self.config)
        code_samples = analyzer.analyze()
        
        # Should not find files in node_modules
        file_paths = [s['file_path'] for s in code_samples]
        self.assertNotIn('node_modules/package.js', file_paths)
    
    def test_get_statistics(self):
        """Test getting repository statistics."""
        analyzer = CodeAnalyzer(self.config)
        code_samples = analyzer.analyze()
        stats = analyzer.get_statistics(code_samples)
        
        self.assertEqual(stats['total_files'], 2)
        self.assertIn('python', stats['languages'])
        self.assertIn('javascript', stats['languages'])


if __name__ == '__main__':
    unittest.main()
