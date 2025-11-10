"""
Configuration management for Turbo Code GPT.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class Config:
    """Load and manage configuration from YAML file."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation, e.g., 'model.huggingface.model_name')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_model_source(self) -> str:
        """Get the configured model source (ollama or huggingface)."""
        return self.get('model_source', 'huggingface')
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration based on the selected source."""
        source = self.get_model_source()
        return self.get(f'model.{source}', {})
    
    def get_repository_config(self) -> Dict[str, Any]:
        """
        Get repository analysis configuration with path validation.

        Returns:
            Repository configuration dictionary with validated path

        Raises:
            ValueError: If repository path is invalid
        """
        repo_config = self.get('repository', {}).copy()

        # Get and validate repository path
        repo_path = repo_config.get('path', '.')

        # Resolve to absolute path
        repo_path = Path(repo_path).resolve()

        # Security: Validate path exists
        if not repo_path.exists():
            raise ValueError(
                f"Repository path does not exist: {repo_path}\n"
                "Please check your configuration and ensure the path is correct."
            )

        # Security: Validate it's a directory
        if not repo_path.is_dir():
            raise ValueError(
                f"Repository path is not a directory: {repo_path}\n"
                "Please provide a valid directory path."
            )

        # Update config with validated absolute path
        repo_config['path'] = str(repo_path)

        logger.debug(f"Validated repository path: {repo_path}")

        return repo_config
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.get('training', {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data preparation configuration."""
        return self.get('data', {})
    
    def get_export_config(self) -> Dict[str, Any]:
        """Get model export configuration."""
        return self.get('export', {})
