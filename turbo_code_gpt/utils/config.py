"""
Configuration management for Turbo Code GPT.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


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
        """Get repository analysis configuration."""
        return self.get('repository', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.get('training', {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data preparation configuration."""
        return self.get('data', {})
    
    def get_export_config(self) -> Dict[str, Any]:
        """Get model export configuration."""
        return self.get('export', {})
