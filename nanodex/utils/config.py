"""
Configuration management for nanodex.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any
from pydantic import ValidationError

from .schemas import NanodexConfig

logger = logging.getLogger(__name__)


class Config:
    """Load and manage configuration from YAML file with Pydantic validation."""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration with validation.

        Args:
            config_path: Path to the YAML configuration file

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValidationError: If config validation fails
        """
        self.config_path = Path(config_path)
        self.config_dict = self._load_yaml()
        self.config = self._validate_config()
    
    def _load_yaml(self) -> Dict[str, Any]:
        """Load raw YAML configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}\n"
                f"Please create a config.yaml file. See config.example.yaml for reference."
            )

        try:
            with open(self.config_path, 'r') as f:
                config_dict = yaml.safe_load(f)

            if not config_dict:
                raise ValueError(
                    f"Configuration file is empty: {self.config_path}\n"
                    "Please provide valid configuration. See config.example.yaml for reference."
                )

            return config_dict

        except yaml.YAMLError as e:
            raise ValueError(
                f"Invalid YAML syntax in {self.config_path}:\n{e}\n"
                "Please check your configuration file for syntax errors."
            )

    def _validate_config(self) -> NanodexConfig:
        """
        Validate configuration using Pydantic schemas.

        Returns:
            Validated NanodexConfig object

        Raises:
            ValidationError: If validation fails with detailed error messages
        """
        try:
            validated_config = NanodexConfig(**self.config_dict)
            logger.info("Configuration validated successfully")
            return validated_config

        except ValidationError as e:
            # Format validation errors in a user-friendly way
            error_lines = ["Configuration validation failed:", ""]

            for error in e.errors():
                field_path = " -> ".join(str(x) for x in error['loc'])
                error_msg = error['msg']
                error_type = error['type']

                error_lines.append(f"  Field: {field_path}")
                error_lines.append(f"  Error: {error_msg}")
                error_lines.append(f"  Type: {error_type}")
                error_lines.append("")

            error_lines.append(f"Please fix the errors in {self.config_path}")
            error_lines.append("See config.example.yaml for reference.")

            raise ValidationError.from_exception_data(
                title="Configuration Validation Error",
                line_errors=e.errors()
            ) from e
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key using dot notation.

        Args:
            key: Configuration key (supports dot notation, e.g., 'model.huggingface.model_name')
            default: Default value if key not found

        Returns:
            Configuration value

        Examples:
            >>> config.get('model_source')
            'huggingface'
            >>> config.get('model.huggingface.model_name')
            'deepseek-ai/deepseek-coder-6.7b-base'
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if hasattr(value, k):
                value = getattr(value, k)
            elif isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value
    
    def get_model_source(self) -> str:
        """Get the configured model source (ollama or huggingface)."""
        return self.config.model_source

    def get_model_config(self) -> Dict[str, Any]:
        """
        Get model configuration based on the selected source.

        Returns:
            Model configuration as dictionary
        """
        source = self.get_model_source()

        if source == 'huggingface':
            return self.config.model.huggingface.model_dump()
        elif source == 'ollama':
            return self.config.model.ollama.model_dump()
        else:
            raise ValueError(f"Unknown model source: {source}")
    
    def get_repository_config(self) -> Dict[str, Any]:
        """
        Get repository analysis configuration.

        Path validation is handled by Pydantic schema during config loading.

        Returns:
            Repository configuration dictionary with validated path
        """
        # Path is already validated by Pydantic schema
        return self.config.repository.model_dump()
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.config.training.model_dump()

    def get_data_config(self) -> Dict[str, Any]:
        """Get data preparation configuration."""
        return self.config.data.model_dump()

    def get_export_config(self) -> Dict[str, Any]:
        """Get model export configuration."""
        return self.config.export.model_dump()
