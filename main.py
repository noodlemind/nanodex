#!/usr/bin/env python3
"""
Main CLI for nanodex.

This script orchestrates the entire pipeline:
1. Analyze code repository
2. Prepare training data
3. Load and fine-tune model
4. Export model for chatbot use
"""

import argparse
import logging
import sys
from pathlib import Path

from nanodex.utils import Config
from nanodex.analyzers import CodeAnalyzer
from nanodex.trainers import DataPreparer, ModelTrainer
from nanodex.models import ModelLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_repository(config: Config):
    """
    Analyze code repository and extract code samples.
    
    Args:
        config: Configuration object
    """
    logger.info("=" * 60)
    logger.info("STEP 1: Analyzing Code Repository")
    logger.info("=" * 60)
    
    repo_config = config.get_repository_config()
    analyzer = CodeAnalyzer(repo_config)
    
    code_samples = analyzer.analyze()
    stats = analyzer.get_statistics(code_samples)
    
    logger.info("\nRepository Statistics:")
    logger.info(f"  Total files: {stats['total_files']}")
    logger.info(f"  Total lines: {stats['total_lines']}")
    logger.info(f"  Languages:")
    for lang, lang_stats in stats['languages'].items():
        logger.info(f"    {lang}: {lang_stats['files']} files, {lang_stats['lines']} lines")
    
    return code_samples


def prepare_training_data(config: Config, code_samples):
    """
    Prepare training data from code samples.
    
    Args:
        config: Configuration object
        code_samples: List of code samples
    """
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Preparing Training Data")
    logger.info("=" * 60)
    
    data_config = config.get_data_config()
    preparer = DataPreparer(data_config)
    
    train_dataset, val_dataset = preparer.prepare_data(code_samples)
    
    logger.info(f"\nDatasets created:")
    logger.info(f"  Training: {len(train_dataset)} examples")
    logger.info(f"  Validation: {len(val_dataset)} examples")
    
    return train_dataset, val_dataset


def train_model(config: Config, train_dataset, val_dataset):
    """
    Load and fine-tune the model.
    
    Args:
        config: Configuration object
        train_dataset: Training dataset
        val_dataset: Validation dataset
    """
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Loading and Fine-tuning Model")
    logger.info("=" * 60)
    
    model_source = config.get_model_source()
    
    if model_source == 'huggingface':
        model_config = config.get_model_config()
        training_config = config.get_training_config()
        
        # Load model
        trust_remote_code = model_config.get('trust_remote_code', False)
        loader = ModelLoader(model_config, training_config, trust_remote_code=trust_remote_code)
        model, tokenizer = loader.load_huggingface_model()
        
        # Apply LoRA
        model = loader.apply_lora(model)
        
        # Train
        trainer = ModelTrainer(model, tokenizer, training_config)
        trainer.train(train_dataset, val_dataset)
        
        logger.info("\nModel fine-tuning completed!")
        
    elif model_source == 'ollama':
        logger.warning("Ollama model fine-tuning not yet implemented")
        logger.info("For Ollama models, use the Ollama CLI tools for model creation")
    else:
        logger.error(f"Unknown model source: {model_source}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='nanodex - Fine-tune models on your codebase'
    )
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--analyze-only',
        action='store_true',
        help='Only analyze the repository without training'
    )
    parser.add_argument(
        '--prepare-only',
        action='store_true',
        help='Only prepare data without training'
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = Config(args.config)
        
        # Step 1: Analyze repository
        code_samples = analyze_repository(config)
        
        if args.analyze_only:
            logger.info("\nAnalysis complete. Exiting (--analyze-only specified)")
            return 0
        
        # Step 2: Prepare training data
        train_dataset, val_dataset = prepare_training_data(config, code_samples)
        
        if args.prepare_only:
            logger.info("\nData preparation complete. Exiting (--prepare-only specified)")
            return 0
        
        # Step 3: Train model
        train_model(config, train_dataset, val_dataset)
        
        logger.info("\n" + "=" * 60)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 60)
        logger.info("\nYour fine-tuned model is ready for chatbot integration.")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
