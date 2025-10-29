"""Configuration loading utilities."""

import yaml
import argparse
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_args(args=None):
    """Parse command line arguments."""
    if args is None:
        parser = argparse.ArgumentParser(description='Train Joint RoBERTa Suicide Detection Model')
        parser.add_argument('--config', type=str, required=True,
                            help='Path to configuration YAML file')
        parser.add_argument('--data-path', dest='data_path', type=str, required=True,
                            help='Path to pickle data file')
        parser.add_argument('--output-dir', dest='output_dir', type=str, default='./checkpoints',
                            help='Output directory for model checkpoints')
        parser.add_argument('--gpu-id', dest='gpu_id', type=str, default='0',
                            help='GPU ID to use for training')
        args = parser.parse_args()

    # load config file and update with command line arguments
    config = load_config(args.config)
    config = update_config_with_args(config, args)
    return config


def update_config_with_args(config: Dict[str, Any], args) -> Dict[str, Any]:
    """Update config with command line arguments."""
    config['paths']['data_path'] = args.data_path
    config['paths']['output_dir'] = args.output_dir
    return config
