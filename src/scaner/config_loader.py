"""Configuration loading utilities."""
import yaml
from typing import Dict, Any


def load_config(config_path: str, data_path=None, output_dir=None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    if data_path:
        config['paths']['data_path'] = data_path
    if output_dir:
        config['paths']['output_dir'] = output_dir
    return config
