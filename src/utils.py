import yaml
import logging
from pathlib import Path


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def setup_logging(log_dir):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(str(Path(log_dir) / 'training.log'), encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


def create_directories(config):
    paths = [
        Path(config['data']['pdf_dir']),
        Path(config['data']['processed_dir']),
        Path(config['model_dir']),
        Path(config['log_dir'])
    ]

    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
