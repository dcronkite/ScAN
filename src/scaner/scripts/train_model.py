import argparse
import json
from pathlib import Path

from scaner.config_loader import load_config
from scaner.helpers import setup_gpu, print_data_statistics
from scaner.trainer import SuicideDetectionTrainer


def load_jsonl_data(path: Path):
    with open(path, encoding='utf8') as fh:
        for line in fh:
            yield json.loads(line)

def main():
    parser = argparse.ArgumentParser(description='Train Joint RoBERTa Suicide Detection Model')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration YAML file')
    parser.add_argument('--output-dir', dest='output_dir', type=str, default='./checkpoints',
                        help='Output directory for model checkpoints')
    parser.add_argument('--gpu-id', dest='gpu_id', type=str, default='0',
                        help='GPU ID to use for training')
    parser.add_argument('--train-path', dest='train_path', type=Path, required=True,
                        help='Path to train.jsonl generated from `create_scan_corpus.py`')
    parser.add_argument('--test-path', dest='test_path', type=Path, default=None,
                        help='Path to train.jsonl generated from `create_scan_corpus.py`')
    parser.add_argument('--val-path', dest='val_path', type=Path, default=None,
                        help='Path to train.jsonl generated from `create_scan_corpus.py`')
    args = parser.parse_args()
    config = load_config(args.config)
    device = setup_gpu(args.gpu_id)
    data = {}
    data['train'] = list(load_jsonl_data(args.train_path))
    if args.test_path:
        data['test'] = list(load_jsonl_data(args.test_path))
    if args.val_path:
        data['val'] = list(load_jsonl_data(args.val_path))
    print_data_statistics(data)
    trainer = SuicideDetectionTrainer(config, device)
    results = trainer.train(data)
    print(results.keys())
    trainer.save_model(
            output_dir=config['paths']['output_dir'],
            model_name=config['paths']['model_name']
    )


if __name__ == '__main__':
    main()
