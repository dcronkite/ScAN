"""Helper utilities for training."""

import os
import pickle as pkl
import numpy as np
import torch
from typing import Dict, Any
from collections import Counter


def get_class_weights(dict_cnt: Dict[str, int], alpha: float = 15) -> Dict[str, float]:
    """Calculate class weights based on inverse frequency."""
    tot_cnt = sum([dict_cnt[x] for x in dict_cnt])
    wt_ = {}
    for each_cat in dict_cnt:
        wt_[each_cat] = np.log(alpha * tot_cnt / dict_cnt[each_cat])
    return wt_


def load_data(data_path: str) -> Dict[str, Any]:
    """Load data from pickle file."""
    with open(data_path, 'rb') as f:
        data = pkl.load(f)
    return data


def save_model(model, tokenizer, output_dir: str, model_name: str):
    """Save model and tokenizer."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(os.path.join(output_dir, model_name))
    tokenizer.save_pretrained(os.path.join(output_dir, model_name))
    
    # Also save the full model for easy loading
    torch.save(model, os.path.join(output_dir, model_name, 'full_model.pt'))
    
    print(f'Model and tokenizer saved to: {os.path.join(output_dir, model_name)}')


def setup_gpu(gpu_id: str):
    """Setup GPU environment."""
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    return device


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_data_statistics(data: Dict[str, Any]):
    """Print dataset statistics."""
    for split in ['train', 'test']:
        if split in data:
            print(f'\n{split.upper()} SET STATISTICS:')
            print(f'Total samples: {len(data[split])}')
            print(f'Relevance: {Counter([x["relevance"] for x in data[split]])}')
            print(f'Suicide Attempt: {Counter([x["suicide_attempt"] for x in data[split]])}')
            print(f'Suicide Ideation: {Counter([x["suicide_ideation"] for x in data[split]])}')
