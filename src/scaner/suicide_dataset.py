"""Suicide detection dataset classes."""

import torch
import random
from torch.utils.data import Dataset
from typing import List, Dict, Any
from transformers import AutoTokenizer


class SuicideRobertaDataset(Dataset):
    """Dataset class for suicide detection with RoBERTa tokenization."""
    
    def __init__(self, 
                 list_data: List[Dict[str, Any]], 
                 sa2id: Dict[str, int], 
                 si2id: Dict[str, int], 
                 rel2id: Dict[str, int], 
                 sa2wt: Dict[str, float],
                 si2wt: Dict[str, float],
                 rel2wt: Dict[str, float],
                 tokenizer: AutoTokenizer, 
                 max_length: int = 400,
                 downsample_to: float = 1.0):
        """
        Initialize the dataset.
        
        Args:
            list_data: List of data samples with keys: 'text', 'suicide_attempt', 'suicide_ideation', 'relevance'
            sa2id: Mapping from suicide attempt labels to IDs
            si2id: Mapping from suicide ideation labels to IDs  
            rel2id: Mapping from relevance labels to IDs
            sa2wt: Class weights for suicide attempt
            si2wt: Class weights for suicide ideation
            rel2wt: Class weights for relevance
            tokenizer: RoBERTa tokenizer
            max_length: Maximum sequence length
            downsample_to: Ratio for downsampling (1.0 = no downsampling)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sa2id = sa2id
        self.si2id = si2id
        self.rel2id = rel2id
        self.sa2wt = sa2wt
        self.si2wt = si2wt
        self.rel2wt = rel2wt
        
        # Ensure tokenizer has proper padding token for RoBERTa
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Apply downsampling if needed
        if downsample_to < 1.0:
            self.data = self._downsample_data(list_data, downsample_to)
        else:
            self.data = list_data
            
    def _downsample_data(self, data: List[Dict], ratio: float) -> List[Dict]:
        """Randomly downsample negative samples."""
        # Separate positive and negative samples based on relevance
        positive_samples = [x for x in data if x['relevance'] == 'pos']
        negative_samples = [x for x in data if x['relevance'] == 'neg']
        
        # Calculate how many negative samples to keep
        n_negative_to_keep = int(len(negative_samples) * ratio)
        
        # Randomly sample negative examples
        random.shuffle(negative_samples)
        sampled_negative = negative_samples[:n_negative_to_keep]
        
        # Combine positive and sampled negative
        downsampled_data = positive_samples + sampled_negative
        random.shuffle(downsampled_data)
        
        print(f"Downsampling: {len(data)} -> {len(downsampled_data)} samples")
        print(f"Positive: {len(positive_samples)}, Negative: {len(sampled_negative)}")
        
        return downsampled_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        text = sample['text']
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Get labels
        sa_label = self.sa2id[sample['suicide_attempt']]
        si_label = self.si2id[sample['suicide_ideation']]
        rel_label = self.rel2id[sample['relevance']]
        
        # Get class weights
        sa_weight = self.sa2wt[sample['suicide_attempt']]
        si_weight = self.si2wt[sample['suicide_ideation']]
        rel_weight = self.rel2wt[sample['relevance']]
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'sa_label': torch.tensor(sa_label, dtype=torch.long),
            'si_label': torch.tensor(si_label, dtype=torch.long),
            'rel_label': torch.tensor(rel_label, dtype=torch.long),
            'sa_wts': torch.tensor(sa_weight, dtype=torch.float),
            'si_wts': torch.tensor(si_weight, dtype=torch.float),
            'rel_wts': torch.tensor(rel_weight, dtype=torch.float)
        }
