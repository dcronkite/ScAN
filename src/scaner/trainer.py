"""Training module for joint RoBERTa suicide detection model."""

import logging
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from tqdm import tqdm
from typing import Dict, Any, List

from scaner.helpers import count_parameters, save_model
from scaner.joint_roberta import create_joint_roberta_model
from scaner.suicide_dataset import SuicideRobertaDataset


class SuicideDetectionTrainer:
    """Trainer class for joint suicide detection model."""
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        """
        Initialize the trainer.
        
        Args:
            config: Configuration dictionary
            device: PyTorch device
        """
        self.config = config
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
        
        # Initialize model
        self.model = create_joint_roberta_model(config['model']['name'], config)
        self.model.to(device)
        
        # Setup multi-GPU if available
        if torch.cuda.device_count() > 1:
            self.logger.info(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
        
        self.logger.info(f"Model has {count_parameters(self.model):,} trainable parameters")
        
        # Extract label mappings and weights from config
        self.sa2id = config['tasks']['suicide_attempt']['label_mapping']
        self.si2id = config['tasks']['suicide_ideation']['label_mapping']
        self.rel2id = config['tasks']['relevance']['label_mapping']
        
        self.sa2wt = config['tasks']['suicide_attempt']['class_weights']
        self.si2wt = config['tasks']['suicide_ideation']['class_weights']
        self.rel2wt = config['tasks']['relevance']['class_weights']
        
        # Task weights for multi-task loss
        self.task_weights = {
            'attempt': config['tasks']['suicide_attempt']['task_weight'],
            'ideation': config['tasks']['suicide_ideation']['task_weight'],
            'relevance': config['tasks']['relevance']['task_weight']
        }
        
    def create_datasets(self, data: Dict[str, List[Dict]]) -> Dict[str, SuicideRobertaDataset]:
        """Create train and test datasets."""
        datasets = {}
        
        # Merge train and val as done in the notebook
        if 'val' in data:
            train_data = data['train'] + data['val']
            self.logger.info(f"Merged train and val: {len(data['train'])} + {len(data['val'])} = {len(train_data)}")
        else:
            train_data = data['train']
        
        # Create training dataset with downsampling
        datasets['train'] = SuicideRobertaDataset(
            list_data=train_data,
            sa2id=self.sa2id,
            si2id=self.si2id,
            rel2id=self.rel2id,
            sa2wt=self.sa2wt,
            si2wt=self.si2wt,
            rel2wt=self.rel2wt,
            tokenizer=self.tokenizer,
            max_length=self.config['model']['max_length'],
            downsample_to=self.config['dataset']['train_downsample_ratio']
        )
        
        # Create test dataset without downsampling
        if 'test' in data:
            datasets['test'] = SuicideRobertaDataset(
                list_data=data['test'],
                sa2id=self.sa2id,
                si2id=self.si2id,
                rel2id=self.rel2id,
                sa2wt=self.sa2wt,
                si2wt=self.si2wt,
                rel2wt=self.rel2wt,
                tokenizer=self.tokenizer,
                max_length=self.config['model']['max_length'],
                downsample_to=self.config['dataset']['test_downsample_ratio']
            )
        
        return datasets
    
    def create_data_loaders(self, datasets: Dict[str, SuicideRobertaDataset]) -> Dict[str, DataLoader]:
        loaders = {}
        
        if 'train' in datasets:
            loaders['train'] = DataLoader(
                datasets['train'],
                batch_size=self.config['training']['batch_size'],
                shuffle=True,
                num_workers=1  # Set to 1 to avoid multiprocessing issues
            )
        
        if 'test' in datasets:
            loaders['test'] = DataLoader(
                datasets['test'],
                batch_size=self.config['training']['batch_size'],
                shuffle=False,
                num_workers=1
            )
        
        return loaders
    
    def setup_optimizer_and_scheduler(self, train_loader: DataLoader):
        """Setup optimizer and learning rate scheduler."""
        # Calculate total training steps
        t_total = len(train_loader) // self.config['training']['gradient_accumulation_steps'] * self.config['training']['epochs']
        warmup_steps = int(self.config['training']['warmup_ratio'] * t_total)
        
        self.logger.info(f"Total training steps: {t_total}, Warmup steps: {warmup_steps}")
        
        # Setup optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config['training']['weight_decay']
            },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 
                'weight_decay': 0.0
            }
        ]
        
        optimizer = AdamW(
            optimizer_grouped_parameters, 
            lr=self.config['training']['learning_rate'], 
            eps=self.config['training']['adam_epsilon']
        )
        
        # Setup scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps,
            num_training_steps=t_total
        )
        
        return optimizer, scheduler
    
    def train_epoch(self, train_loader: DataLoader, optimizer, scheduler) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            input_batch = {
                'input_ids': batch['input_ids'].to(self.device),
                'attention_mask': batch['attention_mask'].to(self.device)
            }
            
            # Forward pass
            outputs = self.model(**input_batch)
            
            # Calculate losses for each task
            loss_sa = self.model.module.loss_fct(outputs[0], batch['sa_label'].view(-1).to(self.device)) if hasattr(self.model, 'module') else self.model.loss_fct(outputs[0], batch['sa_label'].view(-1).to(self.device))
            loss_si = self.model.module.loss_fct(outputs[1], batch['si_label'].view(-1).to(self.device)) if hasattr(self.model, 'module') else self.model.loss_fct(outputs[1], batch['si_label'].view(-1).to(self.device))
            loss_rel = self.model.module.loss_fct(outputs[2], batch['rel_label'].view(-1).to(self.device)) if hasattr(self.model, 'module') else self.model.loss_fct(outputs[2], batch['rel_label'].view(-1).to(self.device))
            
            # Apply class weights
            loss_sa = (loss_sa * batch['sa_wts'].view(-1).to(self.device)).mean()
            loss_si = (loss_si * batch['si_wts'].view(-1).to(self.device)).mean()
            loss_rel = (loss_rel * batch['rel_wts'].view(-1).to(self.device)).mean()
            
            # Combine losses with task weights
            total_batch_loss = (
                self.task_weights['attempt'] * loss_sa +
                self.task_weights['ideation'] * loss_si +
                self.task_weights['relevance'] * loss_rel
            )
            
            # Backward pass
            self.model.zero_grad()
            total_batch_loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += total_batch_loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{total_batch_loss.item():.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}'
            })
        
        return total_loss / num_batches
    
    def train(self, data: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """
        Main training loop.
        
        Args:
            data: Dictionary with 'train' and optionally 'test' keys
            
        Returns:
            Dictionary with training results
        """
        self.logger.info("Starting training...")
        
        # Create datasets and data loaders
        datasets = self.create_datasets(data)
        data_loaders = self.create_data_loaders(datasets)
        
        # Setup optimizer and scheduler
        optimizer, scheduler = self.setup_optimizer_and_scheduler(data_loaders['train'])
        
        # Training loop
        train_losses = []
        
        for epoch in range(self.config['training']['epochs']):
            self.logger.info(f"Epoch {epoch + 1}/{self.config['training']['epochs']}")
            
            # Train for one epoch
            avg_loss = self.train_epoch(data_loaders['train'], optimizer, scheduler)
            train_losses.append(avg_loss)
            
            self.logger.info(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")
        
        self.logger.info("Training completed!")
        
        return {
            'train_losses': train_losses,
            'model': self.model,
            'tokenizer': self.tokenizer
        }
    
    def save_model(self, output_dir: str, model_name: str):
        """Save the trained model."""
        save_model(self.model, self.tokenizer, output_dir, model_name)
