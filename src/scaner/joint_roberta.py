"""Joint RoBERTa model for multi-task suicide detection."""

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import RobertaPreTrainedModel, RobertaModel
from typing import Dict, Any, Optional


class JointRobertaModel(RobertaPreTrainedModel):
    """
    Joint RoBERTa model for multi-task suicide detection.
    
    Performs three classification tasks:
    1. Suicide Attempt (SA) classification
    2. Suicide Ideation (SI) classification  
    3. Relevance classification
    """
    
    def __init__(self, config, specific_config: Dict[str, Any]):
        """
        Initialize the joint model.
        
        Args:
            config: RoBERTa configuration
            specific_config: Task-specific configuration with keys:
                - sa_labels: Number of suicide attempt labels
                - si_labels: Number of suicide ideation labels
                - rel_labels: Number of relevance labels
        """
        super(JointRobertaModel, self).__init__(config)
        self.specific_config = specific_config
        self.num_labels = config.num_labels

        # Shared RoBERTa encoder
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Task-specific classification heads
        self.sa_classifier = nn.Linear(config.hidden_size, self.specific_config['sa_labels'])
        self.si_classifier = nn.Linear(config.hidden_size, self.specific_config['si_labels'])
        self.rel_classifier = nn.Linear(config.hidden_size, self.specific_config['rel_labels'])
        
        # Loss function
        self.loss_fct = CrossEntropyLoss(reduction='none')

        self.init_weights()

    def forward(self, 
                input_ids: Optional[torch.Tensor] = None, 
                attention_mask: Optional[torch.Tensor] = None, 
                token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None, 
                head_mask: Optional[torch.Tensor] = None, 
                inputs_embeds: Optional[torch.Tensor] = None, 
                labels: Optional[torch.Tensor] = None):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs (not used in RoBERTa)
            position_ids: Position IDs
            head_mask: Head mask
            inputs_embeds: Input embeddings
            labels: Labels (not used in this implementation)
            
        Returns:
            Tuple of (logits_sa, logits_si, logits_rel, hidden_states, attentions)
        """
        # Get RoBERTa outputs
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )

        # Get pooled output (CLS token representation)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        
        # Get logits for each task
        logits_sa = self.sa_classifier(pooled_output)
        logits_si = self.si_classifier(pooled_output)
        logits_rel = self.rel_classifier(pooled_output)

        # Return logits and additional outputs
        outputs = (logits_sa, logits_si, logits_rel,) + outputs[2:]  # Add hidden states and attention if they are here

        if labels is not None:
            # This is kept for compatibility but not used in our training loop
            if self.num_labels == 1:
                # Regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits_sa.view(-1), labels.view(-1))
            else:
                # Classification
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits_sa.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits_sa, logits_si, logits_rel, (hidden_states), (attentions)


def create_joint_roberta_model(model_name: str, config: Dict[str, Any]):
    """
    Create and initialize a joint RoBERTa model.
    
    Args:
        model_name: Name of the pre-trained RoBERTa model
        config: Configuration dictionary with task specifications
        
    Returns:
        Initialized JointRobertaModel
    """
    from transformers import RobertaConfig
    
    # Load RoBERTa configuration
    roberta_config = RobertaConfig.from_pretrained(
        model_name,
        num_labels=0,  # We handle labels separately
        finetuning_task='SuicideDataset',
        cache_dir=None,
        output_attentions=True,
        output_hidden_states=True
    )
    
    # Create task-specific configuration
    specific_config = {
        'sa_labels': len(config['tasks']['suicide_attempt']['labels']),
        'si_labels': len(config['tasks']['suicide_ideation']['labels']),
        'rel_labels': len(config['tasks']['relevance']['labels'])
    }
    
    # Initialize model
    model = JointRobertaModel(config=roberta_config, specific_config=specific_config)
    
    return model
