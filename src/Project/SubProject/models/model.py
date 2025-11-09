"""
Dual-Head Reasoning Distillation (DHRD) Model Implementation

Based on the paper: "Dual-Head Reasoning Distillation: Improving Classifier
Accuracy with Train-Time-Only Reasoning" (arXiv:2509.21487)

This implementation includes:
1. A pooled classification head (used during training and inference)
2. A reasoning head for language modeling (used only during training)
3. Combined loss function for both heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig
from typing import Optional, Tuple, Dict


class PooledClassificationHead(nn.Module):
    """
    Pooled classification head that operates on the final hidden state.
    Used during both training and inference.
    """
    def __init__(self, hidden_size: int, num_labels: int, dropout_prob: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states: torch.Tensor, pooling_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            pooling_mask: [batch_size, seq_len] - mask for pooling (e.g., last token position)
        Returns:
            logits: [batch_size, num_labels]
        """
        # Pool the hidden states (use last token or mean pooling)
        if pooling_mask is not None:
            # Use the last valid token position for each sequence
            batch_size = hidden_states.size(0)
            last_token_indices = pooling_mask.sum(dim=1) - 1  # [batch_size]
            pooled = hidden_states[torch.arange(batch_size), last_token_indices]
        else:
            # Mean pooling as fallback
            pooled = hidden_states.mean(dim=1)

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


class ReasoningHead(nn.Module):
    """
    Language modeling head for generating rationales.
    Used only during training for knowledge distillation.
    """
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
        Returns:
            lm_logits: [batch_size, seq_len, vocab_size]
        """
        return self.lm_head(hidden_states)


class DualHeadReasoningModel(nn.Module):
    """
    Dual-Head Reasoning Distillation Model.

    Combines a pooled classification head with a reasoning (LM) head.
    During training, both heads are active and supervised.
    During inference, only the classification head is used for fast predictions.
    """
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        dropout_prob: float = 0.1,
        loss_weight_alpha: float = 0.5,
    ):
        """
        Args:
            model_name_or_path: Pretrained model identifier (e.g., 'gpt2', 'meta-llama/Llama-2-7b-hf')
            num_labels: Number of classification labels
            dropout_prob: Dropout probability for classification head
            loss_weight_alpha: Weight for classification loss (1-alpha for LM loss)
        """
        super().__init__()

        # Load base transformer model (decoder-only LM)
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.transformer = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            config=self.config
        )

        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size

        # Initialize dual heads
        self.pooled_classification_head = PooledClassificationHead(
            hidden_size=hidden_size,
            num_labels=num_labels,
            dropout_prob=dropout_prob
        )

        self.reasoning_head = ReasoningHead(
            hidden_size=hidden_size,
            vocab_size=vocab_size
        )

        # Loss weight
        self.loss_weight_alpha = loss_weight_alpha
        self.num_labels = num_labels

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        rationale_input_ids: Optional[torch.Tensor] = None,
        rationale_attention_mask: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the dual-head model.

        Args:
            input_ids: [batch_size, seq_len] - Input token IDs
            attention_mask: [batch_size, seq_len] - Attention mask for input
            labels: [batch_size] - Classification labels (optional)
            rationale_input_ids: [batch_size, rationale_seq_len] - Input+rationale token IDs (optional)
            rationale_attention_mask: [batch_size, rationale_seq_len] - Attention mask for rationale (optional)
            training: Whether in training mode (affects which heads are used)

        Returns:
            Dictionary containing:
                - classification_logits: [batch_size, num_labels]
                - loss: Combined loss (if labels provided)
                - classification_loss: Classification loss component
                - lm_loss: Language modeling loss component (if training)
        """
        outputs = {}

        # Get hidden states from input
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        hidden_states = transformer_outputs.hidden_states[-1]  # Last layer

        # Always compute classification logits (used in both train and inference)
        classification_logits = self.pooled_classification_head(
            hidden_states=hidden_states,
            pooling_mask=attention_mask
        )
        outputs['classification_logits'] = classification_logits

        # Compute losses if labels are provided
        if labels is not None:
            classification_loss = F.cross_entropy(classification_logits, labels)
            outputs['classification_loss'] = classification_loss

            # During training, also compute LM loss with rationale
            if training and rationale_input_ids is not None:
                # Forward pass with input+rationale
                rationale_outputs = self.transformer(
                    input_ids=rationale_input_ids,
                    attention_mask=rationale_attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
                rationale_hidden_states = rationale_outputs.hidden_states[-1]

                # Compute LM logits
                lm_logits = self.reasoning_head(rationale_hidden_states)

                # Shift for next-token prediction
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = rationale_input_ids[..., 1:].contiguous()

                # Compute LM loss (only on rationale tokens, not input tokens)
                lm_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100  # Ignore padding tokens
                )
                outputs['lm_loss'] = lm_loss

                # Combined loss
                combined_loss = (
                    self.loss_weight_alpha * classification_loss +
                    (1 - self.loss_weight_alpha) * lm_loss
                )
                outputs['loss'] = combined_loss
            else:
                # Inference mode or no rationale: only classification loss
                outputs['loss'] = classification_loss

        return outputs

    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Fast inference using only the classification head (no rationale generation).

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]

        Returns:
            predictions: [batch_size] - Predicted class indices
        """
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                training=False
            )
            predictions = torch.argmax(outputs['classification_logits'], dim=-1)
        return predictions

