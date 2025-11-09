"""
Training engine for Dual-Head Reasoning Distillation (DHRD)

Handles the training loop with both classification and reasoning heads.
"""

import torch
from torch.utils.data import DataLoader
from typing import Optional, Dict, Callable
from tqdm import tqdm
import os


class DHRDTrainer:
    """
    Trainer for Dual-Head Reasoning Distillation models.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        max_grad_norm: float = 1.0,
        log_interval: int = 10,
        eval_interval: int = 100,
        save_dir: str = './outputs',
        logger: Optional[Callable] = None,
    ):
        """
        Args:
            model: DualHeadReasoningModel instance
            train_dataloader: Training data loader
            eval_dataloader: Evaluation data loader (optional)
            optimizer: Optimizer (if None, AdamW will be used)
            scheduler: Learning rate scheduler (optional)
            device: Device to train on
            max_grad_norm: Maximum gradient norm for clipping
            log_interval: Steps between logging
            eval_interval: Steps between evaluation
            save_dir: Directory to save checkpoints
            logger: Custom logging function (optional)
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_dir = save_dir
        self.logger = logger or print

        # Initialize optimizer if not provided
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=5e-5,
                weight_decay=0.01
            )
        else:
            self.optimizer = optimizer

        self.scheduler = scheduler

        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)

        # Tracking metrics
        self.global_step = 0
        self.best_eval_accuracy = 0.0

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        total_loss = 0.0
        total_cls_loss = 0.0
        total_lm_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch}",
            disable=False
        )

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Get rationale data if available
            rationale_input_ids = batch.get('rationale_input_ids')
            rationale_attention_mask = batch.get('rationale_attention_mask')

            if rationale_input_ids is not None:
                rationale_input_ids = rationale_input_ids.to(self.device)
                rationale_attention_mask = rationale_attention_mask.to(self.device)

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                rationale_input_ids=rationale_input_ids,
                rationale_attention_mask=rationale_attention_mask,
                training=True
            )

            loss = outputs['loss']

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm
            )

            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            # Update metrics
            total_loss += loss.item()
            total_cls_loss += outputs['classification_loss'].item()
            if 'lm_loss' in outputs:
                total_lm_loss += outputs['lm_loss'].item()
            num_batches += 1
            self.global_step += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'cls_loss': outputs['classification_loss'].item(),
                'lm_loss': outputs.get('lm_loss', torch.tensor(0.0)).item()
            })

            # Log metrics
            if self.global_step % self.log_interval == 0:
                avg_loss = total_loss / num_batches
                avg_cls_loss = total_cls_loss / num_batches
                avg_lm_loss = total_lm_loss / num_batches if total_lm_loss > 0 else 0.0

                log_msg = (
                    f"Step {self.global_step} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Cls Loss: {avg_cls_loss:.4f} | "
                    f"LM Loss: {avg_lm_loss:.4f}"
                )
                self.logger(log_msg)

            # Evaluate
            if self.eval_dataloader is not None and self.global_step % self.eval_interval == 0:
                eval_metrics = self.evaluate()
                self.logger(f"Eval at step {self.global_step}: {eval_metrics}")

                # Save best model
                if eval_metrics['accuracy'] > self.best_eval_accuracy:
                    self.best_eval_accuracy = eval_metrics['accuracy']
                    self.save_checkpoint('best_model.pt')
                    self.logger(f"New best model saved with accuracy: {self.best_eval_accuracy:.4f}")

                self.model.train()

        # Epoch metrics
        epoch_metrics = {
            'loss': total_loss / num_batches,
            'classification_loss': total_cls_loss / num_batches,
            'lm_loss': total_lm_loss / num_batches if total_lm_loss > 0 else 0.0,
        }

        return epoch_metrics

    def train(self, num_epochs: int):
        """
        Train the model for multiple epochs.

        Args:
            num_epochs: Number of epochs to train
        """
        self.logger(f"Starting training for {num_epochs} epochs...")
        self.logger(f"Device: {self.device}")
        self.logger(f"Number of training batches: {len(self.train_dataloader)}")

        for epoch in range(1, num_epochs + 1):
            self.logger(f"\n{'=' * 50}")
            self.logger(f"Epoch {epoch}/{num_epochs}")
            self.logger(f"{'=' * 50}")

            epoch_metrics = self.train_epoch(epoch)

            self.logger(f"\nEpoch {epoch} Summary:")
            for key, value in epoch_metrics.items():
                self.logger(f"  {key}: {value:.4f}")

            # Evaluate at end of epoch
            if self.eval_dataloader is not None:
                eval_metrics = self.evaluate()
                self.logger(f"\nEpoch {epoch} Evaluation:")
                for key, value in eval_metrics.items():
                    self.logger(f"  {key}: {value:.4f}")

                # Save best model
                if eval_metrics['accuracy'] > self.best_eval_accuracy:
                    self.best_eval_accuracy = eval_metrics['accuracy']
                    self.save_checkpoint('best_model.pt')
                    self.logger(f"New best model saved with accuracy: {self.best_eval_accuracy:.4f}")

            # Save checkpoint at end of epoch
            self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')

        self.logger(f"\nTraining completed!")
        self.logger(f"Best evaluation accuracy: {self.best_eval_accuracy:.4f}")

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on the evaluation dataset.

        Returns:
            Dictionary of evaluation metrics
        """
        if self.eval_dataloader is None:
            return {}

        self.model.eval()

        total_correct = 0
        total_samples = 0
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Forward pass (inference mode - no rationale)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    training=False
                )

                # Get predictions
                predictions = torch.argmax(outputs['classification_logits'], dim=-1)

                # Update metrics
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)
                total_loss += outputs['loss'].item()

        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        avg_loss = total_loss / len(self.eval_dataloader)

        metrics = {
            'accuracy': accuracy,
            'loss': avg_loss,
        }

        return metrics

    def save_checkpoint(self, filename: str):
        """
        Save model checkpoint.

        Args:
            filename: Name of checkpoint file
        """
        checkpoint_path = os.path.join(self.save_dir, filename)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'best_eval_accuracy': self.best_eval_accuracy,
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        self.logger(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)
        self.best_eval_accuracy = checkpoint.get('best_eval_accuracy', 0.0)

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.logger(f"Checkpoint loaded from {checkpoint_path}")
        self.logger(f"Resuming from step {self.global_step}")
