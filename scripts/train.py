#!/usr/bin/env python3
"""
Training script for Dual-Head Reasoning Distillation (DHRD)

Usage:
    python scripts/train.py --config configs/dhrd_config.yaml
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from pathlib import Path

from Project.SubProject.models.model import DualHeadReasoningModel
from Project.SubProject.data.dataset import load_json_dataset, create_dataloaders
from Project.SubProject.engine.train_engine import DHRDTrainer
from Project.SubProject.utils.config import DHRDConfig, create_argument_parser, override_config
from Project.SubProject.utils.seed import set_seed
from Project.SubProject.utils.log import get_logger
from Project.SubProject.utils.mlflow_utils import configure_mlflow, enable_autologging, mlflow_run


def main():
    """Main training function."""
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()

    # Load config
    config = DHRDConfig.from_yaml(args.config)
    config = override_config(config, args)

    # Initialize logger
    logger = get_logger(__name__)
    logger.info("=" * 60)
    logger.info("Dual-Head Reasoning Distillation (DHRD) Training")
    logger.info("=" * 60)
    logger.info(f"\n{config}")

    # Set seed for reproducibility
    set_seed(config.seed)
    logger.info(f"Random seed set to: {config.seed}")

    # Setup MLflow if enabled
    if config.use_mlflow:
        configure_mlflow(
            tracking_uri=config.tracking_uri,
            experiment=config.experiment_name
        )
        enable_autologging()

    # Load tokenizer
    logger.info(f"Loading tokenizer: {config.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)

    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = load_json_dataset(
        file_path=config.train_file,
        tokenizer=tokenizer,
        max_input_length=config.max_input_length,
        max_rationale_length=config.max_rationale_length,
        include_rationale=True
    )

    eval_dataset = None
    if Path(config.eval_file).exists():
        eval_dataset = load_json_dataset(
            file_path=config.eval_file,
            tokenizer=tokenizer,
            max_input_length=config.max_input_length,
            max_rationale_length=config.max_rationale_length,
            include_rationale=False  # No rationale needed for evaluation
        )

    logger.info(f"Training samples: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"Evaluation samples: {len(eval_dataset)}")

    # Create dataloaders
    dataloaders = create_dataloaders(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )

    # Initialize model
    logger.info("Initializing DHRD model...")
    model = DualHeadReasoningModel(
        model_name_or_path=config.model_name_or_path,
        num_labels=config.num_labels,
        dropout_prob=config.dropout_prob,
        loss_weight_alpha=config.loss_weight_alpha
    )

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=tuple(config.optimizer_betas),
        eps=config.optimizer_eps,
        weight_decay=config.weight_decay
    )

    # Initialize scheduler
    total_steps = len(dataloaders['train']) * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Initialize trainer
    trainer = DHRDTrainer(
        model=model,
        train_dataloader=dataloaders['train'],
        eval_dataloader=dataloaders.get('eval'),
        optimizer=optimizer,
        scheduler=scheduler,
        device=config.device,
        max_grad_norm=config.max_grad_norm,
        log_interval=config.log_interval,
        eval_interval=config.eval_interval,
        save_dir=config.save_dir,
        logger=logger.info
    )

    # Load checkpoint if provided
    if args.checkpoint:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)

    # Start training
    if config.use_mlflow:
        with mlflow_run(
            run_name="dhrd_training",
            tags={"model": config.model_name_or_path},
            params=config.to_dict()
        ):
            trainer.train(num_epochs=config.num_epochs)
    else:
        trainer.train(num_epochs=config.num_epochs)

    logger.info("Training completed!")
    logger.info(f"Best evaluation accuracy: {trainer.best_eval_accuracy:.4f}")
    logger.info(f"Checkpoints saved to: {config.save_dir}")


if __name__ == "__main__":
    main()
