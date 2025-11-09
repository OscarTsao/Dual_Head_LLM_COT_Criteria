#!/usr/bin/env python3
"""
Evaluation script for Dual-Head Reasoning Distillation (DHRD)

Usage:
    python scripts/evaluate.py --config configs/dhrd_config.yaml --checkpoint outputs/dhrd_experiment/best_model.pt
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from transformers import AutoTokenizer
from pathlib import Path
import json

from Project.SubProject.models.model import DualHeadReasoningModel
from Project.SubProject.data.dataset import load_json_dataset, create_dataloaders
from Project.SubProject.engine.eval_engine import DHRDEvaluator
from Project.SubProject.utils.config import DHRDConfig, create_argument_parser, override_config
from Project.SubProject.utils.seed import set_seed
from Project.SubProject.utils.log import get_logger


def main():
    """Main evaluation function."""
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()

    if args.checkpoint is None:
        raise ValueError("--checkpoint argument is required for evaluation")

    # Load config
    config = DHRDConfig.from_yaml(args.config)
    config = override_config(config, args)

    # Initialize logger
    logger = get_logger(__name__)
    logger.info("=" * 60)
    logger.info("Dual-Head Reasoning Distillation (DHRD) Evaluation")
    logger.info("=" * 60)

    # Set seed for reproducibility
    set_seed(config.seed)

    # Load tokenizer
    logger.info(f"Loading tokenizer: {config.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)

    # Load test dataset
    logger.info("Loading test dataset...")
    test_dataset = load_json_dataset(
        file_path=config.test_file,
        tokenizer=tokenizer,
        max_input_length=config.max_input_length,
        max_rationale_length=config.max_rationale_length,
        include_rationale=False  # No rationale needed for evaluation
    )

    logger.info(f"Test samples: {len(test_dataset)}")

    # Create dataloader
    from torch.utils.data import DataLoader
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    # Initialize model
    logger.info("Initializing DHRD model...")
    model = DualHeadReasoningModel(
        model_name_or_path=config.model_name_or_path,
        num_labels=config.num_labels,
        dropout_prob=config.dropout_prob,
        loss_weight_alpha=config.loss_weight_alpha
    )

    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.device)

    logger.info("Model loaded successfully!")

    # Initialize evaluator
    evaluator = DHRDEvaluator(
        model=model,
        device=config.device,
        logger=logger.info
    )

    # Run evaluation
    logger.info("\nRunning evaluation...")
    results = evaluator.evaluate(
        dataloader=test_dataloader,
        return_predictions=True,
        compute_throughput=config.compute_throughput
    )

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Results")
    logger.info("=" * 60)
    logger.info(f"Accuracy:  {results['accuracy']:.4f}")
    logger.info(f"Precision: {results['precision']:.4f}")
    logger.info(f"Recall:    {results['recall']:.4f}")
    logger.info(f"F1 Score:  {results['f1']:.4f}")

    if config.compute_throughput:
        logger.info(f"\nThroughput: {results['throughput_qps']:.2f} QPS")
        logger.info(f"Avg Latency: {results['avg_latency_ms']:.3f} ms")

    logger.info("\nClassification Report:")
    logger.info(results['classification_report'])

    # Run benchmark
    if config.benchmark_runs > 0:
        logger.info("\n" + "=" * 60)
        logger.info("Running Inference Benchmark")
        logger.info("=" * 60)
        benchmark_results = evaluator.benchmark_inference(
            dataloader=test_dataloader,
            num_runs=config.benchmark_runs
        )

    # Save results to file
    output_dir = Path(config.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "evaluation_results.json"

    # Remove non-serializable items for JSON
    save_results = {
        'accuracy': results['accuracy'],
        'precision': results['precision'],
        'recall': results['recall'],
        'f1': results['f1'],
        'throughput_qps': results.get('throughput_qps', 0.0),
        'avg_latency_ms': results.get('avg_latency_ms', 0.0),
        'classification_report': results['classification_report'],
    }

    if config.benchmark_runs > 0:
        save_results['benchmark'] = benchmark_results

    with open(results_file, 'w') as f:
        json.dump(save_results, f, indent=2)

    logger.info(f"\nResults saved to: {results_file}")
    logger.info("\nEvaluation completed!")


if __name__ == "__main__":
    main()
