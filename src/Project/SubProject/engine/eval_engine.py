"""
Evaluation engine for Dual-Head Reasoning Distillation (DHRD)

Evaluates model using only the classification head (no rationale generation).
This demonstrates the key advantage of DHRD: train-time reasoning with
inference-time efficiency.
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import time


class DHRDEvaluator:
    """
    Evaluator for Dual-Head Reasoning Distillation models.

    Uses only the pooled classification head for fast inference,
    demonstrating the throughput advantage over CoT methods.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        logger: Optional[callable] = None,
    ):
        """
        Args:
            model: DualHeadReasoningModel instance
            device: Device to evaluate on
            logger: Custom logging function (optional)
        """
        self.model = model.to(device)
        self.device = device
        self.logger = logger or print

    def evaluate(
        self,
        dataloader: DataLoader,
        return_predictions: bool = False,
        compute_throughput: bool = True,
    ) -> Dict[str, any]:
        """
        Evaluate the model on a dataset.

        Args:
            dataloader: DataLoader for evaluation
            return_predictions: Whether to return predictions and labels
            compute_throughput: Whether to compute throughput metrics

        Returns:
            Dictionary containing:
                - accuracy: Overall accuracy
                - precision: Macro precision
                - recall: Macro recall
                - f1: Macro F1 score
                - confusion_matrix: Confusion matrix (if applicable)
                - classification_report: Detailed classification report
                - throughput: Queries per second (if compute_throughput=True)
                - predictions: List of predictions (if return_predictions=True)
                - labels: List of true labels (if return_predictions=True)
        """
        self.model.eval()

        all_predictions = []
        all_labels = []
        total_samples = 0
        start_time = time.time()

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", disable=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Fast inference using only classification head
                predictions = self.model.predict(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                all_predictions.extend(predictions.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                total_samples += labels.size(0)

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        # Compute metrics
        metrics = self._compute_metrics(all_predictions, all_labels)

        # Add throughput metric
        if compute_throughput:
            throughput = total_samples / elapsed_time
            metrics['throughput_qps'] = throughput
            metrics['avg_latency_ms'] = (elapsed_time / total_samples) * 1000

        # Add predictions if requested
        if return_predictions:
            metrics['predictions'] = all_predictions.tolist()
            metrics['labels'] = all_labels.tolist()

        return metrics

    def _compute_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, any]:
        """
        Compute evaluation metrics.

        Args:
            predictions: Array of predictions
            labels: Array of true labels

        Returns:
            Dictionary of metrics
        """
        # Basic metrics
        accuracy = accuracy_score(labels, predictions)

        # Precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            predictions,
            average='macro',
            zero_division=0
        )

        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support = (
            precision_recall_fscore_support(
                labels,
                predictions,
                average=None,
                zero_division=0
            )
        )

        # Confusion matrix
        conf_matrix = confusion_matrix(labels, predictions)

        # Classification report
        unique_labels = sorted(set(labels.tolist()))
        report = classification_report(
            labels,
            predictions,
            labels=unique_labels,
            target_names=[f"Class_{i}" for i in unique_labels],
            zero_division=0
        )

        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'support': support.tolist(),
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': report,
        }

        return metrics

    def compare_with_baseline(
        self,
        dataloader: DataLoader,
        baseline_results: Dict[str, float],
    ) -> Dict[str, any]:
        """
        Compare DHRD model with baseline results.

        Args:
            dataloader: DataLoader for evaluation
            baseline_results: Dictionary with baseline metrics (accuracy, throughput, etc.)

        Returns:
            Dictionary with comparison metrics and improvements
        """
        # Evaluate DHRD model
        dhrd_results = self.evaluate(dataloader, compute_throughput=True)

        # Compute improvements
        comparison = {
            'dhrd_accuracy': dhrd_results['accuracy'],
            'baseline_accuracy': baseline_results.get('accuracy', 0.0),
            'accuracy_improvement': dhrd_results['accuracy'] - baseline_results.get('accuracy', 0.0),
            'relative_accuracy_gain': (
                (dhrd_results['accuracy'] - baseline_results.get('accuracy', 0.0))
                / baseline_results.get('accuracy', 1.0) * 100
                if baseline_results.get('accuracy', 0.0) > 0 else 0.0
            ),
            'dhrd_throughput_qps': dhrd_results['throughput_qps'],
            'baseline_throughput_qps': baseline_results.get('throughput_qps', 0.0),
            'throughput_speedup': (
                dhrd_results['throughput_qps'] / baseline_results.get('throughput_qps', 1.0)
                if baseline_results.get('throughput_qps', 0.0) > 0 else 0.0
            ),
        }

        self.logger("\n" + "=" * 60)
        self.logger("DHRD vs Baseline Comparison")
        self.logger("=" * 60)
        self.logger(f"DHRD Accuracy:     {comparison['dhrd_accuracy']:.4f}")
        self.logger(f"Baseline Accuracy: {comparison['baseline_accuracy']:.4f}")
        self.logger(f"Absolute Gain:     {comparison['accuracy_improvement']:.4f}")
        self.logger(f"Relative Gain:     {comparison['relative_accuracy_gain']:.2f}%")
        self.logger(f"\nDHRD Throughput:     {comparison['dhrd_throughput_qps']:.2f} QPS")
        self.logger(f"Baseline Throughput: {comparison['baseline_throughput_qps']:.2f} QPS")
        self.logger(f"Speedup:             {comparison['throughput_speedup']:.2f}x")
        self.logger("=" * 60 + "\n")

        return comparison

    def benchmark_inference(
        self,
        dataloader: DataLoader,
        num_runs: int = 3,
    ) -> Dict[str, float]:
        """
        Benchmark inference throughput and latency.

        Args:
            dataloader: DataLoader for benchmarking
            num_runs: Number of runs for averaging

        Returns:
            Dictionary with benchmark metrics
        """
        self.logger(f"Running inference benchmark ({num_runs} runs)...")

        throughputs = []
        latencies = []

        for run in range(num_runs):
            self.logger(f"  Run {run + 1}/{num_runs}...")

            total_samples = 0
            start_time = time.time()

            with torch.no_grad():
                for batch in dataloader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)

                    # Inference
                    _ = self.model.predict(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )

                    total_samples += input_ids.size(0)

            end_time = time.time()
            elapsed_time = end_time - start_time

            throughput = total_samples / elapsed_time
            avg_latency = (elapsed_time / total_samples) * 1000  # ms

            throughputs.append(throughput)
            latencies.append(avg_latency)

        benchmark_results = {
            'avg_throughput_qps': float(np.mean(throughputs)),
            'std_throughput_qps': float(np.std(throughputs)),
            'avg_latency_ms': float(np.mean(latencies)),
            'std_latency_ms': float(np.std(latencies)),
            'num_runs': num_runs,
        }

        self.logger("\nBenchmark Results:")
        self.logger(f"  Throughput: {benchmark_results['avg_throughput_qps']:.2f} ± "
                   f"{benchmark_results['std_throughput_qps']:.2f} QPS")
        self.logger(f"  Latency:    {benchmark_results['avg_latency_ms']:.3f} ± "
                   f"{benchmark_results['std_latency_ms']:.3f} ms")

        return benchmark_results
