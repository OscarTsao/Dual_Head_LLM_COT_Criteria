"""
Configuration utilities for DHRD experiments
"""

import yaml
from typing import Dict, Any
from pathlib import Path
import argparse


class DHRDConfig:
    """
    Configuration class for DHRD experiments.
    """

    def __init__(self, config_dict: Dict[str, Any]):
        """
        Args:
            config_dict: Dictionary containing configuration
        """
        self._config = config_dict

        # Model config
        self.model_name_or_path = config_dict['model']['name_or_path']
        self.num_labels = config_dict['model']['num_labels']
        self.dropout_prob = config_dict['model']['dropout_prob']
        self.loss_weight_alpha = config_dict['model']['loss_weight_alpha']

        # Data config
        self.train_file = config_dict['data']['train_file']
        self.eval_file = config_dict['data']['eval_file']
        self.test_file = config_dict['data']['test_file']
        self.max_input_length = config_dict['data']['max_input_length']
        self.max_rationale_length = config_dict['data']['max_rationale_length']
        self.batch_size = config_dict['data']['batch_size']
        self.num_workers = config_dict['data']['num_workers']

        # Training config
        self.num_epochs = config_dict['training']['num_epochs']
        self.learning_rate = config_dict['training']['learning_rate']
        self.weight_decay = config_dict['training']['weight_decay']
        self.warmup_steps = config_dict['training']['warmup_steps']
        self.max_grad_norm = config_dict['training']['max_grad_norm']
        self.log_interval = config_dict['training']['log_interval']
        self.eval_interval = config_dict['training']['eval_interval']
        self.save_dir = config_dict['training']['save_dir']

        # Optimizer config
        self.optimizer_type = config_dict['training']['optimizer']['type']
        self.optimizer_betas = config_dict['training']['optimizer']['betas']
        self.optimizer_eps = config_dict['training']['optimizer']['eps']

        # Scheduler config
        self.scheduler_type = config_dict['training']['scheduler']['type']
        self.warmup_ratio = config_dict['training']['scheduler']['warmup_ratio']

        # Evaluation config
        self.eval_batch_size = config_dict['evaluation']['batch_size']
        self.compute_throughput = config_dict['evaluation']['compute_throughput']
        self.benchmark_runs = config_dict['evaluation']['benchmark_runs']

        # Hardware config
        self.device = config_dict['hardware']['device']
        self.mixed_precision = config_dict['hardware']['mixed_precision']
        self.gradient_accumulation_steps = config_dict['hardware']['gradient_accumulation_steps']

        # SuperGLUE config
        self.superglue_task_name = config_dict['superglue']['task_name']
        self.use_superglue = config_dict['superglue']['use_superglue']

        # Logging config
        self.use_mlflow = config_dict['logging']['use_mlflow']
        self.experiment_name = config_dict['logging']['experiment_name']
        self.tracking_uri = config_dict['logging']['tracking_uri']
        self.log_model = config_dict['logging']['log_model']

        # Reproducibility
        self.seed = config_dict['seed']

    @classmethod
    def from_yaml(cls, config_path: str) -> 'DHRDConfig':
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            DHRDConfig instance
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of config
        """
        return self._config

    def __repr__(self) -> str:
        """String representation of config."""
        lines = ["DHRDConfig:"]
        for key, value in self._config.items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create argument parser for DHRD training/evaluation scripts.

    Returns:
        ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Dual-Head Reasoning Distillation (DHRD) Training/Evaluation"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/dhrd_config.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Override model name from config"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size from config"
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Override learning rate from config"
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Override number of epochs from config"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device from config (cuda/cpu)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed from config"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory from config"
    )

    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Only run evaluation (no training)"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to load"
    )

    return parser


def override_config(config: DHRDConfig, args: argparse.Namespace) -> DHRDConfig:
    """
    Override configuration with command-line arguments.

    Args:
        config: DHRDConfig instance
        args: Parsed command-line arguments

    Returns:
        Updated DHRDConfig instance
    """
    # Override model name
    if args.model_name is not None:
        config.model_name_or_path = args.model_name

    # Override batch size
    if args.batch_size is not None:
        config.batch_size = args.batch_size

    # Override learning rate
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate

    # Override num epochs
    if args.num_epochs is not None:
        config.num_epochs = args.num_epochs

    # Override device
    if args.device is not None:
        config.device = args.device

    # Override seed
    if args.seed is not None:
        config.seed = args.seed

    # Override output directory
    if args.output_dir is not None:
        config.save_dir = args.output_dir

    return config
