"""
Dataset classes for Dual-Head Reasoning Distillation (DHRD)

Handles input-label-rationale triplets for training and evaluation.
"""

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import Dict, List, Optional, Union
import json


class DHRDDataset(Dataset):
    """
    Dataset for Dual-Head Reasoning Distillation.

    Expected data format:
    - input_text: The input text for classification
    - label: The classification label (integer or string)
    - rationale: Chain-of-thought reasoning text (optional, for training)
    """

    def __init__(
        self,
        data: List[Dict[str, Union[str, int]]],
        tokenizer: PreTrainedTokenizer,
        max_input_length: int = 512,
        max_rationale_length: int = 512,
        label2id: Optional[Dict[str, int]] = None,
        include_rationale: bool = True,
    ):
        """
        Args:
            data: List of dictionaries with keys: 'input_text', 'label', 'rationale'
            tokenizer: Tokenizer for encoding text
            max_input_length: Maximum length for input sequences
            max_rationale_length: Maximum length for input+rationale sequences
            label2id: Mapping from label strings to IDs (if labels are strings)
            include_rationale: Whether to include rationale (set False for inference)
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_rationale_length = max_rationale_length
        self.include_rationale = include_rationale

        # Build label mapping if not provided
        if label2id is None:
            unique_labels = sorted(set(item['label'] for item in data))
            if isinstance(unique_labels[0], str):
                self.label2id = {label: idx for idx, label in enumerate(unique_labels)}
            else:
                self.label2id = {label: label for label in unique_labels}
        else:
            self.label2id = label2id

        self.id2label = {v: k for k, v in self.label2id.items()}
        self.num_labels = len(self.label2id)

        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary containing:
                - input_ids: Tokenized input
                - attention_mask: Attention mask for input
                - labels: Classification label
                - rationale_input_ids: Tokenized input+rationale (if available)
                - rationale_attention_mask: Attention mask for rationale (if available)
        """
        item = self.data[idx]

        # Tokenize input
        input_encoding = self.tokenizer(
            item['input_text'],
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Prepare output
        output = {
            'input_ids': input_encoding['input_ids'].squeeze(0),
            'attention_mask': input_encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.label2id[item['label']], dtype=torch.long)
        }

        # Include rationale if available and requested
        if self.include_rationale and 'rationale' in item and item['rationale']:
            # Combine input and rationale
            combined_text = f"{item['input_text']}\n\nReasoning: {item['rationale']}"

            rationale_encoding = self.tokenizer(
                combined_text,
                max_length=self.max_rationale_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            output['rationale_input_ids'] = rationale_encoding['input_ids'].squeeze(0)
            output['rationale_attention_mask'] = rationale_encoding['attention_mask'].squeeze(0)

        return output


class SuperGLUEDataset(DHRDDataset):
    """
    Dataset wrapper for SuperGLUE tasks.

    Handles various SuperGLUE task formats and converts them to the DHRD format.
    """

    TASK_FORMATS = {
        'boolq': {
            'text_fields': ['passage', 'question'],
            'label_field': 'label',
        },
        'cb': {
            'text_fields': ['premise', 'hypothesis'],
            'label_field': 'label',
        },
        'copa': {
            'text_fields': ['premise', 'question', 'choice1', 'choice2'],
            'label_field': 'label',
        },
        'multirc': {
            'text_fields': ['paragraph', 'question', 'answer'],
            'label_field': 'label',
        },
        'rte': {
            'text_fields': ['premise', 'hypothesis'],
            'label_field': 'label',
        },
        'wic': {
            'text_fields': ['sentence1', 'sentence2', 'word'],
            'label_field': 'label',
        },
        'wsc': {
            'text_fields': ['text', 'span1_text', 'span2_text'],
            'label_field': 'label',
        },
    }

    @classmethod
    def from_superglue(
        cls,
        task_name: str,
        raw_data: List[Dict],
        tokenizer: PreTrainedTokenizer,
        **kwargs
    ):
        """
        Create dataset from SuperGLUE task data.

        Args:
            task_name: Name of the SuperGLUE task
            raw_data: Raw data from SuperGLUE
            tokenizer: Tokenizer
            **kwargs: Additional arguments for DHRDDataset
        """
        if task_name.lower() not in cls.TASK_FORMATS:
            raise ValueError(f"Unknown task: {task_name}. Supported tasks: {list(cls.TASK_FORMATS.keys())}")

        task_format = cls.TASK_FORMATS[task_name.lower()]

        # Convert to DHRD format
        formatted_data = []
        for item in raw_data:
            # Combine text fields
            text_parts = [str(item.get(field, '')) for field in task_format['text_fields']]
            input_text = ' '.join(filter(None, text_parts))

            formatted_item = {
                'input_text': input_text,
                'label': item[task_format['label_field']],
                'rationale': item.get('rationale', '')
            }
            formatted_data.append(formatted_item)

        return cls(data=formatted_data, tokenizer=tokenizer, **kwargs)


def load_json_dataset(
    file_path: str,
    tokenizer: PreTrainedTokenizer,
    **kwargs
) -> DHRDDataset:
    """
    Load dataset from a JSON file.

    Expected JSON format:
    [
        {
            "input_text": "...",
            "label": 0 or "label_name",
            "rationale": "..." (optional)
        },
        ...
    ]

    Args:
        file_path: Path to JSON file
        tokenizer: Tokenizer
        **kwargs: Additional arguments for DHRDDataset

    Returns:
        DHRDDataset instance
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return DHRDDataset(data=data, tokenizer=tokenizer, **kwargs)


def create_dataloaders(
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    batch_size: int = 8,
    num_workers: int = 0,
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create data loaders for training and evaluation.

    Args:
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset (optional)
        batch_size: Batch size
        num_workers: Number of workers for data loading

    Returns:
        Dictionary with 'train' and optionally 'eval' dataloaders
    """
    from torch.utils.data import DataLoader

    dataloaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
    }

    if eval_dataset is not None:
        dataloaders['eval'] = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    return dataloaders
