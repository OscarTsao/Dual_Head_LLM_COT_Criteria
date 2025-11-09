# Dual-Head Reasoning Distillation (DHRD)

Implementation of **Dual-Head Reasoning Distillation: Improving Classifier Accuracy with Train-Time-Only Reasoning** ([arXiv:2509.21487](https://arxiv.org/abs/2509.21487))

## Overview

DHRD is a novel training method that improves classifier accuracy by leveraging Chain-of-Thought (CoT) reasoning **only during training**, avoiding the inference-time throughput penalty of traditional CoT methods.

### Key Features

- **Dual-Head Architecture**:
  - Pooled classification head (used during training and inference)
  - Reasoning head for language modeling (used only during training)
- **Combined Loss Function**: Weighted sum of classification and language modeling losses
- **High Throughput**: 96-142x faster than CoT decoding at inference time
- **Improved Accuracy**: 0.65-5.47% relative gains over pooled baselines on SuperGLUE tasks

## Architecture

```
Input Text
    ↓
[Transformer Encoder]
    ↓
    ├─→ [Pooled Classification Head] → Predictions (Train & Inference)
    └─→ [Reasoning Head] → Rationale Generation (Train Only)
```

During training, both heads are supervised:
- Classification head learns from labels
- Reasoning head learns from teacher rationales

During inference, only the classification head is used for fast predictions.

## Installation

Python 3.10+ recommended. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e '.[dev]'
```

## Quick Start

### Option A: Using ReDSM5 Dataset (Depression Criteria Detection)

We provide a pre-configured setup for the **ReDSM5** dataset - a Reddit dataset for DSM-5 depression detection with expert clinical rationales.

#### 1. Prepare ReDSM5 Data

```bash
python scripts/prepare_redsm5.py
```

This will create train/eval/test splits in `data/redsm5_processed/`:
- **1,038 training samples** (788 positive, 250 negative)
- **223 evaluation samples** (169 positive, 54 negative)
- **223 test samples** (169 positive, 54 negative)

#### 2. Train on ReDSM5

```bash
python scripts/train.py --config configs/redsm5_config.yaml
```

#### 3. Evaluate on ReDSM5

```bash
python scripts/evaluate.py \
  --config configs/redsm5_config.yaml \
  --checkpoint outputs/redsm5_dhrd_experiment/best_model.pt
```

### Option B: Using Custom Data

#### 1. Prepare Your Data

Create a JSON file with input-label-rationale triplets:

```json
[
  {
    "input_text": "Question or input text here",
    "label": 0,
    "rationale": "Chain-of-thought reasoning explanation"
  }
]
```

See `data/example_train.json` for examples.

#### 2. Configure Your Experiment

Edit `configs/dhrd_config.yaml`:

```yaml
model:
  name_or_path: "gpt2"  # or any decoder-only LM
  num_labels: 2
  loss_weight_alpha: 0.5  # Weight for classification vs LM loss

data:
  train_file: "data/train.json"
  eval_file: "data/eval.json"
  batch_size: 8

training:
  num_epochs: 3
  learning_rate: 5.0e-5
```

#### 3. Train the Model

```bash
python scripts/train.py --config configs/dhrd_config.yaml
```

Optional arguments:
- `--batch_size`: Override batch size
- `--learning_rate`: Override learning rate
- `--num_epochs`: Override number of epochs
- `--device`: cuda or cpu
- `--checkpoint`: Resume from checkpoint

#### 4. Evaluate the Model

```bash
python scripts/evaluate.py \
  --config configs/dhrd_config.yaml \
  --checkpoint outputs/dhrd_experiment/best_model.pt
```

## Project Structure

```
.
├── configs/
│   └── dhrd_config.yaml          # Experiment configuration
├── data/
│   └── example_train.json        # Example dataset
├── scripts/
│   ├── train.py                  # Training script
│   └── evaluate.py               # Evaluation script
├── src/Project/SubProject/
│   ├── models/
│   │   └── model.py              # Dual-Head model implementation
│   ├── data/
│   │   └── dataset.py            # Dataset classes
│   ├── engine/
│   │   ├── train_engine.py       # Training engine
│   │   └── eval_engine.py        # Evaluation engine
│   └── utils/
│       ├── config.py             # Configuration utilities
│       ├── log.py                # Logging utilities
│       ├── seed.py               # Random seed utilities
│       └── mlflow_utils.py       # MLflow integration
├── outputs/                       # Saved models and results
└── mlruns/                        # MLflow tracking data
```

## Key Components

### DualHeadReasoningModel

The main model class combining classification and reasoning heads:

```python
from Project.SubProject.models.model import DualHeadReasoningModel

model = DualHeadReasoningModel(
    model_name_or_path="gpt2",
    num_labels=2,
    dropout_prob=0.1,
    loss_weight_alpha=0.5
)
```

### DHRDDataset

Dataset class for handling input-label-rationale triplets:

```python
from Project.SubProject.data.dataset import load_json_dataset

dataset = load_json_dataset(
    file_path="data/train.json",
    tokenizer=tokenizer,
    max_input_length=512,
    max_rationale_length=1024
)
```

### DHRDTrainer

Training engine with dual-head training logic:

```python
from Project.SubProject.engine.train_engine import DHRDTrainer

trainer = DHRDTrainer(
    model=model,
    train_dataloader=train_loader,
    eval_dataloader=eval_loader,
    optimizer=optimizer,
    device="cuda"
)

trainer.train(num_epochs=3)
```

### DHRDEvaluator

Evaluation engine using only the classification head:

```python
from Project.SubProject.engine.eval_engine import DHRDEvaluator

evaluator = DHRDEvaluator(model=model, device="cuda")
results = evaluator.evaluate(test_loader)
```

## Performance Metrics

The paper reports the following improvements on SuperGLUE tasks:

| Task | Baseline Acc | DHRD Acc | Relative Gain | Throughput Speedup |
|------|--------------|----------|---------------|-------------------|
| BoolQ | - | - | 0.65-5.47% | 96-142x |
| CB | - | - | Notable gains | 96-142x |
| COPA | - | - | Notable gains | 96-142x |
| RTE | - | - | Notable gains | 96-142x |

## MLflow Integration

Track experiments with MLflow:

```python
from Project.SubProject.utils import configure_mlflow, enable_autologging, mlflow_run

configure_mlflow(tracking_uri="file:./mlruns", experiment="dhrd")
enable_autologging()

with mlflow_run("training", tags={"model": "gpt2"}):
    trainer.train(num_epochs=3)
```

View results:
```bash
mlflow ui
```

## Development

Run linters/formatters:
```bash
ruff check src tests
black src tests
```

Run tests:
```bash
pytest
```

## ReDSM5 Dataset

This repository includes support for the **ReDSM5** dataset - a corpus of 1,484 Reddit posts annotated for DSM-5 major depressive episode symptoms by licensed psychologists.

### Dataset Statistics

- **Total Posts**: 1,484
- **Binary Classification**:
  - Positive (criteria present): 1,126 posts (75.9%)
  - Negative (no criteria): 358 posts (24.1%)
- **Clinical Rationales**: Expert explanations from psychologists
- **Average Post Length**: 295 words
- **Average Rationale Length**: 77 words

### DSM-5 Symptoms Covered

The dataset includes annotations for 9 DSM-5 major depressive episode symptoms:
- Depressed mood
- Anhedonia (loss of interest)
- Appetite changes
- Sleep issues
- Psychomotor changes
- Fatigue
- Worthlessness/guilt
- Cognitive issues
- Suicidal thoughts

### Data Preparation

```bash
# Convert ReDSM5 to DHRD format
python scripts/prepare_redsm5.py

# This creates:
# - data/redsm5_processed/train.json (1,038 samples)
# - data/redsm5_processed/eval.json (223 samples)
# - data/redsm5_processed/test.json (223 samples)
```

### Training on ReDSM5

```bash
# Train DHRD model on ReDSM5
python scripts/train.py --config configs/redsm5_config.yaml

# Evaluate
python scripts/evaluate.py \
  --config configs/redsm5_config.yaml \
  --checkpoint outputs/redsm5_dhrd_experiment/best_model.pt
```

See `data/redsm5_processed/README.md` for detailed documentation.

## Citation

If you use this implementation, please cite the DHRD paper:

```bibtex
@article{xu2025dualhead,
  title={Dual-Head Reasoning Distillation: Improving Classifier Accuracy with Train-Time-Only Reasoning},
  author={Xu, Jillian and Zhou, Dylan and Shukla, Vinay and Yang, Yang and Ruan, Junrui and Lin, Shuhuai and Zou, Wenfei and Liu, Yinxiao and Lakshmanan, Karthik},
  journal={arXiv preprint arXiv:2509.21487},
  year={2025}
}
```

If you use the ReDSM5 dataset, please also cite:

```bibtex
@misc{bao2025redsm5,
  title        = {ReDSM5: A Reddit Dataset for DSM-5 Depression Detection},
  author       = {Eliseo Bao and Anxo Pérez and Javier Parapar},
  year         = {2025},
  eprint       = {2508.03399},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL},
  url          = {https://arxiv.org/abs/2508.03399},
  note         = {Accepted at CIKM 2025}
}
```

## License

MIT License

