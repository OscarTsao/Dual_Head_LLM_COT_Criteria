# DHRD Architecture Documentation

## Overview

This document provides a detailed technical overview of the Dual-Head Reasoning Distillation (DHRD) implementation based on the paper "Dual-Head Reasoning Distillation: Improving Classifier Accuracy with Train-Time-Only Reasoning" (arXiv:2509.21487).

## Problem Statement

Traditional Chain-of-Thought (CoT) prompting improves classification accuracy but introduces a significant inference-time throughput penalty due to rationale generation. DHRD addresses this trade-off by using reasoning only during training.

## Architecture Components

### 1. Base Transformer Model

The foundation is a decoder-only language model (e.g., GPT-2, LLaMA):

```python
self.transformer = AutoModelForCausalLM.from_pretrained(model_name_or_path)
```

Key characteristics:
- Decoder-only architecture (causal attention)
- Pretrained on large text corpora
- Produces contextualized hidden states

### 2. Pooled Classification Head

**Purpose**: Generate classification predictions from input representations

**Architecture**:
```python
class PooledClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout_prob):
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)
```

**Pooling Strategy**:
- Uses the last valid token position for each sequence
- Falls back to mean pooling if no mask provided
- Applies dropout for regularization

**Usage**:
- Active during both training and inference
- Supervised by classification labels
- Enables fast predictions without rationale generation

### 3. Reasoning Head

**Purpose**: Learn to generate rationales during training

**Architecture**:
```python
class ReasoningHead(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
```

**Characteristics**:
- Standard language modeling head
- Predicts next token probabilities
- Supervised by teacher rationales

**Usage**:
- Active only during training
- Disabled at inference time
- Distills reasoning knowledge into the base model

## Training Process

### Forward Pass

1. **Input Processing**:
   - Input text → Tokenization → Token IDs
   - Attention mask for padding handling

2. **Classification Branch**:
   ```python
   hidden_states = transformer(input_ids, attention_mask)
   classification_logits = pooled_head(hidden_states)
   ```

3. **Reasoning Branch** (training only):
   ```python
   # Combine input + rationale
   combined_text = input_text + "\n\nReasoning: " + rationale
   rationale_hidden = transformer(combined_ids)
   lm_logits = reasoning_head(rationale_hidden)
   ```

### Loss Computation

**Combined Loss**:
```
L = α × L_cls + (1 - α) × L_lm
```

Where:
- **L_cls**: Cross-entropy loss for classification
  ```python
  L_cls = CrossEntropy(classification_logits, labels)
  ```

- **L_lm**: Language modeling loss for rationale generation
  ```python
  L_lm = CrossEntropy(shift_logits, shift_labels)
  ```

- **α**: Loss weight hyperparameter (default: 0.5)

**Loss Balancing**:
- α = 0.0: Pure language modeling (baseline LM)
- α = 0.5: Equal weighting (recommended)
- α = 1.0: Pure classification (baseline classifier)

### Optimization

- **Optimizer**: AdamW
  - Learning rate: 5e-5
  - Weight decay: 0.01
  - Betas: (0.9, 0.999)

- **Scheduler**: Linear warmup + decay
  - Warmup ratio: 0.1
  - Total steps: num_batches × num_epochs

- **Gradient Clipping**: Max norm = 1.0

## Inference Process

### Fast Classification

During inference, DHRD uses only the classification head:

```python
def predict(self, input_ids, attention_mask):
    with torch.no_grad():
        hidden_states = self.transformer(input_ids, attention_mask)
        classification_logits = self.pooled_head(hidden_states)
        predictions = argmax(classification_logits)
    return predictions
```

**Advantages**:
- No rationale generation → Low latency
- Single forward pass → High throughput
- Same efficiency as pooled baseline

**Performance**:
- **96-142x faster** than CoT decoding
- **0.65-5.47% accuracy gain** over pooled baseline

## Data Format

### Training Data

Input-label-rationale triplets:

```json
{
  "input_text": "The question or input to classify",
  "label": 0,
  "rationale": "Step-by-step reasoning leading to the answer"
}
```

### Rationale Sources

1. **Teacher Models**: Generate rationales using larger LMs
2. **Human Annotations**: Expert-written reasoning chains
3. **Synthetic Generation**: Programmatic rationale creation

## Implementation Details

### Tokenization

```python
# Input tokenization
input_encoding = tokenizer(
    input_text,
    max_length=512,
    padding='max_length',
    truncation=True
)

# Input + rationale tokenization
combined_text = f"{input_text}\n\nReasoning: {rationale}"
rationale_encoding = tokenizer(
    combined_text,
    max_length=1024,
    padding='max_length',
    truncation=True
)
```

### Memory Optimization

1. **Gradient Accumulation**: Simulate larger batch sizes
2. **Mixed Precision**: Use FP16 for faster training
3. **Gradient Checkpointing**: Reduce memory at cost of speed

### Hyperparameters

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| loss_weight_alpha | 0.5 | [0.0, 1.0] | Balance classification vs LM loss |
| learning_rate | 5e-5 | [1e-5, 1e-4] | Higher for smaller models |
| batch_size | 8 | [4, 32] | Depends on GPU memory |
| max_input_length | 512 | [128, 2048] | Shorter for efficiency |
| max_rationale_length | 1024 | [256, 4096] | Longer for complex reasoning |
| dropout_prob | 0.1 | [0.0, 0.3] | Regularization strength |

## Evaluation Metrics

### Classification Metrics

- **Accuracy**: Overall correctness
- **Precision**: Class-specific correctness
- **Recall**: Class-specific coverage
- **F1 Score**: Harmonic mean of precision and recall

### Throughput Metrics

- **QPS (Queries Per Second)**: Inference throughput
- **Latency (ms)**: Average time per prediction
- **Speedup**: Improvement over CoT baseline

## Comparison with Baselines

### vs. Pooled Classifier

- **Accuracy**: +0.65-5.47% (DHRD wins)
- **Throughput**: ~Same (tied)
- **Conclusion**: DHRD improves accuracy without sacrificing speed

### vs. CoT Prompting

- **Accuracy**: Comparable (similar)
- **Throughput**: 96-142x faster (DHRD wins)
- **Conclusion**: DHRD achieves similar accuracy with much higher throughput

## Best Practices

### 1. Data Preparation

- Ensure high-quality rationales
- Balance rationale length and informativeness
- Include diverse reasoning patterns

### 2. Model Selection

- Start with smaller models (GPT-2) for prototyping
- Scale to larger models (LLaMA) for production
- Match model size to task complexity

### 3. Hyperparameter Tuning

- Grid search over α ∈ {0.3, 0.5, 0.7}
- Adjust learning rate based on model size
- Monitor both classification and LM losses

### 4. Evaluation

- Always benchmark throughput vs. CoT baseline
- Report accuracy on held-out test set
- Analyze per-class performance

## Limitations

1. **Rationale Dependency**: Requires access to rationales during training
2. **Task Specificity**: Best for tasks where reasoning improves accuracy
3. **Model Capacity**: Requires sufficient model capacity to learn from rationales

## Future Extensions

1. **Multi-Task Learning**: Train on multiple classification tasks simultaneously
2. **Active Learning**: Select which examples need rationales
3. **Self-Distillation**: Generate rationales with the model itself
4. **Continuous Learning**: Update model with new rationales over time

## References

- **Paper**: [Dual-Head Reasoning Distillation (arXiv:2509.21487)](https://arxiv.org/abs/2509.21487)
- **Authors**: Jillian Xu, Dylan Zhou, Vinay Shukla, et al.
- **Published**: September 2025
