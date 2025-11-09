# ReDSM5 Processed Dataset for DHRD

This directory contains the processed ReDSM5 dataset in DHRD format for binary classification of DSM-5 depression criteria.

## Dataset Overview

- **Task**: Binary classification for depression criteria detection
- **Source**: ReDSM5 dataset (arXiv:2508.03399)
- **Total Posts**: 1,484
- **Class Distribution**:
  - Status=1 (criteria present): 1,126 posts (75.9%)
  - Status=0 (no criteria): 358 posts (24.1%)

## Data Splits

| Split | Samples | Status=1 | Status=0 |
|-------|---------|----------|----------|
| Train | 1,038   | 788      | 250      |
| Eval  | 223     | 169      | 54       |
| Test  | 223     | 169      | 54       |

## File Format

### JSON Files (train.json, eval.json, test.json)

Each file contains a list of dictionaries with the following structure:

```json
{
  "input_text": "Full Reddit post text...",
  "label": 1,
  "rationale": "Clinical explanation from psychologist..."
}
```

**Fields**:
- `input_text` (str): Full post text from Reddit
- `label` (int): Binary label
  - `1`: Post contains evidence of DSM-5 depression criteria
  - `0`: Post does not contain depression criteria
- `rationale` (str): Clinical explanation from licensed psychologist (training set only)

### CSV File (redsm5_post_level.csv)

Complete dataset with additional metadata:

| Column | Description |
|--------|-------------|
| post_id | Unique post identifier |
| input_text | Full post text |
| label | Binary label (0 or 1) |
| rationale | Combined clinical explanations |
| DSM5_symptom | Comma-separated list of symptoms detected |

## Dataset Statistics

### Post Length (words)
- Mean: 294.8
- Median: 101.0
- Min: 2
- Max: 6,990

### Rationale Length (words)
- Mean: 76.6
- Median: 60.0
- Min: 0
- Max: 996

## Aggregation Logic

The original ReDSM5 dataset provides sentence-level annotations. We aggregated to post-level using:

1. **Label Aggregation**: If ANY sentence in a post has status=1, the post is labeled 1
2. **Rationale Aggregation**: All clinical explanations for a post are concatenated

## Usage

### Training

```bash
python scripts/train.py --config configs/redsm5_config.yaml
```

### Evaluation

```bash
python scripts/evaluate.py \
  --config configs/redsm5_config.yaml \
  --checkpoint outputs/redsm5_dhrd_experiment/best_model.pt
```

### Data Regeneration

To regenerate this processed dataset:

```bash
python scripts/prepare_redsm5.py \
  --posts_path data/redsm5/redsm5_posts.csv \
  --annotations_path data/redsm5/redsm5_annotations.csv \
  --output_dir data/redsm5_processed
```

## DSM-5 Criteria Categories

The dataset includes annotations for 9 DSM-5 major depressive episode symptoms:

1. **DEPRESSED_MOOD**: Persistent sad or empty mood
2. **ANHEDONIA**: Loss of interest or pleasure
3. **APPETITE_CHANGE**: Significant weight or appetite changes
4. **SLEEP_ISSUES**: Insomnia or hypersomnia
5. **PSYCHOMOTOR**: Psychomotor agitation or retardation
6. **FATIGUE**: Loss of energy or fatigue
7. **WORTHLESSNESS**: Feelings of worthlessness or guilt
8. **COGNITIVE_ISSUES**: Diminished concentration or indecisiveness
9. **SUICIDAL_THOUGHTS**: Recurrent thoughts of death or suicide

Plus:
- **SPECIAL_CASE**: Non-DSM-5 clinical or positive discriminations

## Citation

If you use this processed dataset, please cite both papers:

**ReDSM5 Dataset**:
```bibtex
@misc{bao2025redsm5,
  title        = {ReDSM5: A Reddit Dataset for DSM-5 Depression Detection},
  author       = {Eliseo Bao and Anxo Pérez and Javier Parapar},
  year         = {2025},
  eprint       = {2508.03399},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL},
  url          = {https://arxiv.org/abs/2508.03399}
}
```

**DHRD Method**:
```bibtex
@article{xu2025dualhead,
  title={Dual-Head Reasoning Distillation: Improving Classifier Accuracy with Train-Time-Only Reasoning},
  author={Xu, Jillian and Zhou, Dylan and Shukla, Vinay and Yang, Yang and Ruan, Junrui and Lin, Shuhuai and Zou, Wenfei and Liu, Yinxiao and Lakshmanan, Karthik},
  journal={arXiv preprint arXiv:2509.21487},
  year={2025}
}
```

## License

This processed dataset inherits the license from the original ReDSM5 dataset (Apache 2.0).
