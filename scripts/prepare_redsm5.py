#!/usr/bin/env python3
"""
Data preparation script for ReDSM5 dataset

Converts ReDSM5 dataset to DHRD format for binary classification:
- Input: Full post text from redsm5_posts.csv
- Label: Binary status (0 or 1) aggregated from annotations
- Rationale: Clinical explanations from expert annotations

Usage:
    python scripts/prepare_redsm5.py --output_dir data/redsm5_processed
"""

import pandas as pd
import json
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def load_redsm5_data(posts_path: str, annotations_path: str) -> pd.DataFrame:
    """
    Load and merge ReDSM5 posts and annotations.

    Args:
        posts_path: Path to redsm5_posts.csv
        annotations_path: Path to redsm5_annotations.csv

    Returns:
        Merged dataframe
    """
    print(f"Loading posts from {posts_path}...")
    posts_df = pd.read_csv(posts_path)
    print(f"  Loaded {len(posts_df)} posts")

    print(f"Loading annotations from {annotations_path}...")
    annotations_df = pd.read_csv(annotations_path)
    print(f"  Loaded {len(annotations_df)} annotations")

    return posts_df, annotations_df


def aggregate_annotations(posts_df: pd.DataFrame, annotations_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sentence-level annotations to post-level for binary classification.

    Binary classification logic:
    - If ANY annotation for a post has status=1, the post is labeled 1
    - Otherwise, the post is labeled 0
    - Combine all explanations as the rationale

    Args:
        posts_df: Posts dataframe
        annotations_df: Annotations dataframe

    Returns:
        Aggregated dataframe with columns: post_id, text, label, rationale
    """
    print("\nAggregating annotations to post level...")

    # Group annotations by post_id
    post_annotations = annotations_df.groupby('post_id').agg({
        'status': 'max',  # If any status=1, post is labeled 1
        'explanation': lambda x: ' '.join(x.astype(str)),  # Combine all explanations
        'DSM5_symptom': lambda x: ', '.join(x.unique()),  # List unique symptoms
    }).reset_index()

    # Merge with posts
    merged_df = posts_df.merge(post_annotations, on='post_id', how='left')

    # Fill NaN values for posts without annotations
    merged_df['status'] = merged_df['status'].fillna(0).astype(int)
    merged_df['explanation'] = merged_df['explanation'].fillna('')
    merged_df['DSM5_symptom'] = merged_df['DSM5_symptom'].fillna('NONE')

    # Rename columns to match DHRD format
    merged_df = merged_df.rename(columns={
        'text': 'input_text',
        'status': 'label',
        'explanation': 'rationale'
    })

    print(f"  Total posts: {len(merged_df)}")
    print(f"  Posts with status=1: {(merged_df['label'] == 1).sum()}")
    print(f"  Posts with status=0: {(merged_df['label'] == 0).sum()}")

    return merged_df[['post_id', 'input_text', 'label', 'rationale', 'DSM5_symptom']]


def create_train_eval_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    eval_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> tuple:
    """
    Split data into train, eval, and test sets.

    Args:
        df: Input dataframe
        train_ratio: Proportion for training set
        eval_ratio: Proportion for evaluation set
        test_ratio: Proportion for test set
        random_state: Random seed

    Returns:
        Tuple of (train_df, eval_df, test_df)
    """
    assert abs(train_ratio + eval_ratio + test_ratio - 1.0) < 1e-6, \
        "Train, eval, and test ratios must sum to 1.0"

    print(f"\nSplitting data (train={train_ratio}, eval={eval_ratio}, test={test_ratio})...")

    # First split: train vs (eval + test)
    train_df, temp_df = train_test_split(
        df,
        train_size=train_ratio,
        random_state=random_state,
        stratify=df['label']
    )

    # Second split: eval vs test
    eval_size = eval_ratio / (eval_ratio + test_ratio)
    eval_df, test_df = train_test_split(
        temp_df,
        train_size=eval_size,
        random_state=random_state,
        stratify=temp_df['label']
    )

    print(f"  Train set: {len(train_df)} samples")
    print(f"    - Status=1: {(train_df['label'] == 1).sum()}")
    print(f"    - Status=0: {(train_df['label'] == 0).sum()}")
    print(f"  Eval set: {len(eval_df)} samples")
    print(f"    - Status=1: {(eval_df['label'] == 1).sum()}")
    print(f"    - Status=0: {(eval_df['label'] == 0).sum()}")
    print(f"  Test set: {len(test_df)} samples")
    print(f"    - Status=1: {(test_df['label'] == 1).sum()}")
    print(f"    - Status=0: {(test_df['label'] == 0).sum()}")

    return train_df, eval_df, test_df


def save_to_json(df: pd.DataFrame, output_path: str, include_rationale: bool = True):
    """
    Save dataframe to JSON format for DHRD.

    Args:
        df: Input dataframe
        output_path: Output JSON file path
        include_rationale: Whether to include rationale field
    """
    records = []
    for _, row in df.iterrows():
        record = {
            'input_text': row['input_text'],
            'label': int(row['label']),
        }

        if include_rationale and row['rationale']:
            record['rationale'] = row['rationale']

        records.append(record)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"  Saved {len(records)} records to {output_path}")


def print_statistics(df: pd.DataFrame):
    """Print dataset statistics."""
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    print(f"Total samples: {len(df)}")
    print(f"Label distribution:")
    print(f"  Status=1: {(df['label'] == 1).sum()} ({(df['label'] == 1).sum() / len(df) * 100:.1f}%)")
    print(f"  Status=0: {(df['label'] == 0).sum()} ({(df['label'] == 0).sum() / len(df) * 100:.1f}%)")
    print(f"\nPost length statistics (words):")
    df['word_count'] = df['input_text'].str.split().str.len()
    print(f"  Mean: {df['word_count'].mean():.1f}")
    print(f"  Median: {df['word_count'].median():.1f}")
    print(f"  Min: {df['word_count'].min()}")
    print(f"  Max: {df['word_count'].max()}")
    print(f"\nRationale length statistics (words):")
    df['rationale_words'] = df['rationale'].str.split().str.len()
    print(f"  Mean: {df['rationale_words'].mean():.1f}")
    print(f"  Median: {df['rationale_words'].median():.1f}")
    print(f"  Min: {df['rationale_words'].min()}")
    print(f"  Max: {df['rationale_words'].max()}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare ReDSM5 dataset for DHRD training"
    )
    parser.add_argument(
        '--posts_path',
        type=str,
        default='data/redsm5/redsm5_posts.csv',
        help='Path to redsm5_posts.csv'
    )
    parser.add_argument(
        '--annotations_path',
        type=str,
        default='data/redsm5/redsm5_annotations.csv',
        help='Path to redsm5_annotations.csv'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/redsm5_processed',
        help='Output directory for processed JSON files'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.7,
        help='Training set ratio'
    )
    parser.add_argument(
        '--eval_ratio',
        type=float,
        default=0.15,
        help='Evaluation set ratio'
    )
    parser.add_argument(
        '--test_ratio',
        type=float,
        default=0.15,
        help='Test set ratio'
    )
    parser.add_argument(
        '--random_seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    posts_df, annotations_df = load_redsm5_data(
        args.posts_path,
        args.annotations_path
    )

    # Aggregate annotations to post level
    merged_df = aggregate_annotations(posts_df, annotations_df)

    # Print statistics
    print_statistics(merged_df)

    # Create train/eval/test splits
    train_df, eval_df, test_df = create_train_eval_test_split(
        merged_df,
        train_ratio=args.train_ratio,
        eval_ratio=args.eval_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_seed
    )

    # Save to JSON
    print("\nSaving datasets to JSON...")
    save_to_json(
        train_df,
        output_dir / 'train.json',
        include_rationale=True
    )
    save_to_json(
        eval_df,
        output_dir / 'eval.json',
        include_rationale=False  # No rationale for evaluation
    )
    save_to_json(
        test_df,
        output_dir / 'test.json',
        include_rationale=False  # No rationale for test
    )

    # Also save a version with all data for reference
    print("\nSaving complete dataset for reference...")
    merged_df.to_csv(output_dir / 'redsm5_post_level.csv', index=False)
    print(f"  Saved to {output_dir / 'redsm5_post_level.csv'}")

    print("\n" + "=" * 60)
    print("Data preparation completed successfully!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - {output_dir / 'train.json'}")
    print(f"  - {output_dir / 'eval.json'}")
    print(f"  - {output_dir / 'test.json'}")
    print(f"  - {output_dir / 'redsm5_post_level.csv'}")
    print(f"\nYou can now train the DHRD model using:")
    print(f"  python scripts/train.py --config configs/redsm5_config.yaml")


if __name__ == "__main__":
    main()
