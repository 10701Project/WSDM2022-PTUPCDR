"""
Data cleaning utilities for CDR Ensemble project.

This module provides functions to clean and filter review datasets
for cross-domain recommendation tasks.
"""

import gzip
import json
import os
import sys
from collections import Counter
from typing import Dict, List

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data_paths import (
    get_trimed_data_path,
    get_cleaned_data_dir,
    get_cross_domain_dir_name,
    get_cleaned_data_filename,
    ensure_dir_exists
)


def cleaning_data(source_dataset: str, target_dataset: str) -> None:
    """
    Clean review datasets for cross-domain recommendation.

    This function performs the following cleaning steps:
    1. Load review data from both source and target datasets
    2. Keep only reviews with user_id that appears in both datasets
    3. Keep only reviews whose user_id has at least 5 interactions in both domains
    4. Store the processed data under original/cleaned/{dir_name}/{file_name}

    Args:
        source_dataset: Name of the source dataset (e.g., 'Digital_Music')
        target_dataset: Name of the target dataset (e.g., 'Magazine_Subscriptions')

    Examples:
        >>> cleaning_data('Digital_Music', 'Magazine_Subscriptions')
        # Cleans and saves data to:
        # data_set/original/cleaned/Digital_Music_to_Magazine_Subscriptions/
        #   - cleaned_review_Digital_Music.jsonl
        #   - cleaned_review_Magazine_Subscriptions.jsonl
    """
    # Load review data
    print(f"Loading review data for {source_dataset}...")
    source_reviews = _load_reviews(source_dataset)
    print(f"  Loaded {len(source_reviews)} source reviews")

    print(f"Loading review data for {target_dataset}...")
    target_reviews = _load_reviews(target_dataset)
    print(f"  Loaded {len(target_reviews)} target reviews")

    # Step 1: Find users that appear in both datasets
    print("Finding overlapping users...")
    source_users = set(review['user_id'] for review in source_reviews)
    target_users = set(review['user_id'] for review in target_reviews)
    overlapping_users = source_users & target_users
    print(f"  Found {len(overlapping_users)} overlapping users")

    # Filter to keep only overlapping users
    source_reviews_filtered = [r for r in source_reviews if r['user_id'] in overlapping_users]
    target_reviews_filtered = [r for r in target_reviews if r['user_id'] in overlapping_users]
    print(f"  Source reviews after overlap filter: {len(source_reviews_filtered)}")
    print(f"  Target reviews after overlap filter: {len(target_reviews_filtered)}")

    # Step 2: Count interactions per user in both domains
    print("Counting user interactions...")
    source_user_counts = Counter(r['user_id'] for r in source_reviews_filtered)
    target_user_counts = Counter(r['user_id'] for r in target_reviews_filtered)

    # Find users with at least 5 interactions in BOTH domains
    qualified_users = set()
    for user_id in overlapping_users:
        if source_user_counts[user_id] >= 5 and target_user_counts[user_id] >= 5:
            qualified_users.add(user_id)

    print(f"  Found {len(qualified_users)} users with >= 5 interactions in both domains")

    # Filter to keep only qualified users
    source_reviews_cleaned = [r for r in source_reviews_filtered if r['user_id'] in qualified_users]
    target_reviews_cleaned = [r for r in target_reviews_filtered if r['user_id'] in qualified_users]
    print(f"  Source reviews after minimum interaction filter: {len(source_reviews_cleaned)}")
    print(f"  Target reviews after minimum interaction filter: {len(target_reviews_cleaned)}")

    # Step 3: Save cleaned data
    print("Saving cleaned data...")
    dir_name = get_cross_domain_dir_name(source_dataset, target_dataset)
    output_dir = os.path.join(get_cleaned_data_dir(), dir_name)
    ensure_dir_exists(output_dir)

    # Save source dataset
    source_filename = get_cleaned_data_filename(source_dataset, 'review')
    source_output_path = os.path.join(output_dir, source_filename)
    _save_reviews(source_reviews_cleaned, source_output_path)
    print(f"  Saved {len(source_reviews_cleaned)} source reviews to {source_output_path}")

    # Save target dataset
    target_filename = get_cleaned_data_filename(target_dataset, 'review')
    target_output_path = os.path.join(output_dir, target_filename)
    _save_reviews(target_reviews_cleaned, target_output_path)
    print(f"  Saved {len(target_reviews_cleaned)} target reviews to {target_output_path}")

    print("Cleaning completed successfully!")


def _load_reviews(dataset_name: str) -> List[Dict]:
    """
    Load reviews from a trimed review file (gzip compressed).

    Args:
        dataset_name: Name of the dataset

    Returns:
        List of review dictionaries
    """
    file_path = get_trimed_data_path(dataset_name, 'review')
    reviews = []

    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                review = json.loads(line)
                reviews.append(review)

    return reviews


def _save_reviews(reviews: List[Dict], output_path: str) -> None:
    """
    Save reviews to a gzip compressed JSONL file.

    Args:
        reviews: List of review dictionaries
        output_path: Path to save the reviews
    """
    with gzip.open(output_path, 'wt', encoding='utf-8') as f:
        for review in reviews:
            f.write(json.dumps(review) + '\n')


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) != 3:
        print("Usage: python cleaning.py <source_dataset> <target_dataset>")
        print("Example: python cleaning.py Digital_Music Magazine_Subscriptions")
        sys.exit(1)

    source = sys.argv[1]
    target = sys.argv[2]

    cleaning_data(source, target)
