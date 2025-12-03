"""
Data cleaning utilities for CDR Ensemble project.

This module provides functions to clean and filter review datasets
for cross-domain recommendation tasks.
"""

import gzip
import json
import os
import shutil
import sys
from collections import Counter

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

# Batch size for processing large datasets (number of reviews per batch)
BATCH_SIZE = 100000


def cleaning_data(source_dataset: str, target_dataset: str) -> None:
    """
    Clean review datasets for cross-domain recommendation.

    This function performs the following cleaning steps:
    1. Filter each dataset to keep only users with >= 5 reviews (batch processing)
    2. Find common user_ids across both filtered datasets
    3. Keep only reviews from common users
    4. Store the processed data under data/cleaned/{dir_name}/{file_name}

    Args:
        source_dataset: Name of the source dataset (e.g., 'Books')
        target_dataset: Name of the target dataset (e.g., 'Movies_and_TV')

    Examples:
        >>> cleaning_data('Books', 'Movies_and_TV')
        # Cleans and saves data to:
        # data/cleaned/Books_to_Movies_and_TV/
        #   - cleaned_review_Books.jsonl.gz
        #   - cleaned_review_Movies_and_TV.jsonl.gz
    """
    temp_dir = os.path.join(get_cleaned_data_dir(), '_temp')
    ensure_dir_exists(temp_dir)

    try:
        # Step 1: Filter each dataset to keep users with >= 5 reviews
        print("=" * 80)
        print("STEP 1: Filtering datasets (keeping users with >= 5 reviews)")
        print("=" * 80)

        temp_source_path = os.path.join(temp_dir, f"filtered_{source_dataset}.jsonl.gz")
        temp_target_path = os.path.join(temp_dir, f"filtered_{target_dataset}.jsonl.gz")

        source_users = _filter_by_frequency(source_dataset, temp_source_path, min_count=5)
        target_users = _filter_by_frequency(target_dataset, temp_target_path, min_count=5)

        # Step 2: Find common users
        print("\n" + "=" * 80)
        print("STEP 2: Finding common users")
        print("=" * 80)
        common_users = source_users & target_users
        print(f"Found {len(common_users)} common users")

        if len(common_users) == 0:
            print("WARNING: No common users found. Cleaning aborted.")
            return

        # Step 3: Filter by common users and save final results
        print("\n" + "=" * 80)
        print("STEP 3: Filtering by common users and saving final results")
        print("=" * 80)

        dir_name = get_cross_domain_dir_name(source_dataset, target_dataset)
        output_dir = os.path.join(get_cleaned_data_dir(), dir_name)
        ensure_dir_exists(output_dir)

        # Process source dataset
        source_filename = get_cleaned_data_filename(source_dataset, 'review')
        source_output_path = os.path.join(output_dir, source_filename)
        source_count = _filter_by_user_set(temp_source_path, source_output_path, common_users)
        print(f"Saved {source_count} source reviews to: {source_output_path}")

        # Process target dataset
        target_filename = get_cleaned_data_filename(target_dataset, 'review')
        target_output_path = os.path.join(output_dir, target_filename)
        target_count = _filter_by_user_set(temp_target_path, target_output_path, common_users)
        print(f"Saved {target_count} target reviews to: {target_output_path}")

        print("\n" + "=" * 80)
        print("CLEANING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Source dataset: {source_count} reviews from {len(common_users)} users")
        print(f"Target dataset: {target_count} reviews from {len(common_users)} users")

    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            print(f"\nCleaning up temporary files in: {temp_dir}")
            shutil.rmtree(temp_dir)
            print("Temporary files removed.")


def _filter_by_frequency(dataset_name: str, output_path: str, min_count: int = 5) -> set:
    """
    Filter dataset to keep only users with >= min_count reviews.
    Processes data in batches to handle large files.

    Args:
        dataset_name: Name of the dataset
        output_path: Path to save filtered reviews
        min_count: Minimum number of reviews per user

    Returns:
        Set of user_ids that have >= min_count reviews
    """
    input_path = get_trimed_data_path(dataset_name, 'review')
    print(f"\nProcessing {dataset_name}...")
    print(f"  Input: {input_path}")

    # Pass 1: Count user frequencies in batches
    print("  Pass 1: Counting user frequencies...")
    user_counts = Counter()
    total_reviews = 0

    with gzip.open(input_path, 'rt', encoding='utf-8') as f:
        batch = []
        for line in f:
            line = line.strip()
            if line:
                review = json.loads(line)
                batch.append(review)
                total_reviews += 1

                if len(batch) >= BATCH_SIZE:
                    # Process batch
                    for r in batch:
                        user_counts[r['user_id']] += 1
                    batch = []
                    print(f"    Processed {total_reviews:,} reviews...", end='\r')

        # Process remaining reviews
        for r in batch:
            user_counts[r['user_id']] += 1

    print(f"    Processed {total_reviews:,} reviews total")

    # Find qualified users
    qualified_users = {user_id for user_id, count in user_counts.items() if count >= min_count}
    print(f"  Found {len(qualified_users):,} users with >= {min_count} reviews")

    # Pass 2: Filter and save reviews from qualified users
    print("  Pass 2: Filtering and saving qualified reviews...")
    saved_count = 0

    with gzip.open(input_path, 'rt', encoding='utf-8') as f_in, \
         gzip.open(output_path, 'wt', encoding='utf-8') as f_out:

        batch = []
        for line in f_in:
            line = line.strip()
            if line:
                review = json.loads(line)
                if review['user_id'] in qualified_users:
                    batch.append(review)

                if len(batch) >= BATCH_SIZE:
                    # Write batch
                    for r in batch:
                        f_out.write(json.dumps(r) + '\n')
                        saved_count += 1
                    batch = []
                    print(f"    Saved {saved_count:,} reviews...", end='\r')

        # Write remaining reviews
        for r in batch:
            f_out.write(json.dumps(r) + '\n')
            saved_count += 1

    print(f"    Saved {saved_count:,} reviews total")
    print(f"  Output: {output_path}")

    return qualified_users


def _filter_by_user_set(input_path: str, output_path: str, user_set: set) -> int:
    """
    Filter reviews to keep only those from users in the given set.
    Processes data in batches.

    Args:
        input_path: Path to input reviews file
        output_path: Path to save filtered reviews
        user_set: Set of user_ids to keep

    Returns:
        Number of reviews saved
    """
    print(f"\n  Processing: {os.path.basename(input_path)}")
    saved_count = 0

    with gzip.open(input_path, 'rt', encoding='utf-8') as f_in, \
         gzip.open(output_path, 'wt', encoding='utf-8') as f_out:

        batch = []
        for line in f_in:
            line = line.strip()
            if line:
                review = json.loads(line)
                if review['user_id'] in user_set:
                    batch.append(review)

                if len(batch) >= BATCH_SIZE:
                    # Write batch
                    for r in batch:
                        f_out.write(json.dumps(r) + '\n')
                        saved_count += 1
                    batch = []
                    print(f"    Saved {saved_count:,} reviews...", end='\r')

        # Write remaining reviews
        for r in batch:
            f_out.write(json.dumps(r) + '\n')
            saved_count += 1

    print(f"    Saved {saved_count:,} reviews total")

    return saved_count


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
