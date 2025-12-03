"""
Data trimming utilities for CDR Ensemble project.

This module provides functions to trim raw datasets by keeping only
essential fields to reduce memory usage and storage requirements.
"""

import gzip
import json
import os
import sys

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data_paths import (
    get_raw_data_path,
    get_trimed_data_path,
    get_trimed_data_dir,
    ensure_dir_exists
)


def trim_dataset(dataset_name: str, data_type: str = 'review') -> None:
    """
    Trim a dataset by keeping only essential fields.

    This function reads the original dataset and creates a trimmed version
    containing only the following fields:
    - rating
    - user_id
    - parent_asin
    - timestamp

    The trimming is done in a streaming manner to handle large datasets
    efficiently without loading all data into memory at once.

    Args:
        dataset_name: Name of the dataset (e.g., 'Books', 'Movies_and_TV')
        data_type: Type of data ('review' or 'meta'), default is 'review'

    Examples:
        >>> trim_dataset('Books', 'review')
        # Reads from: data/original/raw_review_Books.jsonl.gz
        # Writes to: data/trimed/trimed_review_Books.jsonl.gz
    """
    # Get input and output paths
    input_path = get_raw_data_path(dataset_name, data_type)
    output_path = get_trimed_data_path(dataset_name, data_type)

    # Ensure output directory exists
    output_dir = get_trimed_data_dir()
    ensure_dir_exists(output_dir)

    # Check if input file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Trimming dataset: {dataset_name}")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")

    # Fields to keep
    fields_to_keep = ['rating', 'user_id', 'parent_asin', 'timestamp']

    # Process the file in a streaming manner
    record_count = 0
    trimmed_count = 0

    try:
        with gzip.open(input_path, 'rt', encoding='utf-8') as fin:
            with gzip.open(output_path, 'wt', encoding='utf-8') as fout:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        # Parse JSON record
                        record = json.loads(line)
                        record_count += 1

                        # Extract only the fields we need
                        trimmed_record = {}
                        for field in fields_to_keep:
                            if field in record:
                                trimmed_record[field] = record[field]

                        # Only write if we have all required fields
                        if len(trimmed_record) == len(fields_to_keep):
                            fout.write(json.dumps(trimmed_record) + '\n')
                            trimmed_count += 1

                        # Progress indicator
                        if record_count % 100000 == 0:
                            print(f"  Processed {record_count:,} records, kept {trimmed_count:,}...")

                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse JSON at line {record_count}: {e}")
                        continue

        print(f"Trimming completed!")
        print(f"  Total records processed: {record_count:,}")
        print(f"  Total records kept: {trimmed_count:,}")
        print(f"  Records skipped (missing fields): {record_count - trimmed_count:,}")

    except Exception as e:
        print(f"Error during trimming: {e}")
        # Clean up partial output file if error occurs
        if os.path.exists(output_path):
            os.remove(output_path)
        raise


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python triming.py <dataset_name> [data_type]")
        print("Examples:")
        print("  python triming.py Books")
        print("  python triming.py Movies_and_TV review")
        sys.exit(1)

    dataset = sys.argv[1]
    dtype = sys.argv[2] if len(sys.argv) > 2 else 'review'

    trim_dataset(dataset, dtype)
