#!/usr/bin/env python3
"""
Download Amazon Reviews 2023 Dataset from Hugging Face

This script downloads the Amazon Reviews 2023 dataset for specified categories.
You can download raw reviews and/or raw metadata.
"""

import argparse
import os
from pathlib import Path
import urllib.request
import gzip
import json
from huggingface_hub import hf_hub_download

# Import centralized data path utilities
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_paths import get_raw_data_dir, get_raw_data_filename

# Get default output directory
DEFAULT_OUTPUT_DIR = get_raw_data_dir()


def load_all_categories():
    """Load list of all available categories from Hugging Face."""
    category_filepath = hf_hub_download(
        repo_id='McAuley-Lab/Amazon-Reviews-2023',
        filename='all_categories.txt',
        repo_type='dataset'
    )
    with open(category_filepath, 'r') as file:
        all_categories = [_.strip() for _ in file.readlines()]
    return all_categories


def download_raw_reviews(category, output_dir=None):
    """Download raw review data for a specific category."""
    print(f"\n{'='*60}")
    print(f"Downloading raw reviews for: {category}")
    print(f"{'='*60}")

    # Direct download URL from UCSD data repository
    url = f"https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/{category}.jsonl.gz"

    if output_dir:
        # Use centralized naming convention
        filename = get_raw_data_filename(category, 'review')
        output_path = os.path.join(output_dir, filename + '.gz')
        print(f"Downloading from: {url}")
        try:
            # Download the gzipped file
            urllib.request.urlretrieve(url, output_path)
            print(f"✓ Downloaded to: {output_path}")
            print(f"✓ Saved to: {output_path}")

        except urllib.error.HTTPError as e:
            raise Exception(f"HTTP Error {e.code}: {e.reason}")
        except Exception as e:
            raise Exception(f"Download failed: {str(e)}")

    return output_path if output_dir else None

def parse_args():
    parser = argparse.ArgumentParser(
        description='Download Amazon Reviews 2023 Dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available categories
  python download_dataset.py --list-categories

  # Download raw reviews and metadata for a single category
  python download_dataset.py --category All_Beauty

  # Download raw reviews only for a category
  python download_dataset.py --category All_Beauty

  # Download raw metadata only for a category
  python download_dataset.py --category All_Beauty

  # Download for multiple categories
  python download_dataset.py --category All_Beauty Toys_and_Games

  # Download for all categories (WARNING: This is very large!)
  python download_dataset.py --all-categories
        """
    )

    parser.add_argument(
        '--category',
        type=str,
        nargs='+',
        help='Category/domain name(s) (e.g., All_Beauty, Toys_and_Games)'
    )

    parser.add_argument(
        '--all-categories',
        action='store_true',
        help='Download for all available categories (WARNING: very large!)'
    )

    parser.add_argument(
        '--list-categories',
        action='store_true',
        help='List all available categories and exit'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f'Output directory to save downloaded data (default: {DEFAULT_OUTPUT_DIR})'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Load all available categories
    all_categories = load_all_categories()

    if args.list_categories:
        print("\nAvailable categories:")
        print("=" * 60)
        for i, cat in enumerate(all_categories, 1):
            print(f"{i:3d}. {cat}")
        print(f"\nTotal: {len(all_categories)} categories")
        return

    # Determine which categories to download
    if args.all_categories:
        categories = all_categories
        print(f"\n⚠️  WARNING: Downloading ALL {len(categories)} categories!")
        print("This will download a very large amount of data.")
        response = input("Do you want to continue? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Download cancelled.")
            return
    elif args.category:
        categories = args.category
        # Validate categories
        invalid_cats = [cat for cat in categories if cat not in all_categories]
        if invalid_cats:
            print(f"\n❌ Error: Invalid category/categories: {', '.join(invalid_cats)}")
            print(f"\nUse --list-categories to see all available categories.")
            return
    else:
        print("❌ Error: Please specify --category or --all-categories")
        print("Use --help for usage information")
        return

    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    print(f"\n📁 Output directory: {args.output}")

    # Download data for each category
    print(f"\n🚀 Starting download for {len(categories)} category/categories...")

    for i, category in enumerate(categories, 1):
        print(f"\n{'#'*60}")
        print(f"# Category {i}/{len(categories)}: {category}")
        print(f"{'#'*60}")

        try:
            download_raw_reviews(category, args.output)

        except Exception as e:
            print(f"\n❌ Error downloading {category}: {str(e)}")
            continue

    print(f"\n{'='*60}")
    print("✅ Download complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
