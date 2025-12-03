"""
Centralized data path management for CDR Ensemble project.

This module provides a single source of truth for all data-related paths
and naming conventions used throughout the project.
"""

import os
from pathlib import Path
from typing import Optional


def get_project_root() -> str:
    """
    Get the project root directory.

    Returns:
        Absolute path to the project root
    """
    # Navigate up from utils/data_paths.py to project root
    current_file = os.path.abspath(__file__)
    utils_dir = os.path.dirname(current_file)
    project_root = os.path.dirname(utils_dir)
    return project_root


def get_raw_data_dir() -> str:
    """
    Get the directory where raw data is stored.

    Returns:
        Absolute path to raw data directory
    """
    return os.path.join(get_project_root(), 'data', 'original')


def get_cleaned_data_dir() -> str:
    """
    Get the directory where cleaned data is stored.

    Returns:
        Absolute path to cleaned data directory
    """
    return os.path.join(get_project_root(), 'data', 'cleaned')


def get_processed_data_dir() -> str:
    """
    Get the base directory where processed data is stored.

    Returns:
        Absolute path to processed data directory
    """
    return os.path.join(get_project_root(), 'data', 'processed')


def get_raw_data_filename(dataset_name: str, data_type: str = 'review') -> str:
    """
    Get the filename for raw data following the project's naming convention.

    The convention is: raw_{data_type}_{dataset_name}.jsonl.gz

    Args:
        dataset_name: Name of the dataset (e.g., 'Magazine_Subscriptions')
        data_type: Type of data ('review' or 'meta')

    Returns:
        Filename string (not full path)

    Examples:
        >>> get_raw_data_filename('Magazine_Subscriptions', 'review')
        'raw_review_Magazine_Subscriptions.jsonl.gz'
        >>> get_raw_data_filename('Magazine_Subscriptions', 'meta')
        'raw_meta_Magazine_Subscriptions.jsonl.gz'
    """
    return f"raw_{data_type}_{dataset_name}.jsonl.gz"

def get_cleaned_data_filename(dataset_name: str, data_type: str = 'review') -> str:
    """
    Get the filename for cleaned data following the project's naming convention.

    The convention is: cleaned_{data_type}_{dataset_name}.jsonl.gz

    Args:
        dataset_name: Name of the dataset (e.g., 'Magazine_Subscriptions')
        data_type: Type of data ('review' or 'meta')
    
    Returns:
        Filename string (not full path) 
    
    Examples:
        >>> get_cleaned_data_filename('Magazine_Subscriptions', 'review')
        'cleaned_review_Magazine_Subscriptions.jsonl.gz'
        >>> get_cleaned_data_filename('Magazine_Subscriptions', 'meta')
        'cleaned_meta_Magazine_Subscriptions.jsonl.gz'
    """
    return f"cleaned_{data_type}_{dataset_name}.jsonl.gz"

def get_raw_data_path(dataset_name: str, data_type: str = 'review') -> str:
    """
    Get the full path to a raw data file.

    Args:
        dataset_name: Name of the dataset (e.g., 'Magazine_Subscriptions')
        data_type: Type of data ('review' or 'meta')

    Returns:
        Absolute path to the raw data file

    Examples:
        >>> get_raw_data_path('Magazine_Subscriptions', 'review')
        '/path/to/CDR_ensemble/data/original/raw_review_Magazine_Subscriptions.jsonl.gz'
    """
    raw_dir = get_raw_data_dir()
    filename = get_raw_data_filename(dataset_name, data_type)
    return os.path.join(raw_dir, filename)

def get_cross_domain_dir_name(source_dataset: str, target_dataset: str) -> str:
    """
    Generate a directory name for cross-domain scenarios.

    This function creates a standardized directory name format for cross-domain
    recommendation tasks, following the convention: {source_dataset}_to_{target_dataset}

    Args:
        source_dataset: Name of the source dataset (e.g., 'Books')
        target_dataset: Name of the target dataset (e.g., 'Movies')

    Returns:
        Directory name string in the format '{source_dataset}_to_{target_dataset}'

    Examples:
        >>> get_cross_domain_dir_name('Books', 'Movies')
        'Books_to_Movies'
        >>> get_cross_domain_dir_name('Magazine_Subscriptions', 'Digital_Music')
        'Magazine_Subscriptions_to_Digital_Music'
    """
    return f"{source_dataset}_to_{target_dataset}"

def ensure_dir_exists(path: str) -> str:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to ensure exists

    Returns:
        The same path (for chaining)
    """
    os.makedirs(path, exist_ok=True)
    return path

def list_available_raw_datasets() -> list:
    """
    List all available raw datasets in the original data directory.

    Returns:
        List of (dataset_name, data_type) tuples

    Examples:
        >>> list_available_raw_datasets()
        [('Magazine_Subscriptions', 'review'), ('Magazine_Subscriptions', 'meta'), ...]
    """
    raw_dir = get_raw_data_dir()
    datasets = []

    if not os.path.exists(raw_dir):
        return datasets

    for filename in os.listdir(raw_dir):
        if filename.startswith('raw_') and filename.endswith('.jsonl.gz'):
            # Parse filename: raw_{type}_{dataset}.jsonl.gz
            parts = filename[4:-9].split('_', 1)  # Remove 'raw_' and '.jsonl.gz'
            if len(parts) == 2:
                data_type, dataset_name = parts
                datasets.append((dataset_name, data_type))

    return sorted(set(datasets))


def list_processed_datasets(model_name: Optional[str] = None) -> list:
    """
    List all processed datasets, optionally filtered by model.

    Args:
        model_name: Optional model name to filter by

    Returns:
        List of (model_name, dataset_name) tuples

    Examples:
        >>> list_processed_datasets()
        [('emcdr', 'Magazine_Subscriptions'), ...]
        >>> list_processed_datasets('emcdr')
        [('emcdr', 'Magazine_Subscriptions'), ...]
    """
    processed_dir = get_processed_data_dir()
    datasets = []

    if not os.path.exists(processed_dir):
        return datasets

    if model_name:
        # Only look in specific model directory
        model_dir = os.path.join(processed_dir, model_name)
        if os.path.exists(model_dir):
            for dataset_name in os.listdir(model_dir):
                dataset_path = os.path.join(model_dir, dataset_name)
                if os.path.isdir(dataset_path):
                    datasets.append((model_name, dataset_name))
    else:
        # Look in all model directories
        for model in os.listdir(processed_dir):
            model_dir = os.path.join(processed_dir, model)
            if os.path.isdir(model_dir):
                for dataset_name in os.listdir(model_dir):
                    dataset_path = os.path.join(model_dir, dataset_name)
                    if os.path.isdir(dataset_path):
                        datasets.append((model, dataset_name))

    return sorted(datasets)
