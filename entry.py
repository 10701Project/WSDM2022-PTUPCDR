import os
import torch
import numpy as np
import random
import argparse
import json
import pandas as pd
from preprocessing import DataPreprocessingMid, DataPreprocessingReady
from run import Run
from data.data_paths import get_cross_domain_dir_name


def calculate_uid_iid(root, src_name, tgt_name, ratio):
    """
    Calculate the total number of unique users (uid) and items (iid) from processed mid data.
    This should match the logic in DataPreprocessingReady.mapper() method.

    Args:
        root: Root data directory
        src_name: Source domain name
        tgt_name: Target domain name
        ratio: Data split ratio

    Returns:
        tuple: (uid_all, iid_all) - total unique users and items
    """
    # Read mid data
    src_path = root + 'mid/' + src_name + '.csv'
    tgt_path = root + 'mid/' + tgt_name + '.csv'

    # Check if files exist
    if not os.path.exists(src_path) or not os.path.exists(tgt_path):
        print(f"Warning: Mid data files not found. Please run --process_data_mid first.")
        print(f"  Source: {src_path}")
        print(f"  Target: {tgt_path}")
        return None, None

    src = pd.read_csv(src_path)
    tgt = pd.read_csv(tgt_path)

    # Calculate uid_all: total unique users (union of src and tgt)
    all_uid = set(src.uid) | set(tgt.uid)
    uid_all = len(all_uid)

    # Calculate iid_all: total unique items (sum of src and tgt items)
    # This matches the logic in mapper() where src items get [0, n_src)
    # and tgt items get [n_src, n_src + n_tgt)
    iid_all = len(set(src.iid)) + len(set(tgt.iid))

    print(f"Calculated uid_all: {uid_all}, iid_all: {iid_all}")
    return uid_all, iid_all


def prepare(config_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--process_data_mid', default=0)
    parser.add_argument('--process_data_ready', default=0)
    parser.add_argument('--base_model', default='MF')
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--ratio', default=[0.8, 0.2])
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    with open(config_path, 'r') as f:
        config = json.load(f)
        config['base_model'] = args.base_model
        config['ratio'] = args.ratio
        config['epoch'] = args.epoch
        config['lr'] = args.lr
    return args, config


if __name__ == '__main__':
    config_path = 'config.json'
    args, config = prepare(config_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Extract source and target domains from new config format
    src_domain = config['src_tgt_pairs']['src']
    tgt_domain = config['src_tgt_pairs']['tgt']
    cross_domain_dir_name = get_cross_domain_dir_name(src_domain, tgt_domain)

    if args.process_data_mid:
        # Process both source and target datasets
        print(f"Processing mid data for cross-domain: {cross_domain_dir_name}")
        for dataset_name in [src_domain, tgt_domain]:
            print(f"\nProcessing dataset: {dataset_name}")
            DataPreprocessingMid(config['root'], cross_domain_dir_name, dataset_name).main()

    if args.process_data_ready:
        # Process ready data with the single experiment configuration
        for ratio in [[0.8, 0.2], [0.5, 0.5], [0.2, 0.8]]:
            print(f"\nProcessing ready data with ratio: {ratio}")
            DataPreprocessingReady(config['root'], config['src_tgt_pairs'], ratio).main()

    print('src:{}; tgt:{}; model:{}; ratio:{}; epoch:{}; lr:{}; gpu:{}; seed:{};'.
          format(src_domain, tgt_domain, args.base_model, args.ratio, args.epoch, args.lr, args.gpu, args.seed))

    if not args.process_data_mid and not args.process_data_ready:
        # Calculate uid and iid from mid data and inject into config
        uid_all, iid_all = calculate_uid_iid(config['root'], src_domain, tgt_domain, config['ratio'])

        if uid_all is None or iid_all is None:
            print("Error: Cannot calculate uid and iid. Please run data preprocessing first:")
            print("  python entry.py --process_data_mid 1")
            print("  python entry.py --process_data_ready 1")
            exit(1)

        # Inject calculated values into config
        config['src_tgt_pairs']['uid'] = uid_all
        config['src_tgt_pairs']['iid'] = iid_all

        Run(config).main()
