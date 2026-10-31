#!/usr/bin/env python3
import argparse
import os
from datasets import load_dataset_builder
from trainers.utils import load_config

def main():
    parser = argparse.ArgumentParser(description="Download Raw Dataset to Local Cache")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    cli_args = parser.parse_args()
    
    args = load_config(cli_args.config)
    
    # Same base cache directory as the benchmark scripts
    raw_cache_dir = f"/raid/datasets/{args.dataset_name}"
    os.makedirs(raw_cache_dir, exist_ok=True)
    
    print(f"Initializing download builder for: {args.dataset_name}")
    print(f"Target raw storage path: {raw_cache_dir}")
    
    # 1. Initialize the builder without loading into memory
    builder = load_dataset_builder(
        path=args.dataset_name,
        name=args.data_dir, # maps to sub-directories or configs like 'en' in C4
        cache_dir=raw_cache_dir
    )
    
    # 2. Force the download and generation of raw local cache shards
    print("Starting download phase. This might take a while depending on link speed...")
    builder.download_and_prepare()
    print(f"\nSuccess! Raw data securely stored in: {raw_cache_dir}")

if __name__ == "__main__":
    main()