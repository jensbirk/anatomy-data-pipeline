import os
import shutil
import argparse
import json
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def process_data_file(input_path, target_path, target_format):
    try:
        # Load JSON data
        # Assumption: JSON is either a list of dicts (records) or a single dict
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Normalize to DataFrame
        if isinstance(data, dict):
            data = [data] # Wrap single object in list
        
        df = pd.DataFrame(data)
        
        # Write output based on format
        if target_format == "arrow":
            # Using Feather format which is the on-disk representation of Arrow
            df.to_feather(target_path)

        elif target_format == "parquet":
            df.to_parquet(target_path, index=False)

        elif target_format == "jsonl":
            df.to_json(target_path, orient='records', lines=True, force_ascii=False)

        elif target_format == "csv":
            df.to_csv(target_path, index=False)

        return True, target_path.name

    except Exception as e:
        return False, f"{input_path.name} (Error: {e})"

def main():
    parser = argparse.ArgumentParser(description="Mirror a directory and convert .json files to Arrow, Parquet, CSV, or JSONL.")
    parser.add_argument("-s", "--source", required=True, help="Path to the source directory containing .json files.")
    parser.add_argument("-t", "--target", required=True, help="Path to the base target directory.")
    parser.add_argument("-f", "--format", choices=["arrow", "parquet", "csv", "jsonl", "all"], default="parquet", 
                        help="Choose output format. (Default: parquet)")
    parser.add_argument("-w", "--workers", type=int, default=4, 
                        help="Number of parallel processing threads.")
    
    args = parser.parse_args()
    
    source_path = Path(args.source)
    target_path = Path(args.target)
    
    if not source_path.exists() or not source_path.is_dir():
        print(f"Error: Source directory '{args.source}' does not exist.")
        return

    # Define requested formats
    supported_formats = ["arrow", "parquet", "csv", "jsonl"]
    selected_formats = supported_formats if args.format == "all" else [args.format]
    
    # Extension mapping
    ext_map = {
        "arrow": ".arrow",
        "parquet": ".parquet",
        "csv": ".csv",
        "jsonl": ".jsonl"
    }

    data_tasks = []
    
    print("\nScanning directory and mirroring existing files...")
    
    for item in source_path.rglob("*"):
        if not item.is_file():
            continue
        
        relative_path = item.relative_to(source_path)
        
        # Handle JSON conversion
        if item.suffix.lower() == '.json':
            base_name = item.stem 
            
            for fmt in selected_formats:
                format_base_dir = target_path / fmt
                
                # Construct new path with correct extension
                new_ext = ext_map[fmt]
                new_path = format_base_dir / relative_path.parent / f"{base_name}{new_ext}"
                new_path.parent.mkdir(parents=True, exist_ok=True)
                
                data_tasks.append((item, new_path, fmt))
        
        # Copy TXT and CSV files
        elif item.suffix.lower() in ['.csv', '.txt']:
            for fmt in selected_formats:
                format_base_dir = target_path / fmt
                target_file = format_base_dir / relative_path
                target_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Skip files that are already copied
                shutil.copy2(item, target_file)

    if data_tasks:
        print(f"\nStarting parallel conversion using {args.workers} threads...")
        success_count = 0
        
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_file = {
                executor.submit(process_data_file, src, dest, fmt): dest 
                for src, dest, fmt in data_tasks
            }
            
            for future in tqdm(as_completed(future_to_file), total=len(data_tasks), desc="Converting Data"):
                success, result = future.result()
                if success:
                    success_count += 1
                else:
                    tqdm.write(f"Failed: {result}") 

        print(f"\nSuccessfully converted {success_count}/{len(data_tasks)} files.")
    else:
        print("\nNo .json files found to convert.")

if __name__ == "__main__":
    main()