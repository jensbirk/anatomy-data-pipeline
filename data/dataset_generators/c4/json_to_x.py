import os
import argparse
import gzip
import json
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as pq
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def process_data_file(input_path, target_path, target_format):
    try:
        texts = []
        with gzip.open(input_path, 'rt', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                texts.append(obj.get('text', ''))

        table = pa.table({'text': pa.array(texts, type=pa.string())})

        if target_format == "arrow":
            feather.write_feather(table, target_path)

        elif target_format == "parquet":
            pq.write_table(table, target_path)

        elif target_format == "jsonl":
            with open(target_path, 'w', encoding='utf-8') as out:
                for t in texts:
                    out.write(json.dumps({'text': t}) + '\n')

        elif target_format == "csv":
            import csv
            with open(target_path, 'w', newline='', encoding='utf-8') as out:
                w = csv.writer(out)
                w.writerow(['text'])
                for t in texts:
                    w.writerow([t])

        return True, target_path.name

    except Exception as e:
        return False, f"{Path(input_path).name} (Error: {e})"


def main():
    parser = argparse.ArgumentParser(
        description="Convert C4 .json.gz files to Arrow, Parquet, CSV, or JSONL."
    )
    parser.add_argument(
        "-s", "--source", required=True,
        help="Path to source directory containing .json.gz files.",
    )
    parser.add_argument(
        "-t", "--target", required=True,
        help="Path to the base target directory.",
    )
    parser.add_argument(
        "-f", "--format", choices=["arrow", "parquet", "csv", "jsonl", "all"],
        default="arrow",
        help="Choose output format. (Default: arrow)",
    )
    parser.add_argument(
        "-w", "--workers", type=int, default=16,
        help="Number of parallel processes.",
    )

    args = parser.parse_args()

    source_path = Path(args.source)
    target_path = Path(args.target)

    if not source_path.exists() or not source_path.is_dir():
        print(f"Error: Source directory '{args.source}' does not exist.")
        return

    supported_formats = ["arrow", "parquet", "csv", "jsonl"]
    selected_formats = supported_formats if args.format == "all" else [args.format]

    ext_map = {
        "arrow": ".arrow",
        "parquet": ".parquet",
        "csv": ".csv",
        "jsonl": ".jsonl",
    }

    data_tasks = []

    print("\nScanning directory for .json.gz files...")

    for item in source_path.rglob("*.json.gz"):
        relative_path = item.relative_to(source_path)
        base_name = item.stem.replace('.json', '')

        for fmt in selected_formats:
            format_base_dir = target_path / fmt
            new_ext = ext_map[fmt]
            new_path = format_base_dir / relative_path.parent / f"{base_name}{new_ext}"
            new_path.parent.mkdir(parents=True, exist_ok=True)
            data_tasks.append((item, new_path, fmt))

    if data_tasks:
        print(f"\nStarting parallel conversion using {args.workers} threads...")
        success_count = 0

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
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
        print("\nNo .json.gz files found to convert.")


if __name__ == "__main__":
    main()
