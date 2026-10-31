#!/usr/bin/env python3

import csv
import gzip
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "/raid/datasets")
OUT = ROOT / "data" / "dataset_sizes" / "dataset_files.csv"
C4_COUNTS = ROOT / "paper" / "plots" / "sample_distributions" / "sample_sizes" / "c4_arrow_counts.csv.gz"


def files(rel, suffix):
    out = []
    for d, _, names in os.walk(DATA / rel):
        out += [os.path.join(d, n) for n in names if n.endswith(suffix) and not n.startswith(".")]
    return sorted(out)


def total(paths):
    return sum(os.stat(p).st_size for p in paths)


def rows_arrow(path):
    import pyarrow as pa
    try:
        return pa.ipc.open_file(pa.memory_map(path)).read_all().num_rows
    except pa.ArrowInvalid:  # Hugging Face datasets write the Arrow stream format
        return pa.ipc.open_stream(pa.memory_map(path)).read_all().num_rows


def rows_parquet(path):
    import pyarrow.parquet as pq
    return pq.ParquetFile(path).metadata.num_rows


def rows_jsonl(path):
    with open(path, "rb") as f:
        return sum(1 for line in f if line.strip())


def rows_csv(path):
    import pandas as pd
    return len(pd.read_csv(path))


def c4_records():
    with gzip.open(C4_COUNTS, "rt") as f:
        return sum(int(r["count"]) for r in csv.DictReader(f))


def main():
    out = []

    def add(dataset, fmt, paths, samples):
        out.append({"dataset": dataset, "format": fmt, "files": len(paths),
                    "bytes": total(paths), "samples": samples})
        print(f"{dataset:12s} {fmt:12s} {len(paths):>9,d} files {out[-1]['bytes'] / 2**30:9.2f} GiB "
              f"{samples:>13,d} samples", flush=True)

    for fmt, name, rows in (("Arrow", "arrow/alpaca_data.arrow", rows_arrow),
                            ("Parquet", "parquet/alpaca_data.parquet", rows_parquet),
                            ("JSONL", "jsonl/alpaca_data.jsonl", rows_jsonl),
                            ("CSV", "csv/alpaca_data.csv", rows_csv)):
        p = str(DATA / "alpaca" / name)
        add("Alpaca", fmt, [p], rows(p))

    c4 = [p for p in files("c4/converted/arrow", ".arrow") if os.path.basename(p).startswith("c4-train.")]
    add("C4", "Arrow", c4, c4_records())

    jpg = files("imagenet-1k/jpg", ".JPEG")
    add("ImageNet-1K", "JPEG", jpg, len(jpg))
    png = files("imagenet-1k/png", ".png")
    if png:
        add("ImageNet-1K", "PNG", png, len(png))
    add("ImageNet-1K", "WebDataset", [p for p in files("imagenet-1k/wds", ".tar")], len(jpg))
    add("ImageNet-1K", "Beton", files("imagenet-1k/ffcv", ".beton"), len(jpg))

    # a BraTS sample is a 4-channel image volume plus its label volume
    for fmt, rel, suffix in (("NIfTI", "brats/nii/Task01_BrainTumour", ".nii"),
                             ("NIfTI-GZ", "brats/nii.gz", ".nii.gz")):
        images, labels = files(f"{rel}/imagesTr", suffix), files(f"{rel}/labelsTr", suffix)
        assert len(images) == len(labels), (fmt, len(images), len(labels))
        add("BraTS", fmt, images + labels, len(images))

    # a LibriSpeech sample is an utterance plus its share of the chapter transcript
    flac = files("librispeech/flac", ".flac")
    add("LibriSpeech", "Arrow", files("librispeech/arrow", ""), len(flac))
    for fmt, suffix in (("WAV", ".wav"), ("FLAC", ".flac"), ("MP3", ".mp3")):
        audio = files(f"librispeech/{suffix[1:]}", suffix)
        add("LibriSpeech", fmt, audio + files(f"librispeech/{suffix[1:]}", ".trans.txt"), len(audio))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
