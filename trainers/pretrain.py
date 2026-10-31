#!/usr/bin/env python3
import os
import argparse
import csv
import glob
import torch
import time
from pprint import pprint
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from tqdm import tqdm
import pyarrow.feather as feather
from trainers.utils import load_config

SEED = 42

STAGE_ORDER = ["disk", "decode", "preprocess", "transfer", "train"]


# Stage-specific Datasets (C4 / pretraining)

class DiskOnlyDataset(Dataset):
    def __init__(self, data_dir):
        if os.path.isdir(data_dir):
            pattern = os.path.join(data_dir, "**", "*.arrow")
            self.files = sorted(glob.glob(pattern, recursive=True))
        else:
            self.files = sorted(glob.glob(data_dir))
        if not self.files:
            raise ValueError(f"No .arrow files found for {data_dir}")

        self.file_sizes = [os.path.getsize(f) for f in self.files]

        print(f"[DiskOnly] {len(self.files)} files, "
              f"{sum(self.file_sizes) / 1024**2:.1f} MB")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(self.files[idx], 'rb') as f:
            _ = f.read()
        return 1


class DiskDecodeDataset(Dataset):
    def __init__(self, data_dir):
        if os.path.isdir(data_dir):
            pattern = os.path.join(data_dir, "**", "*.arrow")
            self.files = sorted(glob.glob(pattern, recursive=True))
        else:
            self.files = sorted(glob.glob(data_dir))
        if not self.files:
            raise ValueError(f"No .arrow files found for {data_dir}")

        print(f"[DiskDecode] {len(self.files)} files")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        table = feather.read_table(self.files[idx], memory_map=True)
        texts = table['text'].to_pylist()
        return len(texts)


def make_hf_tokenized_dataset(data_dir, tokenizer, max_seq_length):
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.arrow"), recursive=True))
    if not files:
        raise ValueError(f"No .arrow files found for {data_dir}")
    print(f"[HF Tokenized] Loading {len(files)} Arrow files via HF datasets...")
    ds = load_dataset("arrow", data_files=files, split="train",
                      cache_dir="/raid/datasets/c4/hf_scratch")

    def transform_fn(example):
        tok = tokenizer(example["text"], truncation=True, padding="max_length",
                        max_length=max_seq_length, return_tensors="pt")
        return {
            "input_ids": tok["input_ids"].squeeze(0),
            "attention_mask": tok["attention_mask"].squeeze(0),
            "labels": tok["input_ids"].squeeze(0).clone(),
        }

    ds.set_transform(transform_fn)
    print(f"[HF Tokenized] Dataset ready: {len(ds)} samples")
    return ds


# Collate functions

def collate_disk_only(batch):
    return sum(batch)


def collate_disk_decode(batch):
    return sum(batch)


# Helpers

def make_loader(dataset, args, collate_fn, num_workers, batch_size=None):
    g = torch.Generator()
    g.manual_seed(SEED)
    if batch_size is None:
        batch_size = args.batch_size
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "collate_fn": collate_fn,
        "pin_memory": getattr(args, "pin_memory", True),
        "generator": g,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = getattr(args, "prefetch_factor", 2)
        loader_kwargs["persistent_workers"] = getattr(args, "persistent_workers", True)
    return DataLoader(dataset, **loader_kwargs)


# Benchmark functions

def benchmark(loader, max_steps, name):
    print(f"\n{'='*50}")
    print(f"Benchmarking: {name}")
    print(f"{'='*50}")

    count = 0
    start = time.perf_counter()

    effective = max_steps if max_steps > 0 else float('inf')
    progress = tqdm(loader, total=effective if effective != float('inf') else None, desc=name)

    for i, batch in enumerate(progress):
        if max_steps > 0 and i >= max_steps:
            break
        if isinstance(batch, dict):
            count += batch["input_ids"].shape[0]
        else:
            count += batch
        if (i + 1) % 10 == 0:
            elapsed_so_far = time.perf_counter() - start
            progress.set_postfix({"samp/s": f"{count / elapsed_so_far:.1f}"})

    elapsed = time.perf_counter() - start
    throughput = count / elapsed

    print(f"Processed {count} samples in {elapsed:.2f}s")
    print(f"Throughput: {throughput:.4f} samples/sec")
    return throughput, elapsed


def benchmark_transfer(loader, max_steps, device):
    print(f"\n{'='*50}")
    print(f"Benchmarking: Disk + Decode + Preprocess + GPU Transfer")
    print(f"{'='*50}")

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    count = 0
    start = time.perf_counter()

    for i, batch in enumerate(loader):
        if max_steps > 0 and i >= max_steps:
            break
        batch_dict = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        count += batch_dict["input_ids"].shape[0]

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    throughput = count / elapsed

    print(f"Processed {count} samples in {elapsed:.2f}s")
    print(f"Throughput: {throughput:.4f} samples/sec")
    return throughput, elapsed


def benchmark_training(loader, max_steps, model, device, args):
    print(f"\n{'='*50}")
    print(f"Benchmarking: Full Training")
    print(f"{'='*50}")

    model.train()
    lr = getattr(args, "lr", 2e-5)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.999),
        eps=1e-8, weight_decay=0.1,
    )
    if max_steps > 0:
        effective_steps = max_steps
    elif hasattr(loader.dataset, '__len__'):
        effective_steps = len(loader)
    else:
        effective_steps = float('inf')

    epochs = getattr(args, "epochs", 1)
    total_steps = effective_steps * epochs if effective_steps != float('inf') else 1000
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=getattr(args, "warmup_steps", 0),
        num_training_steps=total_steps,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    count = 0
    global_step = 0
    start = time.perf_counter()

    total_for_tqdm = effective_steps * epochs if effective_steps != float('inf') else None
    progress = tqdm(loader, total=total_for_tqdm, desc="Training")

    for epoch in range(epochs):
        for i, batch in enumerate(progress):
            if max_steps > 0 and global_step >= max_steps:
                break

            batch_dict = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            outputs = model(**batch_dict)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            count += batch_dict["input_ids"].shape[0]
            global_step += 1

            if global_step % 10 == 0:
                elapsed_so_far = time.perf_counter() - start
                current_sps = count / elapsed_so_far
                progress.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "samp/s": f"{current_sps:.1f}",
                })

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    throughput = count / elapsed

    print(f"Processed {count} samples in {elapsed:.2f}s")
    print(f"Throughput: {throughput:.4f} samples/sec")
    return throughput, elapsed


# Orchestration

def run_benchmarks_for_workers(args, num_workers, tokenizer, device, model):
    print(f"\n{'#'*60}")
    print(f"# NUM_WORKERS = {num_workers}")
    print(f"{'#'*60}")

    results = {}
    mode = getattr(args, "mode", "train")
    max_steps = getattr(args, "max_steps", -1)
    if max_steps is None or max_steps < 0:
        max_steps = getattr(args, "num_batches", -1)
    if max_steps is None or max_steps < 0:
        max_steps = 0
    max_seq_length = getattr(args, "max_seq_length", 512)

    if mode in ["disk", "all"]:
        ds = DiskOnlyDataset(args.data_dir)
        loader = make_loader(ds, args, collate_disk_only, num_workers, batch_size=args.batch_size)
        results["disk"] = benchmark(loader, max_steps, "Disk I/O Only")

    if mode in ["decode", "all"]:
        ds = DiskDecodeDataset(args.data_dir)
        loader = make_loader(ds, args, collate_disk_decode, num_workers, batch_size=args.batch_size)
        results["decode"] = benchmark(loader, max_steps, "Disk + Decode")

    needs_tokenized = mode in ["preprocess", "transfer", "train", "all"]
    hf_ds = make_hf_tokenized_dataset(args.data_dir, tokenizer, max_seq_length) if needs_tokenized else None

    if mode in ["preprocess", "all"] and hf_ds is not None:
        loader = DataLoader(hf_ds, batch_size=args.batch_size, num_workers=num_workers,
                            pin_memory=getattr(args, "pin_memory", True))
        results["preprocess"] = benchmark(loader, max_steps, "Disk + Decode + Preprocess")

    if mode in ["transfer", "all"] and hf_ds is not None:
        loader = DataLoader(hf_ds, batch_size=args.batch_size, num_workers=num_workers,
                            pin_memory=getattr(args, "pin_memory", True))
        results["transfer"] = benchmark_transfer(loader, max_steps, device)

    if mode in ["train", "all"] and hf_ds is not None:
        loader = DataLoader(hf_ds, batch_size=args.batch_size, num_workers=num_workers,
                            pin_memory=getattr(args, "pin_memory", True))
        results["train"] = benchmark_training(loader, max_steps, model, device, args)

    return results


def main():
    parser = argparse.ArgumentParser(description="C4 Pretraining - Throughput Benchmark")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--log_dir", type=str, help="Directory to save logs")
    cli_args = parser.parse_args()

    args = load_config(cli_args.config)
    args.log_dir = cli_args.log_dir

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print(f"\n{'='*50}")
    print(f"Throughput Benchmark: {cli_args.config}")
    print(f"{'='*50}")
    pprint(vars(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    mode = getattr(args, "mode", "train")

    model = None
    if mode in ["train", "all"]:
        print(f"Loading model: {args.model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    num_workers_config = getattr(args, "num_workers", 0)
    if isinstance(num_workers_config, list):
        worker_counts = num_workers_config
    else:
        worker_counts = [num_workers_config]

    all_results = {}
    for nw in worker_counts:
        all_results[nw] = run_benchmarks_for_workers(
            args, nw, tokenizer, device, model,
        )

    # Summary
    if len(worker_counts) > 1:
        mode_names = list(all_results[worker_counts[0]].keys())
        print(f"\n{'='*70}")
        print("Summary (samples/s)")
        print(f"{'='*70}")
        header = f"{'workers':<10}" + "".join(f"{m:>14}" for m in mode_names)
        print(header)
        print("-" * len(header))
        for nw in worker_counts:
            row = f"{nw:<10}"
            for m in mode_names:
                samp_s, _ = all_results[nw][m]
                row += f"{samp_s:>14.1f}"
            print(row)
    else:
        results = all_results[worker_counts[0]]
        if len(results) > 1:
            print(f"\n{'='*50}")
            print("Summary")
            print(f"{'='*50}")
            for name, s in results.items():
                samp_s, _ = s
                print(f"  {name:20s}: {samp_s:>8.1f} samp/s")

    # Save CSV
    log_dir = cli_args.log_dir or getattr(args, "output_dir", None)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        csv_path = os.path.join(log_dir, f"benchmark_results_{int(time.time())}.csv")
        print(f"\nSaving results to: {csv_path}")
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["num_workers", "mode", "throughput_samp_s", "elapsed_s"])
            for nw, mode_results in all_results.items():
                for mode_name, (samp_s, elapsed_s) in mode_results.items():
                    writer.writerow([nw, mode_name, f"{samp_s:.4f}", f"{elapsed_s:.2f}"])
        print("[Success]")


if __name__ == "__main__":
    main()
