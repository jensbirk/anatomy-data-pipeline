#!/usr/bin/env python3
import os
import sys
import time
import argparse
import csv
import torch
from pprint import pprint
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import torch.distributed.checkpoint as dcp
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup

from trainers.utils import load_config

import warnings
warnings.filterwarnings("ignore", message=".*Corrupt EXIF data.*")

SEED = 42

# Alpaca Prompt Template
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ),
}

# Stage-specific Datasets

class DiskOnlyDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset
        print(f"[DiskOnly] Loaded {len(self.dataset)} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # Safely fetch, defaulting to empty string if None
        _ = item.get('instruction') or ""
        _ = item.get('input') or ""
        _ = item.get('output') or ""
        return idx


class DiskDecodeDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset
        print(f"[DiskDecode] Loaded {len(self.dataset)} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Safely fetch, defaulting to empty string if None
        inst = item.get('instruction') or ""
        inp = item.get('input') or ""
        out = item.get('output') or ""

        # "Decode": format raw fields into prompt strings
        if inp != "":
            full_text = PROMPT_DICT["prompt_input"].format_map({'instruction': inst, 'input': inp}) + out
            prompt_text = PROMPT_DICT["prompt_input"].format_map({'instruction': inst, 'input': inp})
        else:
            full_text = PROMPT_DICT["prompt_no_input"].format_map({'instruction': inst}) + out
            prompt_text = PROMPT_DICT["prompt_no_input"].format_map({'instruction': inst})

        return full_text, prompt_text


class DiskDecodePreprocessDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_seq_length):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        print(f"[DiskDecodePreprocess] Loaded {len(self.dataset)} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Safely fetch, defaulting to empty string if None
        inst = item.get('instruction') or ""
        inp = item.get('input') or ""
        out = item.get('output') or ""

        # Decode: format into prompt strings
        if inp != "":
            full_text = PROMPT_DICT["prompt_input"].format_map({'instruction': inst, 'input': inp}) + out
            prompt_text = PROMPT_DICT["prompt_input"].format_map({'instruction': inst, 'input': inp})
        else:
            full_text = PROMPT_DICT["prompt_no_input"].format_map({'instruction': inst}) + out
            prompt_text = PROMPT_DICT["prompt_no_input"].format_map({'instruction': inst})

        full_text += self.tokenizer.eos_token

        # Preprocess: tokenize
        tokenized_full = self.tokenizer(
            full_text, truncation=True, max_length=self.max_seq_length,
            padding="max_length", return_tensors="pt",
        )
        tokenized_prompt = self.tokenizer(
            prompt_text, truncation=True, max_length=self.max_seq_length,
            add_special_tokens=True,
        )

        input_ids = tokenized_full["input_ids"][0]
        attention_mask = tokenized_full["attention_mask"][0]
        labels = input_ids.clone()
        prompt_len = len(tokenized_prompt["input_ids"])
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        return input_ids, attention_mask, labels


# Collate functions

def collate_disk_only(batch):
    return len(batch)


def collate_disk_decode(batch):
    return len(batch)


def collate_full(batch):
    input_ids = torch.stack([item[0] for item in batch])
    attention_mask = torch.stack([item[1] for item in batch])
    labels = torch.stack([item[2] for item in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# Helpers

def make_loader(dataset, args, collate_fn, num_workers):
    g = torch.Generator()
    g.manual_seed(SEED)

    loader_kwargs = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': num_workers,
        'collate_fn': collate_fn,
        'pin_memory': getattr(args, 'pin_memory', True),
        'generator': g,
    }

    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = getattr(args, 'prefetch_factor', 2)
        loader_kwargs['persistent_workers'] = getattr(args, 'persistent_workers', True)

    return DataLoader(dataset, **loader_kwargs)


def load_model(args, device):
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Loading model: {args.model_name} (dtype={dtype})")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        use_cache=False,
    )
    model.to(device)

    if getattr(args, 'torch_compile', False):
        model = torch.compile(model)

    return model


def load_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_hf_dataset(args):
    dataset_path = args.dataset_name
    cache_dir = getattr(args, 'cache_dir', None)
    
    # Handle local files explicitly based on extension
    if os.path.isfile(dataset_path):
        ext = os.path.splitext(dataset_path)[-1].lower()
        if ext in ['.json', '.jsonl']:
            dataset = load_dataset("json", data_files=dataset_path, cache_dir=cache_dir)
        elif ext == '.parquet':
            dataset = load_dataset("parquet", data_files=dataset_path, cache_dir=cache_dir)
        elif ext == '.arrow':
            dataset = load_dataset("arrow", data_files=dataset_path, cache_dir=cache_dir)
        elif ext == '.csv':
            dataset = load_dataset("csv", data_files=dataset_path, cache_dir=cache_dir)
        else:
            dataset = load_dataset(dataset_path, cache_dir=cache_dir)
    else:
        dataset = load_dataset(
            path=dataset_path,
            data_dir=getattr(args, 'data_dir', None),
            cache_dir=cache_dir,
        )

    if hasattr(dataset, "keys"):
        if "train" in dataset:
            train_data = dataset["train"]
        else:
            train_data = dataset[list(dataset.keys())[0]]
    else:
        train_data = dataset

    if getattr(args, 'max_samples', None):
        train_data = train_data.select(range(args.max_samples))
        
    return train_data


# Benchmark functions

def benchmark(loader, num_batches, name):
    print(f"\n{'='*50}")
    print(f"Benchmarking: {name}")
    print(f"{'='*50}")

    count = 0
    start = time.perf_counter()

    for i, batch in enumerate(loader):
        if num_batches > 0 and i >= num_batches:
            break
        if isinstance(batch, int):
            count += batch
        else:
            count += batch["input_ids"].shape[0]

    elapsed = time.perf_counter() - start
    throughput = count / elapsed

    print(f"Processed {count} samples in {elapsed:.2f}s")
    print(f"Throughput: {throughput:.1f} samples/sec")

    return throughput


def benchmark_transfer(loader, num_batches, device):
    print(f"\n{'='*50}")
    print(f"Benchmarking: Disk + Decode + Preprocess + GPU Transfer")
    print(f"{'='*50}")

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    count = 0
    start = time.perf_counter()

    for i, batch_dict in enumerate(loader):
        if num_batches > 0 and i >= num_batches:
            break

        batch_dict = {k: v.to(device, non_blocking=True) for k, v in batch_dict.items()}
        count += batch_dict["input_ids"].shape[0]

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    throughput = count / elapsed

    print(f"Processed {count} samples in {elapsed:.2f}s")
    print(f"Throughput: {throughput:.1f} samples/sec")

    return throughput


def benchmark_training(loader, num_batches, model, device, args):
    print(f"\n{'='*50}")
    print(f"Benchmarking: Full Training (Disk + Decode + Preprocess + Transfer + Compute)")
    print(f"{'='*50}")

    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.1,
    )

    total_steps = num_batches if num_batches > 0 else len(loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    count = 0
    start = time.perf_counter()

    for i, batch_dict in enumerate(loader):
        if num_batches > 0 and i >= num_batches:
            break

        batch_dict = {k: v.to(device, non_blocking=True) for k, v in batch_dict.items()}
        
        outputs = model(**batch_dict)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        count += batch_dict["input_ids"].shape[0]
        
        if args.save_steps > 0 and i % args.save_steps == 0:
            os.makedirs(args.output_dir, exist_ok=True)

            checkpoint_state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": i,
            }

            if getattr(args, 'async_checkpoint', False):
                dcp.async_save(
                    state_dict=checkpoint_state,
                    storage_writer=dcp.FileSystemWriter(args.output_dir),
                )
            else:
                torch.save(checkpoint_state, os.path.join(args.output_dir, "checkpoint.pt"))
            print("Saved checkpoint")

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    throughput = count / elapsed

    print(f"Processed {count} samples in {elapsed:.2f}s")
    print(f"Throughput: {throughput:.1f} samples/sec")

    return throughput


# Orchestration

def run_benchmarks_for_workers(args, num_workers, tokenizer, hf_train, device):
    print(f"\n{'#'*60}")
    print(f"# NUM_WORKERS = {num_workers}")
    print(f"{'#'*60}")

    results = {}
    mode = getattr(args, 'mode', 'all')
    num_batches = getattr(args, 'num_batches', 0)
    max_seq_length = getattr(args, 'max_seq_length', 512)

    if mode in ["disk", "all"]:
        dataset = DiskOnlyDataset(hf_train)
        loader = make_loader(dataset, args, collate_disk_only, num_workers)
        results["disk"] = benchmark(loader, num_batches, "Disk I/O Only")

    if mode in ["decode", "all"]:
        dataset = DiskDecodeDataset(hf_train)
        loader = make_loader(dataset, args, collate_disk_decode, num_workers)
        results["decode"] = benchmark(loader, num_batches, "Disk + Decode")

    if mode in ["preprocess", "all"]:
        dataset = DiskDecodePreprocessDataset(hf_train, tokenizer, max_seq_length)
        loader = make_loader(dataset, args, collate_full, num_workers)
        results["preprocess"] = benchmark(loader, num_batches, "Disk + Decode + Preprocess")

    if mode in ["transfer", "all"]:
        dataset = DiskDecodePreprocessDataset(hf_train, tokenizer, max_seq_length)
        loader = make_loader(dataset, args, collate_full, num_workers)
        results["transfer"] = benchmark_transfer(loader, num_batches, device)

    if mode in ["train", "all"]:
        dataset = DiskDecodePreprocessDataset(hf_train, tokenizer, max_seq_length)
        loader = make_loader(dataset, args, collate_full, num_workers)
        model = load_model(args, device)
        results["train"] = benchmark_training(loader, num_batches, model, device, args)

    return results


def main():
    parser = argparse.ArgumentParser(description="LLM Fine-tuning - Throughput Benchmark")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--log_dir", type=str, help="Directory to save logs")
    cli_args = parser.parse_args()

    args = load_config(cli_args.config)

    # Seeding
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

    # Load tokenizer and dataset once
    tokenizer = load_tokenizer(args)
    hf_train = load_hf_dataset(args)
    print(f"Dataset size: {len(hf_train)} samples")

    # Get worker counts
    num_workers_config = getattr(args, 'num_workers', getattr(args, 'workers', 0))
    if isinstance(num_workers_config, list):
        worker_counts = num_workers_config
    else:
        worker_counts = [num_workers_config]

    all_results = {}
    for num_workers in worker_counts:
        all_results[num_workers] = run_benchmarks_for_workers(
            args, num_workers, tokenizer, hf_train, device,
        )

    # Print summary
    if len(worker_counts) > 1:
        mode_names = list(all_results[worker_counts[0]].keys())

        print(f"\n{'='*70}")
        print("Summary (samples/s)")
        print(f"{'='*70}")
        header = f"{'workers':<10}" + "".join(f"{mode:>14}" for mode in mode_names)
        print(header)
        print("-" * len(header))
        for nw in worker_counts:
            row = f"{nw:<10}"
            for mode in mode_names:
                samp_tp = all_results[nw][mode]
                row += f"{samp_tp:>14.1f}"
            print(row)
    else:
        results = all_results[worker_counts[0]]
        if len(results) > 1:
            print(f"\n{'='*50}")
            print("Summary")
            print(f"{'='*50}")
            for name, samp_tp in results.items():
                print(f"  {name:20s}: {samp_tp:>8.1f} samp/s")

    # Save CSV
    log_dir = cli_args.log_dir or getattr(args, 'output_dir', None)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        csv_path = os.path.join(log_dir, f"benchmark_results_{int(time.time())}.csv")

        print(f"\n{'='*50}")
        print(f"Saving results to: {csv_path}")

        try:
            with open(csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['num_workers', 'mode', 'throughput_samp_s'])
                for nw, mode_results in all_results.items():
                    for mode_name, samp_s in mode_results.items():
                        writer.writerow([nw, mode_name, f"{samp_s:.4f}"])
            print("[Success]")
        except Exception as e:
            print(f"[Error saving CSV: {e}]")


if __name__ == "__main__":
    main()