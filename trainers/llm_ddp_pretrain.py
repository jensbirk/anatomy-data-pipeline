#!/usr/bin/env python3
import os
import argparse
import csv
import torch
import time
from pprint import pprint
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from tqdm import tqdm
from trainers.utils import load_config

SEED = 42

STAGE_ORDER = ["disk", "decode", "preprocess", "transfer", "train"]


# Stage-specific Datasets (HF C4 / pretraining)

class DiskOnlyDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset
        print(f"[DiskOnly] Loaded {len(self.dataset)} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]['text']
        return 1, len(text.encode('utf-8'))


class DiskDecodeDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset
        print(f"[DiskDecode] Loaded {len(self.dataset)} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]['text']
        return text, len(text.encode('utf-8'))


class DiskDecodePreprocessDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_seq_length):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        print(f"[DiskDecodePreprocess] Loaded {len(self.dataset)} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]['text']

        tokenized = self.tokenizer(
            text, truncation=True, max_length=self.max_seq_length,
            padding="max_length", return_tensors="pt",
        )

        input_ids = tokenized["input_ids"][0]
        attention_mask = tokenized["attention_mask"][0]
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return input_ids, attention_mask, labels, len(text.encode('utf-8'))


# Collate functions

def collate_disk_only(batch):
    count = sum(item[0] for item in batch)
    total_bytes = sum(item[1] for item in batch)
    return count, total_bytes


def collate_disk_decode(batch):
    count = len(batch)
    total_bytes = sum(item[1] for item in batch)
    return count, total_bytes


def collate_full(batch):
    input_ids = torch.stack([item[0] for item in batch])
    attention_mask = torch.stack([item[1] for item in batch])
    labels = torch.stack([item[2] for item in batch])
    total_bytes = sum(item[3] for item in batch)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }, total_bytes


# Helpers

def print_rank0(*args, **kwargs):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)


def load_hf_dataset(args):
    dataset = load_dataset(
        path=args.dataset_name,
        name=getattr(args, 'data_dir', None),
        split="train",
        cache_dir=getattr(args, 'cache_dir', None),
    )
    if getattr(args, 'max_samples', None):
        dataset = dataset.select(range(args.max_samples))
    return dataset


def load_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# Benchmark functions

def benchmark(loader, max_steps, name, local_rank=0):
    print_rank0(f"\n{'='*50}")
    print_rank0(f"Benchmarking: {name}")
    print_rank0(f"{'='*50}")

    count = 0
    total_bytes = 0
    start = time.perf_counter()

    effective = max_steps if max_steps > 0 else float('inf')
    progress = tqdm(loader, total=effective if effective != float('inf') else None,
                    desc=name, disable=local_rank != 0)

    for i, batch in enumerate(progress):
        if max_steps > 0 and i >= max_steps:
            break
        if isinstance(batch[0], int):
            count += batch[0]
            total_bytes += batch[1]
        else:
            count += batch[0]["input_ids"].shape[0]
            total_bytes += batch[1]
        if (i + 1) % 10 == 0:
            elapsed_so_far = time.perf_counter() - start
            progress.set_postfix({"samp/s": f"{count / elapsed_so_far:.1f}"})

    elapsed = time.perf_counter() - start
    throughput = count / elapsed
    throughput_mbs = (total_bytes / (1024 * 1024)) / elapsed

    print_rank0(f"Processed {count} samples ({total_bytes / (1024*1024):.1f} MB) in {elapsed:.2f}s")
    print_rank0(f"Throughput: {throughput:.4f} samples/sec | {throughput_mbs:.1f} MB/s")
    return throughput, elapsed


def benchmark_transfer(loader, max_steps, device, local_rank=0):
    print_rank0(f"\n{'='*50}")
    print_rank0(f"Benchmarking: Disk + Decode + Preprocess + GPU Transfer")
    print_rank0(f"{'='*50}")

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    count = 0
    start = time.perf_counter()

    for i, batch in enumerate(loader):
        if max_steps > 0 and i >= max_steps:
            break
        batch_dict, _ = batch
        batch_dict = {k: v.to(device, non_blocking=True) for k, v in batch_dict.items()}
        count += batch_dict["input_ids"].shape[0]

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    throughput = count / elapsed

    print_rank0(f"Processed {count} samples in {elapsed:.2f}s")
    print_rank0(f"Throughput: {throughput:.4f} samples/sec")
    return throughput, elapsed


def benchmark_training(loader, max_steps, model, device, args):
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    is_main = rank == 0

    print_rank0(f"\n{'='*50}")
    print_rank0(f"Benchmarking: Full Training (DDP, {world_size} GPUs)")
    print_rank0(f"{'='*50}")

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
    total_bytes = 0
    global_step = 0
    start = time.perf_counter()

    total_for_tqdm = effective_steps * epochs if effective_steps != float('inf') else None
    progress = tqdm(loader, total=total_for_tqdm, desc="Training", disable=not is_main)

    for epoch in range(epochs):
        if hasattr(loader.sampler, 'set_epoch'):
            loader.sampler.set_epoch(epoch)

        for i, batch in enumerate(progress):
            if max_steps > 0 and global_step >= max_steps:
                break

            batch_dict, batch_bytes = batch
            batch_dict = {k: v.to(device, non_blocking=True) for k, v in batch_dict.items()}
            total_bytes += batch_bytes

            outputs = model(**batch_dict)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            count += batch_dict["input_ids"].shape[0]
            global_step += 1

            if is_main and global_step % 10 == 0:
                elapsed_so_far = time.perf_counter() - start
                current_sps = (count * world_size) / elapsed_so_far
                progress.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "samp/s": f"{current_sps:.1f}",
                })

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    count_tensor = torch.tensor([count], dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
    total_count = int(count_tensor.item())

    throughput = total_count / elapsed
    throughput_mbs = (total_bytes * world_size / (1024 * 1024)) / elapsed

    print_rank0(f"Processed {total_count} samples ({total_bytes * world_size / (1024*1024):.1f} MB) in {elapsed:.2f}s")
    print_rank0(f"Total throughput: {throughput:.4f} samples/sec (per-GPU: {throughput / world_size:.4f})")
    print_rank0(f"Global batch size: {args.batch_size * world_size}")
    return throughput, elapsed


# Orchestration

def run_benchmarks_for_workers(args, num_workers, tokenizer, hf_train, device, model):
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    is_main = rank == 0

    print_rank0(f"\n{'#'*60}")
    print_rank0(f"# NUM_WORKERS = {num_workers}  |  RANK = {rank}  |  WORLD_SIZE = {world_size}")
    print_rank0(f"{'#'*60}")

    results = {}
    mode = getattr(args, "mode", "train")
    max_steps = getattr(args, "max_steps", -1)
    if max_steps is None or max_steps < 0:
        max_steps = getattr(args, "num_batches", -1)
    if max_steps is None or max_steps < 0:
        max_steps = 0
    max_seq_length = getattr(args, "max_seq_length", 512)

    if mode in ["disk", "all"]:
        ds = DiskOnlyDataset(hf_train)
        loader = DataLoader(ds, batch_size=args.batch_size, num_workers=num_workers,
                            collate_fn=collate_disk_only, pin_memory=getattr(args, "pin_memory", True))
        results["disk"] = benchmark(loader, max_steps, "Disk I/O Only", rank)

    if mode in ["decode", "all"]:
        ds = DiskDecodeDataset(hf_train)
        loader = DataLoader(ds, batch_size=args.batch_size, num_workers=num_workers,
                            collate_fn=collate_disk_decode, pin_memory=getattr(args, "pin_memory", True))
        results["decode"] = benchmark(loader, max_steps, "Disk + Decode", rank)

    needs_tokenized = mode in ["preprocess", "transfer", "train", "all"]
    preprocess_ds = DiskDecodePreprocessDataset(hf_train, tokenizer, max_seq_length) if needs_tokenized else None

    if mode in ["preprocess", "all"] and preprocess_ds is not None:
        loader = DataLoader(preprocess_ds, batch_size=args.batch_size, num_workers=num_workers,
                            collate_fn=collate_full, pin_memory=getattr(args, "pin_memory", True))
        results["preprocess"] = benchmark(loader, max_steps, "Disk + Decode + Preprocess", rank)

    if mode in ["transfer", "all"] and preprocess_ds is not None:
        loader = DataLoader(preprocess_ds, batch_size=args.batch_size, num_workers=num_workers,
                            collate_fn=collate_full, pin_memory=getattr(args, "pin_memory", True))
        results["transfer"] = benchmark_transfer(loader, max_steps, device, rank)

    if mode in ["train", "all"] and preprocess_ds is not None:
        if world_size > 1:
            sampler = DistributedSampler(
                preprocess_ds,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
            )
            loader = DataLoader(preprocess_ds, batch_size=args.batch_size, sampler=sampler,
                                num_workers=num_workers, collate_fn=collate_full,
                                pin_memory=getattr(args, "pin_memory", True))
        else:
            loader = DataLoader(preprocess_ds, batch_size=args.batch_size, num_workers=num_workers,
                                collate_fn=collate_full, pin_memory=getattr(args, "pin_memory", True))
        results["train"] = benchmark_training(loader, max_steps, model, device, args)

    if dist.is_initialized():
        dist.barrier()
    return results


def main():
    parser = argparse.ArgumentParser(description="LLM Pretraining - DDP Throughput Benchmark")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--log_dir", type=str, help="Directory to save logs")
    cli_args = parser.parse_args()

    args = load_config(cli_args.config)
    args.log_dir = cli_args.log_dir

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    distributed = world_size > 1

    if distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0

    torch.manual_seed(SEED + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED + rank)

    print_rank0(f"\n{'='*50}")
    print_rank0(f"DDP Throughput Benchmark: {cli_args.config}")
    print_rank0(f"{'='*50}")
    if is_main:
        pprint(vars(args))
    print_rank0(f"World size: {world_size}, Local rank: {local_rank}, Rank: {rank}")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    tokenizer = load_tokenizer(args)
    print_rank0(f"Loading dataset: {args.dataset_name} ({getattr(args, 'data_dir', 'en')})")
    hf_train = load_hf_dataset(args)
    print_rank0(f"Dataset size: {len(hf_train)} samples")

    mode = getattr(args, "mode", "train")

    model = None
    if mode in ["train", "all"]:
        print_rank0(f"Loading model: {args.model_name}")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            use_cache=False,
        )
        model.to(device)

        if distributed:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
            print_rank0("Model wrapped with DistributedDataParallel")

        if getattr(args, 'torch_compile', False) and not distributed:
            model = torch.compile(model)

        print_rank0(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    num_workers_config = getattr(args, "num_workers", 0)
    if isinstance(num_workers_config, list):
        worker_counts = num_workers_config
    else:
        worker_counts = [num_workers_config]

    all_results = {}
    for nw in worker_counts:
        all_results[nw] = run_benchmarks_for_workers(
            args, nw, tokenizer, hf_train, device, model,
        )

    if is_main:
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

        log_dir = cli_args.log_dir or getattr(args, "output_dir", None)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            csv_path = os.path.join(log_dir, f"benchmark_results_{int(time.time())}.csv")
            print(f"\nSaving results to: {csv_path}")
            with open(csv_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["num_workers", "mode", "throughput_samp_s", "throughput_mb_s"])
                for nw, mode_results in all_results.items():
                    for mode_name, (samp_s, mb_s) in mode_results.items():
                        writer.writerow([nw, mode_name, f"{samp_s:.4f}", f"{mb_s:.4f}"])
            print("[Success]")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
