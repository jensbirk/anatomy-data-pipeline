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
from trainers.ckpt_timing import CheckpointTimer
from system.loggers.metrics import CSVLogger
# from utils.checkpoint import BatchedAsyncCheckpointer

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
        self.byte_sizes = []
        for i in range(len(hf_dataset)):
            item = hf_dataset[i]
            size = len(item['instruction'].encode('utf-8'))
            size += len(item['input'].encode('utf-8'))
            size += len(item['output'].encode('utf-8'))
            self.byte_sizes.append(size)
        print(f"[DiskOnly] Loaded {len(self.dataset)} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        _ = item['instruction']
        _ = item['input']
        _ = item['output']
        return idx, self.byte_sizes[idx]


class DiskDecodeDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset
        self.byte_sizes = []
        for i in range(len(hf_dataset)):
            item = hf_dataset[i]
            size = len(item['instruction'].encode('utf-8'))
            size += len(item['input'].encode('utf-8'))
            size += len(item['output'].encode('utf-8'))
            self.byte_sizes.append(size)
        print(f"[DiskDecode] Loaded {len(self.dataset)} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        inst, inp, out = item['instruction'], item['input'], item['output']

        # "Decode": format raw fields into prompt strings
        if inp != "":
            full_text = PROMPT_DICT["prompt_input"].format_map({'instruction': inst, 'input': inp}) + out
            prompt_text = PROMPT_DICT["prompt_input"].format_map({'instruction': inst, 'input': inp})
        else:
            full_text = PROMPT_DICT["prompt_no_input"].format_map({'instruction': inst}) + out
            prompt_text = PROMPT_DICT["prompt_no_input"].format_map({'instruction': inst})

        return full_text, prompt_text, self.byte_sizes[idx]


class DiskDecodePreprocessDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_seq_length):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.byte_sizes = []
        for i in range(len(hf_dataset)):
            item = hf_dataset[i]
            size = len(item['instruction'].encode('utf-8'))
            size += len(item['input'].encode('utf-8'))
            size += len(item['output'].encode('utf-8'))
            self.byte_sizes.append(size)
        print(f"[DiskDecodePreprocess] Loaded {len(self.dataset)} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        inst, inp, out = item['instruction'], item['input'], item['output']

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

        return input_ids, attention_mask, labels, self.byte_sizes[idx]


# Collate functions

def collate_disk_only(batch):
    total_bytes = sum(item[1] for item in batch)
    return len(batch), total_bytes


def collate_disk_decode(batch):
    total_bytes = sum(item[2] for item in batch)
    return len(batch), total_bytes


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
    dataset = load_dataset(
        path=args.dataset_name,
        data_dir=getattr(args, 'data_dir', None),
        cache_dir=getattr(args, 'cache_dir', None),
    )
    if getattr(args, 'max_samples', None):
        dataset["train"] = dataset["train"].select(range(args.max_samples))
    return dataset["train"]


# Benchmark functions

def benchmark(loader, max_steps, name):
    print(f"\n{'='*50}")
    print(f"Benchmarking: {name}")
    print(f"{'='*50}")

    count = 0
    total_bytes = 0
    start = time.perf_counter()

    for i, batch in enumerate(loader):
        if max_steps > 0 and i >= max_steps:
            break
        if isinstance(batch[0], int):
            count += batch[0]
            total_bytes += batch[1]
        else:
            count += batch[0]["input_ids"].shape[0]
            total_bytes += batch[1]

    elapsed = time.perf_counter() - start
    throughput = count / elapsed
    throughput_mbs = (total_bytes / (1024 * 1024)) / elapsed

    print(f"Processed {count} samples ({total_bytes / (1024*1024):.1f} MB) in {elapsed:.2f}s")
    print(f"Throughput: {throughput:.1f} samples/sec | {throughput_mbs:.1f} MB/s")

    return throughput, throughput_mbs


def benchmark_transfer(loader, max_steps, device):
    print(f"\n{'='*50}")
    print(f"Benchmarking: Disk + Decode + Preprocess + GPU Transfer")
    print(f"{'='*50}")

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    count = 0
    total_bytes = 0
    start = time.perf_counter()

    for i, batch in enumerate(loader):
        if max_steps > 0 and i >= max_steps:
            break

        batch_dict, batch_bytes = batch
        batch_dict = {k: v.to(device, non_blocking=True) for k, v in batch_dict.items()}
        total_bytes += batch_bytes
        count += batch_dict["input_ids"].shape[0]

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    throughput = count / elapsed
    throughput_mbs = (total_bytes / (1024 * 1024)) / elapsed

    print(f"Processed {count} samples ({total_bytes / (1024*1024):.1f} MB) in {elapsed:.2f}s")
    print(f"Throughput: {throughput:.1f} samples/sec | {throughput_mbs:.1f} MB/s")

    return throughput, throughput_mbs


def benchmark_training(loader, max_steps, model, device, args):
    print(f"\n{'='*50}")
    print(f"Benchmarking: Full Training (Disk + Decode + Preprocess + Transfer + Compute)")
    print(f"{'='*50}")

    csv_logger = None
    if getattr(args, 'log_loss', False) and getattr(args, 'log_dir', None):
        csv_logger = CSVLogger(
            filepath=os.path.join(args.log_dir, "trainer_metrics.csv"),
            fieldnames=["step", "epoch", "loss"],
            append=False,
        )

    model.train()

    lr = getattr(args, 'lr', None)
    if lr is None:
        lr = 2e-5
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.1,
    )

    effective_steps = max_steps if max_steps > 0 else len(loader)
    epochs = getattr(args, 'epochs', 1)
    total_steps = effective_steps * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    # batched_ckpter = BatchedAsyncCheckpointer() if getattr(args, 'batched_checkpoint', False) else None

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    count = 0
    total_bytes = 0
    global_step = 0
    timer = CheckpointTimer(args)
    start = time.perf_counter()

    for epoch in range(epochs):
        for i, batch in enumerate(loader):
            if max_steps > 0 and global_step >= max_steps * (epoch + 1):
                break

            batch_dict, batch_bytes = batch
            batch_dict = {k: v.to(device, non_blocking=True) for k, v in batch_dict.items()}
            total_bytes += batch_bytes

            outputs = model(**batch_dict)
            loss = outputs.loss

            if csv_logger is not None and global_step % 10 == 0:
                csv_logger.log("step", global_step)
                csv_logger.log("epoch", epoch + 1)
                csv_logger.log("loss", loss.item())
                csv_logger.step()

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            count += batch_dict["input_ids"].shape[0]
            global_step += 1

            timer.step_done(global_step)
            if timer.should_save(global_step, args.save_steps):
                os.makedirs(args.output_dir, exist_ok=True)

                checkpoint_state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": global_step,
                }

                # if batched_ckpter is not None:
                #     ckpt_path = os.path.join(args.output_dir, "checkpoint.ckpt")
                #     batched_ckpter.save(checkpoint_state, ckpt_path)
                if timer.enabled:
                    timer.save(checkpoint_state, global_step)
                elif args.async_checkpoint:
                    dcp.async_save(
                        state_dict=checkpoint_state,
                        storage_writer=dcp.FileSystemWriter(args.output_dir),
                    )
                else:
                    torch.save(checkpoint_state, os.path.join(args.output_dir, "checkpoint.pt"))
                print("Saved checkpoint")

    # if batched_ckpter is not None:
    #     batched_ckpter.wait()

    if csv_logger is not None:
        csv_logger.close()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    timer.finish()
    throughput = count / elapsed
    throughput_mbs = (total_bytes / (1024 * 1024)) / elapsed

    print(f"Processed {count} samples ({total_bytes / (1024*1024):.1f} MB) in {elapsed:.2f}s")
    print(f"Throughput: {throughput:.1f} samples/sec | {throughput_mbs:.1f} MB/s")

    return throughput, throughput_mbs


# Orchestration

def run_benchmarks_for_workers(args, num_workers, tokenizer, hf_train, device):
    print(f"\n{'#'*60}")
    print(f"# NUM_WORKERS = {num_workers}")
    print(f"{'#'*60}")

    results = {}
    mode = getattr(args, 'mode', 'all')
    max_steps = getattr(args, 'max_steps', None)
    if max_steps is None:
        max_steps = getattr(args, 'num_batches', 0)
    max_seq_length = getattr(args, 'max_seq_length', 512)

    if mode in ["disk", "all"]:
        dataset = DiskOnlyDataset(hf_train)
        loader = make_loader(dataset, args, collate_disk_only, num_workers)
        results["disk"] = benchmark(loader, max_steps, "Disk I/O Only")

    if mode in ["decode", "all"]:
        dataset = DiskDecodeDataset(hf_train)
        loader = make_loader(dataset, args, collate_disk_decode, num_workers)
        results["decode"] = benchmark(loader, max_steps, "Disk + Decode")

    if mode in ["preprocess", "all"]:
        dataset = DiskDecodePreprocessDataset(hf_train, tokenizer, max_seq_length)
        loader = make_loader(dataset, args, collate_full, num_workers)
        results["preprocess"] = benchmark(loader, max_steps, "Disk + Decode + Preprocess")

    if mode in ["transfer", "all"]:
        dataset = DiskDecodePreprocessDataset(hf_train, tokenizer, max_seq_length)
        loader = make_loader(dataset, args, collate_full, num_workers)
        results["transfer"] = benchmark_transfer(loader, max_steps, device)

    if mode in ["train", "all"]:
        dataset = DiskDecodePreprocessDataset(hf_train, tokenizer, max_seq_length)
        loader = make_loader(dataset, args, collate_full, num_workers)
        model = load_model(args, device)
        results["train"] = benchmark_training(loader, max_steps, model, device, args)

    return results


def main():
    parser = argparse.ArgumentParser(description="LLM Fine-tuning - Throughput Benchmark")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--log_dir", type=str, help="Directory to save logs")
    cli_args = parser.parse_args()

    args = load_config(cli_args.config)

    args.log_dir = cli_args.log_dir

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

    # Load tokenizer and dataset once (shared across all benchmarks)
    tokenizer = load_tokenizer(args)
    hf_train = load_hf_dataset(args)
    print(f"Dataset size: {len(hf_train)} samples")

    # Get worker counts (support list or single value)
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

        # samples/s table
        print(f"\n{'='*70}")
        print("Summary (samples/s)")
        print(f"{'='*70}")
        header = f"{'workers':<10}" + "".join(f"{mode:>14}" for mode in mode_names)
        print(header)
        print("-" * len(header))
        for nw in worker_counts:
            row = f"{nw:<10}"
            for mode in mode_names:
                samp_tp, _ = all_results[nw][mode]
                row += f"{samp_tp:>14.1f}"
            print(row)

        # MB/s table
        print(f"\n{'='*70}")
        print("Summary (MB/s)")
        print(f"{'='*70}")
        print(header)
        print("-" * len(header))
        for nw in worker_counts:
            row = f"{nw:<10}"
            for mode in mode_names:
                _, mb_tp = all_results[nw][mode]
                row += f"{mb_tp:>14.1f}"
            print(row)
    else:
        results = all_results[worker_counts[0]]
        if len(results) > 1:
            print(f"\n{'='*50}")
            print("Summary")
            print(f"{'='*50}")
            for name, (samp_tp, mb_tp) in results.items():
                print(f"  {name:20s}: {samp_tp:>8.1f} samp/s | {mb_tp:>8.1f} MB/s")

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
                writer.writerow(['num_workers', 'mode', 'throughput_samp_s', 'throughput_mb_s'])
                for nw, mode_results in all_results.items():
                    for mode_name, (samp_s, mb_s) in mode_results.items():
                        writer.writerow([nw, mode_name, f"{samp_s:.4f}", f"{mb_s:.4f}"])
            print("[Success]")
        except Exception as e:
            print(f"[Error saving CSV: {e}]")


if __name__ == "__main__":
    main()