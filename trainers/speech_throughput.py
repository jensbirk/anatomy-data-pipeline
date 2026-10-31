#!/usr/bin/env python3
import os
import time
import argparse
import csv
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed.checkpoint as dcp
import torchaudio
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    set_seed,
)
from datasets import load_from_disk, Dataset as HFDataset, DatasetDict

from trainers.utils import load_config
from trainers.ckpt_timing import CheckpointTimer
from system.loggers.metrics import CSVLogger

import warnings
warnings.filterwarnings("ignore")

SEED = 42


# Universal source + dataset builders

class UniversalAudioSource:
    def __init__(
        self,
        source_type: str,
        source_path: str,
        target_sr: int = 16000,
        audio_column: str = "audio",
        text_column: str = "text",
        arrow_split: Optional[str] = None,
    ):
        self.source_type = source_type.lower()
        self.source_path = source_path
        self.target_sr = target_sr
        self.audio_column = audio_column
        self.text_column = text_column
        self.arrow_split = arrow_split
        self._resamplers: Dict[Tuple[int, int], torchaudio.transforms.Resample] = {}

        self.dataset: Optional[HFDataset] = None
        if self.source_type == "arrow":
            obj = load_from_disk(source_path)

            if isinstance(obj, DatasetDict):
                if arrow_split is None:
                    available = list(obj.keys())
                    raise ValueError(
                        f"dataset_root points to a DatasetDict. Please set args.dataset_split "
                        f"to one of: {available}"
                    )
                if arrow_split not in obj and len(obj) == 1:
                    # Older configs set dataset_split to the benchmark mode
                    # (e.g. "decode"); a single-split dataset has only one choice.
                    print(f"Split '{arrow_split}' not in dataset; using its only split "
                          f"'{next(iter(obj))}'")
                    arrow_split = next(iter(obj))
                if arrow_split not in obj:
                    available = list(obj.keys())
                    raise ValueError(
                        f"Split '{arrow_split}' not found in DatasetDict. Available: {available}"
                    )
                self.dataset = obj[arrow_split]
            else:
                self.dataset = obj

    def _get_resampler(self, src_sr: int):
        key = (src_sr, self.target_sr)
        if key not in self._resamplers:
            self._resamplers[key] = torchaudio.transforms.Resample(src_sr, self.target_sr)
        return self._resamplers[key]

    def load_audio_file(self, path: str):
        waveform, sr = torchaudio.load(path)  # [channels, time]

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.target_sr:
            waveform = self._get_resampler(sr)(waveform)
            sr = self.target_sr

        return waveform.squeeze(0).numpy(), sr

    def load_audio(self, sample: dict):
        fmt = sample["format"]

        if fmt in {"flac", "mp3", "wav"}:
            return self.load_audio_file(sample["audio"])

        if fmt == "arrow":
            if self.dataset is None:
                raise RuntimeError("Arrow dataset not initialized")

            item = self.dataset[sample["hf_index"]]
            audio = item[self.audio_column]

            audio_array = audio["array"]
            sr = audio["sampling_rate"]

            if sr != self.target_sr:
                waveform = torch.tensor(audio_array, dtype=torch.float32)
                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0)
                elif waveform.ndim == 2 and waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)

                waveform = self._get_resampler(sr)(waveform)
                audio_array = waveform.squeeze(0).numpy()
                sr = self.target_sr

            return audio_array, sr

        raise ValueError(f"Unsupported format: {fmt}")


def build_librispeech_file_samples(root_dir: str, ext: str) -> List[dict]:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"dataset_root does not exist: {root_dir}")

    samples: List[dict] = []
    trans_files = sorted(root.rglob("*.trans.txt"))

    if not trans_files:
        raise RuntimeError(
            f"No transcript files (*.trans.txt) found under {root_dir}. "
            f"Expected a LibriSpeech-style directory tree."
        )

    for trans_file in trans_files:
        with open(trans_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                utt_id, text = line.split(" ", 1)
                audio_path = trans_file.parent / f"{utt_id}.{ext}"

                if audio_path.exists():
                    samples.append({
                        "id": utt_id,
                        "audio": str(audio_path),
                        "text": text,
                        "format": ext,
                    })

    if not samples:
        raise RuntimeError(
            f"Found transcript files under {root_dir}, but no matching .{ext} audio files."
        )

    return samples


def build_arrow_samples(
    dataset: HFDataset,
    audio_column: str = "audio",
    text_column: str = "text",
) -> List[dict]:
    if audio_column not in dataset.column_names:
        raise ValueError(
            f"Audio column '{audio_column}' not found in arrow dataset. "
            f"Available columns: {dataset.column_names}"
        )
    if text_column not in dataset.column_names:
        raise ValueError(
            f"Text column '{text_column}' not found in arrow dataset. "
            f"Available columns: {dataset.column_names}"
        )

    samples: List[dict] = []
    for i in range(len(dataset)):
        item = dataset[i]
        text = item[text_column]

        samples.append({
            "id": str(i),
            "audio": None,          # no direct filesystem audio path guaranteed
            "text": text,
            "format": "arrow",
            "hf_index": i,
        })

    return samples


class UnifiedSpeechDataset(Dataset):
    def __init__(self, samples: List[dict], source: UniversalAudioSource):
        self.samples = samples
        self.source = source

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio_array, sr = self.source.load_audio(sample)

        return {
            "id": sample["id"],
            "audio": {
                "path": sample.get("audio"),
                "array": audio_array,
                "sampling_rate": sr,
            },
            "text": sample["text"],
            "format": sample["format"],
        }


def load_audio_dataset(args):
    dataset_type = getattr(args, "dataset_type", None)
    dataset_root = getattr(args, "dataset_root", None)

    if dataset_type is None:
        raise ValueError("Missing required config: dataset_type")
    if dataset_root is None:
        raise ValueError("Missing required config: dataset_root")

    dataset_type = dataset_type.lower()
    target_sr = getattr(args, "sampling_rate", 16000)
    audio_column = getattr(args, "audio_column", "audio")
    text_column = getattr(args, "text_column", "text")
    dataset_split = getattr(args, "dataset_split", None)

    if dataset_type == "arrow":
        source = UniversalAudioSource(
            source_type="arrow",
            source_path=dataset_root,
            target_sr=target_sr,
            audio_column=audio_column,
            text_column=text_column,
            arrow_split=dataset_split,
        )
        if source.dataset is None:
            raise RuntimeError("Failed to initialize arrow dataset")

        samples = build_arrow_samples(
            source.dataset,
            audio_column=audio_column,
            text_column=text_column,
        )
        return UnifiedSpeechDataset(samples, source)

    if dataset_type in {"flac", "mp3", "wav"}:
        source = UniversalAudioSource(
            source_type=dataset_type,
            source_path=dataset_root,
            target_sr=target_sr,
            audio_column=audio_column,
            text_column=text_column,
        )
        samples = build_librispeech_file_samples(dataset_root, dataset_type)
        return UnifiedSpeechDataset(samples, source)

    raise ValueError(
        f"Unsupported dataset_type: {dataset_type}. "
        f"Expected one of: arrow, flac, mp3, wav"
    )


# Stage-specific benchmark datasets

class DiskOnlyDataset(Dataset):
    def __init__(self, base_dataset, audio_column="audio"):
        self.dataset = base_dataset
        self.audio_column = audio_column

        self.paths = []
        self.file_sizes = []
        self.formats = []

        for sample in self.dataset.samples:
            path = sample.get("audio")
            fmt = sample.get("format")
            self.formats.append(fmt)

            if path and os.path.exists(str(path)):
                self.paths.append(path)
                self.file_sizes.append(os.path.getsize(path))
            else:
                self.paths.append(None)
                self.file_sizes.append(0)

        print(f"[DiskOnly] Loaded {len(self.dataset)} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        path = self.paths[idx]

        if path:
            with open(path, "rb") as f:
                _ = f.read()
        else:
            _ = self.dataset[idx]["audio"]

        return idx, self.file_sizes[idx]


class DiskDecodeDataset(Dataset):
    def __init__(self, base_dataset, audio_column="audio"):
        self.dataset = base_dataset
        self.audio_column = audio_column

        self.paths = []
        self.file_sizes = []

        for sample in self.dataset.samples:
            # Safely extract the path string depending on dataset structure
            audio_data = sample.get(self.audio_column)
            path = audio_data.get("path") if isinstance(audio_data, dict) else audio_data
            
            if path and os.path.exists(str(path)):
                self.paths.append(str(path))
                self.file_sizes.append(os.path.getsize(path))
            else:
                # Raise an error early during initialization if a file is missing
                raise FileNotFoundError(f"Audio file not found or invalid path: {path}")

        print(f"[DiskDecode] Successfully loaded {len(self.paths)} samples for decoding.")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        
        waveform, sr = torchaudio.load(path)
                
        return waveform, sr, self.file_sizes[idx]


class DiskDecodePreprocessDataset(Dataset):
    def __init__(self, base_dataset, processor, audio_column="audio", text_column="text"):
        self.dataset = base_dataset
        self.processor = processor
        self.audio_column = audio_column
        self.text_column = text_column

        self.paths = []
        self.file_sizes = []

        for sample in self.dataset.samples:
            path = sample.get("audio")
            if path and os.path.exists(str(path)):
                self.paths.append(path)
                self.file_sizes.append(os.path.getsize(path))
            else:
                self.paths.append(None)
                self.file_sizes.append(0)

        print(f"[DiskDecodePreprocess] Loaded {len(self.dataset)} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        audio_array = item[self.audio_column]["array"]
        sr = item[self.audio_column]["sampling_rate"]
        text = item[self.text_column]

        input_features = self.processor.feature_extractor(
            audio_array,
            sampling_rate=sr,
        ).input_features[0]
        labels = self.processor.tokenizer(text).input_ids

        return input_features, labels, self.file_sizes[idx]


# Collate functions

def collate_disk_only(batch):
    total_bytes = sum(item[1] for item in batch)
    return len(batch), total_bytes


def collate_disk_decode(batch):
    total_bytes = sum(item[2] for item in batch)
    return len(batch), total_bytes


def make_collate_full(processor, decoder_start_token_id):
    model_input_name = processor.model_input_names[0]

    def collate_full(batch):
        input_features_list = [item[0] for item in batch]
        labels_list = [item[1] for item in batch]
        total_bytes = sum(item[2] for item in batch)

        input_features_padded = processor.feature_extractor.pad(
            [{model_input_name: feat} for feat in input_features_list],
            return_tensors="pt",
        )

        labels_padded = processor.tokenizer.pad(
            [{"input_ids": lab} for lab in labels_list],
            return_tensors="pt",
        )
        labels = labels_padded["input_ids"].masked_fill(
            labels_padded.attention_mask.ne(1), -100,
        )

        if labels.shape[1] > 0 and (labels[:, 0] == decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        return {
            "input_features": input_features_padded["input_features"],
            "labels": labels,
        }, total_bytes

    return collate_full


# Helpers

def make_loader(dataset, args, collate_fn, num_workers):
    g = torch.Generator()
    g.manual_seed(SEED)

    loader_kwargs = {
        "batch_size": args.batch_size,
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


def load_model_and_processor(args, device):
    print(f"Loading processor for: {args.model_name}")
    processor = AutoProcessor.from_pretrained(
        args.model_name,
        language=getattr(args, "language", "english"),
        task="transcribe",
    )

    print(f"Loading model: {args.model_name}")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_name,
        low_cpu_mem_usage=True,
        use_cache=False,
    )
    model.to(device)

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    if getattr(args, "freeze_encoder", False):
        model.freeze_encoder()
        model.model.encoder.gradient_checkpointing = False

    return model, processor


# Benchmark functions

def benchmark(loader, max_steps, name):
    print(f"\n{'=' * 50}")
    print(f"Benchmarking: {name}")
    print(f"{'=' * 50}")

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
            count += batch[0]["input_features"].shape[0]
            total_bytes += batch[1]

    elapsed = time.perf_counter() - start
    throughput = count / elapsed
    throughput_mbs = (total_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0.0

    print(f"Processed {count} samples ({total_bytes / (1024 * 1024):.1f} MB) in {elapsed:.2f}s")
    print(f"Throughput: {throughput:.1f} samples/sec | {throughput_mbs:.1f} MB/s")

    return throughput, throughput_mbs


def benchmark_transfer(loader, max_steps, device):
    print(f"\n{'=' * 50}")
    print("Benchmarking: Disk + Decode + Preprocess + GPU Transfer")
    print(f"{'=' * 50}")

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    count = 0
    total_bytes = 0
    start = time.perf_counter()

    for i, batch in enumerate(loader):
        if max_steps > 0 and i >= max_steps:
            break

        batch_dict, batch_bytes = batch
        input_features = batch_dict["input_features"].to(device, non_blocking=True)
        labels = batch_dict["labels"].to(device, non_blocking=True)

        _ = labels
        total_bytes += batch_bytes
        count += input_features.shape[0]

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    throughput = count / elapsed
    throughput_mbs = (total_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0.0

    print(f"Processed {count} samples ({total_bytes / (1024 * 1024):.1f} MB) in {elapsed:.2f}s")
    print(f"Throughput: {throughput:.1f} samples/sec | {throughput_mbs:.1f} MB/s")

    return throughput, throughput_mbs


def benchmark_training(loader, max_steps, model, device, args):
    print(f"\n{'=' * 50}")
    print("Benchmarking: Full Training (Disk + Decode + Preprocess + Transfer + Compute)")
    print(f"{'=' * 50}")

    csv_logger = None
    if getattr(args, 'log_loss', False) and getattr(args, 'log_dir', None):
        csv_logger = CSVLogger(
            filepath=os.path.join(args.log_dir, "trainer_metrics.csv"),
            fieldnames=["step", "epoch", "loss"],
            append=False,
        )

    model.train()

    lr = getattr(args, "lr", None)
    if lr is None:
        lr = 1e-5
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.01,
    )

    effective_steps = max_steps if max_steps > 0 else len(loader)
    epochs = getattr(args, 'epochs', 1)
    total_steps = effective_steps * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None

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
            input_features = batch_dict["input_features"].to(device, non_blocking=True)
            labels = batch_dict["labels"].to(device, non_blocking=True)
            total_bytes += batch_bytes

            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.autocast("cuda"):
                    outputs = model(input_features=input_features, labels=labels)
                    loss = outputs.loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_features=input_features, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

            if csv_logger is not None and global_step % 10 == 0:
                csv_logger.log("step", global_step)
                csv_logger.log("epoch", epoch + 1)
                csv_logger.log("loss", loss.item())
                csv_logger.step()

            scheduler.step()
            count += input_features.shape[0]
            global_step += 1

            save_steps = getattr(args, 'save_steps', 0) or 0
            timer.step_done(global_step)
            if timer.should_save(global_step, save_steps):
                os.makedirs(args.output_dir, exist_ok=True)
                checkpoint_state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": global_step,
                }
                if timer.enabled:
                    timer.save(checkpoint_state, global_step)
                elif getattr(args, 'async_checkpoint', False):
                    dcp.async_save(
                        state_dict=checkpoint_state,
                        storage_writer=dcp.FileSystemWriter(args.output_dir),
                    )
                else:
                    torch.save(checkpoint_state, os.path.join(args.output_dir, "checkpoint.pt"))

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    timer.finish()
    throughput = count / elapsed
    throughput_mbs = (total_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0.0

    print(f"Processed {count} samples ({total_bytes / (1024 * 1024):.1f} MB) in {elapsed:.2f}s")
    print(f"Throughput: {throughput:.1f} samples/sec | {throughput_mbs:.1f} MB/s")

    return throughput, throughput_mbs


# Orchestration

def run_benchmarks_for_workers(args, num_workers, processor, decoder_start_token_id, train_dataset, device):
    print(f"\n{'#' * 60}")
    print(f"# NUM_WORKERS = {num_workers}")
    print(f"{'#' * 60}")

    results = {}
    mode = getattr(args, "mode", "all")
    max_steps = getattr(args, 'max_steps', None)
    if max_steps is None:
        max_steps = getattr(args, "num_batches", 0)
    audio_column = getattr(args, "audio_column", "audio")
    text_column = getattr(args, "text_column", "text")
    dataset_type = getattr(args, "dataset_type", "").lower()

    collate_fn = make_collate_full(processor, decoder_start_token_id)

    if mode in ["disk", "all"]:
        if dataset_type == "arrow":
            print("[Info] 'disk' benchmark for arrow has no original file path; falling back to dataset item access.")
        dataset = DiskOnlyDataset(train_dataset, audio_column)
        loader = make_loader(dataset, args, collate_disk_only, num_workers)
        results["disk"] = benchmark(loader, max_steps, "Disk I/O Only")

    if mode in ["decode", "all"]:
        dataset = DiskDecodeDataset(train_dataset, audio_column)
        loader = make_loader(dataset, args, collate_disk_decode, num_workers)
        results["decode"] = benchmark(loader, max_steps, "Disk + Decode")

    if mode in ["preprocess", "all"]:
        dataset = DiskDecodePreprocessDataset(train_dataset, processor, audio_column, text_column)
        loader = make_loader(dataset, args, collate_fn, num_workers)
        results["preprocess"] = benchmark(loader, max_steps, "Disk + Decode + Preprocess")

    if mode in ["transfer", "all"]:
        dataset = DiskDecodePreprocessDataset(train_dataset, processor, audio_column, text_column)
        loader = make_loader(dataset, args, collate_fn, num_workers)
        results["transfer"] = benchmark_transfer(loader, max_steps, device)

    if mode in ["train", "all"]:
        dataset = DiskDecodePreprocessDataset(train_dataset, processor, audio_column, text_column)
        loader = make_loader(dataset, args, collate_fn, num_workers)
        model, _ = load_model_and_processor(args, device)
        results["train"] = benchmark_training(loader, max_steps, model, device, args)

    return results


def main():
    parser = argparse.ArgumentParser(description="Speech Recognition - Throughput Benchmark")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--log_dir", type=str, help="Directory to save logs")
    cli_args = parser.parse_args()

    args = load_config(cli_args.config)

    args.log_dir = cli_args.log_dir

    set_seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print(f"\n{'=' * 50}")
    print(f"Throughput Benchmark: {cli_args.config}")
    print(f"{'=' * 50}")
    pprint(vars(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    print("Loading processor and model config...")
    processor = AutoProcessor.from_pretrained(
        args.model_name,
        language=getattr(args, "language", "english"),
        task="transcribe",
    )

    model_config = AutoConfig.from_pretrained(args.model_name)
    decoder_start_token_id = model_config.decoder_start_token_id

    train_dataset = load_audio_dataset(args)
    print(f"Dataset size: {len(train_dataset)} samples")

    num_workers_config = getattr(args, "num_workers", getattr(args, "workers", 0))
    if isinstance(num_workers_config, list):
        worker_counts = num_workers_config
    else:
        worker_counts = [num_workers_config]

    all_results = {}
    for num_workers in worker_counts:
        all_results[num_workers] = run_benchmarks_for_workers(
            args, num_workers, processor, decoder_start_token_id, train_dataset, device,
        )

    if len(worker_counts) > 1:
        mode_names = list(all_results[worker_counts[0]].keys())

        print(f"\n{'=' * 70}")
        print("Summary (samples/s)")
        print(f"{'=' * 70}")
        header = f"{'workers':<10}" + "".join(f"{mode:>14}" for mode in mode_names)
        print(header)
        print("-" * len(header))
        for nw in worker_counts:
            row = f"{nw:<10}"
            for mode in mode_names:
                samp_tp, _ = all_results[nw][mode]
                row += f"{samp_tp:>14.1f}"
            print(row)

        print(f"\n{'=' * 70}")
        print("Summary (MB/s)")
        print(f"{'=' * 70}")
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
            print(f"\n{'=' * 50}")
            print("Summary")
            print(f"{'=' * 50}")
            for name, (samp_tp, mb_tp) in results.items():
                print(f"  {name:20s}: {samp_tp:>8.1f} samp/s | {mb_tp:>8.1f} MB/s")

    log_dir = cli_args.log_dir or getattr(args, "output_dir", None)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        csv_path = os.path.join(log_dir, f"benchmark_results_{int(time.time())}.csv")

        print(f"\n{'=' * 50}")
        print(f"Saving results to: {csv_path}")

        try:
            with open(csv_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["num_workers", "mode", "throughput_samp_s", "throughput_mb_s"])
                for nw, mode_results in all_results.items():
                    for mode_name, (samp_s, mb_s) in mode_results.items():
                        writer.writerow([nw, mode_name, f"{samp_s:.4f}", f"{mb_s:.4f}"])
            print("[Success]")
        except Exception as e:
            print(f"[Error saving CSV: {e}]")


if __name__ == "__main__":
    main()