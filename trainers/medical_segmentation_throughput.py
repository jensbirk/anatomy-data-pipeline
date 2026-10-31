import os
import time
import argparse
import csv
import glob

import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from monai.apps import DecathlonDataset
from monai.losses import DiceLoss
from monai.networks.nets import SegResNet, UNet, SwinUNETR
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    Orientationd,
    Spacingd,
    RandSpatialCropd,
    RandFlipd,
    NormalizeIntensityd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    MapTransform,
)
from monai.utils import set_determinism

from trainers.utils import load_config
from trainers.ckpt_timing import CheckpointTimer
from system.loggers.metrics import CSVLogger
from pprint import pprint

SEED = 42
ROI_SIZE = [224, 224, 128]


class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            result.append(torch.logical_or(d[key] == 2, d[key] == 3))
            result.append(torch.logical_or(torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1))
            result.append(d[key] == 2)
            d[key] = torch.stack(result, axis=0).float()
        return d


# Stage-specific Datasets

class DiskOnlyDataset(Dataset):
    def __init__(self, root_dir, task, section="training"):
        # Use DecathlonDataset just to get the file list
        ds = DecathlonDataset(
            root_dir=root_dir, task=task, section=section,
            download=False, cache_rate=0.0,
            transform=None,
        )
        self.items = ds.data  # list of dicts with "image" and "label" paths
        # Pre-compute file sizes
        self.file_sizes = []
        for item in self.items:
            size = 0
            for path in item["image"] if isinstance(item["image"], list) else [item["image"]]:
                size += os.path.getsize(path)
            size += os.path.getsize(item["label"])
            self.file_sizes.append(size)
        print(f"[DiskOnly] Loaded {len(self.items)} samples")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        # Read all image modality files + label as raw bytes
        paths = item["image"] if isinstance(item["image"], list) else [item["image"]]
        paths = paths + [item["label"]]
        for path in paths:
            with open(path, 'rb') as f:
                _ = f.read()
        return idx, self.file_sizes[idx]


class DiskDecodeDataset(Dataset):
    def __init__(self, root_dir, task, section="training"):
        ds = DecathlonDataset(
            root_dir=root_dir, task=task, section=section,
            download=False, cache_rate=0.0,
            transform=None,
        )
        self.items = ds.data
        self.loader = Compose([
            LoadImaged(keys=["image", "label"]),
            # EnsureChannelFirstd(keys="image"),
            # EnsureTyped(keys=["image", "label"]),
            # ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        ])
        self.file_sizes = []
        for item in self.items:
            size = 0
            for path in item["image"] if isinstance(item["image"], list) else [item["image"]]:
                size += os.path.getsize(path)
            size += os.path.getsize(item["label"])
            self.file_sizes.append(size)
        print(f"[Diskdecode] Loaded {len(self.items)} samples")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = dict(self.items[idx])
        d = self.loader(item)
        return d["image"], d["label"], self.file_sizes[idx]


class DiskdecodePreprocessDataset(Dataset):
    def __init__(self, root_dir, task, transform, section="training"):
        ds = DecathlonDataset(
            root_dir=root_dir, task=task, section=section,
            download=False, cache_rate=0.0,
            transform=None,
        )
        self.items = ds.data
        self.transform = transform
        self.file_sizes = []
        for item in self.items:
            size = 0
            for path in item["image"] if isinstance(item["image"], list) else [item["image"]]:
                size += os.path.getsize(path)
            size += os.path.getsize(item["label"])
            self.file_sizes.append(size)
        print(f"[DiskdecodePreprocess] Loaded {len(self.items)} samples")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = dict(self.items[idx])
        d = self.transform(item)
        return d["image"], d["label"], self.file_sizes[idx]


# Collate functions

def collate_disk_only(batch):
    total_bytes = sum(item[1] for item in batch)
    return len(batch), total_bytes


def collate_decode(batch):
    total_bytes = sum(item[2] for item in batch)
    return len(batch), total_bytes


def collate_full(batch):
    images = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    total_bytes = sum(item[2] for item in batch)
    return images, labels, total_bytes


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
    else:
        loader_kwargs['prefetch_factor'] = None

    return DataLoader(dataset, **loader_kwargs)


def get_train_transform():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        RandSpatialCropd(keys=["image", "label"], roi_size=ROI_SIZE, random_size=False),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
    ])


def load_model(model_name, device):
    print(f"Loading model: {model_name}")

    if model_name == "SegResNet":
        model = SegResNet(
            blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1],
            init_filters=16, in_channels=4, out_channels=3, dropout_prob=0.2,
        )
    elif model_name == "UNet":
        model = UNet(
            spatial_dims=3, in_channels=4, out_channels=3,
            channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2, norm=("instance", {"affine": True})
        )
    elif model_name == "SwinUNETR":
        model = SwinUNETR(
            in_channels=4, out_channels=3, feature_size=48, use_checkpoint=True,
        )
    else:
        raise ValueError(f"Model {model_name} not recognized.")

    return model.to(device)


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
            count += batch[0].shape[0]
            total_bytes += batch[2]

    elapsed = time.perf_counter() - start
    throughput = count / elapsed
    throughput_mbs = (total_bytes / (1024 * 1024)) / elapsed

    print(f"Processed {count} volumes ({total_bytes / (1024**2):.1f} MB) in {elapsed:.2f}s")
    print(f"Throughput: {throughput:.2f} volumes/sec | {throughput_mbs:.1f} MB/s")

    return throughput, throughput_mbs


def benchmark_transfer(loader, max_steps, device):
    print(f"\n{'='*50}")
    print(f"Benchmarking: Disk + decode + Preprocess + GPU Transfer")
    print(f"{'='*50}")

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    count = 0
    total_bytes = 0
    start = time.perf_counter()

    for i, batch in enumerate(loader):
        if max_steps > 0 and i >= max_steps:
            break
        images = batch[0].to(device, non_blocking=True)
        labels = batch[1].to(device, non_blocking=True)
        count += images.shape[0]
        total_bytes += batch[2]

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    throughput = count / elapsed
    throughput_mbs = (total_bytes / (1024 * 1024)) / elapsed

    print(f"Processed {count} volumes ({total_bytes / (1024**2):.1f} MB) in {elapsed:.2f}s")
    print(f"Throughput: {throughput:.2f} volumes/sec | {throughput_mbs:.1f} MB/s")

    return throughput, throughput_mbs


def benchmark_training(loader, max_steps, model, device, args=None):
    print(f"\n{'='*50}")
    print(f"Benchmarking: Full Training (Disk + decode + Preprocess + Transfer + Compute)")
    print(f"{'='*50}")

    csv_logger = None
    if getattr(args, 'log_loss', False) and getattr(args, 'log_dir', None):
        csv_logger = CSVLogger(
            filepath=os.path.join(args.log_dir, "trainer_metrics.csv"),
            fieldnames=["step", "epoch", "loss"],
            append=False,
        )

    model.train()
    loss_function = DiceLoss(
        smooth_nr=0, smooth_dr=1e-5, squared_pred=True,
        to_onehot_y=False, sigmoid=True,
    )
    lr = getattr(args, 'lr', None)
    if lr is None:
        lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scaler = torch.GradScaler("cuda")

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    count = 0
    total_bytes = 0
    global_step = 0
    epochs = getattr(args, 'epochs', 1) if args else 1
    timer = CheckpointTimer(args)
    start = time.perf_counter()

    for epoch in range(epochs):
        for i, batch in enumerate(loader):
            if max_steps > 0 and global_step >= max_steps * (epoch + 1):
                break

            images = batch[0].to(device, non_blocking=True)
            labels = batch[1].to(device, non_blocking=True)
            total_bytes += batch[2]

            optimizer.zero_grad()
            with torch.autocast("cuda"):
                outputs = model(images)
                loss = loss_function(outputs, labels)

                if csv_logger is not None and global_step % 10 == 0:
                    csv_logger.log("step", global_step)
                    csv_logger.log("epoch", epoch + 1)
                    csv_logger.log("loss", loss.item())
                    csv_logger.step()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            count += images.shape[0]
            global_step += 1

            save_steps = (getattr(args, 'save_steps', 0) or 0) if args else 0
            timer.step_done(global_step)
            if timer.should_save(global_step, save_steps):
                os.makedirs(args.output_dir, exist_ok=True)
                checkpoint_state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": i,
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
    throughput_mbs = (total_bytes / (1024 * 1024)) / elapsed

    print(f"Processed {count} volumes ({total_bytes / (1024**2):.1f} MB) in {elapsed:.2f}s")
    print(f"Throughput: {throughput:.2f} volumes/sec | {throughput_mbs:.1f} MB/s")

    return throughput, throughput_mbs


# Orchestration

def run_benchmarks_for_workers(args, num_workers, device):
    print(f"\n{'#'*60}")
    print(f"# NUM_WORKERS = {num_workers}")
    print(f"{'#'*60}")

    results = {}
    mode = getattr(args, 'mode', 'all')
    max_steps = getattr(args, 'max_steps', None)
    if max_steps is None:
        max_steps = getattr(args, 'num_batches', 0) or 0
    task = args.task

    if mode in ["disk", "all"]:
        dataset = DiskOnlyDataset(args.cache_dir, task)
        loader = make_loader(dataset, args, collate_disk_only, num_workers)
        results["disk"] = benchmark(loader, max_steps, "Disk I/O Only")

    if mode in ["decode", "all"]:
        dataset = DiskDecodeDataset(args.cache_dir, task)
        loader = make_loader(dataset, args, collate_decode, num_workers)
        results["decode"] = benchmark(loader, max_steps, "Disk + decode (NIfTI)")

    if mode in ["preprocess", "all"]:
        transform = get_train_transform()
        dataset = DiskdecodePreprocessDataset(args.cache_dir, task, transform)
        loader = make_loader(dataset, args, collate_full, num_workers)
        results["preprocess"] = benchmark(loader, max_steps, "Disk + decode + Preprocess")

    if mode in ["transfer", "all"]:
        transform = get_train_transform()
        dataset = DiskdecodePreprocessDataset(args.cache_dir, task, transform)
        loader = make_loader(dataset, args, collate_full, num_workers)
        results["transfer"] = benchmark_transfer(loader, max_steps, device)

    if mode in ["train", "all"]:
        transform = get_train_transform()
        dataset = DiskdecodePreprocessDataset(args.cache_dir, task, transform)
        loader = make_loader(dataset, args, collate_full, num_workers)
        model_name = getattr(args, 'model', 'SegResNet')
        model = load_model(model_name, device)
        results["train"] = benchmark_training(loader, max_steps, model, device, args=args)

    return results


def main():
    parser = argparse.ArgumentParser(description="Medical 3D Segmentation - Throughput Benchmark")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--log_dir", type=str, help="Directory to save logs")
    cli_args = parser.parse_args()

    args = load_config(cli_args.config)

    args.log_dir = cli_args.log_dir

    # Seeding
    set_determinism(seed=SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print(f"\n{'='*50}")
    print(f"Throughput Benchmark: {cli_args.config}")
    print(f"{'='*50}")
    pprint(vars(args))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Get worker counts (support list or single value)
    num_workers_config = getattr(args, 'num_workers', 0)
    if isinstance(num_workers_config, list):
        worker_counts = num_workers_config
    else:
        worker_counts = [num_workers_config]

    all_results = {}
    for num_workers in worker_counts:
        all_results[num_workers] = run_benchmarks_for_workers(args, num_workers, device)

    # Print summary
    if len(worker_counts) > 1:
        mode_names = list(all_results[worker_counts[0]].keys())

        # volumes/s table
        print(f"\n{'='*70}")
        print("Summary (volumes/s)")
        print(f"{'='*70}")
        header = f"{'workers':<10}" + "".join(f"{mode:>14}" for mode in mode_names)
        print(header)
        print("-" * len(header))
        for nw in worker_counts:
            row = f"{nw:<10}"
            for mode in mode_names:
                vol_tp, _ = all_results[nw][mode]
                row += f"{vol_tp:>14.2f}"
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
            for name, (vol_tp, mb_tp) in results.items():
                print(f"  {name:20s}: {vol_tp:>8.2f} vol/s | {mb_tp:>8.1f} MB/s")

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
                writer.writerow(['num_workers', 'mode', 'throughput_vol_s', 'throughput_mb_s'])
                for nw, mode_results in all_results.items():
                    for mode_name, (vol_s, mb_s) in mode_results.items():
                        writer.writerow([nw, mode_name, f"{vol_s:.4f}", f"{mb_s:.4f}"])
            print("[Success]")
        except Exception as e:
            print(f"[Error saving CSV: {e}]")


if __name__ == "__main__":
    main()