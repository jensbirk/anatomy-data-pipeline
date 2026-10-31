import os
import time
import argparse
import csv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from datasets import load_dataset
from transformers import ViTForImageClassification, ViTConfig, AutoModelForImageClassification, AutoConfig

from trainers.utils import load_config
from pprint import pprint

import warnings
warnings.filterwarnings("ignore", message=".*Corrupt EXIF data.*")


class HFImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def get_dataloaders(args):
    print(f"Loading dataset from {args.cache_dir}...")
    dataset_train = load_dataset(args.dataset_name, split="train", cache_dir=args.cache_dir)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    num_workers = getattr(args, 'num_workers', 0)
    loader = DataLoader(
        HFImageNetDataset(dataset_train, train_transform),
        batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=getattr(args, 'persistent_workers', False) and num_workers > 0,
    )
    return loader


def load_model(model_name, device):
    print(f"Loading model: {model_name}")
    if "resnet" in model_name.lower():
        model = models.resnet50(weights=None)
    elif "vit" in model_name.lower():
        config = ViTConfig.from_pretrained(model_name)
        config.num_labels = 1000
        model = ViTForImageClassification(config)
    elif "mamba" in model_name.lower():
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.num_labels = 1000
        model = AutoModelForImageClassification.from_config(config, trust_remote_code=True)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model.to(device)


def main():
    parser = argparse.ArgumentParser(description="Image Classification - Epoch Boundary Timing")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--log_dir", required=False, help="Path to log directory")
    cli_args = parser.parse_args()

    args = load_config(cli_args.config)

    print(f"\n{'='*50}")
    print(f"Epoch Boundary Benchmark: {cli_args.config}")
    print(f"{'='*50}")
    pprint(vars(args))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    num_workers = getattr(args, 'num_workers', 0)
    max_steps = getattr(args, 'num_batches', 0)
    epochs = getattr(args, 'epochs', 3)

    print(f"Device: {device}, Workers: {num_workers}, Epochs: {epochs}, Batch: {args.batch_size}")

    loader = get_dataloaders(args)
    model = load_model(args.model_name, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Initialize GradScaler for AMP
    scaler = torch.GradScaler("cuda")

    log_dir = cli_args.log_dir or getattr(args, 'output_dir', None)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        csv_path = os.path.join(log_dir, "trainer_metrics.csv")
        csv_file = open(csv_path, mode='w', newline='')
        writer = csv.writer(csv_file)
        writer.writerow(["event", "epoch", "global_step", "timestamp", "duration"])
    else:
        writer = None
        csv_path = None

    def log_row(event_name, epoch_idx, step, ts, duration=""):
        if writer is not None:
            writer.writerow([event_name, epoch_idx, step, f"{ts:.6f}", duration])
            csv_file.flush()

    global_step = 0

    for epoch in range(epochs):
        epoch_start_ts = time.time()
        log_row("epoch_start", epoch + 1, 0, epoch_start_ts)
        print(f"\n--- Epoch {epoch+1}/{epochs} start at t={epoch_start_ts:.3f} ---")

        model.train()

        for images, labels in loader:
            if max_steps > 0 and global_step >= max_steps:
                break

            batch_start = time.time()
            global_step += 1

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # Autocast context for forward pass
            with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
                outputs = model(images)
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                elif hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                loss = criterion(logits, labels)

            # Scale loss and backpropagate
            scaler.scale(loss).backward()
            
            # Unscale gradients prior to clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Step optimizer and update scaler
            scaler.step(optimizer)
            scaler.update()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            batch_end = time.time()
            log_row("batch", epoch + 1, global_step, batch_start, f"{batch_end - batch_start:.6f}")

        if max_steps > 0 and global_step >= max_steps:
            break

        epoch_end_ts = time.time()
        log_row("epoch_end", epoch + 1, global_step, epoch_end_ts)
        print(f"  Epoch {epoch+1} end at t={epoch_end_ts:.3f}")

    if csv_path:
        csv_file.close()
        print(f"\nMetrics saved to: {csv_path}")

    print(f"Epoch boundary experiment complete.")


if __name__ == "__main__":
    main()