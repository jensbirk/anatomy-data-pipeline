import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from transformers import ViTForImageClassification, ViTConfig, AutoModelForImageClassification, AutoConfig
from pprint import pprint
from trainers.utils import load_config
from system.loggers.metrics import CSVLogger
from tqdm import tqdm
from PIL import Image
import io

import warnings
warnings.filterwarnings("ignore", message=".*Corrupt EXIF data.*")


class ImageFolderWithTiming(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        
        # Use ImageFolder to get file paths and labels
        self._imagefolder = ImageFolder(root=root, transform=None)
        self.samples = self._imagefolder.samples  # List of (path, label) tuples
        self.classes = self._imagefolder.classes
        self.class_to_idx = self._imagefolder.class_to_idx
        
        print(f"Loaded {len(self.samples)} images from {root}")
        print(f"Found {len(self.classes)} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        
        # Pure disk I/O
        # Read raw bytes only - no parsing, no decoding
        disk_start = time.perf_counter()
        with open(file_path, 'rb') as f:
            raw_bytes = f.read()
        disk_time = time.perf_counter() - disk_start
        
        # Decompression
        # Decode the raw bytes to PIL image
        decomp_start = time.perf_counter()
        image = Image.open(io.BytesIO(raw_bytes))
        image.load()  # Force full decode of pixel data
        if image.mode != 'RGB':
            image = image.convert('RGB')
        decomp_time = time.perf_counter() - decomp_start
        
        # Preprocessing
        preprocess_time = 0.0
        if self.transform:
            preprocess_start = time.perf_counter()
            image = self.transform(image)
            preprocess_time = time.perf_counter() - preprocess_start
        
        return {
            'image': image,
            'label': label,
            'latency_disk': torch.tensor(disk_time, dtype=torch.float32),
            'latency_decompression': torch.tensor(decomp_time, dtype=torch.float32),
            'latency_preprocessing': torch.tensor(preprocess_time, dtype=torch.float32),
        }


def collate_fn_with_timing(batch):
    images = torch.stack([item['image'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch])
    
    # Sum latencies across batch
    latency_disk = torch.stack([item['latency_disk'] for item in batch]).sum()
    latency_decompression = torch.stack([item['latency_decompression'] for item in batch]).sum()
    latency_preprocessing = torch.stack([item['latency_preprocessing'] for item in batch]).sum()
    
    return {
        'image': images,
        'label': labels,
        'latency_disk': latency_disk,
        'latency_decompression': latency_decompression,
        'latency_preprocessing': latency_preprocessing,
    }


def main():
    parser = argparse.ArgumentParser(description="YAML training configuration")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--log_dir", required=False, help="Path to log directory")
    cli_args = parser.parse_args()

    args = load_config(cli_args.config)

    print(f"\n{'='*40}")
    print(f"Training Configuration: {cli_args.config}")
    print(f"{'='*40}")
    pprint(vars(args))
    print(f"{'='*40}\n")

    # Seeding
    seed_val = 0
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)
    
    g = torch.Generator()
    g.manual_seed(seed_val)

    def get_dataloaders():
        print(f"Loading ImageNet from {args.data_dir}...")
        
        if not os.path.exists(args.data_dir):
            raise ValueError(f"Data directory not found: {args.data_dir}")

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        # Load full dataset
        full_dataset = ImageFolderWithTiming(args.data_dir, train_transform)
        
        # Split into train/val (default 90/10 split)
        val_split = getattr(args, 'val_split', 0.1)
        total_size = len(full_dataset)
        val_size = int(total_size * val_split)
        train_size = total_size - val_size
        
        # Use generator for reproducible split
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        print(f"Split dataset: {train_size} train, {val_size} val")

        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=0,  # Must be 0 to capture timing in main process
            pin_memory=True,
            generator=g,
            collate_fn=collate_fn_with_timing,
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn_with_timing,
        )
        
        return train_loader, val_loader

    def load_model(device):
        print(f"Loading model: {args.model_name}")
        
        if "resnet" in args.model_name.lower():
            model = models.resnet50(weights=None) 
        
        elif "vit" in args.model_name.lower():
            print("Initializing ViT from scratch (Random Weights)...")
            config = ViTConfig.from_pretrained(args.model_name)
            config.num_labels = 1000
            model = ViTForImageClassification(config) 
            
        elif "mamba" in args.model_name.lower():
            print("Initializing MambaVision from scratch (Random Weights)...")
            config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
            config.num_labels = 1000
            model = AutoModelForImageClassification.from_config(config, trust_remote_code=True)
            
        model = model.to(device)
        return model

    if cli_args.log_dir:
        log_file = os.path.join(cli_args.log_dir, "trainer_metrics.csv")
    else:
        log_file = os.path.join(args.output_dir, "trainer_metrics.csv")

    csv_logger = CSVLogger(
        filepath=log_file,
        fieldnames=[
            "step", "epoch", "loss", "lr",
            "latency_disk",
            "latency_compression",  # Named to match medical segmentation script
            "latency_preprocessing",
            "latency_gpu_transfer",
            "latency_gpu_compute"
        ],
        append=False
    )

    def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch_index, global_step):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(loader, desc=f"Training Epoch {epoch_index}")

        for i, batch_data in enumerate(progress_bar):
            if hasattr(args, 'max_steps') and args.max_steps is not None and global_step >= args.max_steps:
                break
            
            # Extract timing from dataloader
            disk_read_time = batch_data['latency_disk'].item()
            decompression_time = batch_data['latency_decompression'].item()
            preprocessing_time = batch_data['latency_preprocessing'].item()

            # GPU transfer time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            transfer_start = time.perf_counter()
            
            images = batch_data['image'].to(device, non_blocking=True)
            labels = batch_data['label'].to(device, non_blocking=True)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            gpu_transfer_time = time.perf_counter() - transfer_start

            # GPU compute time
            compute_start = time.perf_counter()
            
            optimizer.zero_grad()
            
            with torch.autocast("cuda"):
                outputs = model(images)
                
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                elif hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs    
                    
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            
            # Clip gradients to avoid NaN in ViT
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            gpu_compute_time = time.perf_counter() - compute_start
            
            global_step += 1
            running_loss += loss.item()
            
            # Log metrics
            csv_logger.log("step", global_step)
            csv_logger.log("epoch", epoch_index)
            csv_logger.log("loss", loss.item())
            csv_logger.log("lr", optimizer.param_groups[0]['lr'])
            csv_logger.log("latency_disk", disk_read_time)
            csv_logger.log("latency_compression", decompression_time)
            csv_logger.log("latency_preprocessing", preprocessing_time)
            csv_logger.log("latency_gpu_transfer", gpu_transfer_time)
            csv_logger.log("latency_gpu_compute", gpu_compute_time)
            csv_logger.step()

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                compute=f"{gpu_compute_time*1000:.1f}ms",
                xfer=f"{gpu_transfer_time*1000:.1f}ms"
            )
        
        return global_step 

    def validate(model, loader, device):
        model.eval()
        correct = 0
        total = 0
        
        progress_bar = tqdm(loader, desc="Validating")
        
        with torch.no_grad():
            for batch_data in progress_bar:
                images = batch_data['image'].to(device)
                labels = batch_data['label'].to(device)
                outputs = model(images)
                
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                elif hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                    
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f"Validation Accuracy: {acc:.2f}%")
        return acc

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    train_loader, val_loader = get_dataloaders()
    model = load_model(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.GradScaler("cuda")
    
    global_step = 0

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        global_step = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch+1, global_step)
        # validate(model, val_loader, device)
        lr_scheduler.step()
        
        if hasattr(args, 'max_steps') and args.max_steps is not None and global_step >= args.max_steps:
            print(f"\nReached max_steps ({args.max_steps}), stopping training.")
            break
    
    csv_logger.close()
    print("Training complete.")


if __name__ == "__main__":
    main()