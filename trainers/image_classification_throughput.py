import os
import time
import argparse
import torch
import torch.distributed.checkpoint as dcp
import glob
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from transformers import ViTForImageClassification, ViTConfig, AutoModelForImageClassification, AutoConfig
from PIL import Image
import io
import csv
from pprint import pprint
import warnings
import numpy as np



import torch.multiprocessing as mp

# torchvision v2 transforms for the mmap dataset
from torchvision.transforms import v2

# WebDataset import
try:
    import webdataset as wds
except ImportError:
    wds = None

# FFCV import
try:
    from ffcv.loader import Loader, OrderOption
    from ffcv.transforms import ToTensor, ToDevice, ToTorchImage, NormalizeImage, RandomHorizontalFlip, Convert
    from ffcv.transforms.common import Squeeze
    from ffcv.fields.decoders import CenterCropRGBImageDecoder, IntDecoder, RandomResizedCropRGBImageDecoder
except ImportError:
    Loader = None

from trainers.utils import load_config
from trainers.ckpt_timing import CheckpointTimer
from system.loggers.metrics import CSVLogger

warnings.filterwarnings("ignore", message=".*Corrupt EXIF data.*")

from PIL import PngImagePlugin
PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024 

SEED = 42

# Original ImageFolder Datasets

class DiskOnlyDataset(Dataset):
    def __init__(self, root):
        self._imagefolder = ImageFolder(root=root, transform=None)
        self.samples = self._imagefolder.samples
        self.file_sizes = [os.path.getsize(path) for path, _ in self.samples]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        with open(file_path, 'rb') as f:
            raw_bytes = f.read()
        return raw_bytes, label, self.file_sizes[idx]

class DiskDecodeDataset(Dataset):
    def __init__(self, root):
        self._imagefolder = ImageFolder(root=root, transform=None)
        self.samples = self._imagefolder.samples
        self.file_sizes = [os.path.getsize(path) for path, _ in self.samples]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        with open(file_path, 'rb') as f:
            raw_bytes = f.read()
        image = Image.open(io.BytesIO(raw_bytes))
        image.load()
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image, label, self.file_sizes[idx]

class DiskDecodePreprocessDataset(Dataset):
    def __init__(self, root, transform):
        self._imagefolder = ImageFolder(root=root, transform=None)
        self.samples = self._imagefolder.samples
        self.transform = transform
        self.file_sizes = [os.path.getsize(path) for path, _ in self.samples]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        with open(file_path, 'rb') as f:
            raw_bytes = f.read()
        image = Image.open(io.BytesIO(raw_bytes))
        image.load()
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = self.transform(image)
        return image, label, self.file_sizes[idx]


class DiskDecodeCropUint8Dataset(Dataset):
    def __init__(self, root):
        self._imagefolder = ImageFolder(root=root, transform=None)
        self.samples = self._imagefolder.samples
        self.crop = transforms.RandomResizedCrop(224)
        self.flip = transforms.RandomHorizontalFlip()
        self.file_sizes = [os.path.getsize(path) for path, _ in self.samples]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        with open(file_path, 'rb') as f:
            raw_bytes = f.read()
        image = Image.open(io.BytesIO(raw_bytes))
        image.load()
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = self.crop(image)
        image = self.flip(image)
        arr = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).reshape(224, 224, 3).permute(2, 0, 1)
        return arr, label, self.file_sizes[idx]

class MMapDataset(Dataset):
    def __init__(self, data_dir):
        data_dir = os.path.dirname(data_dir) if os.path.isfile(data_dir) else data_dir
        dat_files = sorted(glob.glob(os.path.join(data_dir, "*.dat")))
        if not dat_files:
            raise ValueError(f"No .dat files found in {data_dir}")
        dat_path = dat_files[0]
        label_path = dat_path.replace('_uint8.dat', '_labels.npy')

        mm = np.memmap(dat_path, dtype=np.uint8, mode='r')
        total_pixels = mm.size
        N = total_pixels // (3 * 224 * 224)
        self.data = mm.reshape(N, 3, 224, 224)
        self.labels = np.load(label_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.data[idx].copy())
        label = int(self.labels[idx])
        return img, label, 0

import queue
import threading

class MMapStream:
    def __init__(self, data_dir, batch_size, device, gpu_transform):
        ds = MMapDataset(data_dir)
        self.data = ds.data
        self.labels = ds.labels
        self.N = len(self.labels)
        self.batch_size = batch_size
        self.device = device
        self.transform = gpu_transform

        self.queue = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._producer, daemon=True)
        self._thread.start()

    def _producer(self):
        perm = torch.randperm(self.N)
        pos = 0
        while not self._stop.is_set():
            if pos + self.batch_size > len(perm):
                perm = torch.randperm(self.N)
                pos = 0
            indices = perm[pos:pos + self.batch_size]
            pos += self.batch_size

            imgs = torch.stack([torch.from_numpy(self.data[i].copy()) for i in indices])
            labels = torch.tensor([int(self.labels[i]) for i in indices])
            self.queue.put((imgs, labels))

    def __iter__(self):
        return self

    def __next__(self):
        imgs, labels = self.queue.get()
        imgs = imgs.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)
        if self.transform is not None:
            imgs = self.transform(imgs)
        return imgs, labels, 0

    def __len__(self):
        return self.N // self.batch_size

    def __del__(self):
        self._stop.set()

# Collate Functions

def collate_disk_only(batch):
    total_bytes = sum(item[2] for item in batch)
    return len(batch), total_bytes

def collate_disk_decode(batch):
    total_bytes = sum(item[2] for item in batch)
    return len(batch), total_bytes

def collate_full(batch):
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    total_bytes = sum(item[2] for item in batch)
    return images, labels, total_bytes

# WebDataset Builder

def build_wds_pipeline(args, mode, transform):
    if wds is None:
        raise ImportError("WebDataset not installed. Run `pip install webdataset`")

    if os.path.isdir(args.data_dir):
        urls = sorted(glob.glob(os.path.join(args.data_dir, "*.tar")))
        if not urls:
            raise ValueError(f"No .tar files found in {args.data_dir}")
    else:
        urls = args.data_dir

    dataset = wds.WebDataset(urls).shuffle(1000)

    if mode == 'disk':
        def _get_bytes(sample):
            img_data = sample.get('jpg', sample.get('jpeg', sample.get('png', b'')))
            cls_data = sample.get('cls', b'0')
            return img_data, int(cls_data), len(img_data)
        dataset = dataset.map(_get_bytes)
        collate_fn = collate_disk_only

    elif mode == 'decode':
        dataset = dataset.decode("pil").to_tuple("jpg;jpeg;png", "cls")
        def _add_metadata(sample):
            img, label = sample
            return img, int(label), 0 
        dataset = dataset.map(_add_metadata)
        collate_fn = collate_disk_decode

    else:
        dataset = dataset.decode("pil").to_tuple("jpg;jpeg;png", "cls")
        def _apply_transform(sample):
            img, label = sample
            return transform(img), int(label), 0
        dataset = dataset.map(_apply_transform)
        collate_fn = collate_full

    return dataset, collate_fn

# FFCV Builder 

class FFCVLoaderWrapper:
    def __init__(self, loader, mode):
        self.loader = loader
        self.mode = mode
        
    def __iter__(self):
        for images, labels in self.loader:
            if self.mode in ['disk', 'decode']:
                yield images.shape[0], 0  # count, total_bytes (0 because FFCV obscures it)
            else:
                yield images, labels, 0   # images, labels, total_bytes
                
    def __len__(self):
        return len(self.loader)

def build_ffcv_pipeline(args, mode, device, num_workers):
    if Loader is None:
        raise ImportError("FFCV not installed. Run `pip install ffcv`")

    # Locate the .beton file
    beton_path = args.data_dir
    if os.path.isdir(beton_path):
        beton_files = glob.glob(os.path.join(beton_path, "*.beton"))
        if not beton_files:
            raise ValueError(f"No .beton files found in {args.data_dir}")
        beton_path = beton_files[0]

    # Image normalization constants (scaled for FFCV)
    mean = np.array([0.485, 0.456, 0.406]) * 255
    std = np.array([0.229, 0.224, 0.225]) * 255

    # Define Label Pipeline
    label_pipeline = [IntDecoder(), ToTensor(), Squeeze()]

    # Define Image Pipeline based on benchmark mode
    if mode == 'disk':
        raise ValueError("FFCV Loader does not support pure raw byte reading for RGBImageField. Skip 'disk' mode for FFCV.")

    if mode =='decode':
        # FFCV natively decodes. We stop at ToTensor to simulate pure CPU loading
        image_pipeline = [CenterCropRGBImageDecoder((224, 224), ratio=1.0), ToTensor(), ToTorchImage()]
        
    else: # preprocess, transfer, train
        # FFCV fuses decoding and resizing for massive speedups
        image_pipeline = [
            RandomResizedCropRGBImageDecoder((224, 224)),
            RandomHorizontalFlip(),
            # NormalizeImage(mean, std, np.float32),
            ToTensor(),
            ToTorchImage(),

        ]
        
        if mode in ['transfer', 'train']:
            # Push normalization and tensors to GPU immediately
            image_pipeline.extend([
                ToDevice(device, non_blocking=True),
                # Convert(torch.float32),
                NormalizeImage(mean, std, np.float32),
            ])
            label_pipeline.append(ToDevice(device, non_blocking=True))

    loader = Loader(
        beton_path,
        batch_size=args.batch_size,
        num_workers=max(1, num_workers), # FFCV requires at least 1 worker usually
        order=OrderOption.QUASI_RANDOM,
        pipelines={'image': image_pipeline, 'label': label_pipeline},
        drop_last=True
    )
    return loader

# Loaders & Benchmarks

def make_loader(dataset, args, collate_fn, num_workers, is_wds=False):
    g = torch.Generator()
    g.manual_seed(SEED)
    
    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': num_workers,
        'collate_fn': collate_fn,
        'pin_memory': getattr(args, 'pin_memory', True),
        'generator': g,
    }
    
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = getattr(args, 'prefetch_factor', 2)
        loader_kwargs['persistent_workers'] = getattr(args, 'persistent_workers', True)

    if is_wds:
        loader_kwargs['shuffle'] = False
    else:
        loader_kwargs['shuffle'] = True
        
    return DataLoader(dataset, **loader_kwargs)

def load_model(model_name, device, channels_last=False):
    print(f"Loading model: {model_name} (channels_last={channels_last})")
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
        model = models.resnet50(weights=None)
    model = model.to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    return model

def benchmark_generic(loader, num_batches, name, device=None, is_train=False, model=None, args=None, channels_last=False):
    print(f"\n{'='*50}")
    print(f"Benchmarking: {name}")
    print(f"{'='*50}")
    
    gpu_transform = None

    csv_logger = None
    if is_train and getattr(args, 'log_loss', False) and getattr(args, 'log_dir', None):
        csv_logger = CSVLogger(
            filepath=os.path.join(args.log_dir, "trainer_metrics.csv"),
            fieldnames=["step", "epoch", "loss"],
            append=False,
        )

    epochs = getattr(args, 'epochs', 1) if not is_train else getattr(args, 'epochs', 1)

    if is_train:
        model.train()
        criterion = nn.CrossEntropyLoss()
        lr = getattr(args, 'lr', None)
        if lr is None:
            lr = 1e-3
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        scaler = torch.GradScaler("cuda")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    count = 0
    total_bytes = 0
    global_step = 0
    timer = CheckpointTimer(args)
    start = time.perf_counter()
    
    for epoch in range(epochs):
        for i, batch in enumerate(loader):
            if num_batches > 0 and global_step >= num_batches * (epoch + 1):
                break
            if len(batch) == 2:
                b_count, b_bytes = batch
                count += b_count
                total_bytes += b_bytes
            else:
                images, labels, b_bytes = batch
                count += images.shape[0]
                total_bytes += b_bytes
                
                if device:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    if channels_last and images.dtype != torch.uint8:
                        images = images.contiguous(memory_format=torch.channels_last)

                    # GPU transforms for raw uint8 input
                    if images.dtype == torch.uint8:
                        if gpu_transform is None:
                            from torchvision.transforms import v2
                            if images.shape[-2:] == (224, 224):
                                # pre-cropped
                                gpu_transform = v2.Compose([
                                    v2.ToDtype(torch.float32, scale=True),
                                    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ])
                            else:
                                # full-res
                                gpu_transform = v2.Compose([
                                    v2.RandomResizedCrop((224, 224)),
                                    v2.RandomHorizontalFlip(),
                                    v2.ToDtype(torch.float32, scale=True),
                                    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ])
                        images = gpu_transform(images)
                
                    if is_train:
                        optimizer.zero_grad()
                        with torch.autocast("cuda"):
                            outputs = model(images)
                            if hasattr(outputs, 'logits'): logits = outputs.logits
                            elif isinstance(outputs, dict): logits = outputs['logits']
                            else: logits = outputs
                            loss = criterion(logits, labels)
                            if csv_logger is not None and global_step % 10 == 0:
                                csv_logger.log("step", global_step)
                                csv_logger.log("epoch", epoch + 1)
                                csv_logger.log("loss", loss.item())
                                csv_logger.step()
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        global_step += 1

                        save_steps = getattr(args, 'save_steps', None) or 0
                        timer.step_done(global_step)
                        if timer.should_save(global_step, save_steps):
                            os.makedirs(args.output_dir, exist_ok=True)
                            checkpoint_state = {
                                "model": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "scaler": scaler.state_dict(),
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

    if csv_logger is not None:
        csv_logger.close()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    timer.finish()
    throughput = count / elapsed
    throughput_mbs = (total_bytes / (1024 * 1024)) / elapsed if total_bytes > 0 else 0
    
    print(f"Processed {count} images ({total_bytes / (1024*1024):.1f} MB) in {elapsed:.2f}s")
    print(f"Throughput: {throughput:.1f} images/sec | {throughput_mbs:.1f} MB/s")
    return throughput, throughput_mbs

def run_benchmarks_for_workers(args, cli_args, num_workers, transform, device):
    print(f"\n{'#'*60}")
    print(f"# NUM_WORKERS = {num_workers} | DATASET = {args.dataset_type}")
    print(f"{'#'*60}")
    
    results = {}
    is_wds = (args.dataset_type == 'webdataset')
    is_ffcv = (args.dataset_type == 'ffcv')
    is_mmap = (args.dataset_type == 'mmap')
    is_img_uint8 = (args.dataset_type == 'imagefolder_uint8')

    def get_loader(mode):
        if is_wds:
            ds, collate = build_wds_pipeline(args, mode, transform)
            return make_loader(ds, args, collate, num_workers, is_wds=True)
        elif is_ffcv:
            ffcv_loader = build_ffcv_pipeline(args, mode, device, num_workers)
            return FFCVLoaderWrapper(ffcv_loader, mode)
        elif is_img_uint8:
            if mode in ['preprocess', 'transfer', 'train']:
                ds = DiskDecodeCropUint8Dataset(args.data_dir)
                return make_loader(ds, args, collate_full, num_workers, is_wds=False)
            else:
                raise ValueError(f"Mode '{mode}' not supported for imagefolder_uint8")
        elif is_mmap:
            if mode in ['preprocess', 'transfer', 'train']:
                if mode == 'train':
                    gpu_transform = v2.Compose([
                        v2.RandomResizedCrop((224, 224)),
                        v2.RandomHorizontalFlip(),
                        v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
                else:
                    gpu_transform = None
                loader = MMapStream(args.data_dir, args.batch_size, device, gpu_transform)
                return loader
            else:
                raise ValueError(f"Mode '{mode}' not supported for mmap dataset")
        else:
            if mode == 'disk':
                ds = DiskOnlyDataset(args.data_dir)
                return make_loader(ds, args, collate_disk_only, num_workers, is_wds=False)
            elif mode == 'decode':
                ds = DiskDecodeDataset(args.data_dir)
                return make_loader(ds, args, collate_disk_decode, num_workers, is_wds=False)
            elif mode in ['preprocess', 'transfer', 'train']:
                ds = DiskDecodePreprocessDataset(args.data_dir, transform)
                return make_loader(ds, args, collate_full, num_workers, is_wds=False)

    # 1. Disk
    if args.mode in ["disk", "all"]:
        loader = get_loader('disk')
        results["disk"] = benchmark_generic(loader, args.num_batches, "Disk I/O Only")

    # 2. Decode
    if args.mode in ["decode", "all"]:
        loader = get_loader('decode')
        results["decode"] = benchmark_generic(loader, args.num_batches, "Disk + Decode")

    # 3. Preprocess
    if args.mode in ["preprocess", "all"]:
        loader = get_loader('preprocess')
        results["preprocess"] = benchmark_generic(loader, args.num_batches, "Disk + Decode + Preprocess")

    # 4. Transfer
    if args.mode in ["transfer", "all"]:
        loader = get_loader('transfer')
        results["transfer"] = benchmark_generic(loader, args.num_batches, "Disk + ... + GPU Transfer", device=device)

    # 5. Train
    if args.mode in ["train", "all"]:
        loader = get_loader('train')
        channels_last = getattr(args, 'channels_last', False) or False
        model = load_model(getattr(args, 'model_name', 'resnet50'), device, channels_last=channels_last)
        results["train"] = benchmark_generic(loader, args.num_batches, "Full Training", device=device, is_train=True, model=model, args=args, channels_last=channels_last)

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--log_dir", type=str, help="Directory to save logs")
    cli_args = parser.parse_args()

    args = load_config(cli_args.config)

    if getattr(args, "dataset_type", None) is None:
        args.dataset_type = "imagefolder"
    args.log_dir = cli_args.log_dir

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print(f"\n{'='*50}")
    print(f"Throughput Benchmark: {cli_args.config}")
    print(f"Dataset Type: {args.dataset_type}")
    print(f"{'='*50}")

    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    num_workers_config = getattr(args, 'num_workers', 4)
    worker_counts = num_workers_config if isinstance(num_workers_config, list) else [num_workers_config]

    all_results = {}
    for num_workers in worker_counts:
        all_results[num_workers] = run_benchmarks_for_workers(args, cli_args, num_workers, transform, device)

    # Print summary
    if len(worker_counts) > 0:
        mode_names = list(all_results[worker_counts[0]].keys())
        print(f"\n{'='*70}")
        print(f"Summary (img/s) - {args.dataset_type}")
        print(f"{'='*70}")
        header = f"{'workers':<10}" + "".join(f"{mode:>12}" for mode in mode_names)
        print(header)
        print("-" * len(header))
        for num_workers in worker_counts:
            row = f"{num_workers:<10}"
            for mode in mode_names:
                img_throughput, _ = all_results[num_workers][mode]
                row += f"{img_throughput:>12.1f}"
            print(row)

    if cli_args.log_dir:
        os.makedirs(cli_args.log_dir, exist_ok=True)
        csv_filename = f"benchmark_{args.dataset_type}_{int(time.time())}.csv"
        csv_path = os.path.join(cli_args.log_dir, csv_filename)
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['num_workers', 'mode', 'throughput_img_s', 'throughput_mb_s'])
            for num_workers, mode_results in all_results.items():
                for mode_name, (img_s, mb_s) in mode_results.items():
                    writer.writerow([num_workers, mode_name, f"{img_s:.4f}", f"{mb_s:.4f}"])
        print(f"\nSaved to {csv_path}")

if __name__ == "__main__":
    main()