import os
import time
import argparse
import torch
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

# DALI import
try:
    from nvidia.dali import pipeline_def, fn, types
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
except ImportError:
    pipeline_def = None

from trainers.utils import load_config

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
                yield images.shape[0], 0
            else:
                yield images, labels, 0
                
    def __len__(self):
        return len(self.loader)

def build_ffcv_pipeline(args, mode, device, num_workers):
    if Loader is None:
        raise ImportError("FFCV not installed. Run `pip install ffcv`")

    beton_path = args.data_dir
    if os.path.isdir(beton_path):
        beton_files = glob.glob(os.path.join(beton_path, "*.beton"))
        if not beton_files:
            raise ValueError(f"No .beton files found in {args.data_dir}")
        beton_path = beton_files[0]

    mean = np.array([0.485, 0.456, 0.406]) * 255
    std = np.array([0.229, 0.224, 0.225]) * 255

    label_pipeline = [IntDecoder(), ToTensor(), Squeeze()]

    if mode == 'disk':
        raise ValueError("FFCV Loader does not support pure raw byte reading for RGBImageField. Skip 'disk' mode for FFCV.")

    if mode =='decode':
        image_pipeline = [CenterCropRGBImageDecoder((224, 224), ratio=1.0), ToTensor(), ToTorchImage()]
        
    else: 
        image_pipeline = [
            RandomResizedCropRGBImageDecoder((224, 224)),
            RandomHorizontalFlip(),
            ToTensor(),
            ToTorchImage(),
        ]
        
        if mode in ['transfer', 'train']:
            image_pipeline.extend([
                ToDevice(device, non_blocking=True),
                NormalizeImage(mean, std, np.float32),
            ])
            label_pipeline.append(ToDevice(device, non_blocking=True))

    loader = Loader(
        beton_path,
        batch_size=args.batch_size,
        num_workers=max(1, num_workers),
        order=OrderOption.QUASI_RANDOM,
        pipelines={'image': image_pipeline, 'label': label_pipeline},
        drop_last=True
    )
    return loader

# DALI Builder

class DALILoaderWrapper:
    def __init__(self, loader, mode):
        self.loader = loader
        self.mode = mode
        
    def __iter__(self):
        for data in self.loader:
            batch = data[0]
            images = batch["data"]
            # DALI returns labels with shape [B, 1], convert to [B]
            labels = batch["label"].squeeze(-1).long()
            
            if self.mode in ['disk', 'decode']:
                # DALI dynamic shape tensors in these modes don't always behave like standard batched tensors.
                # Getting the length of the labels is the safest way to count the batch size.
                yield len(labels), 0
            else:
                yield images, labels, 0
                
    def __len__(self):
        return len(self.loader)

def build_dali_pipeline(args, mode, device, num_workers):
    if pipeline_def is None:
        raise ImportError("NVIDIA DALI not installed. Run `pip install nvidia-dali-cuda110` (or 120)")

    gds_enabled = getattr(args, 'gds', False)

    @pipeline_def
    def _dali_pipeline():
        # Read encoded images (Raw Bytes)
        jpegs, labels = fn.readers.file(
            file_root=args.data_dir, random_shuffle=True, name="Reader",
            # gpu_prefetch=gds_enabled,
        )

        if mode == 'disk':
            return jpegs, labels

        # With GDS data is already on GPU, so decode must be mixed
        decode_device = "mixed" if (mode in ['transfer', 'train'] or gds_enabled) else "cpu"
        
        if mode == 'decode':
            images = fn.decoders.image(jpegs, device=decode_device, output_type=types.RGB)
            return images, labels
            
        # Preprocess / Transfer / Train
        images = fn.decoders.image_random_crop(
            jpegs,
            device=decode_device,
            output_type=types.RGB,
            random_aspect_ratio=[0.75, 1.3333],
            random_area=[0.08, 1.0],
            num_attempts=100
        )
        
        images = fn.resize(
            images,
            device="gpu" if decode_device == "mixed" else "cpu",
            resize_x=224,
            resize_y=224
        )

        mirror = fn.random.coin_flip(probability=0.5)
        images = fn.crop_mirror_normalize(
            images,
            dtype=types.UINT8,
            output_layout="CHW",
            mirror=mirror,
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0]
        )
        
        if mode in ['transfer', 'train']:
            labels = labels.gpu()

        return images, labels

    device_id = device.index if device.index is not None else 0
    pipe = _dali_pipeline(
        batch_size=args.batch_size,
        num_threads=max(1, num_workers),
        device_id=device_id if torch.cuda.is_available() else None,
        prefetch_queue_depth=2
    )
    pipe.build()

    # If we are in disk or decode modes, resolutions/shapes are dynamic. 
    dynamic_shape = True if mode in ['disk', 'decode'] else False

    dali_iter = DALIGenericIterator(
        [pipe],
        output_map=["data", "label"],
        reader_name="Reader",
        auto_reset=True,
        dynamic_shape=dynamic_shape
    )
    
    return DALILoaderWrapper(dali_iter, mode)


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

def benchmark_generic(loader, num_batches, name, device=None, is_train=False, model=None, channels_last=False):
    print(f"\n{'='*50}")
    print(f"Benchmarking: {name}")
    print(f"{'='*50}")
    
    gpu_transform = None

    if is_train:
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.GradScaler("cuda")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    count = 0
    total_bytes = 0
    start = time.perf_counter()
    
    for i, batch in enumerate(loader):
        if i >= num_batches and num_batches > 0:
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

                if images.dtype == torch.uint8:
                    if gpu_transform is None:
                        from torchvision.transforms import v2
                        gpu_transform = v2.Compose([
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
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
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
    is_dali = (args.dataset_type == 'dali')

    def get_loader(mode):
        if is_wds:
            ds, collate = build_wds_pipeline(args, mode, transform)
            return make_loader(ds, args, collate, num_workers, is_wds=True)
        elif is_ffcv:
            ffcv_loader = build_ffcv_pipeline(args, mode, device, num_workers)
            return FFCVLoaderWrapper(ffcv_loader, mode)
        elif is_dali:
            return build_dali_pipeline(args, mode, device, num_workers)
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

    if args.mode in ["disk", "all"]:
        loader = get_loader('disk')
        if not (is_ffcv and args.mode == 'disk'): # Avoid FFCV's known ValueError
            results["disk"] = benchmark_generic(loader, args.num_batches, "Disk I/O Only")

    if args.mode in ["decode", "all"]:
        loader = get_loader('decode')
        results["decode"] = benchmark_generic(loader, args.num_batches, "Disk + Decode")

    if args.mode in ["preprocess", "all"]:
        loader = get_loader('preprocess')
        results["preprocess"] = benchmark_generic(loader, args.num_batches, "Disk + Decode + Preprocess")

    if args.mode in ["transfer", "all"]:
        loader = get_loader('transfer')
        results["transfer"] = benchmark_generic(loader, args.num_batches, "Disk + ... + GPU Transfer", device=device)

    if args.mode in ["train", "all"]:
        loader = get_loader('train') 
        channels_last = getattr(args, 'channels_last', False) or False
        model = load_model(getattr(args, 'model_name', 'resnet50'), device, channels_last=channels_last)
        results["train"] = benchmark_generic(loader, args.num_batches, "Full Training", device=device, is_train=True, model=model, channels_last=channels_last)

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--log_dir", type=str, help="Directory to save logs")
    cli_args = parser.parse_args()

    args = load_config(cli_args.config)

    if args.dataset_type == None:
        args.dataset_type = "imagefolder"

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
                # Catch empty results if the mode was skipped (e.g. FFCV and disk)
                if mode in all_results[num_workers]:
                    img_throughput, _ = all_results[num_workers][mode]
                    row += f"{img_throughput:>12.1f}"
                else:
                    row += f"{'N/A':>12}"
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