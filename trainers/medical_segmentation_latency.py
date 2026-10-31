import os
import shutil
import tempfile
import time
import argparse
import matplotlib.pyplot as plt
import torch
import torch.profiler
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader, decollate_batch
from monai.handlers.utils import from_engine
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet, UNet, SwinUNETR
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
)
from monai.utils import set_determinism
from tqdm import tqdm
from trainers.utils import load_config
from system.loggers.metrics import CSVLogger

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

class SeparatedLoadImaged(MapTransform):
    def __init__(self, keys, reader=None):
        super().__init__(keys)
        self.native_loader = LoadImaged(keys=keys, reader=reader)

    def __call__(self, data):
        d = dict(data)
        disk_time_accum = 0.0
        decomp_time_accum = 0.0

        for key in self.keys:
            if key in d:
                path = d[key]
                if isinstance(path, (str, os.PathLike)):
                    t0 = time.perf_counter()
                    with open(path, 'rb') as f:
                        _ = f.read()
                    disk_time_accum += (time.perf_counter() - t0)

        t1 = time.perf_counter()
        d = self.native_loader(d)
        decomp_time_accum = (time.perf_counter() - t1)

        d["latency_disk"] = torch.tensor(disk_time_accum, dtype=torch.float32)
        d["latency_decompression"] = torch.tensor(decomp_time_accum, dtype=torch.float32)
        return d

class TimedCompose(Compose):
    def __call__(self, input_):
        preprocess_time_accum = 0.0
        for transform in self.transforms:
            if isinstance(transform, SeparatedLoadImaged):
                input_ = transform(input_)
                continue
            start = time.perf_counter()
            input_ = transform(input_)
            end = time.perf_counter()
            preprocess_time_accum += (end - start)

        if isinstance(input_, dict):
            input_["latency_preprocessing"] = torch.tensor(preprocess_time_accum, dtype=torch.float32)
        return input_

def main():
    parser = argparse.ArgumentParser(description="Medical 3D Segmentation - Profiling & Timing")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--log_dir", required=False, help="Path to log directory")
    cli_args = parser.parse_args()
    args = load_config(cli_args.config)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Seeding step 1: global determinism (transforms)
    # This locks down RandSpatialCropd, RandFlipd, etc.
    seed_val = 0
    set_determinism(seed=seed_val)

    # Seeding step 2: isolated generator (shuffling)
    # This locks down the order of images in the batches.
    g = torch.Generator()
    g.manual_seed(seed_val)

    roi_size = [224, 224, 128]

    train_transform = TimedCompose([
        SeparatedLoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        RandSpatialCropd(keys=["image", "label"], roi_size=roi_size, random_size=False),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
    ])
    
    train_ds = DecathlonDataset(
        root_dir=args.cache_dir,
        task=args.task,
        transform=train_transform,
        section="training",
        download=True,
        cache_rate=0.0,
    )
    
    print(f"Batch Size: {args.batch_size}")
    
    # DataLoader
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0,      # main process only
        generator=g,        # Forces the shuffle order to be the same every run
        pin_memory=True,
        prefetch_factor=None # Must be None if num_workers=0
    )

    model_name = args.model
    if model_name == "SegResNet":
        model = SegResNet(blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1], init_filters=16, in_channels=4, out_channels=3, dropout_prob=0.2).to(device)
    elif model_name == "UNet":
        model = UNet(spatial_dims=3, in_channels=4, out_channels=3, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2).to(device)
    elif model_name == "SwinUNETR":
        model = SwinUNETR(in_channels=4, out_channels=3, feature_size=48, use_checkpoint=True).to(device)
    else:
        raise ValueError(f"Model {model_name} not recognized.")

    loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.GradScaler("cuda")
    torch.backends.cudnn.benchmark = True

    log_file = os.path.join(cli_args.log_dir if cli_args.log_dir else args.output_dir, "trainer_metrics.csv")
    
    csv_logger = CSVLogger(
        filepath=log_file,
        fieldnames=[
            "step", "epoch", "loss", "lr", 
            "latency_disk", 
            "latency_compression", 
            "latency_preprocessing", 
            "latency_gpu_transfer", 
            "latency_gpu_compute"
        ],
        append=False
    )

    print(f"\n=== Training ===")
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        step_in_epoch = 0
        iterator = iter(train_loader)
        
        while global_step < args.max_steps and step_in_epoch < len(train_loader):
            batch_data = next(iterator)

            disk_read_time = batch_data["latency_disk"].sum().item()
            decompression_time = batch_data["latency_decompression"].sum().item()
            preprocessing_time = batch_data["latency_preprocessing"].sum().item()

            if torch.cuda.is_available(): torch.cuda.synchronize()
            transfer_start = time.perf_counter()
            
            inputs, labels = (
                batch_data["image"].to(device, non_blocking=True),
                batch_data["label"].to(device, non_blocking=True),
            )
            
            if torch.cuda.is_available(): torch.cuda.synchronize()
            gpu_transfer_time = time.perf_counter() - transfer_start

            compute_start = time.perf_counter()
            
            optimizer.zero_grad()
            with torch.autocast("cuda"):
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if torch.cuda.is_available(): torch.cuda.synchronize()
            gpu_compute_time = time.perf_counter() - compute_start

            global_step += 1
            step_in_epoch += 1
            
            csv_logger.log("step", global_step)
            csv_logger.log("epoch", epoch + 1)
            csv_logger.log("loss", loss.item())
            csv_logger.log("lr", optimizer.param_groups[0]['lr'])
            
            csv_logger.log("latency_disk", disk_read_time)
            csv_logger.log("latency_compression", decompression_time)
            csv_logger.log("latency_preprocessing", preprocessing_time)
            csv_logger.log("latency_gpu_transfer", gpu_transfer_time)
            csv_logger.log("latency_gpu_compute", gpu_compute_time)
            
            csv_logger.step()

            print(f"Epoch {epoch+1} [{step_in_epoch}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f} | "
                  f"Compute: {gpu_compute_time*1000:.1f}ms | "
                  f"Xfer: {gpu_transfer_time*1000:.1f}ms")

        lr_scheduler.step()

    csv_logger.close()
    print("Training complete.")

if __name__ == "__main__":
    main()