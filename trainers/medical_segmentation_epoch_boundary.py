import os
import time
import argparse
import csv

import torch
from torch.utils.data import DataLoader
from monai.apps import DecathlonDataset
from monai.losses import DiceLoss
from monai.networks.nets import UNet
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


def main():
    parser = argparse.ArgumentParser(description="Medical 3D Segmentation - Epoch Boundary Timing")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--log_dir", required=False, help="Path to log directory")
    cli_args = parser.parse_args()

    args = load_config(cli_args.config)

    set_determinism(seed=SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

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

    transform = get_train_transform()
    train_ds = DecathlonDataset(
        root_dir=args.cache_dir, task=args.task,
        transform=transform, section="training",
        download=False, cache_rate=0.0,
    )
    loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
    )

    model = UNet(
        spatial_dims=3, in_channels=4, out_channels=3,
        channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2,
    ).to(device)

    loss_function = DiceLoss(
        smooth_nr=0, smooth_dr=1e-5, squared_pred=True,
        to_onehot_y=False, sigmoid=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
    scaler = torch.GradScaler("cuda")

    log_dir = cli_args.log_dir or getattr(args, 'output_dir', None)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        csv_path = os.path.join(log_dir, "trainer_metrics.csv")
    else:
        csv_path = None

    if csv_path:
        csv_file = open(csv_path, mode='w', newline='')
        writer = csv.writer(csv_file)
        writer.writerow(["event", "epoch", "global_step", "timestamp", "duration"])
    else:
        writer = None

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

        for batch_data in loader:
            if max_steps > 0 and global_step >= max_steps:
                break

            batch_start = time.time()
            global_step += 1

            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            with torch.autocast("cuda"):
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
            scaler.scale(loss).backward()
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

    print(f"\nEpoch boundary experiment complete.")


if __name__ == "__main__":
    main()
