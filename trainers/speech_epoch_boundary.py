import os
import time
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, set_seed
from dataclasses import dataclass
from typing import Any

from trainers.utils import load_config
from pprint import pprint

SEED = 42


class UniversalAudioSource:
    def __init__(self, source_type: str, target_sr: int = 16000):
        self.source_type = source_type.lower()
        self.target_sr = target_sr
        self._resamplers: Dict[Tuple[int, int], torchaudio.transforms.Resample] = {}

    def _get_resampler(self, src_sr: int):
        key = (src_sr, self.target_sr)
        if key not in self._resamplers:
            self._resamplers[key] = torchaudio.transforms.Resample(src_sr, self.target_sr)
        return self._resamplers[key]

    def load_audio_file(self, path: str):
        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.target_sr:
            waveform = self._get_resampler(sr)(waveform)
            sr = self.target_sr
        return waveform.squeeze(0).numpy(), sr

    def load_audio(self, sample: dict):
        return self.load_audio_file(sample["audio"])


def build_librispeech_samples(root_dir: str, ext: str) -> List[dict]:
    root = Path(root_dir)
    samples: List[dict] = []
    trans_files = sorted(root.rglob("*.trans.txt"))
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
    return samples


class FlacSpeechDataset(Dataset):
    def __init__(self, samples: List[dict], source: UniversalAudioSource, processor: Any):
        self.samples = samples
        self.source = source
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio_array, sr = self.source.load_audio(sample)
        input_features = self.processor.feature_extractor(
            audio_array, sampling_rate=sr
        ).input_features[0]
        labels = self.processor.tokenizer(sample["text"]).input_ids
        return {"input_features": input_features, "labels": labels}


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        model_input_name = self.processor.model_input_names[0]
        input_features = [{model_input_name: f[model_input_name]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def main():
    parser = argparse.ArgumentParser(description="Speech - Epoch Boundary Timing")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--log_dir", required=False, help="Path to log directory")
    cli_args = parser.parse_args()

    args = load_config(cli_args.config)

    set_seed(SEED)

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
    freeze_encoder = getattr(args, 'freeze_encoder', False)
    dataset_root = getattr(args, 'dataset_root', None)
    dataset_type = getattr(args, 'dataset_type', 'flac')
    sampling_rate = getattr(args, 'sampling_rate', 16000)

    print(f"Device: {device}, Workers: {num_workers}, Epochs: {epochs}, Batch: {args.batch_size}")
    print(f"Dataset: {dataset_root} ({dataset_type})")

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        args.model_name, language=args.language, task="transcribe"
    )

    print("Building dataset from local files...")
    samples = build_librispeech_samples(dataset_root, dataset_type)
    print(f"  Found {len(samples)} samples")

    source = UniversalAudioSource(source_type=dataset_type, target_sr=sampling_rate)
    dataset = FlacSpeechDataset(samples, source, processor)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=processor.tokenizer.bos_token_id or 50257,
    )

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=data_collator,
        pin_memory=True, prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=getattr(args, 'persistent_workers', False) and num_workers > 0,
    )

    print("Loading model...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_name, low_cpu_mem_usage=True, use_cache=False,
    )
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    if freeze_encoder:
        model.freeze_encoder()
        model.model.encoder.gradient_checkpointing = False
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=getattr(args, 'lr', 1e-5))
    scaler = torch.GradScaler("cuda") if torch.cuda.is_available() else None

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

        for batch in loader:
            if max_steps > 0 and global_step >= max_steps:
                break

            batch_start = time.time()
            global_step += 1

            input_features = batch["input_features"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

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
