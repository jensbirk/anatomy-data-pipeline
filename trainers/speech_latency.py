#!/usr/bin/env python3
import logging
import os
import sys
import time
import argparse
import io
from pprint import pprint
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
from torch.utils.data import DataLoader, Dataset
import evaluate
from datasets import load_dataset, Audio, DatasetDict
import soundfile as sf

from transformers import (
    AutoConfig,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    set_seed,
)

from trainers.utils import load_config
from system.loggers.metrics import CSVLogger
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class SpeechDatasetWithTiming(Dataset):
    def __init__(self, hf_dataset, processor, audio_column="audio", text_column="text"):
        self.dataset = hf_dataset
        self.processor = processor
        self.audio_column = audio_column
        self.text_column = text_column
        
        # Store file paths for direct disk access
        # HuggingFace audio datasets typically have 'path' in the audio dict
        self._has_paths = False
        
        # Check if we can access file paths
        try:
            sample = hf_dataset[0]
            audio_info = sample[audio_column]
            if isinstance(audio_info, dict) and 'path' in audio_info and audio_info['path']:
                self._has_paths = True
                print(f"  Found audio file paths for direct disk I/O measurement")
        except Exception as e:
            print(f"  Could not access audio paths: {e}")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item[self.text_column]
        audio_info = item[self.audio_column]
        
        disk_time = 0.0
        decomp_time = 0.0
        
        # Check if we have file path for pure disk I/O measurement
        audio_path = None
        if isinstance(audio_info, dict) and 'path' in audio_info:
            audio_path = audio_info.get('path')
        
        if audio_path and os.path.exists(str(audio_path)):
            # Pure disk I/O
            # Read raw audio bytes from disk (no decoding)
            disk_start = time.perf_counter()
            with open(audio_path, 'rb') as f:
                raw_bytes = f.read()
            disk_time = time.perf_counter() - disk_start
            
            # Decompression (audio decoding)
            # Decode audio bytes to waveform (FLAC/MP3/WAV -> numpy array)
            decomp_start = time.perf_counter()
            audio_array, sampling_rate = sf.read(io.BytesIO(raw_bytes))
            decomp_time = time.perf_counter() - decomp_start
            
        else:
            # Fallback: HuggingFace already decoded the audio
            # disk_time includes decode in this case
            disk_start = time.perf_counter()
            # Re-fetch to measure
            _ = self.dataset[idx][self.audio_column]
            disk_time = time.perf_counter() - disk_start
            
            decomp_start = time.perf_counter()
            audio_array = audio_info['array']
            sampling_rate = audio_info['sampling_rate']
            decomp_time = time.perf_counter() - decomp_start
        
        return {
            'audio_array': audio_array,
            'sampling_rate': sampling_rate,
            'text': text,
            'latency_disk': torch.tensor(disk_time, dtype=torch.float32),
            'latency_decompression': torch.tensor(decomp_time, dtype=torch.float32),
        }


def collate_fn_with_timing(batch, processor, decoder_start_token_id):
    # Aggregate disk and decompression times
    disk_time = sum(item['latency_disk'].item() for item in batch)
    decomp_time = sum(item['latency_decompression'].item() for item in batch)
    
    # Preprocessing (feature extraction + tokenization)
    preprocess_start = time.perf_counter()
    
    input_features_list = []
    labels_list = []
    
    for item in batch:
        audio_array = item['audio_array']
        sampling_rate = item['sampling_rate']
        text = item['text']
        
        # Feature extraction (mel spectrogram)
        input_features = processor.feature_extractor(
            audio_array,
            sampling_rate=sampling_rate
        ).input_features[0]
        
        # Tokenization
        labels = processor.tokenizer(text).input_ids
        
        input_features_list.append(input_features)
        labels_list.append(labels)
    
    # Pad input features
    model_input_name = processor.model_input_names[0]
    input_features_padded = processor.feature_extractor.pad(
        [{model_input_name: feat} for feat in input_features_list],
        return_tensors="pt"
    )
    
    # Pad labels
    labels_padded = processor.tokenizer.pad(
        [{"input_ids": lab} for lab in labels_list],
        return_tensors="pt"
    )
    
    # Replace padding with -100 to ignore in loss
    labels = labels_padded["input_ids"].masked_fill(
        labels_padded.attention_mask.ne(1), -100
    )
    
    # If bos token is appended, cut it
    if (labels[:, 0] == decoder_start_token_id).all().cpu().item():
        labels = labels[:, 1:]
    
    preprocess_time = time.perf_counter() - preprocess_start
    
    return {
        "input_features": input_features_padded["input_features"],
        "labels": labels,
        "latency_disk": torch.tensor(disk_time, dtype=torch.float32),
        "latency_decompression": torch.tensor(decomp_time, dtype=torch.float32),
        "latency_preprocessing": torch.tensor(preprocess_time, dtype=torch.float32),
    }


def main():
    parser = argparse.ArgumentParser(description="Speech Recognition Training with Latency Tracking")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--log_dir", required=False, help="Path to log directory")
    cli_args = parser.parse_args()
    args = load_config(cli_args.config)
    
    if hasattr(args, 'seed') and args.seed:
        set_seed(args.seed)

    print(f"\n{'='*40}")
    print(f"Training Configuration: {cli_args.config}")
    print(f"{'='*40}")
    pprint(vars(args))
    print(f"{'='*40}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load datasets
    print("Loading datasets...")
    raw_datasets = DatasetDict()
    raw_datasets["train"] = load_dataset(
        args.dataset_name, 
        args.dataset_config if hasattr(args, 'dataset_config') else "clean",
        split=args.train_split if hasattr(args, 'train_split') else "train.100",
        cache_dir=args.cache_dir,
    )
    
    raw_datasets["eval"] = load_dataset(
        args.dataset_name,
        args.dataset_config if hasattr(args, 'dataset_config') else "clean",
        split=args.eval_split if hasattr(args, 'eval_split') else "test",
        cache_dir=args.cache_dir,
    )

    # Load Processor
    print(f"Loading processor for: {args.model_name}")
    processor = AutoProcessor.from_pretrained(
        args.model_name,
        language=args.language if hasattr(args, 'language') else "english",
        task="transcribe"
    )

    audio_column = args.audio_column if hasattr(args, 'audio_column') else "audio"
    text_column = args.text_column if hasattr(args, 'text_column') else "text"
    
    # Cast audio column to correct sampling rate
    raw_datasets = raw_datasets.cast_column(audio_column, Audio(sampling_rate=16000))

    # Load Model
    print("Loading model...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_name,
        low_cpu_mem_usage=True,
        use_cache=False
    )
    model.to(device)

    # Configure model
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    
    if hasattr(args, 'freeze_encoder') and args.freeze_encoder:
        model.freeze_encoder()
        model.model.encoder.gradient_checkpointing = False

    # Create datasets with timing
    print("Creating datasets with timing instrumentation...")
    train_dataset = SpeechDatasetWithTiming(
        raw_datasets["train"], processor, audio_column, text_column
    )
    eval_dataset = SpeechDatasetWithTiming(
        raw_datasets["eval"], processor, audio_column, text_column
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")

    # Create collate function with processor bound
    decoder_start_token_id = model.config.decoder_start_token_id
    
    def collate_fn(batch):
        return collate_fn_with_timing(batch, processor, decoder_start_token_id)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers if hasattr(args, 'workers') else 0,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers if hasattr(args, 'workers') else 0,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr if hasattr(args, 'lr') else 1e-5,
        weight_decay=0.01
    )
    
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    # Mixed precision
    scaler = torch.GradScaler("cuda") if torch.cuda.is_available() else None

    # Setup logging
    if cli_args.log_dir:
        log_file = os.path.join(cli_args.log_dir, "trainer_metrics.csv")
    else:
        log_file = os.path.join(args.output_dir, "trainer_metrics.csv")
    
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
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

    print(f"\n=== Training Started ===")
    print(f"Total Steps: {total_steps}")
    
    global_step = 0
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for step, batch in enumerate(progress):
            if hasattr(args, 'max_steps') and args.max_steps is not None and global_step >= args.max_steps:
                break

            # Extract timing from batch
            disk_time = batch['latency_disk'].item()
            decomp_time = batch['latency_decompression'].item()
            preprocessing_time = batch['latency_preprocessing'].item()
            
            # Remove timing tensors from batch
            input_features = batch['input_features']
            labels = batch['labels']

            # GPU transfer
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            transfer_start = time.perf_counter()
            
            input_features = input_features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            gpu_transfer_time = time.perf_counter() - transfer_start
            
            global_step += 1

            # GPU compute
            compute_start = time.perf_counter()
            
            optimizer.zero_grad()
            
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
            
            scheduler.step()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            gpu_compute_time = time.perf_counter() - compute_start

            # Update metrics
            current_loss = loss.item()
            epoch_loss += current_loss
            progress.set_postfix({
                "loss": f"{current_loss:.4f}",
                "compute": f"{gpu_compute_time*1000:.1f}ms"
            })

            # Log every step
            current_lr = scheduler.get_last_lr()[0]
            csv_logger.log("step", global_step)
            csv_logger.log("epoch", epoch + 1)
            csv_logger.log("loss", current_loss)
            csv_logger.log("lr", current_lr)
            csv_logger.log("latency_disk", disk_time)
            csv_logger.log("latency_compression", decomp_time)
            csv_logger.log("latency_preprocessing", preprocessing_time)
            csv_logger.log("latency_gpu_transfer", gpu_transfer_time)
            csv_logger.log("latency_gpu_compute", gpu_compute_time)
            csv_logger.step()

        if hasattr(args, 'max_steps') and args.max_steps is not None and global_step >= args.max_steps:
            print(f"\nReached max_steps ({args.max_steps}), stopping training.")
            break

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

    # Save model if requested
    if hasattr(args, 'save_model') and args.save_model:
        print("Saving final model...")
        os.makedirs(args.output_dir, exist_ok=True)
        model.save_pretrained(args.output_dir)
        processor.save_pretrained(args.output_dir)
        print(f"Model saved to {args.output_dir}")
    
    csv_logger.close()
    print("Training complete.")


if __name__ == "__main__":
    main()