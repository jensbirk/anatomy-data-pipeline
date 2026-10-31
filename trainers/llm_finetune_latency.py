#!/usr/bin/env python3
import os
import sys
import time
import argparse
import torch
from pprint import pprint
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from tqdm import tqdm
from system.loggers.metrics import CSVLogger
from trainers.utils import load_config

# Alpaca Prompt Template
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ),
}


class AlpacaDatasetWithTiming(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Disk I/O
        # Fetch raw text from the dataset
        disk_start = time.perf_counter()
        item = self.dataset[idx]
        disk_time = time.perf_counter() - disk_start
        
        return {
            'instruction': item['instruction'],
            'input': item['input'],
            'output': item['output'],
            'latency_disk': torch.tensor(disk_time, dtype=torch.float32),
        }

def collate_fn_with_timing(batch, tokenizer, max_seq_length):
    # Aggregate disk times
    disk_time = sum(item['latency_disk'].item() for item in batch)
    
    # Preprocessing (tokenization)
    preprocess_start = time.perf_counter()
    
    input_ids_list = []
    attention_masks_list = []
    labels_list = []
    
    for item in batch:
        inst = item['instruction']
        inp = item['input']
        out = item['output']
        
        # Format the full text
        if inp != "":
            full_text = PROMPT_DICT["prompt_input"].format_map({'instruction': inst, 'input': inp}) + out
            prompt_text = PROMPT_DICT["prompt_input"].format_map({'instruction': inst, 'input': inp})
        else:
            full_text = PROMPT_DICT["prompt_no_input"].format_map({'instruction': inst}) + out
            prompt_text = PROMPT_DICT["prompt_no_input"].format_map({'instruction': inst})
        
        full_text += tokenizer.eos_token
        
        # Tokenize
        tokenized_full = tokenizer(
            full_text,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        tokenized_prompt = tokenizer(
            prompt_text,
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=True
        )
        
        input_ids = tokenized_full["input_ids"][0]
        attention_mask = tokenized_full["attention_mask"][0]
        
        # Create labels
        labels = input_ids.clone()
        prompt_len = len(tokenized_prompt["input_ids"])
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100
        
        input_ids_list.append(input_ids)
        attention_masks_list.append(attention_mask)
        labels_list.append(labels)
    
    preprocess_time = time.perf_counter() - preprocess_start
    
    return {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_masks_list),
        "labels": torch.stack(labels_list),
        "latency_disk": torch.tensor(disk_time, dtype=torch.float32),
        "latency_preprocessing": torch.tensor(preprocess_time, dtype=torch.float32),
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

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    print(f"Loading model: {args.model_name}")
    print(f"Precision: {dtype}")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        use_cache=False
    )

    model.to("cuda")

    if args.torch_compile: 
        model = torch.compile(model)
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        padding_side="right"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    print(f"\n{'='*60}")
    print("Loading Dataset (raw text, no pre-tokenization)...")
    print(f"{'='*60}\n")
    
    dataset = load_dataset(
        path=args.dataset_name, 
        data_dir=args.data_dir, 
        cache_dir=args.cache_dir
    )

    if args.max_samples:
        print(f"Selecting first {args.max_samples} samples...")
        dataset["train"] = dataset["train"].select(range(args.max_samples))

    # Wrap in our timing dataset (NO pre-tokenization)
    train_data = AlpacaDatasetWithTiming(dataset["train"])
    
    print(f"Dataset size: {len(train_data)} samples")

    def collate_fn(batch):
        return collate_fn_with_timing(batch, tokenizer, args.max_seq_length)

    print("Creating DataLoader (tokenization happens per-batch)...")
    dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=args.pin_memory,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.1
    )

    total_steps = len(dataloader) * args.epochs
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    global_step = 0

    if cli_args.log_dir:
        log_file = os.path.join(cli_args.log_dir, "trainer_metrics.csv")
    else:
        log_file = os.path.join(args.output_dir, "trainer_metrics.csv")
    
    print(f"Logging metrics to: {log_file}")
    
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
    print(f"Warming up with {args.warmup_steps} steps...")
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for step, batch in enumerate(progress):
            if args.max_steps is not None and global_step >= args.max_steps:
                break

            # Extract timing from batch
            disk_time = batch['latency_disk'].item()
            preprocessing_time = batch['latency_preprocessing'].item()
            compression_time = 0.0  # N/A for text
            
            # Remove timing tensors from batch
            batch = {k: v for k, v in batch.items() if not k.startswith('latency_')}

            # GPU transfer
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            transfer_start = time.perf_counter()
            
            batch = {k: v.to(model.device, non_blocking=True) for k, v in batch.items()}
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            gpu_transfer_time = time.perf_counter() - transfer_start
            
            global_step += 1

            # Checkpoint logic
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                model.save_pretrained(ckpt_dir)    
                tokenizer.save_pretrained(ckpt_dir)

            # no-op check
            if hasattr(args, 'no_ops') and args.no_ops:
                continue

            # GPU compute
            compute_start = time.perf_counter()
            
            outputs = model(**batch)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            gpu_compute_time = time.perf_counter() - compute_start

            # Update Metrics
            current_loss = loss.item()
            epoch_loss += current_loss
            progress.set_postfix({"loss": f"{current_loss:.4f}"})

            # Log every step
            current_lr = scheduler.get_last_lr()[0]
            csv_logger.log("step", global_step)
            csv_logger.log("epoch", epoch + 1)
            csv_logger.log("loss", current_loss)
            csv_logger.log("lr", current_lr)
            csv_logger.log("latency_disk", disk_time)
            csv_logger.log("latency_compression", compression_time)
            csv_logger.log("latency_preprocessing", preprocessing_time)
            csv_logger.log("latency_gpu_transfer", gpu_transfer_time)
            csv_logger.log("latency_gpu_compute", gpu_compute_time)
            csv_logger.step()

        if args.max_steps is not None and global_step >= args.max_steps:
            print(f"\nReached max_steps ({args.max_steps}), stopping training.")
            break

        if not (hasattr(args, 'no_ops') and args.no_ops):
            print(f"Epoch {epoch+1} avg loss: {epoch_loss / len(dataloader):.4f}")

    if args.save_model:
        print("Saving final model...")
        os.makedirs(args.output_dir, exist_ok=True)
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Model saved to {args.output_dir}")
    
    csv_logger.close()

if __name__ == "__main__":
    main()