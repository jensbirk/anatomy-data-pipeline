import os
import time
import argparse
import csv

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from trainers.utils import load_config
from pprint import pprint

SEED = 42
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


def format_alpaca(example):
    if example.get("input", "") != "":
        return PROMPT_DICT["prompt_input"].format_map(example) + example["output"]
    else:
        return PROMPT_DICT["prompt_no_input"].format_map(example) + example["output"]


def tokenize_and_format(examples, tokenizer, max_seq_length):
    input_ids_list = []
    attention_masks_list = []
    labels_list = []
    for inst, inp, out in zip(examples['instruction'], examples['input'], examples['output']):
        full_text = format_alpaca({'instruction': inst, 'input': inp, 'output': out})
        full_text += tokenizer.eos_token
        if inp != "":
            prompt_text = PROMPT_DICT["prompt_input"].format_map({'instruction': inst, 'input': inp})
        else:
            prompt_text = PROMPT_DICT["prompt_no_input"].format_map({'instruction': inst})
        tokenized_full = tokenizer(full_text, truncation=True, max_length=max_seq_length, padding="max_length", return_tensors="pt")
        tokenized_prompt = tokenizer(prompt_text, truncation=True, max_length=max_seq_length, add_special_tokens=True)
        input_ids = tokenized_full["input_ids"][0]
        attention_mask = tokenized_full["attention_mask"][0]
        labels = input_ids.clone()
        prompt_len = len(tokenized_prompt["input_ids"])
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100
        input_ids_list.append(input_ids)
        attention_masks_list.append(attention_mask)
        labels_list.append(labels)
    return {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_masks_list),
        "labels": torch.stack(labels_list),
    }


def main():
    parser = argparse.ArgumentParser(description="LLM Finetune - Epoch Boundary Timing")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--log_dir", required=False, help="Path to log directory")
    cli_args = parser.parse_args()

    args = load_config(cli_args.config)

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

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=dtype, trust_remote_code=True, use_cache=False,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    dataset = load_dataset(path=args.dataset_name, cache_dir=args.cache_dir)
    if args.max_samples:
        dataset["train"] = dataset["train"].select(range(args.max_samples))

    tokenized_dataset = dataset.map(
        tokenize_and_format,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "max_seq_length": args.max_seq_length},
        remove_columns=dataset["train"].column_names,
        num_proc=8,
        load_from_cache_file=False,
        desc="Processing",
    )
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    train_data = tokenized_dataset["train"]

    loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=getattr(args, 'pin_memory', True),
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=getattr(args, 'persistent_workers', False) and num_workers > 0,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

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

            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

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
