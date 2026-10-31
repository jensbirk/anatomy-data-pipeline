#!/usr/bin/env python3

import os, time, gc, csv, sys
import numpy as np
import torch
import torchvision.models as torchvision_models
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoModelForImageClassification,
    ViTConfig, ViTForImageClassification,
    WhisperConfig, WhisperForConditionalGeneration,
)
from monai.networks.nets import SegResNet, UNet, SwinUNETR
# The repo root's utils/ package, not trainers/utils.py next to this script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import clear_page_cache

NUM_RUNS = int(os.environ.get("NUM_RUNS", 10))
OUTPUT = os.environ.get("CKPT_BENCH_OUTPUT", "/raid/benchmark/checkpoint_format_all")
RESULTS_CSV = os.path.join(OUTPUT, "checkpoint_format_results.csv")
RAW_CSV = os.path.join(OUTPUT, "checkpoint_format_raw.csv")
os.makedirs(OUTPUT, exist_ok=True)

# Models whose weights (and therefore Adam moments) are bf16; all others are fp32.
BF16_MODELS = {"Llama-3.2-1B", "Llama-3.2-3B", "Qwen3-0.6B", "Qwen3-1.7B"}

# Model factories

def _p(name, model, domain):
    sd = model.state_dict()
    params = sum(p.numel() for p in model.parameters())
    print(f"  {name}: {params:,} params", flush=True)
    return name, sd, domain, params


def make_resnet50():
    m = torchvision_models.resnet50(weights=None)
    return _p("ResNet-50", m, "Image Classification")


def make_vit_base():
    cfg = ViTConfig.from_pretrained("google/vit-base-patch16-224")
    cfg.num_labels = 1000
    m = ViTForImageClassification(cfg)
    return _p("ViT-base", m, "Image Classification")


def make_mambavision_t():
    cfg = AutoConfig.from_pretrained("nvidia/MambaVision-T-1K", trust_remote_code=True)
    cfg.num_labels = 1000
    m = AutoModelForImageClassification.from_config(cfg, trust_remote_code=True)
    return _p("MambaVision-T", m, "Image Classification")


def make_whisper_small():
    cfg = WhisperConfig.from_pretrained("openai/whisper-small")
    m = WhisperForConditionalGeneration(cfg)
    return _p("Whisper-small", m, "Speech")


def make_llama(model_id, name):
    cfg = AutoConfig.from_pretrained(model_id)
    m = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16)
    return _p(name, m, "LLM")


def make_qwen(model_id, name):
    cfg = AutoConfig.from_pretrained(model_id)
    m = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16)
    return _p(name, m, "LLM")


def make_segresnet():
    m = SegResNet(blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1],
                  init_filters=16, in_channels=4, out_channels=3, dropout_prob=0.2)
    return _p("SegResNet", m, "Medical Segmentation")


def make_unet3d():
    m = UNet(spatial_dims=3, in_channels=4, out_channels=3,
             channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2)
    return _p("UNet 3D", m, "Medical Segmentation")


def make_swinunetr():
    m = SwinUNETR(in_channels=4, out_channels=3, feature_size=48, use_checkpoint=True)
    return _p("SwinUNETR", m, "Medical Segmentation")


FACTORIES = [
    make_resnet50,
    make_vit_base,
    make_mambavision_t,
    make_whisper_small,
    lambda: make_llama("meta-llama/Llama-3.2-1B", "Llama-3.2-1B"),
    lambda: make_llama("meta-llama/Llama-3.2-3B", "Llama-3.2-3B"),
    lambda: make_qwen("Qwen/Qwen3-0.6B", "Qwen3-0.6B"),
    lambda: make_qwen("Qwen/Qwen3-1.7B", "Qwen3-1.7B"),
    make_segresnet,
    make_unet3d,
    make_swinunetr,
]

import importlib.util

if importlib.util.find_spec("mamba_ssm") is None:
    print(
        "WARNING: mamba_ssm not installed; skipping MambaVision-T.",
        flush=True,
    )
    FACTORIES = [f for f in FACTORIES if f is not make_mambavision_t]


def unique_tensors(state_dict):
    seen, out = set(), {}
    for k, v in state_dict.items():
        key = (v.untyped_storage().data_ptr(), v.storage_offset(), tuple(v.shape))
        if key in seen:
            continue
        seen.add(key)
        out[k] = v
    return out


def build_checkpoint(state_dict):
    sd = {k: v.clone() for k, v in unique_tensors(state_dict).items()}
    for name in list(sd.keys()):
        tensor = sd[name]
        if not torch.is_floating_point(tensor):
            # Integer buffers (e.g. BatchNorm num_batches_tracked) never get
            # Adam moments; randn_like is not implemented for them.
            continue
        sd[f"optimizer.exp_avg.{name}"] = torch.randn_like(tensor)
        sd[f"optimizer.exp_avg_sq.{name}"] = torch.randn_like(tensor)
    return sd


def benchmark_model(name, domain, params, state_dict):
    safe_key = name.replace(" ", "_")
    safe_path = os.path.join(OUTPUT, f"{safe_key}.safetensors")
    pt_path = os.path.join(OUTPUT, f"{safe_key}.pt")
    device = torch.device("cuda")

    ckpt = build_checkpoint(state_dict)
    n_tensors = len(ckpt)
    total_gb = sum(v.numel() * v.element_size() for v in ckpt.values()) / 1024**3
    model_gb = sum(
        v.numel() * v.element_size()
        for k, v in ckpt.items()
        if not k.startswith("optimizer.")
    ) / 1024**3
    optim_gb = total_gb - model_gb

    print(f"\n{'='*70}", flush=True)
    print(f"{domain}: {name}  |  {n_tensors} tensors  |  {total_gb:.2f} GB", flush=True)
    print(f"{'='*70}", flush=True)

    # Move to GPU
    print("  Moving to GPU ...", flush=True)
    gpu_ckpt = {k: v.to(device) for k, v in ckpt.items()}
    del ckpt; gc.collect()
    torch.cuda.synchronize()

    # Initial write (save both formats from GPU)
    print("  Writing safetensors ...", flush=True)
    t0 = time.perf_counter()
    cpu_ckpt = {k: v.cpu() for k, v in gpu_ckpt.items()}
    torch.cuda.synchronize()
    gpu_to_cpu_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    save_file(cpu_ckpt, safe_path)
    st_init_write = time.perf_counter() - t0 + gpu_to_cpu_s
    del cpu_ckpt; gc.collect()
    print(f"    done: {st_init_write:.1f}s", flush=True)

    print("  Writing torch pickle ...", flush=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    torch.save(gpu_ckpt, pt_path)
    torch.cuda.synchronize()
    pt_init_write = time.perf_counter() - t0
    print(f"    done: {pt_init_write:.1f}s", flush=True)

    st_gb = os.path.getsize(safe_path) / 1024**3
    pt_gb = os.path.getsize(pt_path) / 1024**3

    # Benchmark: writes (overwrite)
    st_writes, pt_writes = [], []

    for r in range(NUM_RUNS):
        # torch.save handles GPU to CPU to disk internally
        clear_page_cache()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        torch.save(gpu_ckpt, pt_path)
        torch.cuda.synchronize()
        pt_writes.append(time.perf_counter() - t0)

        # safetensors: GPU to CPU copy then save
        clear_page_cache()
        t0 = time.perf_counter()
        # cpu_ckpt = {k: v.cpu() for k, v in gpu_ckpt.items()}
        torch.cuda.synchronize()
        copy_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        save_file(gpu_ckpt, safe_path)
        st_writes.append(time.perf_counter() - t0 + copy_s)
        gc.collect()

        print(f"  write run {r+1}/{NUM_RUNS}  st={st_writes[-1]:.2f}s  pt={pt_writes[-1]:.2f}s", flush=True)

    del gpu_ckpt; gc.collect()
    torch.cuda.empty_cache()

    # Benchmark: cold reads (disk to GPU)
    st_reads, pt_reads = [], []

    for r in range(NUM_RUNS):
        clear_page_cache()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        tensors = {}
        with safe_open(safe_path, framework="pt", device=0) as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        _ = sum(v.sum() for v in tensors.values())
        torch.cuda.synchronize()
        st_reads.append(time.perf_counter() - t0)
        del tensors; gc.collect()
        torch.cuda.empty_cache()

        clear_page_cache()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        tensors = torch.load(pt_path, weights_only=True, map_location="cuda")
        _ = sum(v.sum() for v in tensors.values())
        torch.cuda.synchronize()
        pt_reads.append(time.perf_counter() - t0)
        del tensors; gc.collect()
        torch.cuda.empty_cache()

        print(f"  read run {r+1}/{NUM_RUNS}  st={st_reads[-1]:.2f}s  pt={pt_reads[-1]:.2f}s", flush=True)

    # Free disk: the checkpoint pair is only needed for the measurements above.
    for path in (safe_path, pt_path):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

    def stats(arr):
        return np.mean(arr), np.std(arr, ddof=1), np.percentile(arr, 99)

    return {
            "name": name, "domain": domain, "params": params,
            "tensors": n_tensors,
            "model_gb": model_gb, "optim_gb": optim_gb, "total_gb": total_gb,
        "st_gb": st_gb, "pt_gb": pt_gb,
        "st_init_write": st_init_write,
        "pt_init_write": pt_init_write,
        "write_st_mean": stats(st_writes)[0], "write_st_std": stats(st_writes)[1], "write_st_p99": stats(st_writes)[2],
        "write_pt_mean": stats(pt_writes)[0], "write_pt_std": stats(pt_writes)[1], "write_pt_p99": stats(pt_writes)[2],
        "read_st_mean":  stats(st_reads)[0],  "read_st_std":  stats(st_reads)[1],  "read_st_p99":  stats(st_reads)[2],
        "read_pt_mean":  stats(pt_reads)[0],  "read_pt_std":  stats(pt_reads)[1],  "read_pt_p99":  stats(pt_reads)[2],
        "raw_st_write": st_writes, "raw_pt_write": pt_writes,
        "raw_st_read":  st_reads,  "raw_pt_read":  pt_reads,
    }


def main():
    print(f"Checkpoint format benchmark: {NUM_RUNS} runs, GPU, page cache cleared", flush=True)
    print(f"Output dir: {OUTPUT}", flush=True)
    print()

    results = []
    for i, factory in enumerate(FACTORIES):
        print(f"\n[model {i+1}/{len(FACTORIES)}]", flush=True)
        name, state_dict, domain, params = factory()
        results.append(benchmark_model(name, domain, params, state_dict))
        del state_dict; gc.collect()
        torch.cuda.empty_cache()

    # Save raw CSV
    with open(RAW_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "domain", "run", "st_write", "pt_write", "st_read", "pt_read"])
        for r in results:
            for i in range(NUM_RUNS):
                w.writerow([r["name"], r["domain"], i + 1,
                            r["raw_st_write"][i], r["raw_pt_write"][i],
                            r["raw_st_read"][i], r["raw_pt_read"][i]])
    print(f"\nRaw data -> {RAW_CSV}")

    # Save summary CSV
    FIELD_NAMES = ["name", "domain", "params", "tensors", "model_gb", "optim_gb", "total_gb",
                   "read_st_mean", "read_st_std", "read_st_p99",
                   "read_pt_mean", "read_pt_std", "read_pt_p99",
                   "write_st_mean", "write_st_std", "write_st_p99",
                   "write_pt_mean", "write_pt_std", "write_pt_p99"]
    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELD_NAMES)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in FIELD_NAMES})
    print(f"Summary -> {RESULTS_CSV}")


    # Print report
    print(f"\n\n{'='*140}")
    print(f"{'Model':<20} {'Params':>10} {'Total':>8} | "
          f"{'Safetensors Read':>30} {'Pickle Read':>30} | "
          f"{'Safetensors Write':>30} {'Pickle Write':>30}")
    print("-" * 140)
    for r in results:
        print(f"{r['name']:<20} {r['params']/1e6:>5.1f}M  {r['total_gb']:>5.1f}GB | "
              f"\u03bc={r['read_st_mean']:.3f} \u03c3={r['read_st_std']:.3f} p99={r['read_st_p99']:.3f}  "
              f"\u03bc={r['read_pt_mean']:.3f} \u03c3={r['read_pt_std']:.3f} p99={r['read_pt_p99']:.3f} | "
              f"\u03bc={r['write_st_mean']:.2f} \u03c3={r['write_st_std']:.2f} p99={r['write_st_p99']:.2f}  "
              f"\u03bc={r['write_pt_mean']:.2f} \u03c3={r['write_pt_std']:.2f} p99={r['write_pt_p99']:.2f}")

    print("\n\nDone.")


if __name__ == "__main__":
    main()
