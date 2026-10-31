# Experiments

One directory per experiment. Each has a `manifest.yaml` (the runs to execute,
`make run experiments/<dir>`) and one sub-directory per executed run:

```
<run>_<YYYYMMDD_HHMMSS>/
  config.yaml          the run's configuration
  run_metadata.yaml    trainer and run id
  code_snapshot/       config as executed
  logs/                benchmark_*.csv (trainer result), trainer_metrics.csv,
                       psutil.csv, iostat.csv, vmstat.csv, smi.csv, dcgmi.csv
```

All runs used Machine 1 (local NVMe) unless noted. Folder names follow
`<study>_<domain>_<stage>`; stages are cumulative: `disk` (storage reads only),
`decode`, `preprocess`, `transfer` (to GPU), `train` (end-to-end).

## Index

| Directory | Paper | Contents |
|---|---|---|
| `latency_breakdown` | Fig. 7 | per-stage latency with `num_workers=0`, 11 models |
| `scaling_{vision,medical,speech,nlp}_{disk,…,train}` | Fig. 8, Table 4 | throughput per cumulative stage vs. worker count (one reference model per domain; all models for `train`) |
| `formats_{vision,medical,speech,nlp}_{disk,…,train}` | Table 5, Fig. 9 | throughput and resources per file format (16 workers in the table; worker sweeps for Vision `train`) |
| `formats_vision_train_channels_last` | Table 5 | Vision 16-worker rows with `channels_last` |
| `formats_vision_dali` | Table 5, Fig. 9 | ResNet-50 with NVIDIA DALI |
| `memory_budget` | Fig. 10 | throughput and swap under cgroup `MemoryHigh` limits (`*-swap-135GB*`, 8 workers) |
| `epoch_boundary` | Fig. 11 | 3 epochs per model, per-batch timestamps and 1 Hz GPU metrics |
| `read_amplification_c4` | §4.6 | C4 bytes read with/without shuffling, map-style vs. iterable |
| `ddp_llama3.2_3b` | §4.2 | Llama-3.2-3B DDP on 1–8 GPUs (Machine 3, Gefion; `submit_make_run.slurm`) |
| `checkpoint_formats` | Table 6 | Safetensors vs. Pickle, 100 runs per model, tied weights stored once (`trainers/checkpoint_format_benchmark.py`) |
| `checkpoint_cost_nvme` | Fig. 12 | per-step and per-save timing, 3 reps per model |
| `waf_nvme` | §5.2 | SSD write amplification of checkpoint writes: NVMe SMART and OCP logs before/after 20 synchronous Llama-3.2-3B saves (`smart/`) |
| `checkpoint_cost_nfs` | Fig. 12 | same on NFS, DTU P1 cluster (`submit.lsf`); no page-cache drop (no root), no DCGM |
| `reruns_2026_10` | Fig. 8, §4.3 | Speech `transfer`; 3D-UNet NIfTI vs. NIfTI-GZ, 3 repetitions each |

## Notes

- Page cache is dropped before every run on Machine 1 (`utils/utils.py`;
  `DROP_CACHES` overrides the helper path).
- Configs record the original data paths (`/raid/datasets/...` on Machine 1,
  `/dtu/p1/...` on DTU P1).
