# Anatomy of a Data Pipeline in Deep Learning Training: artifact

Benchmark code, run logs and the data behind the figures and tables of the paper.

## Repository layout

```
experiments/<dir>/     manifest.yaml + one directory of logs per run (see experiments/README.md)
paper/plots/<topic>/   data behind each figure (.csv) and the figure itself
paper/tables/<topic>/  data behind each table (.csv)
trainers/              benchmark scripts per domain
system/                experiment runner and system monitors (psutil, iostat, vmstat, nvidia-smi, DCGM)
tools/                 dataset scans
data/                  dataset download/conversion guides and on-disk dataset sizes
utils/                 page-cache helper
```

## Paper → data

| Paper | Data |
|---|---|
| Table 1, Fig. 6 sample sizes | `paper/plots/sample_distributions/sample_sizes/*.csv.gz` |
| Table 2 dataset sizes | `paper/tables/datasets/dataset_table.csv` (from `tools/measure_dataset_sizes.py`) |
| Table 3 models | `paper/tables/models/model_table.csv` |
| Fig. 7 latency breakdown | `paper/plots/latency_breakdown/latency_breakdown.csv`; runs in `experiments/latency_breakdown` |
| Fig. 8 throughput scaling | `experiments/scaling_*`, `experiments/formats_vision_train` |
| Table 5 formats | 16-worker runs in `experiments/formats_*` |
| Fig. 9 throughput vs. host memory, §4.4 shared memory | `paper/plots/img_classification_memory_scaling/*.csv` |
| Fig. 10 memory budget | `paper/plots/memory_budget/memory_budget_data.csv`; runs in `experiments/memory_budget` |
| Fig. 11 epoch boundary | `paper/plots/epoch_boundary/boundary_wait.csv`; runs in `experiments/epoch_boundary` |
| §4.6 read amplification | `paper/plots/read_amplification/read_amplification.csv`; runs in `experiments/read_amplification_c4` |
| Table 6 checkpoint formats | `experiments/checkpoint_formats/checkpoint_format_results.csv` |
| Fig. 12 checkpoint frequency, §5.2 write volume | `paper/plots/checkpoint_cost/*.csv`; runs in `experiments/checkpoint_cost_{nvme,nfs}` |
| §5.2 SSD write amplification | `experiments/waf_nvme` |
| §4.2 multi-GPU | `paper/tables/ddp_scaling/ddp_scaling.csv`; runs in `experiments/ddp_llama3.2_3b` |

## Running experiments

`uv sync` installs Python 3.10 and the pinned dependencies (`uv.lock`). Running the benchmarks needs a CUDA GPU and
the datasets, which are not included; `data/dataset_generators/<dataset>/download.md`
describes how to obtain and convert them. Optional extras:

```bash
# MambaVision: mamba_ssm must be compiled against the installed torch
CUDA_HOME=/usr/local/cuda-13.0 uv pip install --no-build-isolation mamba-ssm==2.3.2.post1
# FFCV: needs the libturbojpeg and OpenCV development packages
uv pip install --no-build-isolation ffcv==1.0.2
hf auth login          # Llama-3.2 models are gated on Hugging Face
```

```bash
make run experiments/<dir>     # every run in <dir>/manifest.yaml
```

`make run` writes one run directory per manifest entry (configuration, git
commit of the code, and logs). It drops the page cache before each run
(`utils/utils.py`), which needs passwordless `sudo` for
`sync; echo 3 > /proc/sys/vm/drop_caches`, or for a script named in `DROP_CACHES`.

Absolute numbers depend on the hardware:

| Machine | Hardware | Experiments |
|---|---|---|
| Machine 1 | 1× H100 PCIe 80 GiB, 2× Xeon Silver 4410Y (48 threads), 256 GiB DDR5, Samsung PM9A3 7.68 TB NVMe (ext4) | everything not listed below |
| Machine 2 | 2× H100 PCIe 80 GiB, 2× EPYC 9354, 768 GiB, NFSv4.2 over RDMA (100 Gb/s InfiniBand) to an NVMe RAID 5 | `checkpoint_cost_nfs` (`submit.lsf`) |
| Machine 3 | 8× H100 SXM 80 GiB, 2× Xeon Platinum 8480C, 2 TiB, WekaFS | `ddp_llama3.2_3b` (`submit_make_run.slurm`) |

## License

MIT, see `LICENSE`.
