#!/bin/bash
#SBATCH --account=iu_0096
#SBATCH --qos=iu_0096
#SBATCH --job-name=llama32_3b_ddp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=defq
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
set -euo pipefail
export PATH="/dcai/users/andjen/.local/bin:$PATH"
PROJECT_ROOT=/dcai/users/andjen/project-angel/p1
cd ${PROJECT_ROOT}
echo "========================================="
echo "SLURM Job: ${SLURM_JOB_ID}"
echo "Command: make run dio/experiments/ddp_llama3.2_3b/"
echo "Node list: ${SLURM_NODELIST:-}"
echo "SLURM_JOB_GPUS: ${SLURM_JOB_GPUS:-}"
echo "SLURM_STEP_GPUS: ${SLURM_STEP_GPUS:-}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}"
echo "========================================="
# Log the Slurm-visible GPU assignment and the physical GPU mapping.
nvidia-smi -L
nvidia-smi --query-gpu=index,uuid,pci.bus_id,name --format=csv
make run dio/experiments/ddp_llama3.2_3b/
