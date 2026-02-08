#!/bin/bash --login
#SBATCH --job-name=chestxray_r50
#SBATCH --output=/mnt/scratch/kodumuru/pari/chestxray_project/work/logs/%x_%j.out
#SBATCH --error=/mnt/scratch/kodumuru/pari/chestxray_project/work/logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

set -euo pipefail

# === paths ===
PROJ="/mnt/scratch/kodumuru/pari/chestxray_project"
REPO="$PROJ/azure-hpc-medical-imaging"
WORK="$PROJ/work"
DATA="$WORK/data/chestxray/raw"

mkdir -p "$WORK/logs" "$WORK/runs"

echo "Host: $(hostname)"
echo "CUDA visible: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

# activate your env (adjust if needed)
source ~/.bashrc
conda activate infra_env

cd "$REPO"

python -m pip install -r requirements.txt -q

python src/training/train.py --config configs/train_resnet50.yaml
