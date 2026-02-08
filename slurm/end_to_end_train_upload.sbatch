#!/bin/bash --login
#SBATCH --job-name=cxr_e2e_r50
#SBATCH --output=/mnt/scratch/kodumuru/pari/chestxray_project/work/logs/%x_%j.out
#SBATCH --error=/mnt/scratch/kodumuru/pari/chestxray_project/work/logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

set -euo pipefail

PROJ="/mnt/scratch/kodumuru/pari/chestxray_project"
REPO="$PROJ/azure-hpc-medical-imaging"
WORK="$PROJ/work"
DATA_DIR="$WORK/data/chestxray/raw"
RUNS_DIR="$WORK/runs"

mkdir -p "$WORK/logs" "$RUNS_DIR" "$WORK/data/chestxray" "$DATA_DIR"

# ---- REQUIRED ENV VARS (export these before sbatch) ----
: "${AZ_DATASETS_SAS_URL:?AZ_DATASETS_SAS_URL not set}"
: "${AZ_ARTIFACTS_SAS_URL:?AZ_ARTIFACTS_SAS_URL not set}"

echo "Host: $(hostname)"
nvidia-smi || true

source ~/.bashrc
conda activate infra_env

cd "$REPO"
python -m pip install -r requirements.txt -q

# --- Download dataset from Azure (Blob -> local) ---
BASE="${AZ_DATASETS_SAS_URL/\?*/}"
TOK="${AZ_DATASETS_SAS_URL#*\?}"

# IMPORTANT: Blob "folder" copy (directory to directory)
azcopy copy \
"${BASE}/chestxray/raw/?${TOK}" \
"${DATA_DIR}/" \
--recursive=true

# remove junk (if any)
find "$DATA_DIR" -name ".DS_Store" -delete || true

# --- Train ---
python src/training/train.py --config configs/train_resnet50.yaml

# latest run folder
RUN_ID=$(ls -1t "$RUNS_DIR" | head -n 1)
RUN_DIR="$RUNS_DIR/$RUN_ID"

# --- Report/plots ---
python src/training/make_report.py --run_dir "$RUN_DIR"

# --- Upload artifacts to Azure ---
BASEA="${AZ_ARTIFACTS_SAS_URL/\?*/}"
TOKA="${AZ_ARTIFACTS_SAS_URL#*\?}"
DEST="${BASEA}/runs/${RUN_ID}/?${TOKA}"

azcopy copy "$RUN_DIR/*" "$DEST" --recursive=true

echo "Uploaded artifacts to: ${BASEA}/runs/${RUN_ID}/"
