#!/bin/bash
# run_ddp_usr.sh

#SBATCH --job-name=discotme_gated
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jiasong@coh.org
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu-v100-dev
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --time=1:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

mkdir -p slurm_logs

echo "=== [Step 1] Environment Setup ==="

MY_HARDCODED_PATH="/coh_labs/dits/nsong/manuscript/DiSCoTME_2026"

if [ -n "${DISCOTME_HOME}" ]; then
    echo "[Info] Using provided env var DISCOTME_HOME"
    export PROJECT_ROOT="${DISCOTME_HOME}"
else
    # Try dynamic detection
    SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    DYNAMIC_ROOT=$(dirname "$SCRIPT_DIR")
    
    # Check if the dynamically detected path contains core files
    if [ -f "${DYNAMIC_ROOT}/scripts/run_train_internal_test.py" ]; then
        echo "[Info] Auto-detected project root: ${DYNAMIC_ROOT}"
        export PROJECT_ROOT="${DYNAMIC_ROOT}"
    else
        # Dynamic detection failed (likely running in SLURM spool), use hardcoded fallback
        echo "[Warn] Dynamic detection failed (likely SLURM spool). Using hardcoded fallback."
        export PROJECT_ROOT="${MY_HARDCODED_PATH}"
    fi
fi

# Final safety check
if [ ! -d "$PROJECT_ROOT" ]; then
    echo "CRITICAL ERROR: Project root not found at: '$PROJECT_ROOT'"
    echo "Please check if MY_HARDCODED_PATH is correct in the script."
    exit 1
fi

# Set other paths
PY_ENTRY="${PROJECT_ROOT}/scripts/run_train_internal_test.py"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "Project Root: ${PROJECT_ROOT}"
echo "Py Entry:     ${PY_ENTRY}"

# === Conda Environment Activation ===
: "${CONDA_BASE_DIR:=${HOME}/anaconda3}"
: "${CONDA_ENV:=gigapath}"

echo "Activating Conda: ${CONDA_ENV}"
if [ -f "${CONDA_BASE_DIR}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_BASE_DIR}/etc/profile.d/conda.sh"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
else
    echo "ERROR: Could not find conda.sh"
    exit 1
fi

conda activate "${CONDA_ENV}"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment '${CONDA_ENV}'"
    exit 1
fi

# === Environment Variables & Paths ===
export LD_LIBRARY_PATH="/usr/lib64:/usr/lib64/nvidia:${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}${CONDA_PREFIX}/lib"
export HF_HOME=/coh_labs/dits/nsong/.cache/huggingface
export HUGGING_FACE_HUB_TOKEN="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"

# === Distributed Network Configuration ===
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((29512 + ($SLURM_JOB_ID % 1000)))
# GPUS_PER_NODE is automatically obtained from SBATCH config (4 in your example)
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-4}}"

echo "Master Addr: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "GPUs/Node:   $GPUS_PER_NODE"
echo "Python Path: $(which python)"

# =========================================================
# === [Step 2] Training Arguments (Modify parameters here) ===
# =========================================================

DATASET_NAME="SPA6_A"
DATA_ROOT="/coh_labs/dits/nsong/manuscript/${DATASET_NAME}"
META_CSV="metadata.csv" 
POS_CSV="tissue_positions.csv"

# 2. Dynamically generate Run Name
# [Modified] Updated Tag to distinguish from previous Standard experiments
MODEL_TAG="dino_gated"
EXTRA_TAG="n15"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="${DATASET_NAME}_${MODEL_TAG}_${EXTRA_TAG}_${TIMESTAMP}"

# 3. Define common save root directory
SAVE_BASE="${PROJECT_ROOT}/checkpoints"

echo "Experiment Name: $RUN_NAME"
echo "Save Path:       $SAVE_BASE/$RUN_NAME"

# ARGS array
ARGS=(
  # --- Data Paths ---
  --data-root "$DATA_ROOT"
  --metadata-csv "$META_CSV"
  --tissue-positions-csv "$POS_CSV"
  
  # --- [Dynamic Paths] ---
  --save-dir-base "$SAVE_BASE"
  --run-name "$RUN_NAME"
  
  # --- Basic Training Hyperparameters ---
  --batch-size 12
  --num-epochs 5
  --learning-rate 1e-4
  --temperature 0.07
  --seed 42
  
  # --- Data Sampling ---
  --num-local 15
  --num-global 0
  --local-distance 400
  
  # --- Model Architecture (CRITICAL CHANGE) ---
  # Options: "standard_discotme" (old) | "factorized_discotme" (new, with reconstruction)
  --model-arch "standard_discotme"

  --image-encoder-type "gated_image_encoder"
  --image-backbone "vit_dino_v1"
  --gene-encoder-type "gated_gene_encoder"
  --context-config "LongNet_for_spatial"
  
  # --- Dimensions ---
  --embed-dim 256
  --proj-dim 128
  
  # --- Distillation/Reconstruction Weights ---
  # Note: In Factorized mode, Recon weight is currently hardcoded as 1.0 in trainer.py
  # If you want to tune this parameter here later, you need to modify run_train.py to pass kwargs
  # --distill-weight 0.0 
)

# =========================================================
# === [Step 3] Launching Training ===
# =========================================================

echo "Starting torchrun..."
echo "Entry Point: ${PY_ENTRY}"

# srun launcher
srun --label --export=ALL --gres=gpu:${GPUS_PER_NODE} \
  $(which torchrun) \
  --nnodes="${SLURM_NNODES}" \
  --nproc_per_node="${GPUS_PER_NODE}" \
  --rdzv_id="${SLURM_JOB_ID}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  "${PY_ENTRY}" \
  "${ARGS[@]}" \
  "$@" # Allow command line to append additional arguments to override ARGS above

EXIT_CODE=$?
echo "Job finished with exit code $EXIT_CODE"
exit $EXIT_CODE