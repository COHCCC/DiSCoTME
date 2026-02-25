#!/bin/bash
# run_train_slurm.sh
# SLURM script for multi-node distributed training of DiSCoTME

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

# =============================================================================
# USER CONFIGURATION - MODIFY THESE
# =============================================================================

PROJECT_ROOT="/coh_labs/dits/nsong/manuscript/unit_test_v2/DiSCoTME"  # <-- ADD THIS
CONDA_ENV="discotme_v2"
DATA_ROOT="/coh_labs/dits/nsong/manuscript/unit_test_v2/demo_data/"                    # <-- CHANGE THIS
META_CSV="metadata.csv"
POS_CSV="tissue_positions.csv"

# =============================================================================
# TRAINING ARGUMENTS - MODIFY AS NEEDED
# =============================================================================

ARGS=(
    # --- Data Paths ---
    --data-root "$DATA_ROOT"
    --metadata-csv "$META_CSV"
    --tissue-positions-csv "$POS_CSV"
    
    # --- Training Hyperparameters ---
    --batch-size 12
    --num-epochs 50
    --learning-rate 0.0001
    --weight-decay 0.00001
    --temperature 0.07
    --seed 42
    
    # --- Data Sampling ---
    --num-local 15
    --num-global 0
    --local-distance 400
    
    # --- Model Architecture ---
    --model-arch "standard_discotme"
    --image-encoder-type "gated_image_encoder"
    --image-backbone "vit_dino_v1"
    --gene-encoder-type "gated_gene_encoder"
    --context-config "LongNet_for_spatial"
    
    # --- Dimensions ---
    --embed-dim 256
    --proj-dim 128
    
    # --- Output ---
    --save-dir-base "checkpoints"
)

# =============================================================================
# ENVIRONMENT SETUP (usually no changes needed below)
# =============================================================================

mkdir -p slurm_logs

echo "============================================================"
echo "DiSCoTME Training Job"
echo "============================================================"

if [ -z "$PROJECT_ROOT" ]; then
    SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    if [ -f "${SCRIPT_DIR}/run_train.py" ]; then
        PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
    elif [ -f "${SCRIPT_DIR}/scripts/run_train.py" ]; then
        PROJECT_ROOT="${SCRIPT_DIR}"
    else
        echo "ERROR: Could not find project root."
        echo "Please set PROJECT_ROOT in the script."
        exit 1
    fi
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "Project Root: ${PROJECT_ROOT}"
echo "Data Root:    ${DATA_ROOT}"

# === Conda Activation ===
CONDA_BASE_DIR="${CONDA_BASE_DIR:-${HOME}/anaconda3}"

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

echo "Conda Env:    ${CONDA_ENV}"
echo "Python:       $(which python)"

# === Library Path ===
export LD_LIBRARY_PATH="/usr/lib64:/usr/lib64/nvidia:${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}${CONDA_PREFIX}/lib"

# === Distributed Configuration ===
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((29500 + ($SLURM_JOB_ID % 1000)))
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-4}}"

echo "Master Addr:  ${MASTER_ADDR}"
echo "Master Port:  ${MASTER_PORT}"
echo "Nodes:        ${SLURM_NNODES}"
echo "GPUs/Node:    ${GPUS_PER_NODE}"
echo "============================================================"

# =============================================================================
# LAUNCH TRAINING
# =============================================================================

cd "${PROJECT_ROOT}"

srun --label --export=ALL --gres=gpu:${GPUS_PER_NODE} \
    $(which torchrun) \
    --nnodes="${SLURM_NNODES}" \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    scripts/run_train_internal_test.py \
    "${ARGS[@]}" \
    "$@"

EXIT_CODE=$?
echo "============================================================"
echo "Job finished with exit code ${EXIT_CODE}"
echo "============================================================"
exit $EXIT_CODE
































