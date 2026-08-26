#!/bin/bash
# run_ddp_hd_HumanCC_g4.sh
# VisiumHD super-bin MIL model on the 10x PUBLIC "Visium HD Human Colon Cancer"
# (spatial-exp 4.0.1) sample -- FULL tissue, grid 4x4 (64um).
# gene = sub-bin set aggregation (gated-MIL) via subbin_gene_encoder_n16.
#   - run eval/hd/preprocess_subbin_HumanCC_g4.slurm first to build DATA_ROOT.
#   - if batch 0 OOMs, drop --batch-size.

#SBATCH --job-name=discotme_HumanCC_mil_g4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email@example.com
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu-v100-dev
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --time=2:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

mkdir -p slurm_logs
echo "=== [Step 1] Environment Setup ==="

# PROJECT_ROOT: set $DISCOTME_HOME, else auto-detect from this script's location
# (scripts/hd/ -> two levels up = repo root), else edit the fallback below.
MY_HARDCODED_PATH="/path/to/DiSCoTME"
if [ -n "${DISCOTME_HOME}" ]; then
    export PROJECT_ROOT="${DISCOTME_HOME}"
else
    SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    DYNAMIC_ROOT=$(dirname "$(dirname "$SCRIPT_DIR")")   # scripts/hd -> scripts -> root
    if [ -f "${DYNAMIC_ROOT}/scripts/run_train_internal_test.py" ]; then
        export PROJECT_ROOT="${DYNAMIC_ROOT}"
    else
        echo "[Warn] Dynamic detection failed. Using hardcoded fallback."
        export PROJECT_ROOT="${MY_HARDCODED_PATH}"
    fi
fi
[ -d "$PROJECT_ROOT" ] || { echo "CRITICAL: Project root not found: '$PROJECT_ROOT'"; exit 1; }

PY_ENTRY="${PROJECT_ROOT}/scripts/hd/run_train_hd.py"   # thin HD wrapper (registers subbin encoders)
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
echo "Project Root: ${PROJECT_ROOT}"

: "${CONDA_BASE_DIR:=${HOME}/anaconda3}"
: "${CONDA_ENV:=spatialdata_hd}"
if [ -f "${CONDA_BASE_DIR}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_BASE_DIR}/etc/profile.d/conda.sh"
fi
conda activate "${CONDA_ENV}"
[ $? -eq 0 ] || { echo "ERROR: cannot activate ${CONDA_ENV}"; exit 1; }

export LD_LIBRARY_PATH="/usr/lib64:/usr/lib64/nvidia:${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}${CONDA_PREFIX}/lib"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
# Optional: only needed for GATED backbones (e.g. GigaPath). Public timm/DINOv2
# weights download anonymously -- leave this empty unless you need a gated model.
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((29512 + ($SLURM_JOB_ID % 1000)))
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-4}}"
echo "Master Addr: $MASTER_ADDR  Port: $MASTER_PORT  GPUs/Node: $GPUS_PER_NODE  Python: $(which python)"

# =========================================================
# === [Step 2] Arguments ===
# =========================================================
DATASET_NAME="VisiumHD_HumanCC_10x_subbin"
# ★ EDIT: point to your preprocessed HD data dir (build it with
#   eval/hd/preprocess_subbin_HumanCC_g4.slurm; grid 4, FULL tissue).
DATA_ROOT="/path/to/Visium_HD_Human_Colon_Cancer/binned_outputs/square_016um/preprocess_subbin_full_g4"

MODEL_TAG="dino_gated_subbinMIL"
EXTRA_TAG="n15_r500_grid4x4_64um_FULLtissue_bs12_ep5_LEARNgate_16gpu_relpos_dev"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="${DATASET_NAME}_${MODEL_TAG}_${EXTRA_TAG}_${TIMESTAMP}"

SAVE_BASE="${PROJECT_ROOT}/checkpoints"
CONTEXT_CONFIG_PATH="${PROJECT_ROOT}/configs/hd/visiumhd_crc_dilated_n120_relpos.yaml"  # relpos config

echo "Experiment Name: $RUN_NAME"
echo "Data Root:       $DATA_ROOT"
echo "Gene encoder:    subbin_gene_encoder_n16 (grid 4x4, agg=MIL)"

ARGS=(
  --data-root "$DATA_ROOT"
  --metadata-csv "metadata.csv"
  --tissue-positions-csv "tissue_positions.csv"

  --save-dir-base "$SAVE_BASE"
  --run-name "$RUN_NAME"

  --batch-size 12
  --num-epochs 5
  --learning-rate 1e-4
  --temperature 0.07
  --seed 42

  --num-local 15
  --num-global 0
  --local-distance 500

  --model-arch "standard_discotme"
  --image-encoder-type "gated_image_encoder"
  --image-backbone "vit_dino_v1"
  --gene-encoder-type "subbin_gene_encoder_n16"   # ★ HD sub-bin gene encoder (n_sub=16, MIL by default)
                                               #   use subbin_gene_encoder_mlp / _mean to switch agg
  --context-config "LongNet_for_spatial"
  --config-path "$CONTEXT_CONFIG_PATH"

  --embed-dim 256
  --proj-dim 128
)
# NOTE: image backbone left UNFROZEN (no --freeze-image-backbone).

echo "Starting torchrun..."
srun --label --export=ALL --gres=gpu:${GPUS_PER_NODE} \
  $(which torchrun) \
  --nnodes="${SLURM_NNODES}" \
  --nproc_per_node="${GPUS_PER_NODE}" \
  --rdzv_id="${SLURM_JOB_ID}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  "${PY_ENTRY}" \
  "${ARGS[@]}" \
  "$@"

echo "Job finished with exit code $?"
