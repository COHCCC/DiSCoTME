#!/bin/bash
#SBATCH --job-name=vis_gated
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jiasong@coh.org
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu-v100-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

mkdir -p slurm_logs
echo "=== [Step 1] Environment Setup ==="

# 1. 路径设置 (混合策略)
MY_HARDCODED_PATH="/coh_labs/dits/nsong/manuscript/DiSCoTME_2026"

if [ -n "${DISCOTME_HOME}" ]; then
    export PROJECT_ROOT="${DISCOTME_HOME}"
else
    SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    DYNAMIC_ROOT=$(dirname "$SCRIPT_DIR")
    # 检查 eval 脚本是否存在
    if [ -f "${DYNAMIC_ROOT}/eval/run_eval_with_alpha.py" ]; then
        export PROJECT_ROOT="${DYNAMIC_ROOT}"
    else
        export PROJECT_ROOT="${MY_HARDCODED_PATH}"
    fi
fi

PY_ENTRY="${PROJECT_ROOT}/eval/run_eval_with_alpha.py"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# 2. Conda 环境激活
: "${CONDA_BASE_DIR:=${HOME}/anaconda3}"
: "${CONDA_ENV:=gigapath}"
if [ -f "${CONDA_BASE_DIR}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_BASE_DIR}/etc/profile.d/conda.sh"
fi
conda activate "${CONDA_ENV}"
export LD_LIBRARY_PATH="/usr/lib64:/usr/lib64/nvidia:${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}${CONDA_PREFIX}/lib"

# =========================================================
# === [Step 2] Default Arguments (可被命令行覆盖) ===
# =========================================================
# 默认数据集和检查点
DATASET_NAME="CRC_07_Tumor"
CHECKPOINT_DIR="CRC_07_Tumor_dino_gated_n15_20260122_180207"

DEFAULT_DATA_ROOT="/coh_labs/dits/nsong/manuscript/${DATASET_NAME}"
DEFAULT_MODEL_PATH="${PROJECT_ROOT}/checkpoints/${CHECKPOINT_DIR}/best_model.pth"
DEFAULT_OUTPUT_DIR="${DEFAULT_DATA_ROOT}/results/eval/dino_gated_n15"
# =========================================================
# === [Step 3] Run Evaluation ===
# =========================================================

echo "Starting Bilateral Gated Evaluation..."
echo "Project Root: $PROJECT_ROOT"
echo "Model Path:   $DEFAULT_MODEL_PATH"

python "${PY_ENTRY}" \
  --data-root "$DEFAULT_DATA_ROOT" \
  --metadata-csv "metadata.csv" \
  --tissue-positions-csv "tissue_positions.csv" \
  --model-path "$DEFAULT_MODEL_PATH" \
  --output-dir "$DEFAULT_OUTPUT_DIR" \
  --output-suffix "fact_e10" \
  --device "cuda" \
  \
  --model-arch "standard_discotme" \
  \
  --image-encoder-type "gated_image_encoder" \
  --image-backbone "vit_dino_v1" \
  --gene-encoder-type "gated_gene_encoder" \
  --context-config "LongNet_for_spatial" \
  \
  --embed-dim 256 \
  --proj-dim 128 \
  \
  --k-min 4 \
  --k-max 12 \
  --batch-size 32 \
  \
  --num-local 15 \
  --num-global 0 \
  --local-distance 400 \
  "$@" 

echo "Done. Gated results and Alpha maps are in $DEFAULT_OUTPUT_DIR"