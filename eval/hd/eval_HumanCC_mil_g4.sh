#!/bin/bash
# Eval for the 10x PUBLIC "Visium HD Human Colon Cancer" g4 FULL-tissue MIL run.
#SBATCH --job-name=eval_HumanCC_mil_g4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu-v100-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=2:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
set -uo pipefail
mkdir -p slurm_logs
EPOCH_TAG="epoch5"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PROJECT_ROOT=$(dirname "$(dirname "$SCRIPT_DIR")")   # eval/hd -> eval -> repo root
PY_ENTRY="${PROJECT_ROOT}/eval/hd/run_eval_hd.py"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
: "${CONDA_BASE_DIR:=${HOME}/anaconda3}"; : "${CONDA_ENV:=spatialdata_hd}"
[ -f "${CONDA_BASE_DIR}/etc/profile.d/conda.sh" ] && source "${CONDA_BASE_DIR}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}" 2>/dev/null || true
export LD_LIBRARY_PATH="/usr/lib64:/usr/lib64/nvidia:${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}${CONDA_PREFIX}/lib"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
PYTHON_BIN="${CONDA_BASE_DIR}/envs/${CONDA_ENV}/bin/python"

# ★ EDIT: square_016um dir of your downloaded sample.
SQ="/path/to/Visium_HD_Human_Colon_Cancer/binned_outputs/square_016um"
DATA_ROOT="${SQ}/preprocess_subbin_full_g4"
CKPT_DIR=$(ls -d ${PROJECT_ROOT}/checkpoints/*HumanCC*subbinMIL*grid4x4*16gpu* 2>/dev/null | head -1)
[ -n "${CKPT_DIR}" ] || { echo "ERROR: no HumanCC MIL g4 checkpoint"; exit 1; }
CHECKPOINT_DIR=$(basename "${CKPT_DIR}")
MODEL_PATH="${CKPT_DIR}/checkpoint_${EPOCH_TAG}.pth"
OUTPUT_DIR="${DATA_ROOT}/results/eval/${CHECKPOINT_DIR}__${EPOCH_TAG}"
CONFIG_PATH="${PROJECT_ROOT}/configs/hd/visiumhd_crc_dilated_n120_relpos.yaml"
[ -f "${MODEL_PATH}" ] || { echo "ERROR: checkpoint not found: ${MODEL_PATH}"; exit 1; }
mkdir -p "${OUTPUT_DIR}"
echo "Eval: ${MODEL_PATH}"

"${PYTHON_BIN}" "${PY_ENTRY}" \
  --data-root "$DATA_ROOT" --metadata-csv "metadata.csv" --tissue-positions-csv "tissue_positions.csv" \
  --model-path "$MODEL_PATH" --output-dir "$OUTPUT_DIR" --output-suffix "hd_subbin_${EPOCH_TAG}" --device "cuda" \
  --model-arch "standard_discotme" --image-encoder-type "gated_image_encoder" --image-backbone "vit_dino_v1" \
  --gene-encoder-type "subbin_gene_encoder_n16" --context-config "LongNet_for_spatial" --config-path "$CONFIG_PATH" \
  --embed-dim 256 --proj-dim 128 --k-min 4 --k-max 12 --pca-dim 20 --batch-size 8 --num-workers 8 \
  --num-local 15 --num-global 0 --local-distance 500
echo "Done eval -> ${OUTPUT_DIR}"
