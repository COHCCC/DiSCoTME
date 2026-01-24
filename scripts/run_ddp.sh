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
    # 尝试动态获取
    SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    DYNAMIC_ROOT=$(dirname "$SCRIPT_DIR")
    
    # 检查动态获取的路径里有没有核心文件
    if [ -f "${DYNAMIC_ROOT}/scripts/run_train_internal_test.py" ]; then
        echo "[Info] Auto-detected project root: ${DYNAMIC_ROOT}"
        export PROJECT_ROOT="${DYNAMIC_ROOT}"
    else
        # 动态获取失败 (说明在 SLURM Spool 里)，使用硬编码保底
        echo "[Warn] Dynamic detection failed (likely SLURM spool). Using hardcoded fallback."
        export PROJECT_ROOT="${MY_HARDCODED_PATH}"
    fi
fi

# 最后的安全检查
if [ ! -d "$PROJECT_ROOT" ]; then
    echo "CRITICAL ERROR: Project root not found at: '$PROJECT_ROOT'"
    echo "Please check if MY_HARDCODED_PATH is correct in the script."
    exit 1
fi

# 设置其他路径
PY_ENTRY="${PROJECT_ROOT}/scripts/run_train_internal_test.py"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "Project Root: ${PROJECT_ROOT}"
echo "Py Entry:     ${PY_ENTRY}"

# === Conda 环境激活 ===
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

# === 环境变量 & 路径 ===
export LD_LIBRARY_PATH="/usr/lib64:/usr/lib64/nvidia:${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}${CONDA_PREFIX}/lib"
export HF_HOME=/coh_labs/dits/nsong/.cache/huggingface
export HUGGING_FACE_HUB_TOKEN="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"

# === 分布式网络配置 ===
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((29512 + ($SLURM_JOB_ID % 1000)))
# 这里的 GPUS_PER_NODE 自动获取 SBATCH 配置 (你的例子是4)
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-4}}"

echo "Master Addr: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "GPUs/Node:   $GPUS_PER_NODE"
echo "Python Path: $(which python)"

# =========================================================
# === [Step 2] Training Arguments (在这里修改参数) ===
# =========================================================

DATASET_NAME="CRC_07_Tumor"
DATA_ROOT="/coh_labs/dits/nsong/manuscript/${DATASET_NAME}"
META_CSV="metadata.csv" 
POS_CSV="tissue_positions.csv"

# 2. 动态生成 Run Name
# [修改] 更新 Tag 以区分之前的 Standard 实验
MODEL_TAG="dino_gated"
EXTRA_TAG="n15"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="${DATASET_NAME}_${MODEL_TAG}_${EXTRA_TAG}_${TIMESTAMP}"

# 3. 定义通用的保存根目录
SAVE_BASE="${PROJECT_ROOT}/checkpoints"

echo "Experiment Name: $RUN_NAME"
echo "Save Path:       $SAVE_BASE/$RUN_NAME"

# ARGS 数组
ARGS=(
  # --- 数据路径 ---
  --data-root "$DATA_ROOT"
  --metadata-csv "$META_CSV"
  --tissue-positions-csv "$POS_CSV"
  
  # --- [动态路径] ---
  --save-dir-base "$SAVE_BASE"
  --run-name "$RUN_NAME"
  
  # --- 基础训练超参 ---
  --batch-size 12
  --num-epochs 5
  --learning-rate 1e-4
  --temperature 0.07
  --seed 42
  
  # --- 数据采样 ---
  --num-local 15
  --num-global 0
  --local-distance 400
  
  # --- 模型架构 (CRITICAL CHANGE) ---
  # 选项: "standard_discotme" (旧) | "factorized_discotme" (新, 带重建)
  --model-arch "standard_discotme"

  --image-encoder-type "gated_image_encoder"
  --image-backbone "vit_dino_v1"
  --gene-encoder-type "gated_gene_encoder"
  --context-config "LongNet_for_spatial"
  
  # --- 维度 ---
  --embed-dim 256
  --proj-dim 128
  
  # --- 蒸馏/重建权重 ---
  # 注意：在 Factorized 模式下，Recon 权重目前在 trainer.py 里硬编码为 1.0
  # 如果之后想在这里调参，需要修改 run_train.py 传入 kwargs
  # --distill-weight 0.0 
)

# =========================================================
# === [Step 3] Launching Training ===
# =========================================================

echo "Starting torchrun..."
echo "Entry Point: ${PY_ENTRY}"

# srun 启动器
srun --label --export=ALL --gres=gpu:${GPUS_PER_NODE} \
  $(which torchrun) \
  --nnodes="${SLURM_NNODES}" \
  --nproc_per_node="${GPUS_PER_NODE}" \
  --rdzv_id="${SLURM_JOB_ID}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  "${PY_ENTRY}" \
  "${ARGS[@]}" \
  "$@" # 允许命令行追加额外参数覆盖上面的 ARGS

EXIT_CODE=$?
echo "Job finished with exit code $EXIT_CODE"
exit $EXIT_CODE