#!/bin/bash
# run_ddp_usr.sh
#############   HOW TO USE THIS SCRIPT  ################
# sbatch run_ddp_usr.sh \
#   --data-root /coh_labs/dits/nsong/CRC_Spatial/CRC_08_Tumor \
#   --metadata-csv metadata.csv \
#   --tissue-positions-csv tissue_positions.csv \
#   --batch-size 12 --num-epochs 5 --learning-rate 3e-5 \
#   --weight-decay 1e-5 --temperature 0.07 \
#   --num-local 15 --num-global 0 --local-distance 400 \
#   --embed-dim 256 --proj-dim 128 \
#   --seed 42 \
#   --save-dir-base checkpoints/crc_08

#############   GOOD LUCK:>  ################

#SBATCH --job-name=crc08_ddp
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jiasong@coh.org
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=8:00:00
#SBATCH --output=slurm_logs/ddp_crc%j.out
#SBATCH --error=slurm_logs/ddp_crc%j.err

mkdir -p slurm_logs

echo "Attempting to activate Conda environment..."

# === 允许从外部覆盖；提供通用默认 ===
: "${CONDA_BASE_DIR:=${HOME}/anaconda3}"
: "${CONDA_ENV:=gigapath}"
: "${PROJECT_ROOT:=$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}"/.. && pwd)}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
: "${PY_ENTRY:=global_local_main_usr.py}"               # 入口脚本（文件名或绝对路径）

# 自动检测 conda.sh
if [ -f "${CONDA_BASE_DIR}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_BASE_DIR}/etc/profile.d/conda.sh"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
else
    echo "ERROR: Could not find conda.sh in ${CONDA_BASE_DIR} or ${HOME}/miniconda3"
    exit 1
fi

conda activate "${CONDA_ENV}"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment '${CONDA_ENV}'"
    echo "Available conda environments:"
    conda env list
    exit 1
fi
echo "Conda environment successfully activated: $CONDA_DEFAULT_ENV (Path: $(which python))"

# ===== 动态库路径（安全追加，避免未定义时报错） =====
# export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}${CONDA_PREFIX}/lib"
export LD_LIBRARY_PATH="/usr/lib64:/usr/lib64/nvidia:${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}${CONDA_PREFIX}/lib"
# ===== PYTHONPATH（软编码项目根；如你的入口脚本已做路径注入，可注释掉） =====
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# 从当前已激活环境抓取 python / torchrun 绝对路径（避免 srun 子进程掉环境）
PY_BIN="$(which python)"
TORCHRUN_BIN="$(which torchrun)"
echo "[INFO] PY_BIN=${PY_BIN}"
echo "[INFO] TORCHRUN_BIN=${TORCHRUN_BIN}"

# --- env ---
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node list: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_NNODES"
echo "Partition: $SLURM_JOB_PARTITION"
echo "CPUs per task on each node: $SLURM_CPUS_PER_TASK"
echo "Memory per node: $SLURM_MEM_PER_NODE"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((29500 + ($SLURM_JOB_ID % 1000)))

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"

# 与 SBATCH --gres 保持一致：优先用 SLURM 提供的变量，回退到 2（本作业申请了2卡）
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-2}}"

cd "$SLURM_SUBMIT_DIR"

echo "Current working directory: $(pwd)"
echo "Using PY_ENTRY: ${PY_ENTRY}"
echo "Using PROJECT_ROOT (for PYTHONPATH): ${PROJECT_ROOT}"
echo "GPUS_PER_NODE: ${GPUS_PER_NODE}"
echo "Starting torchrun at $(date)..."

# （可选）快速环境自检：确保子进程里 torch 可用
echo "=== quick env check (single task) ==="
srun --export=ALL --ntasks=1 --ntasks-per-node=1 bash -lc "
  echo 'HOST:'\$HOSTNAME ' CUDA_VISIBLE_DEVICES='\"\${CUDA_VISIBLE_DEVICES-<unset>}\" ;
  nvidia-smi --query-gpu=name,index --format=csv,noheader || true ;
  ${PY_BIN} - <<'PY'
import sys, torch
print('python:', sys.executable)
print('torch:', getattr(torch, '__version__', 'NOT INSTALLED'))
print('cuda.is_available:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())
PY
"
echo "====================================="
echo "[INFO] PY_ENTRY=${PY_ENTRY}"
echo "[INFO] ARGS: $@"
# 关键点 1：让 srun 继承环境（--export=ALL）
# 关键点 2：使用当前 env 的 torchrun 绝对路径（${TORCHRUN_BIN}）
# 关键点 3：训练参数透传给 Python（"$@"），不再硬编码
srun --label --export=ALL --gres=gpu:${GPUS_PER_NODE} \
  "${TORCHRUN_BIN}" \
  --nnodes="${SLURM_NNODES}" \
  --nproc_per_node="${GPUS_PER_NODE}" \
  --rdzv_id="${SLURM_JOB_ID}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  "${PY_ENTRY}" "$@"

EXIT_CODE=$?
echo "------------------------------------------------------------------------------------"
echo "torchrun finished at $(date) with exit code $EXIT_CODE"
echo "Job finished."
exit $EXIT_CODE