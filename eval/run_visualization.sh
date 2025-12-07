#!/bin/bash
# run_visualization.sh - SLURM script for running visualization
#############   HOW TO USE THIS SCRIPT  ################
# sbatch run_visualization.sh \
#   --data-root /coh_labs/dits/nsong/CRC_Spatial/CRC_08_Tumor \
#   --metadata-csv /coh_labs/dits/nsong/CRC_Spatial/CRC_08_Tumor/metadata.csv \
#   --model-path /coh_labs/dits/nsong/python2024/stHistoCLIP/TME_aware_model/checkpoints/crc_08/geneattn_ddp_20251019_194806/final_model.pth \
#   --output-dir /coh_labs/dits/nsong/CRC_Spatial/CRC_08_Tumor/results \
#   --output-suffix e5
#############   GOOD LUCK:>  ################
#!/bin/bash
# run_visualization.sh - 使用DDP脚本的环境设置逻辑运行可视化

#SBATCH --job-name=discotme_demo
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jiasong@coh.org
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=4:00:00
#SBATCH --output=slurm_logs/discotme_demo_vis_%j.out
#SBATCH --error=slurm_logs/discotme_demo_vis_%j.err

mkdir -p slurm_logs

echo "Attempting to activate Conda environment..."

# === 从DDP脚本复制的环境设置 ===
: "${CONDA_BASE_DIR:=${HOME}/anaconda3}"
: "${CONDA_ENV:=gigapath}"
: "${PROJECT_ROOT:=$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}"/.. && pwd)}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
: "${VIS_SCRIPT:=visualization_usr.py}"  

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

# ===== 动态库路径 =====
export LD_LIBRARY_PATH="/usr/lib64:/usr/lib64/nvidia:${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}${CONDA_PREFIX}/lib"

# 从当前已激活环境抓取 python 绝对路径
PY_BIN="$(which python)"
echo "[INFO] PY_BIN=${PY_BIN}"

# --- env ---
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node list: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_NNODES"
echo "Partition: $SLURM_JOB_PARTITION"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Memory per node: $SLURM_MEM_PER_NODE"

cd "$SLURM_SUBMIT_DIR"

echo "Current working directory: $(pwd)"
echo "Using VIS_SCRIPT: ${VIS_SCRIPT}"
echo "Using PROJECT_ROOT (for PYTHONPATH): ${PROJECT_ROOT}"

# 检查CUDA
echo "=== CUDA check ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=name,index --format=csv,noheader || true
echo "=================="

# 确保脚本存在
if [ ! -f "${VIS_SCRIPT}" ]; then
    echo "ERROR: Cannot find ${VIS_SCRIPT} in current directory"
    echo "Current directory contents:"
    ls -la
    exit 1
fi

echo "[INFO] Running: ${PY_BIN} ${VIS_SCRIPT} $@"
echo "Starting visualization at $(date)..."

# 运行可视化脚本（不使用torchrun）
"${PY_BIN}" "${VIS_SCRIPT}" "$@"

EXIT_CODE=$?
echo "------------------------------------------------------------------------------------"
echo "Visualization finished at $(date) with exit code $EXIT_CODE"
echo "Job finished."
exit $EXIT_CODE