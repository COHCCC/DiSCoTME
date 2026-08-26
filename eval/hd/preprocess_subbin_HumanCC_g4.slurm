#!/bin/bash
# 64um (grid 4x4) super-bin preprocessing for the 10x PUBLIC "Visium HD Human
# Colon Cancer" (spatial-exp 4.0.1) sample, FULL tissue (no quadrant).
# NOTE: for THIS sample the .btf lives at the sample ROOT (one level above
# binned_outputs), NOT inside square_016um -- so --wsi is passed EXPLICITLY
# (autodetect_wsi only searches --root and square_root and would miss it).
#SBATCH --job-name=pp_HumanCC_g4
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
set -euo pipefail
mkdir -p slurm_logs

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(dirname "$(dirname "$SCRIPT_DIR")")   # eval/hd -> eval -> repo root

: "${CONDA_BASE_DIR:=${HOME}/anaconda3}"
[ -f "${CONDA_BASE_DIR}/etc/profile.d/conda.sh" ] && source "${CONDA_BASE_DIR}/etc/profile.d/conda.sh"
conda activate spatialdata_hd 2>/dev/null || true
PYBIN="${CONDA_BASE_DIR}/envs/spatialdata_hd/bin/python"
export HDF5_USE_FILE_LOCKING=FALSE

# ★ EDIT: the downloaded 10x sample folder (contains binned_outputs/ and the .btf).
SAMPLE="/path/to/Visium_HD_Human_Colon_Cancer"
SQ="${SAMPLE}/binned_outputs/square_016um"
WSI="${SAMPLE}/Visium_HD_Human_Colon_Cancer_tissue_image.btf"
echo "Host: $(hostname)  grid=4x4 (64um)  FULL tissue  Start: $(date)"

"${PYBIN}" "${PROJECT_ROOT}/preprocessing/hd/preprocess_visiumhd_subbin.py" \
  --root "${SQ}" \
  --wsi  "${WSI}" \
  --out  "${SQ}/preprocess_subbin_full_g4" \
  --grid 4 \
  --progress-every 2000

echo "Done: $(date)"
