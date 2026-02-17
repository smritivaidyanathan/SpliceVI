#!/usr/bin/env bash
#SBATCH --job-name=subsample_mudata
#SBATCH --output=logs/subsample_mudata_%j.out
#SBATCH --error=logs/subsample_mudata_%j.err
#SBATCH --partition=cpu,bigmem,dev
#SBATCH --mem=300G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00

set -euo pipefail

# User inputs
TRAIN_MDATA_PATH="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025/train_70_30_model_ready_combined_gene_expression_aligned_splicing_20251009_024406_UPDATEDOBS.h5mu"

# INPUT_H5MU=${INPUT_H5MU:-"/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025/train_70_30_model_ready_combined_gene_expression_aligned_splicing_20251009_024406_UPDATEDOBS.h5mu"}

# INPUT_H5MU=${INPUT_H5MU:-"/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025/train_70_30_model_ready_combined_gene_expression_aligned_splicing_20251009_024406_UPDATEDOBS_SUBSAMPLED_N_50000.h5mu"}

INPUT_H5MU=${INPUT_H5MU:-"/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025/train_70_30_model_ready_combined_gene_expression_aligned_splicing_20251009_024406_UPDATEDOBS_SUBSAMPLED_N_50000_SUBSAMPLED_N_25000_SUBSAMPLED_N_10000.h5mu"}

N_CELLS=${N_CELLS:-10000}
STRATIFY_ON=${STRATIFY_ON:-"tissue_celltype age"}        # space-separated list, e.g. "broad_cell_type tissue"
SEED=${SEED:-42}
LOG_DIR=${LOG_DIR:-"logs"}

# Env
CONDA_BASE="/gpfs/commons/home/svaidyanathan/miniconda3"
ENV_NAME="splicevi-env"
SCRIPT_PATH="src/subsample_mudata.py"

mkdir -p "${LOG_DIR}"

echo "[JOB] Input       : ${INPUT_H5MU}"
echo "[JOB] N_CELLS     : ${N_CELLS}"
echo "[JOB] STRATIFY_ON : ${STRATIFY_ON}"
echo "[JOB] SEED        : ${SEED}"
echo "[JOB] Log dir     : ${LOG_DIR}"

echo "[ENV] Activating ${ENV_NAME}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
echo "[ENV] Python: $(which python)"

echo "[RUN] Starting subsample"
set -x
python "${SCRIPT_PATH}" \
  --input "${INPUT_H5MU}" \
  --n_cells "${N_CELLS}" \
  ${STRATIFY_ON:+--stratify_on ${STRATIFY_ON}} \
  --seed "${SEED}"
set +x

echo "[DONE] Subsampling complete"
