#!/usr/bin/env bash
#SBATCH --job-name=mudata_train_test_split
#SBATCH --output=logs/train_test_split_%j.out
#SBATCH --error=logs/train_test_split_%j.err
#SBATCH --partition=bigmem
#SBATCH --mem=500G
#SBATCH --cpus-per-task=1
#SBATCH --time=8:00:00

set -euo pipefail

# train_test_split.sh
#
# Create a combined MuData and (optionally) a stratified train/test split.
#
# Usage:
#   1) Edit USER CONFIGURATION below.
#   2) Run:
#        sbatch train_test_split.sh
#      or
#        bash train_test_split.sh
#
# Notes:
# - Edit the paths to splicing and gene expression anndatas ATSE_BASENAME and GE_BASENAME (cells must be fully paired, anndatas must be in same ROOT_PATH)
# - Edit the name of the combined mudata object (exp + splicing) OUTPUT_BASENAME, will end up in same path as the anndatas
# - Train/test filenames are derived from the combined filename:
#     train_{train_pct}_{test_pct}_{combined_name}.h5mu
#     test_{test_pct}_{train_pct}_{combined_name}.h5mu
# - Set MAX_JUNCTIONS_PER_ATSE="None" to disable ATSE junction filtering.
# - Set STRATIFY_COLS=("none") to disable stratification.

#######################################
# USER CONFIGURATION
#######################################

# 1) Data paths (edit these)
ROOT_PATH="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/MODEL_INPUT/102025"
ATSE_BASENAME="model_ready_aligned_splicing_data_20251009_023419.h5ad"
GE_BASENAME="model_ready_gene_expression_data_20251009_023419.h5ad"
OUTPUT_BASENAME="DO_NOT_USE_THIS_IS_A_TEST_model_ready_combined_gene_expression_aligned_splicing_data_20251009_023419.h5mu"

ATSE_DATA_PATH="${ROOT_PATH}/${ATSE_BASENAME}"
GE_DATA_PATH="${ROOT_PATH}/${GE_BASENAME}"
OUTPUT_MUDATA_PATH="${ROOT_PATH}/${OUTPUT_BASENAME}"

# 2) Split configuration
DO_SPLIT=true
TEST_SIZE=0.30
RANDOM_STATE=42
STRATIFY_COLS=(
  "broad_cell_type"
  "age"
)

# 3) ATSE junction filtering
MAX_JUNCTIONS_PER_ATSE="None"  # use an integer (e.g., 4) to enable filtering

# 4) Conda / script locations
CONDA_BASE="/gpfs/commons/home/svaidyanathan/miniconda3"
ENV_NAME="splicevi-env"
SCRIPT_PATH="/gpfs/commons/home/svaidyanathan/repos/SpliceVI/preprocess/createtesttrainmudata.py"

#######################################
# DERIVED SETTINGS
#######################################


echo "=================================================================="
echo "[JOB] MuData train/test split"
echo "[JOB] Slurm job ID           : ${SLURM_JOB_ID:-N/A}"
echo "[JOB] ATSE data path          : ${ATSE_DATA_PATH}"
echo "[JOB] GE data path            : ${GE_DATA_PATH}"
echo "[JOB] Output MuData path      : ${OUTPUT_MUDATA_PATH}"
echo "[JOB] Do split                : ${DO_SPLIT}"
echo "[JOB] Test size               : ${TEST_SIZE}"
echo "[JOB] Random state            : ${RANDOM_STATE}"
echo "[JOB] Max junctions per ATSE  : ${MAX_JUNCTIONS_PER_ATSE}"
echo "[JOB] Stratify columns:"
for c in "${STRATIFY_COLS[@]}"; do
  echo "         - ${c}"
done
echo "=================================================================="

#######################################
# ENVIRONMENT SETUP
#######################################

echo "[ENV] Activating conda environment '${ENV_NAME}'..."
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
echo "[ENV] Python executable: $(which python)"
echo "[ENV] Python version  : $(python -V)"
echo

#######################################
# LAUNCH
#######################################

echo "[RUN] Launching MuData creation/split script..."
set -x

python "${SCRIPT_PATH}" \
  --root_path "${ROOT_PATH}" \
  --atse_data_path "${ATSE_DATA_PATH}" \
  --ge_data_path "${GE_DATA_PATH}" \
  --output_mudata_path "${OUTPUT_MUDATA_PATH}" \
  --do_split "${DO_SPLIT}" \
  --test_size "${TEST_SIZE}" \
  --random_state "${RANDOM_STATE}" \
  --stratify_cols "${STRATIFY_COLS[@]}" \
  --max_junctions_per_atse "${MAX_JUNCTIONS_PER_ATSE}"

set +x

echo
echo "[DONE] MuData creation/split job finished."
