#!/usr/bin/env bash
#SBATCH --job-name=splicevi_eval_basic
#SBATCH --output=logs/splicevi_eval_%j.out
#SBATCH --error=logs/splicevi_eval_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=300G
#SBATCH --cpus-per-task=4
#SBATCH --time=30:00:00

set -euo pipefail

# eval_splicevi.sh
#
# Minimal Slurm job script to:
#   1. Load TRAIN and TEST MuData
#   2. Load a trained SPLICEVI model from MODEL_DIR
#   3. Run selected evaluation blocks (UMAP, clustering, metrics, imputation)
#   4. Save evaluation figures/CSVs under a per-run directory
#
# Submit with:
#   sbatch eval_splicevi.sh

#######################################
# USER CONFIGURATION
#######################################

# 1) Data paths
TRAIN_MDATA_PATH="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025/train_70_30_model_ready_combined_gene_expression_aligned_splicing_20251009_024406_UPDATEDOBS.h5mu"
TEST_MDATA_PATH="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025/test_30_70_model_ready_combined_gene_expression_aligned_splicing_20251009_024406_UPDATEDOBS.h5mu"

# Optional mapping CSV (can be empty if not used)
MAPPING_CSV="/gpfs/commons/home/svaidyanathan/repos/multivi_tools_splicing/multivi_splice_utils/runfiles/tissue_celltype_mapping.csv"

# Optional masked TEST MuData paths for imputation
# MASKED_TEST_MDATA_PATHS="\
# /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025/MASKED_25_PERCENT_test_30_70_model_ready_combined_gene_expression_aligned_splicing_20251009_024406.h5mu \
# /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025/MASKED_50_PERCENT_test_30_70_model_ready_combined_gene_expression_aligned_splicing_20251009_024406.h5mu \
# /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025/MASKED_75_PERCENT_test_30_70_model_ready_combined_gene_expression_aligned_splicing_20251009_024406.h5mu"


MASKED_TEST_MDATA_PATHS="\
/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025/MASKED_25_PERCENT_test_30_70_model_ready_combined_gene_expression_aligned_splicing_20251009_024406_UPDATEDOBS.h5mu"

# 2) Single model directory to evaluate
# MODEL_DIR="/gpfs/commons/home/svaidyanathan/repos/SpliceVI/models/splicevi_basic_20260127_143801"  #non linear (yes batch key)

MODEL_DIR="/gpfs/commons/home/svaidyanathan/repos/SpliceVI/models/splicevi_basic_20251213_111454"  #linear (yes batch key)

# MODEL_DIR="/gpfs/commons/home/svaidyanathan/repos/SpliceVI/models/splicevi_basic_20260121_123415" #linear (no batch key)




# 3) Evaluation blocks to run
# These must match argparse in eval_splicevi.py (nargs="+")
EVALS=(
  umap
  # clustering
  # train_eval
  # test_eval
  cross_fold_classification
  age_r2_heatmap
  masked_impute
)

# 4) UMAP and imputation config
UMAP_TOP_N_CELLTYPES=15       # kept for compatibility (currently unused in plotting logic)
IMPUTE_BATCH_SIZE=512         # set to -1 to disable batching (one big batch per masked file)

# List of obs keys to use for coloring TRAIN UMAPs for each latent space
UMAP_OBS_KEYS=(
  "broad_cell_type"
  "medium_cell_type"
  "mouse.id"
  "tissue_celltype"
  # "tissue"
  # "tissue_celltype"
)

# 4.5) Cross-fold classification config
CROSS_FOLD_SPLITS="train"  # train | test | both
CROSS_FOLD_TARGETS=(
  "broad_cell_type"
  "mouse.id"
  "tissue_celltype"
)
CROSS_FOLD_K=5
CROSS_FOLD_CLASSIFIERS=(
  "logreg"
  # "rf"
)

# 5) Conda / script locations
CONDA_BASE="/gpfs/commons/home/svaidyanathan/miniconda3"
ENV_NAME="splicevi-env"
SCRIPT_PATH="src/eval_splicevi.py"   # relative to repo root

# 6) W&B configuration (optional)
USE_WANDB=true                        # set to "false" to disable W&B logging
WANDB_PROJECT="MLCB_SUBMISSION"       # required if USE_WANDB=true
WANDB_ENTITY=""                       # optional W&B entity (team)
WANDB_GROUP="splicevi_eval"           # optional W&B group name
WANDB_RUN_NAME_PREFIX="basic_eval"    # prefix for run names
WANDB_LOG_FREQ=1000                   # how often to log from wandb.watch

# 7) Output directory for eval run
BASE_RUN_DIR="/gpfs/commons/home/svaidyanathan/repos/SpliceVI/logs"

#######################################
# DERIVED SETTINGS
#######################################

TS=$(date +"%Y%m%d_%H%M%S")
MODEL_BASENAME=$(basename "${MODEL_DIR}")
RUN_NAME="eval_${MODEL_BASENAME}_${TS}"

RUN_DIR="${BASE_RUN_DIR}/${RUN_NAME}"
FIG_DIR="${RUN_DIR}/figures"
mkdir -p "${FIG_DIR}"

echo "=================================================================="
echo "[JOB] SPLICEVI basic eval job"
echo "[JOB] Slurm job ID           : ${SLURM_JOB_ID:-N/A}"
echo "[JOB] Run name               : ${RUN_NAME}"
echo "[JOB] MODEL_DIR              : ${MODEL_DIR}"
echo "[JOB] TRAIN_MDATA_PATH       : ${TRAIN_MDATA_PATH}"
echo "[JOB] TEST_MDATA_PATH        : ${TEST_MDATA_PATH}"
echo "[JOB] Mapping CSV            : ${MAPPING_CSV}"
echo "[JOB] Eval output directory  : ${RUN_DIR}"
echo "[JOB] Figures directory      : ${FIG_DIR}"
echo "=================================================================="
echo "[JOB] MASKED_TEST_MDATA_PATHS:"
for p in ${MASKED_TEST_MDATA_PATHS}; do
  echo "         - ${p}"
done
echo "[JOB] UMAP_OBS_KEYS:"
for k in "${UMAP_OBS_KEYS[@]}"; do
  echo "         - ${k}"
done
echo "[JOB] CROSS_FOLD_SPLITS : ${CROSS_FOLD_SPLITS}"
echo "[JOB] CROSS_FOLD_TARGETS:"
for t in "${CROSS_FOLD_TARGETS[@]}"; do
  echo "         - ${t}"
done
echo "[JOB] CROSS_FOLD_CLASSIFIERS:"
for c in "${CROSS_FOLD_CLASSIFIERS[@]}"; do
  echo "         - ${c}"
done
echo "[JOB] CROSS_FOLD_K      : ${CROSS_FOLD_K}"
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
# BUILD OPTIONAL W&B ARGUMENT STRING
#######################################

WANDB_ARGS=""
if [ "${USE_WANDB}" = true ]; then
  echo "[W&B] Enabling Weights & Biases logging."
  WANDB_ARGS+=" --use_wandb"
  if [ -n "${WANDB_PROJECT}" ]; then
    WANDB_ARGS+=" --wandb_project ${WANDB_PROJECT}"
  else
    echo "[W&B] ERROR: USE_WANDB=true but WANDB_PROJECT is empty."
    exit 1
  fi
  if [ -n "${WANDB_ENTITY}" ]; then
    WANDB_ARGS+=" --wandb_entity ${WANDB_ENTITY}"
  fi
  if [ -n "${WANDB_GROUP}" ]; then
    WANDB_ARGS+=" --wandb_group ${WANDB_GROUP}"
  fi
  WANDB_RUN_NAME="${WANDB_RUN_NAME_PREFIX}_${MODEL_BASENAME}"
  WANDB_ARGS+=" --wandb_run_name ${WANDB_RUN_NAME}"
  WANDB_ARGS+=" --wandb_log_freq ${WANDB_LOG_FREQ}"
else
  echo "[W&B] W&B logging disabled for this run."
fi
echo

#######################################
# PREPARE MULTI-VALUE ARGUMENTS
#######################################

EVALS_JOINED="${EVALS[*]}"
UMAP_OBS_KEYS_JOINED="${UMAP_OBS_KEYS[*]}"
CROSS_FOLD_TARGETS_JOINED="${CROSS_FOLD_TARGETS[*]}"
CROSS_FOLD_CLASSIFIERS_JOINED="${CROSS_FOLD_CLASSIFIERS[*]}"

#######################################
# LAUNCH EVALUATION
#######################################

echo "[RUN] Launching SPLICEVI eval script..."
set -x

python "${SCRIPT_PATH}" \
  --train_mdata_path "${TRAIN_MDATA_PATH}" \
  --test_mdata_path "${TEST_MDATA_PATH}" \
  --model_dir "${MODEL_DIR}" \
  --fig_dir "${FIG_DIR}" \
  --mapping_csv "${MAPPING_CSV}" \
  --impute_batch_size "${IMPUTE_BATCH_SIZE}" \
  --umap_top_n_celltypes "${UMAP_TOP_N_CELLTYPES}" \
  --umap_obs_keys ${UMAP_OBS_KEYS_JOINED} \
  --cross_fold_splits "${CROSS_FOLD_SPLITS}" \
  --cross_fold_targets ${CROSS_FOLD_TARGETS_JOINED} \
  --cross_fold_k "${CROSS_FOLD_K}" \
  --cross_fold_classifiers ${CROSS_FOLD_CLASSIFIERS_JOINED} \
  --evals ${EVALS_JOINED} \
  ${MASKED_TEST_MDATA_PATHS:+--masked_test_mdata_paths ${MASKED_TEST_MDATA_PATHS}} \
  ${WANDB_ARGS}

set +x

echo
echo "[DONE] SPLICEVI basic eval job finished."
echo "[DONE] Eval outputs in: ${RUN_DIR}"
