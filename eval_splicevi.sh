#!/usr/bin/env bash
#SBATCH --job-name=splicevi_eval_batch
#SBATCH --output=/logs/splicevi_train_%j.out
#SBATCH --error=/logs/splicevi_train_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=230G
#SBATCH --time=30:00:00

set -euo pipefail


# eval_splicevi.sh
#
# Eval-only Slurm driver for SPLICEVI.
#
# For each trained model directory, this script:
#   1. Loads the TRAIN and TEST MuData
#   2. Loads the trained SPLICEVI model from disk
#   3. Runs the requested evaluation blocks (UMAP, clustering, metrics, imputation)
#   4. Saves evaluation figures/CSVs under a per-run directory
#
# This script is designed to be independent of the training script.

# Submit with:
#   sbatch eval_splicevi.sh

#######################################
# USER CONFIGURATION
#######################################

# 1) Data paths
TRAIN_MDATA_PATH="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025/train_70_30_model_ready_combined_gene_expression_aligned_splicing_20251009_024406.h5mu"
TEST_MDATA_PATH="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025/test_30_70_model_ready_combined_gene_expression_aligned_splicing_20251009_024406.h5mu"

# Optional mapping CSV (can be empty if not used)
MAPPING_CSV="/gpfs/commons/home/svaidyanathan/repos/multivi_tools_splicing/multivi_splice_utils/runfiles/tissue_celltype_mapping.csv"

# Optional masked TEST MuData paths for imputation
MASKED_TEST_MDATA_PATHS="\
/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025/MASKED_25_PERCENT_test_30_70_model_ready_combined_gene_expression_aligned_splicing_20251009_024406.h5mu \
/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025/MASKED_50_PERCENT_test_30_70_model_ready_combined_gene_expression_aligned_splicing_20251009_024406.h5mu \
/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025/MASKED_75_PERCENT_test_30_70_model_ready_combined_gene_expression_aligned_splicing_20251009_024406.h5mu"

# 2) Model directories to evaluate (ONE JOB PER MODEL_DIR)
MODEL_DIRS=(
  "/gpfs/commons/home/svaidyanathan/splice_vi_partial_vae_sweep/batch_20251112_131914/mouse_trainandtest_REAL_cd=32_mn=50000_ld=25_lr=1e-5_0_scatter_PartialEncoderEDDI_pool=mean/models"
  "/gpfs/commons/home/svaidyanathan/splice_vi_partial_vae_sweep/batch_20251112_131914/mouse_trainandtest_REAL_cd=32_mn=50000_ld=25_lr=1e-5_2_scatter_PartialEncoderEDDI_pool=mean/models"
  "/gpfs/commons/home/svaidyanathan/splice_vi_partial_vae_sweep/batch_20251111_114606/mouse_trainandtest_REAL_cd=32_mn=50000_ld=25_lr=1e-5_0_scatter_PartialEncoderEDDI_pool=mean/models"
)

# 3) Evaluation blocks to run
EVALS=(
  umap
  # clustering
  # train_eval
  # test_eval
  # age_r2_heatmap
  # masked_impute
)

# 4) UMAP and imputation config
UMAP_TOP_N_CELLTYPES=15       # kept for compatibility (currently unused in plotting logic)
IMPUTE_BATCH_SIZE=256         # set to -1 to disable batching (one big batch per masked file)

# List of obs keys to use for coloring TRAIN UMAPs for each latent space
UMAP_OBS_KEYS=(
  "broad_cell_type"
  "medium_cell_type"
  # "tissue"
  # "tissue_celltype"
)

# 5) Conda / script locations
CONDA_BASE="/gpfs/commons/home/svaidyanathan/miniconda3"
ENV_NAME="splicevi-env"
SCRIPT_PATH="/gpfs/commons/home/svaidyanathan/repos/multivi_tools_splicing/multivi_splice_utils/runfiles/eval_splicevi_basic.py"

# 6) W&B configuration (optional)
USE_WANDB=true                        # set to "false" to disable W&B logging
WANDB_PROJECT="MLCB_SUBMISSION"       # required if USE_WANDB=true
WANDB_ENTITY=""                       # optional W&B entity (team)
WANDB_GROUP="splicevi_eval"           # optional W&B group name
WANDB_RUN_NAME_PREFIX="basic_eval"    # prefix for per-model run names
WANDB_LOG_FREQ=1000                   # how often to log from wandb.watch

# 7) Output directory for eval runs
BASE_RUN_DIR="/gpfs/commons/home/svaidyanathan/splice_vi_partial_vae_eval"
TS=$(date +"%Y%m%d_%H%M%S")
BATCH_RUN_DIR="${BASE_RUN_DIR}/batch_${TS}"
mkdir -p "${BATCH_RUN_DIR}"

echo "=================================================================="
echo "[BATCH] SPLICEVI basic eval batch"
echo "[BATCH] TRAIN_MDATA_PATH = ${TRAIN_MDATA_PATH}"
echo "[BATCH] TEST_MDATA_PATH  = ${TEST_MDATA_PATH}"
echo "[BATCH] MAPPING_CSV      = ${MAPPING_CSV}"
echo "[BATCH] MASKED_TEST_MDATA_PATHS:"
for p in ${MASKED_TEST_MDATA_PATHS}; do
  echo "         - ${p}"
done
echo "[BATCH] MODEL_DIRS:"
for m in "${MODEL_DIRS[@]}"; do
  echo "         - ${m}"
done
echo "[BATCH] UMAP_OBS_KEYS:"
for k in "${UMAP_OBS_KEYS[@]}"; do
  echo "         - ${k}"
done
echo "[BATCH] Output batch dir = ${BATCH_RUN_DIR}"
echo "=================================================================="

#######################################
# ENVIRONMENT SETUP (once)
#######################################

echo "[ENV] Activating conda environment '${ENV_NAME}'..."
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
echo "[ENV] Python executable: $(which python)"
echo "[ENV] Python version  : $(python -V)"
echo

#######################################
# BUILD W&B ARG STRING (shared)
#######################################

WANDB_ARGS=""
if [ "${USE_WANDB}" = true ]; then
  echo "[W&B] W&B logging enabled for all eval jobs."
  if [ -z "${WANDB_PROJECT}" ]; then
    echo "[W&B] ERROR: USE_WANDB=true but WANDB_PROJECT is empty."
    exit 1
  fi
else
  echo "[W&B] W&B logging disabled."
fi
echo

#######################################
# SUBMIT ONE JOB PER MODEL_DIR
#######################################

EVALS_JOINED="${EVALS[*]}"
UMAP_OBS_KEYS_JOINED="${UMAP_OBS_KEYS[*]}"

job_index=0
for MODEL_DIR in "${MODEL_DIRS[@]}"; do
  ((job_index++))

  MODEL_BASENAME=$(basename "${MODEL_DIR}")
  RUN_NAME="eval_${MODEL_BASENAME}_${TS}"

  # Per-model directories
  RUN_SUBDIR="${BATCH_RUN_DIR}/run_$(printf '%02d' "${job_index}")"
  mkdir -p "${RUN_SUBDIR}"

  JOB_NAME="eval_${MODEL_BASENAME}"
  JOB_DIR="${RUN_SUBDIR}/${JOB_NAME}"
  mkdir -p "${JOB_DIR}/figures"
  FIG_DIR="${JOB_DIR}/figures"

  echo "------------------------------------------------------------------"
  echo "[JOB ${job_index}] Submitting eval for model: ${MODEL_DIR}"
  echo "[JOB ${job_index}] JOB_NAME      = ${JOB_NAME}"
  echo "[JOB ${job_index}] JOB_DIR       = ${JOB_DIR}"
  echo "[JOB ${job_index}] FIG_DIR       = ${FIG_DIR}"
  echo "------------------------------------------------------------------"

  # Build per-job W&B args (run name can be unique per model)
  PER_JOB_WANDB_ARGS=""
  if [ "${USE_WANDB}" = true ]; then
    PER_JOB_WANDB_ARGS+=" --use_wandb"
    PER_JOB_WANDB_ARGS+=" --wandb_project ${WANDB_PROJECT}"
    if [ -n "${WANDB_ENTITY}" ]; then
      PER_JOB_WANDB_ARGS+=" --wandb_entity ${WANDB_ENTITY}"
    fi
    if [ -n "${WANDB_GROUP}" ]; then
      PER_JOB_WANDB_ARGS+=" --wandb_group ${WANDB_GROUP}"
    fi
    WANDB_RUN_NAME="${WANDB_RUN_NAME_PREFIX}_${MODEL_BASENAME}"
    PER_JOB_WANDB_ARGS+=" --wandb_run_name ${WANDB_RUN_NAME}"
    PER_JOB_WANDB_ARGS+=" --wandb_log_freq ${WANDB_LOG_FREQ}"
  fi

  # Launch the eval as a child Slurm job
  sbatch \
    --job-name="${JOB_NAME}" \
    --output="${JOB_DIR}/slurm_%j.out" \
    --error="${JOB_DIR}/slurm_%j.err" \
    --partition=gpu \
    --gres=gpu:1 \
    --mem=230G \
    --time=30:00:00 \
    --wrap "$(cat <<EOF
#!/usr/bin/env bash
set -euo pipefail

echo "[EVAL] Starting eval for model: ${MODEL_DIR}"
echo "[EVAL] TRAIN_MDATA_PATH = ${TRAIN_MDATA_PATH}"
echo "[EVAL] TEST_MDATA_PATH  = ${TEST_MDATA_PATH}"
echo "[EVAL] FIG_DIR          = ${FIG_DIR}"
echo "[EVAL] MAPPING_CSV      = ${MAPPING_CSV}"
echo "[EVAL] EVALS            = ${EVALS_JOINED}"
echo "[EVAL] UMAP_OBS_KEYS    = ${UMAP_OBS_KEYS_JOINED}"
echo "[EVAL] IMPUTE_BATCH_SIZE= ${IMPUTE_BATCH_SIZE}"
echo "[EVAL] UMAP_TOP_N_CT    = ${UMAP_TOP_N_CELLTYPES}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

python "${SCRIPT_PATH}" \
  --train_mdata_path "${TRAIN_MDATA_PATH}" \
  --test_mdata_path "${TEST_MDATA_PATH}" \
  --model_dir "${MODEL_DIR}" \
  --fig_dir "${FIG_DIR}" \
  --mapping_csv "${MAPPING_CSV}" \
  --impute_batch_size ${IMPUTE_BATCH_SIZE} \
  --umap_top_n_celltypes ${UMAP_TOP_N_CELLTYPES} \
  --umap_obs_keys ${UMAP_OBS_KEYS_JOINED} \
  --evals ${EVALS_JOINED} \
  ${MASKED_TEST_MDATA_PATHS:+--masked_test_mdata_paths ${MASKED_TEST_MDATA_PATHS}} \
  ${PER_JOB_WANDB_ARGS}
EOF
)"

done

echo
echo "=================================================================="
echo "[BATCH] All eval jobs submitted."
echo "[BATCH] Monitor with: squeue -u \$(whoami)"
echo "[BATCH] Batch outputs in: ${BATCH_RUN_DIR}"
echo "=================================================================="
