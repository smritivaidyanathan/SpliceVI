#!/usr/bin/env bash
#
# train_splicevi.sh
#
# Minimal Slurm job script to:
#   1. Load a MuData file
#   2. Set up a SPLICEVI model using chosen hyperparameters
#   3. Train the model
#   4. Save the trained model to a specified directory
#
# This script is intentionally focused on TRAINING ONLY.
# No evaluation, UMAP plots, clustering, or imputation.
#
# Submit with:
#   sbatch train_splicevi.sh
#

#SBATCH --job-name=splicevi_train_basic
#SBATCH --output=/gpfs/commons/home/svaidyanathan/logs/splicevi_train_basic_%j.out
#SBATCH --error=/gpfs/commons/home/svaidyanathan/logs/splicevi_train_basic_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=230G
#SBATCH --time=24:00:00

set -euo pipefail

#######################################
# USER CONFIGURATION
#######################################

# 1) Paths
TRAIN_MDATA_PATH="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025/train_70_30_model_ready_combined_gene_expression_aligned_splicing_20251009_024406.h5mu"

# Base directory where trained models will be stored
MODEL_DIR_BASE="models"

# Location of the training Python script (in your repo)
SCRIPT_PATH="src/train_splicevi.py"

# 2) Conda / environment
CONDA_BASE="/gpfs/commons/home/svaidyanathan/miniconda3"
ENV_NAME="splicevi-env"

# 3) Core hyperparameters
MAX_EPOCHS=500          # Total training epochs
LR=1e-5                 # Learning rate
BATCH_SIZE=256          # Minibatch size
N_EPOCHS_KL_WARMUP=100  # KL warmup epochs
N_LATENT=25             # Latent dimensionality
DROPOUT_RATE=0.01       # Model dropout rate
SPLICING_LOSS_TYPE="dirichlet_multinomial"  # binomial | beta_binomial | dirichlet_multinomial

# 4) Optional architecture knobs (only passed if you care to set them)
FORWARD_STYLE="scatter"              # per-cell | batched | scatter
MAX_NOBS=50000                       # Subsampling cap for number of observations
ENCODER_HIDDEN_DIM=128
H_HIDDEN_DIM=64
ATSE_EMBEDDING_DIMENSION=16
SPLICING_ARCHITECTURE="partial"      # vanilla | partial
EXPRESSION_ARCHITECTURE="linear"     # vanilla | linear

WEIGHT_DECAY=1e-3
EARLY_STOPPING_PATIENCE=50
LR_SCHEDULER_TYPE="plateau"          # plateau | step
LR_FACTOR=0.5
LR_PATIENCE=30
STEP_SIZE=100
GRADIENT_CLIPPING=true
GRADIENT_CLIPPING_MAX_NORM=5.0

# 5) Optional: W&B configuration
USE_WANDB=true                       # Set to "false" to disable W&B logging
WANDB_PROJECT="MLCB_SUBMISSION"      # Required if USE_WANDB=true
WANDB_ENTITY=""                      # Optional: W&B entity (team). Leave empty if not used.
WANDB_GROUP="splicevi_basic"    # Optional group label
WANDB_RUN_NAME_PREFIX="basic_train"  # Prefix for W&B run names
WANDB_LOG_FREQ=1000                  # Log frequency for wandb.watch


#######################################
# DERIVED SETTINGS
#######################################

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_NAME="splicevi_basic_${TIMESTAMP}"

MODEL_DIR="${MODEL_DIR_BASE}/${RUN_NAME}"
mkdir -p "${MODEL_DIR}"

echo "=================================================================="
echo "[JOB] SPLICEVI basic training job"
echo "[JOB] Slurm job ID          : ${SLURM_JOB_ID:-N/A}"
echo "[JOB] Run name              : ${RUN_NAME}"
echo "[JOB] Training MuData path  : ${TRAIN_MDATA_PATH}"
echo "[JOB] Model output directory: ${MODEL_DIR}"
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
  WANDB_RUN_NAME="${WANDB_RUN_NAME_PREFIX}_${RUN_NAME}"
  WANDB_ARGS+=" --wandb_run_name ${WANDB_RUN_NAME}"
  WANDB_ARGS+=" --wandb_log_freq ${WANDB_LOG_FREQ}"
else
  echo "[W&B] W&B logging disabled for this run."
fi
echo


#######################################
# LAUNCH TRAINING
#######################################

echo "[RUN] Launching SPLICEVI training script..."
set -x

python "${SCRIPT_PATH}" \
  --train_mdata_path "${TRAIN_MDATA_PATH}" \
  --model_dir "${MODEL_DIR}" \
  --max_epochs "${MAX_EPOCHS}" \
  --lr "${LR}" \
  --batch_size "${BATCH_SIZE}" \
  --n_epochs_kl_warmup "${N_EPOCHS_KL_WARMUP}" \
  --n_latent "${N_LATENT}" \
  --dropout_rate "${DROPOUT_RATE}" \
  --splicing_loss_type "${SPLICING_LOSS_TYPE}" \
  --max_nobs "${MAX_NOBS}" \
  --forward_style "${FORWARD_STYLE}" \
  --encoder_hidden_dim "${ENCODER_HIDDEN_DIM}" \
  --h_hidden_dim "${H_HIDDEN_DIM}" \
  --atse_embedding_dimension "${ATSE_EMBEDDING_DIMENSION}" \
  --splicing_architecture "${SPLICING_ARCHITECTURE}" \
  --expression_architecture "${EXPRESSION_ARCHITECTURE}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --early_stopping_patience "${EARLY_STOPPING_PATIENCE}" \
  --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
  --lr_factor "${LR_FACTOR}" \
  --lr_patience "${LR_PATIENCE}" \
  --step_size "${STEP_SIZE}" \
  --gradient_clipping "${GRADIENT_CLIPPING}" \
  --gradient_clipping_max_norm "${GRADIENT_CLIPPING_MAX_NORM}" \
  ${WANDB_ARGS}

set +x

echo
echo "[DONE] SPLICEVI basic training job finished."
echo "[DONE] Trained model saved to: ${MODEL_DIR}"
