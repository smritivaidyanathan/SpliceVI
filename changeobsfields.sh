#!/usr/bin/env bash
# changeobsfields.sh
# Thin wrapper: activate env then call the Python helper.
# Edit the TEMPLATE and MUDATAS defaults below for quick reuse.

#SBATCH --job-name=change_obs_fields
#SBATCH --partition=cpu,dev,bigmem
#SBATCH --cpus-per-task=4
#SBATCH --mem=300G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/commons/home/svaidyanathan/repos/SpliceVI/logs/changeobsfields_%j.out
#SBATCH --error=/gpfs/commons/home/svaidyanathan/repos/SpliceVI/logs/changeobsfields_%j.err

set -euo pipefail

DEFAULT_TEMPLATE="/gpfs/commons/groups/knowles_lab/Smriti/splice_adata_for_figures_mouse_foundation.h5ad"
MUDATAS=(
  "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025/MASKED_25_PERCENT_test_30_70_model_ready_combined_gene_expression_aligned_splicing_20251009_024406.h5mu" 
  "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025/MASKED_50_PERCENT_test_30_70_model_ready_combined_gene_expression_aligned_splicing_20251009_024406.h5mu" 
  "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025/MASKED_75_PERCENT_test_30_70_model_ready_combined_gene_expression_aligned_splicing_20251009_024406.h5mu"
)

TAG="_UPDATEDOBS"
ALLOW_SUBSET="--allow-subset"
LOG_DIR="/gpfs/commons/home/svaidyanathan/repos/SpliceVI/logs"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: changeobsfields.sh --template TEMPLATE_H5AD --mudata-paths PATH1 [PATH2 ...] [--tag _UPDATEDOBS] [--skip-cell-check] [--allow-subset]

If no arguments are provided, the script uses the defaults set near the top.
EOF
  exit 0
fi

CONDA_BASE="/gpfs/commons/home/svaidyanathan/miniconda3"
ENV_NAME="splicevi-env"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

TS=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/changeobsfields_${TS}.log"

if [[ $# -gt 0 ]]; then
  CMD=(python preprocess/changeobsfields.py "$@")
else
  CMD=(python preprocess/changeobsfields.py \
    --template "${DEFAULT_TEMPLATE}" \
    --mudata-paths "${MUDATAS[@]}" \
    --tag "${TAG}" \
    ${ALLOW_SUBSET})
fi

echo "[LOG] Writing to ${LOG_FILE}"
echo "[RUN] ${CMD[*]}"

"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
status=${PIPESTATUS[0]}

echo "[DONE] Updated obs fields."
exit $status
