#!/usr/bin/env bash
# masked_mask_test_mudata.sh
set -euo pipefail

PY="preprocess/make_masked_test_mudata.py"

FILES=(
  "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025/train_70_30_model_ready_combined_gene_expression_aligned_splicing_20251009_024406.h5mu"
)
FRACTIONS=(0.25 0.50 0.75)
SEED=0
BATCH_SIZE=256

# Since the new script is fast, we don't need massive resources per job 
# unless the .h5mu files are enormous (>100GB on disk).
CPUS_PER_TASK=2   
MEM="300G"        # Adjusted based on typical Mudata usage, increase if needed
PARTITION="cpu,dev,bigmem"
TIME="24:00:00"   # The new script should finish much faster than 2 days
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

for f in "${FILES[@]}"; do
  [[ -f "$f" ]] || { echo "Missing input: $f" >&2; continue; }
  bn=$(basename "$f")
  dir=$(dirname "$f")

  for frac in "${FRACTIONS[@]}"; do
    pct=$(python -c "print(int(round($frac*100)))")
    job="Mask_${pct}_${bn:0:10}"
    
    # We pass only ONE file and ONE fraction per sbatch job for maximum parallel efficiency
    cmd="python $PY --inputs \"$f\" --fractions $frac --seed $SEED --batch-size $BATCH_SIZE"

    echo "Submitting: $job"
    sbatch \
        --wrap="$cmd" \
        --cpus-per-task="$CPUS_PER_TASK" \
        --mem="$MEM" \
        --partition="$PARTITION" \
        --time="$TIME" \
        --output="${LOG_DIR}/masking_file_${bn%.h5mu}_${pct}.out" \
        --error="${LOG_DIR}/masking_file_${bn%.h5mu}_${pct}.err" \
        -J "$job"
    done
done
