#!/usr/bin/env bash
# masked_mask_test_mudata.sh
set -euo pipefail

PY="preprocess/make_masked_test_mudata.py"

FILES=(
  #  "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/MODEL_INPUT/102025/test_30_70_model_ready_combined_gene_expression_aligned_splicing_data_20251009_023419.h5mu"
  "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025/train_70_30_model_ready_combined_gene_expression_aligned_splicing_20251009_024406.h5mu"
)
FRACTIONS=(0.25 0.50 0.75)
SEED=0
N_JOBS=4          # threads passed to the masking script
CPUS_PER_TASK=4   # request matching CPUs from Slurm

MEM="500G"
PARTITION="cpu,dev,bigmem"
TIME="2-00:00:00"

for f in "${FILES[@]}"; do
  [[ -f "$f" ]] || { echo "Missing input: $f" >&2; continue; }
  bn=$(basename "$f")
  dir=$(dirname "$f")

  for frac in "${FRACTIONS[@]}"; do
    pct=$(python -c "print(int(round($frac*100)))")   # or the awk line
    job="Mask:${bn}:${pct}"
    cmd="python $PY --inputs \"$f\" --fractions $frac --seed $SEED --n-jobs $N_JOBS"

    echo "Submitting: $job"
    sbatch \
        --wrap="$cmd" \
        --cpus-per-task="$CPUS_PER_TASK" \
        --mem="$MEM" \
        --partition="$PARTITION" \
        --time="$TIME" \
        --output="${dir}/slurm_${bn%.h5mu}_${pct}.out" \
        --error="${dir}/slurm_${bn%.h5mu}_${pct}.err" \
        -J "$job"
    done

done
