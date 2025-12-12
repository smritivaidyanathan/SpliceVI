#!/usr/bin/env bash
#SBATCH --job-name=Mudata_Test_Train_Split
#SBATCH --output=cpu_job_output.txt
#SBATCH --error=cpu_job_error.txt
#SBATCH --partition=bigmem
#SBATCH --mem=500G
#SBATCH --cpus-per-task=1   # Adjust based on how many threads your script uses
#SBATCH --time=8:00:00      # Adjust as needed

# Activate environment (optional)
CONDA_BASE="/gpfs/commons/home/svaidyanathan/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate scvi-env

# Run your Python script
python /gpfs/commons/home/svaidyanathan/repos/multivi_tools_splicing/multivi_splice_utils/runfiles/createtesttrainmudata.py
