# SpliceVI

git clone https://github.com/smritivaidyanathan/SpliceVI

cd SpliceVI

conda create -n splicevi-env python=3.12  # any python 3.11 to 3.13

conda activate splicevi-env

pip install -r requirements.txt

chmod +x train_splicevi.sh

chmod +x eval_splicevi.sh

sbatch train_splicevi.sh

sbatch eval_splicevi.sh