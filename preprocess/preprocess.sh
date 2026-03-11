#!/bin/sh
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -n 1
#SBATCH --mem=400G
#SBATCH -t 24:00:00
#SBATCH -o logs/preprocess%j.out


module load python/3.11


source ../../neuroenv/bin/activate

#cd /users/aiyer51/data/aiyer51/scNeuro/preprocess


python -u qc_norm_rawdata.py

python -u merge_donors.py

python -u HVG.py