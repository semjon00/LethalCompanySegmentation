#!/bin/bash
#SBATCH --job-name=lts_submission
#SBATCH --time=02:00:00
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH--mail-user=semjon00@ut.ee
#SBATCH--mail-type=ALL
#SBATCH --partition=gpu
#SBATCH --exclude=falcon2
#SBATCH --gres=gpu:tesla:1

cd /gpfs/space/home/semjon00/LethalCompanySegmentation/Solution

module load any/python/3.8.3-conda
conda activate transformers-course
pip install pycocotools

srun python -u submission.py themodel.pt
# sbatch hpc_submission.sh
# squeue -u semjon00
# scancel
