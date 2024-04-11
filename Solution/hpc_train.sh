#!/bin/bash
#SBATCH --job-name=lts_training
#SBATCH --time=05:00:00
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
pip install segmentation_models_pytorch

srun python -u training.py
# sbatch hpc_train.sh
# squeue -u semjon00
# scancel
