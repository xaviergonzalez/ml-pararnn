#!/bin/bash
#SBATCH --job-name=transformer_1B
#SBATCH -A marlowe-m000120-pm05
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -G 8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH -t 2-00:00:00
#SBATCH --output=/projects/m000120/xavier18/logs/transformer_%j.out
#SBATCH --error=/projects/m000120/xavier18/logs/transformer_%j.err
#SBATCH --mail-user=xavier18@stanford.edu
#SBATCH --mail-type=ALL

set -e
mkdir -p /projects/m000120/xavier18/logs

eval "$(conda shell.bash hook)"
conda activate pararnn

cd /users/xavier18/ml-pararnn/training

torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    train.py \
    model=transformer \
    optimizer.lr=0.002 \
    optimizer.weight_decay=1e-4 \
    seed=42
