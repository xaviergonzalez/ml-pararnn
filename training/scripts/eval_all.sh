#!/bin/bash
#SBATCH --job-name=pararnn_eval
#SBATCH -A marlowe-m000120-pm05
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH -t 12:00:00
#SBATCH --output=/projects/m000120/xavier18/logs/eval_%j.out
#SBATCH --error=/projects/m000120/xavier18/logs/eval_%j.err
#SBATCH --mail-user=xavier18@stanford.edu
#SBATCH --mail-type=ALL

set -e
mkdir -p /projects/m000120/xavier18/logs

eval "$(conda shell.bash hook)"
conda activate pararnn

cd /users/xavier18/ml-pararnn/training

CKPT_DIR=/projects/m000120/xavier18/pararnn_checkpoints

# Evaluate each model: perplexity + lm-eval + LLE
for model in paragru paralstm mamba2 transformer; do
    echo "=== Evaluating $model ==="
    python evaluate.py \
        model=$model \
        checkpoint_path=${CKPT_DIR}/${model}/final.pt \
        eval_mode=all
done
