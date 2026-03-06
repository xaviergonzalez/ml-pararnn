#!/bin/bash
#SBATCH --job-name=pararnn_build_train
#SBATCH -A marlowe-m000120-pm05
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -G 8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH -t 2-00:00:00
#SBATCH --output=/projects/m000120/xavier18/logs/build_train_%j.out
#SBATCH --error=/projects/m000120/xavier18/logs/build_train_%j.err
#SBATCH --mail-user=xavier18@stanford.edu
#SBATCH --mail-type=ALL

set -e

eval "$(conda shell.bash hook)"
conda activate pararnn

# Verify GPU access
nvidia-smi
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"

# Build pararnn CUDA extensions
echo "=== Building pararnn CUDA extensions ==="
cd /users/xavier18/ml-pararnn
pip install -e . --no-build-isolation 2>&1 | tail -10

# Verify pararnn installation
python -c "
import torch
print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
from pararnn.rnn_cell.gru_diag_mh import GRUDiagMH, GRUDiagMHConfig
print('ParaRNN imports OK')
"

# Now train the requested model
MODEL=${1:-paragru}
echo "=== Training $MODEL ==="

cd /users/xavier18/ml-pararnn/training

# Model-specific overrides
case $MODEL in
    paragru)
        LR=0.003
        ;;
    paralstm)
        LR=0.003
        ;;
    mamba2)
        LR=0.005
        ;;
    transformer)
        LR=0.002
        ;;
    *)
        echo "Unknown model: $MODEL"
        exit 1
        ;;
esac

torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    train.py \
    model=$MODEL \
    optimizer.lr=$LR \
    optimizer.weight_decay=1e-4 \
    seed=42
