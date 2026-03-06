#!/bin/bash
# Setup conda environment for ParaRNN training on Marlowe
set -e

# Create environment
conda create -n pararnn python=3.10 -y
conda activate pararnn

# Install PyTorch with CUDA
pip install torch torchvision torchaudio

# Install pararnn (with CUDA extensions)
cd /users/xavier18/ml-pararnn
pip install -e . --no-build-isolation

# Install training dependencies
pip install \
    transformers \
    datasets \
    tokenizers \
    wandb \
    hydra-core \
    omegaconf \
    lm-eval \
    mamba-ssm \
    causal-conv1d \
    numpy \
    matplotlib

# Login to wandb (do this interactively)
echo "Run 'wandb login' to authenticate"

# Make the wandb project private
python -c "
import wandb
api = wandb.Api()
try:
    project = api.project('paraRNN', entity='xavier_gonzalez')
    # wandb projects are private by default when created via init
    print('Project paraRNN exists or will be created on first run')
except Exception as e:
    print(f'Project will be created on first wandb.init call: {e}')
"

echo "Environment setup complete!"
