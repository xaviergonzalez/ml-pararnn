#!/bin/bash
# Launch all 4 training jobs on Marlowe
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Launching ParaGRU 1B training..."
JOB1=$(sbatch --parsable ${SCRIPT_DIR}/train_paragru.sh)
echo "  Job ID: $JOB1"

echo "Launching ParaLSTM 1B training..."
JOB2=$(sbatch --parsable ${SCRIPT_DIR}/train_paralstm.sh)
echo "  Job ID: $JOB2"

echo "Launching Mamba2 1B training..."
JOB3=$(sbatch --parsable ${SCRIPT_DIR}/train_mamba2.sh)
echo "  Job ID: $JOB3"

echo "Launching Transformer 1B training..."
JOB4=$(sbatch --parsable ${SCRIPT_DIR}/train_transformer.sh)
echo "  Job ID: $JOB4"

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo "After training completes, run evaluation:"
echo "  sbatch ${SCRIPT_DIR}/eval_all.sh"
