#!/bin/bash
#SBATCH --partition=stingy
#SBATCH --nodes=1                # 1 computer nodes
#SBATCH --ntasks-per-node=1      # 1 MPI tasks on EACH NODE
#SBATCH --gres=gpu:1             # Using 2 GPU card
#SBATCH --mem=100GB              # Request memory
#SBATCH --time=0-5:00:00        # Time limit day-hrs:min:sec
#SBATCH --output=./log/eval/gpujob_%j.log   # Standard output
#SBATCH --error=./log/eval/gpujob_%j.err    # Standard error log

conda init bash
source ~/.bashrc
conda activate new_lora

echo "=== Environment Information ==="
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "==========================="

bash commonsense_evaluate.sh