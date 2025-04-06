#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1                # 1 computer nodes
#SBATCH --ntasks-per-node=1      # 1 MPI tasks on EACH NODE
#SBATCH --gres=gpu:2             # Using 2 GPU card
#SBATCH --mem=100GB              # Request memory
#SBATCH --time=2-12:00:00        # Time limit day-hrs:min:sec
#SBATCH --output=./log/aroma/gpujob_%j.log   # Standard output
#SBATCH --error=./log/aroma/gpujob_%j.err    # Standard error log

# conda init bash
# source ~/.bashrc
# conda activate new_lora

# echo "=== Environment Information ==="
# nvidia-smi
# echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
# echo "SLURM_JOB_ID: $SLURM_JOB_ID"
# echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
# echo "==========================="

bash run_llama_commonsense_aroma.sh