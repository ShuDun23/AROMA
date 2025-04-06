#!/bin/bash
#SBATCH --partition=stingy
#SBATCH --nodes=1                # 1 computer nodes
#SBATCH --ntasks-per-node=1      # 1 MPI tasks on EACH NODE
#SBATCH --gres=gpu:1             # Using 2 GPU card
#SBATCH --mem=100GB              # Request memory
#SBATCH --time=0-05:00:00        # Time limit day-hrs:min:sec
#SBATCH --output=./log/glue/gpujob_%j.log   # Standard output
#SBATCH --error=./log/glue/gpujob_%j.err    # Standard error log

# conda init bash
# source ~/.bashrc
# conda activate new_lora

# echo "=== Environment Information ==="
# nvidia-smi
# echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
# echo "SLURM_JOB_ID: $SLURM_JOB_ID"
# echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
# echo "==========================="

bash run_for_seed_mrpc.sh
# bash run_for_seed_rte.sh
# bash run_for_seed_cola.sh
# bash run_for_seed_stsb.sh
# bash run_for_seed_sst2.sh
# bash run_for_seed_qnli.sh
# bash run_for_seed_mnli.sh
# bash run_for_seed_qqp.sh
# bash run_for_seed_wnli.sh