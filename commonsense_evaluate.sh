#!/bin/bash

# ADAPTER="AdaLoRA"
# ADAPTER="LoRA"
ADAPTER="AR1LoRA"
BASE_MODEL="/home/hnsheng2/scratch/R1LoRA-FT-backup/llama3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920"
# LORA_WEIGHTS="None" # vanilla llama3-8B
# LORA_WEIGHTS="/home/hnsheng2/scratch/R1LoRA-FT-backup/experiments/peft_commonsense_lr_3e-5_llama3-8B_r8"
# LORA_WEIGHTS="/home/hnsheng2/scratch/R1LoRA-FT-backup/experiments/adalora_commonsense_llama3-8B_r8"
LORA_WEIGHTS="/home/hnsheng2/scratch/R1LoRA-FT-backup/experiments/commonsense/output/20250120_100708/checkpoint-105000"

SAVE_DIR="/home/hnsheng2/scratch/R1LoRA-FT-backup/evaluation_results/adalora_llama3-8B_r64"

mkdir -p $SAVE_DIR

for dataset in ARC-Easy openbookqa social_i_qa ARC-Challenge winogrande piqa boolq hellaswag
do
    echo "Start evaluating $dataset"
    
    python -u commonsense_evaluate.py \
        --model LLaMA3 \
        --adapter $ADAPTER \
        --dataset $dataset \
        --batch_size 8 \
        --base_model $BASE_MODEL \
        --lora_weights $LORA_WEIGHTS \
        --save_dir $SAVE_DIR
    
    echo "Completed evaluation for $dataset"
    echo "----------------------------------------"
done