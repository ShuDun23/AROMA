#!/bin/bash

# ADAPTER="AdaLoRA"
# ADAPTER="LoRA"
ADAPTER="AROMA"
BASE_MODEL="path/to/your/base/model"
# LORA_WEIGHTS="None" # vanilla llama3-8B
# LORA_WEIGHTS="path/to/your/lora/weights"
# LORA_WEIGHTS="path/to/your/adalora/weights"
LORA_WEIGHTS="path/to/your/ar1lora/weights"

SAVE_DIR="path/to/your/save/directory"

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