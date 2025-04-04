#!/bin/bash

ngpu_per_node=2


RANDOM_PORT=$(shuf -i 20000-65000 -n 1)

MODEL_NAME_OR_PATH="./llama3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920"
OUTPUT_DIR="./experiments/"
DATASET_DIR="/home/hnsheng2/scratch/R1LoRA-FT-backup/data/commonsense170k"
DATASET_NAME="commonsense"
NUM_TRAIN_EPOCHS=20
NUM_TRAINING_STEPS=100000
MAX_SEQ_LENGTH=256
PER_DEVICE_TRAIN_BATCH_SIZE=4
PER_DEVICE_EVAL_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=1e-4
LORA_R=8
LORA_ALPHA=16
RELORA=2000
CYCLE_LENGTH=2000
SCHEDULER_TYPE="cosine_restarts"
RESET_OPTIMIZER_ON_RELORA=True
FIRST_WARMUP_STEPS=100
RESTART_WARMUP_STEPS=50
MIN_LR_RATIO=0.1
DO_TRAIN=True
DO_EVAL=False
DO_PREDICT=False
VAL_SET_SIZE=0
# EVAL_STRATEGY="steps"
# EVAL_STEPS=100
SAVE_STRATEGY="steps"
SAVE_STEPS=10000
SAVE_TOTAL_LIMIT=1
REMOVE_UNUSED_COLUMNS=False
IGNORE_MISMATCHED_SIZES=True
LOGGING_STEPS=10

CONVERGENCE_THRESHOLD=1e-3
CHECK_CONVERGENCE=True
CONVERGENCE_WINDOW=1
CONVERGENCE_PATIENCE=1

LORA_CHECK_FREQUENCY=200
MAX_STEPS_BEFORE_RESET=2000
LORA_CHANGE_THRESHOLD=0.1

# REPORT_TO="wandb"
# WANDB_PROJECT="ar1lora_llama3-8B"
# WANDB_NAME="experiment_lr_1e-4"
# WANDB_TAGS="llama,relora,test"
SAVE_SAFETENSORS=False
BF16=True

NUM_EXPERIMENTS=1

results=()

declare -a results
for ((i=0; i<$NUM_EXPERIMENTS; i++))
do
    SEED=$i

    torchrun --nproc_per_node ${ngpu_per_node} \
        --master_port=$RANDOM_PORT \
        run_llama_commonsense_ar1lora.py \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --dataset_dir $DATASET_DIR \
        --dataset_name $DATASET_NAME \
        --output_dir $OUTPUT_DIR \
        --num_train_epochs $NUM_TRAIN_EPOCHS \
        --num_training_steps $NUM_TRAINING_STEPS \
        --max_seq_length $MAX_SEQ_LENGTH \
        --seed $SEED \
        --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
        --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --learning_rate $LEARNING_RATE \
        --lora_r $LORA_R \
        --lora_alpha $LORA_ALPHA \
        --relora $RELORA \
        --cycle_length $CYCLE_LENGTH \
        --scheduler_type $SCHEDULER_TYPE \
        --reset_optimizer_on_relora $RESET_OPTIMIZER_ON_RELORA \
        --first_warmup_steps $FIRST_WARMUP_STEPS \
        --restart_warmup_steps $RESTART_WARMUP_STEPS \
        --min_lr_ratio $MIN_LR_RATIO \
        --do_train $DO_TRAIN \
        --do_eval $DO_EVAL \
        --do_predict $DO_PREDICT \
        --val_set_size $VAL_SET_SIZE \
        --save_strategy $SAVE_STRATEGY \
        --save_steps $SAVE_STEPS \
        --save_total_limit $SAVE_TOTAL_LIMIT \
        --remove_unused_columns $REMOVE_UNUSED_COLUMNS \
        --ignore_mismatched_sizes $IGNORE_MISMATCHED_SIZES \
        --logging_steps $LOGGING_STEPS \
        --convergence_threshold $CONVERGENCE_THRESHOLD \
        --check_convergence $CHECK_CONVERGENCE \
        --convergence_window $CONVERGENCE_WINDOW \
        --convergence_patience $CONVERGENCE_PATIENCE \
        --lora_check_frequency $LORA_CHECK_FREQUENCY \
        --max_steps_before_reset $MAX_STEPS_BEFORE_RESET \
        --lora_change_threshold $LORA_CHANGE_THRESHOLD \
        --save_safetensors $SAVE_SAFETENSORS \
        --bf16 $BF16

    output_dirs=($(ls -td $OUTPUT_DIR/$TASK_NAME/output/*/))
    accuracy=$(cat "${output_dirs[$i]}/eval_results.json" | grep -o '"eval_accuracy": [0-9.]*' | cut -d' ' -f2)
    results+=($accuracy)
    
    echo "Experiment $i accuracy: $accuracy (from ${output_dirs[$i]})"
done

mean=$(echo "${results[@]}" | awk '{for(i=1;i<=NF;i++) sum+=$i; print sum/NF}')
std=$(echo "${results[@]}" | awk -v mean=$mean '{for(i=1;i<=NF;i++) sum+=($i-mean)^2; print sqrt(sum/NF)}')

echo "Mean Accuracy: $mean"
echo "Standard Deviation: $std"