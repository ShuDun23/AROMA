#!/bin/bash

NPROC_PER_NODE=1

RANDOM_PORT=$(shuf -i 20000-65000 -n 1)

MODEL_NAME_OR_PATH="./roberta-base"
TASK_NAME="mrpc"
OUTPUT_DIR="./experiments/glue/"
NUM_TRAIN_EPOCHS=52
NUM_TRAINING_STEPS=3000
MAX_SEQ_LENGTH=256
PER_DEVICE_TRAIN_BATCH_SIZE=64
PER_DEVICE_EVAL_BATCH_SIZE=64
GRADIENT_ACCUMULATION_STEPS=1
LEARNING_RATE=1e-4
LORA_R=1
LORA_ALPHA=4
RELORA=200
CYCLE_LENGTH=200
EVAL_STEPS=200
SAVE_STEPS=200
SCHEDULER_TYPE="cosine_restarts"
RESET_OPTIMIZER_ON_RELORA=True
FIRST_WARMUP_STEPS=100
RESTART_WARMUP_STEPS=50
MIN_LR_RATIO=0.1
DO_TRAIN=True
DO_EVAL=True
DO_PREDICT=False
EVAL_STRATEGY="steps"
SAVE_STRATEGY="steps"
SAVE_TOTAL_LIMIT=1
REMOVE_UNUSED_COLUMNS=True
IGNORE_MISMATCHED_SIZES=True
LOGGING_STEPS=10
CONVERGENCE_THRESHOLD=0.001
CHECK_CONVERGENCE=True
CONVERGENCE_WINDOW=1
CONVERGENCE_PATIENCE=1
LORA_CHECK_FREQUENCY=10
MAX_STEPS_BEFORE_RESET=200
LORA_CHANGE_THRESHOLD=0.1

NUM_EXPERIMENTS=3

results=()

declare -a results
for ((i=0; i<$NUM_EXPERIMENTS; i++))
do
    SEED=$i

    torchrun --nproc_per_node=$NPROC_PER_NODE \
        --master_port=$RANDOM_PORT \
        run_glue_ar1lora.py \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --task_name $TASK_NAME \
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
        --eval_steps $EVAL_STEPS \
        --save_steps $SAVE_STEPS \
        --scheduler_type $SCHEDULER_TYPE \
        --reset_optimizer_on_relora $RESET_OPTIMIZER_ON_RELORA \
        --first_warmup_steps $FIRST_WARMUP_STEPS \
        --restart_warmup_steps $RESTART_WARMUP_STEPS \
        --min_lr_ratio $MIN_LR_RATIO \
        --do_train $DO_TRAIN \
        --do_eval $DO_EVAL \
        --do_predict $DO_PREDICT \
        --eval_strategy $EVAL_STRATEGY \
        --save_strategy $SAVE_STRATEGY \
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
        --lora_change_threshold $LORA_CHANGE_THRESHOLD

    output_dirs=($(ls -td $OUTPUT_DIR/$TASK_NAME/output/*/))
    accuracy=$(cat "${output_dirs[$i]}/eval_results.json" | grep -o '"eval_accuracy": [0-9.]*' | cut -d' ' -f2)
    results+=($accuracy)
    
    echo "Experiment $i accuracy: $accuracy (from ${output_dirs[$i]})"
done

mean=$(echo "${results[@]}" | awk '{for(i=1;i<=NF;i++) sum+=$i; print sum/NF}')
std=$(echo "${results[@]}" | awk -v mean=$mean '{for(i=1;i<=NF;i++) sum+=($i-mean)^2; print sqrt(sum/NF)}')

echo "Mean Accuracy: $mean"
echo "Standard Deviation: $std"