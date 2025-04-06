#!/bin/bash

NPROC_PER_NODE=1

RANDOM_PORT=$(shuf -i 20000-65000 -n 1)

MODEL_NAME_OR_PATH="path/to/your/roberta-base"
TASK_NAME="qnli"
OUTPUT_DIR="path/to/your/experiments/glue/"
NUM_TRAIN_EPOCHS=10
NUM_TRAINING_STEPS=30000
MAX_SEQ_LENGTH=256
PER_DEVICE_TRAIN_BATCH_SIZE=32
PER_DEVICE_EVAL_BATCH_SIZE=32
GRADIENT_ACCUMULATION_STEPS=1
LORA_R=1
LORA_ALPHA=4
T_IN=1000
CYCLE_LENGTH=1000
SAVE_STEPS=2000
SCHEDULER_TYPE="cosine_restarts"
RESET_OPTIMIZER_ON_RELORA=True
FIRST_WARMUP_STEPS=200
RESTART_WARMUP_STEPS=100
MIN_LR_RATIO=0.1
DO_TRAIN=True
DO_EVAL=True
DO_PREDICT=False
EVAL_STRATEGY="epoch"
SAVE_STRATEGY="steps"
SAVE_TOTAL_LIMIT=1
REMOVE_UNUSED_COLUMNS=True
IGNORE_MISMATCHED_SIZES=True
LOGGING_STEPS=10
CONVERGENCE_THRESHOLD=0.005
CHECK_CONVERGENCE=True
CONVERGENCE_WINDOW=1
CONVERGENCE_PATIENCE=1
LORA_CHECK_FREQUENCY=10
MAX_STEPS_BEFORE_RESET=2000

LEARNING_RATES=1e-4
LORA_CHANGE_THRESHOLDS=0.05

SEED=0

torchrun --nproc_per_node=$NPROC_PER_NODE \
    --master_port=$RANDOM_PORT \
    run_glue_aroma.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --task_name $TASK_NAME \
    --output_dir $CURRENT_OUTPUT_DIR \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --num_training_steps $NUM_TRAINING_STEPS \
    --max_seq_length $MAX_SEQ_LENGTH \
    --seed $SEED \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $lr \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --T_in $T_IN \
    --cycle_length $CYCLE_LENGTH \
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
    --lora_change_threshold $threshold