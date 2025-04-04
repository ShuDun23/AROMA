#!/bin/bash

NPROC_PER_NODE=1

RANDOM_PORT=$(shuf -i 20000-65000 -n 1)

MODEL_NAME_OR_PATH="./roberta-base"
TASK_NAME="rte"
OUTPUT_DIR="./experiments/glue/"
NUM_TRAIN_EPOCHS=62
NUM_TRAINING_STEPS=2400
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
CHECK_CONVERGENCE=True
CONVERGENCE_WINDOW=1
CONVERGENCE_PATIENCE=1
LORA_CHECK_FREQUENCY=10
MAX_STEPS_BEFORE_RESET=200

CONVERGENCE_THRESHOLDS=(0.006)
LORA_CHANGE_THRESHOLDS=(0.05 0.2)
NUM_EXPERIMENTS=3

RESULTS_FILE="grid_search_results_rte.txt"
echo "Grid Search Results" > $RESULTS_FILE
echo "===================" >> $RESULTS_FILE

for conv_threshold in "${CONVERGENCE_THRESHOLDS[@]}"; do
    for lora_threshold in "${LORA_CHANGE_THRESHOLDS[@]}"; do
        echo "Testing CONVERGENCE_THRESHOLD=$conv_threshold, LORA_CHANGE_THRESHOLD=$lora_threshold" | tee -a $RESULTS_FILE
        
        CURRENT_OUTPUT_DIR="${OUTPUT_DIR}/${TASK_NAME}/conv${conv_threshold}_lora${lora_threshold}"
        mkdir -p $CURRENT_OUTPUT_DIR
        
        declare -a results
        
        for ((i=0; i<$NUM_EXPERIMENTS; i++)); do
            echo "Running experiment $i with CONVERGENCE_THRESHOLD=$conv_threshold, LORA_CHANGE_THRESHOLD=$lora_threshold"
            
            SEED=$i
            
            torchrun --nproc_per_node=$NPROC_PER_NODE \
                --master_port=$RANDOM_PORT \
                run_glue_ar1lora.py \
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
                --convergence_threshold $conv_threshold \
                --check_convergence $CHECK_CONVERGENCE \
                --convergence_window $CONVERGENCE_WINDOW \
                --convergence_patience $CONVERGENCE_PATIENCE \
                --lora_check_frequency $LORA_CHECK_FREQUENCY \
                --max_steps_before_reset $MAX_STEPS_BEFORE_RESET \
                --lora_change_threshold $lora_threshold

            output_dirs=($(ls -td $CURRENT_OUTPUT_DIR/output/*/))
            accuracy=$(cat "${output_dirs[0]}/eval_results.json" | grep -o '"eval_accuracy": [0-9.]*' | cut -d' ' -f2)
            results+=($accuracy)
            
            echo "Experiment $i accuracy: $accuracy" | tee -a $RESULTS_FILE
        done

        mean=$(echo "${results[@]}" | awk '{sum=0; for(i=1;i<=NF;i++) sum+=$i; print sum/NF}')
        std=$(echo "${results[@]}" | awk -v mean=$mean '{sum=0; for(i=1;i<=NF;i++) sum+=($i-mean)^2; print sqrt(sum/NF)}')

        echo "Parameters: CONVERGENCE_THRESHOLD=$conv_threshold, LORA_CHANGE_THRESHOLD=$lora_threshold" >> $RESULTS_FILE
        echo "Mean Accuracy: $mean" >> $RESULTS_FILE
        echo "Standard Deviation: $std" >> $RESULTS_FILE
        echo "----------------------------------------" >> $RESULTS_FILE
        
        unset results
        declare -a results
    done
done

echo "Grid search completed. Results saved in $RESULTS_FILE"