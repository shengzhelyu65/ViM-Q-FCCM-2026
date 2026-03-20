#!/bin/bash

# Get the script directory before changing directories
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# Initialize log file (clear it at the start)
LOG_FILE="${SCRIPT_DIR}/ptq-vim-t.txt"
> "$LOG_FILE"  # Clear the log file

# Logging function
log_output() {
    local script_name=$(basename "$0" .sh)
    local log_file="${SCRIPT_DIR}/${script_name}.txt"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "=== Log started at $timestamp ===" | tee -a "$log_file"
    echo "Script: $0" | tee -a "$log_file"
    echo "Working directory: $(pwd)" | tee -a "$log_file"
    echo "========================================" | tee -a "$log_file"
    
    # Execute the command and log its output
    "$@" 2>&1 | tee -a "$log_file"
    
    local exit_code=${PIPESTATUS[0]}
    local end_timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "========================================" | tee -a "$log_file"
    echo "=== Log ended at $end_timestamp (Exit code: $exit_code) ===" | tee -a "$log_file"
    echo "" | tee -a "$log_file"
    
    return $exit_code
}

set -euo pipefail

MODEL="vim_tiny_pretrain"

# Use environment variables set by run.sh
DATA_PATH="$DATA_PATH"
PTH_PATH="$VIM_T_PTH"
ACT_SCALE_DIR="$ACT_SCALES_BASE"
NUM_WORKERS="$NUM_WORKERS"
SCALE_BATCH_SIZE=256
PTQ_BATCH_SIZE=32
CUDA_DEVICES="$CUDA_DEVICES"

# Generate activation scales
mkdir -p "$ACT_SCALE_DIR"

echo "Generating activation scales for $MODEL..."
CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" python a_generate_act_scale.py \
  --resume "$PTH_PATH" \
  --model "$MODEL" \
  --data-path "$DATA_PATH" \
  --batch-size "$SCALE_BATCH_SIZE" \
  --scales-output-path "$ACT_SCALE_DIR"

# PTQ evaluation
echo "[!!!!!] Running PTQ evaluation for $MODEL..."
CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" log_output python main.py --eval \
  --model "$MODEL" \
  --batch-size "$PTQ_BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --data-path "$DATA_PATH" \
  --no_amp \
  --no-pin-mem \
  --resume "$PTH_PATH" \
  --act_scales "$ACT_SCALE_DIR" \
  --ptq \
  --smooth \
  --per-block \
  --weight_bits 4 \
  --block-size 32

echo "PTQ evaluation complete."
