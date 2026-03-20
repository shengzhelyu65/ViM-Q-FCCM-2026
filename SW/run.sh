#!/bin/bash

# ============================================================================
# PATH CONFIGURATION - Defaults come from project_config.sh
# ============================================================================

# Base directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load project-level configuration if available.
. "$PROJECT_ROOT/project_config.sh"

export DATA_PATH="${DATA_PATH:-$PROJECT_ROOT/Data}"
export MODEL_WEIGHTS_BASE="${MODEL_WEIGHTS_BASE:-$PROJECT_ROOT/Checkpoints}"
export ACT_SCALES_BASE="${ACT_SCALES_BASE:-$PROJECT_ROOT/Checkpoints/act_scales}"

# Model-specific checkpoint paths
export VIM_T_PTH="${MODEL_WEIGHTS_BASE}/tiny/vim_t_midclstok_76p1acc.pth"
export VIM_S_PTH="${MODEL_WEIGHTS_BASE}/small/vim_s_midclstok_80p5acc.pth"
export VIM_B_PTH="${MODEL_WEIGHTS_BASE}/base/vim_b_midclstok_81p9acc.pth"

# Common settings
export NUM_WORKERS="${NUM_WORKERS:-8}"
export CUDA_DEVICES="${CUDA_DEVICES:-0}"

# ============================================================================
# SCRIPT EXECUTION
# ============================================================================

# Use script directory as working directory for relative paths
cd "$SCRIPT_DIR" || exit 1

# Baseline scripts
# bash scripts/vim-t/baseline-vim-t.sh
# bash scripts/vim-s/baseline-vim-s.sh
# bash scripts/vim-b/baseline-vim-b.sh

# Main PTQ evaluation scripts
bash scripts/vim-t/ptq-vim-t.sh
# bash scripts/vim-s/ptq-vim-s.sh
# bash scripts/vim-b/ptq-vim-b.sh

# Export model scripts
bash scripts/vim-t/export-model-vim-t.sh
