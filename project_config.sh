#!/bin/bash
# ViM-Q project-level configuration
#
# Edit this file once to configure all HW/SW scripts.

# Resolve repository root from this file location.
VIM_Q_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Core project paths
export VIM_Q_ROOT
export VIM_Q_HW_ROOT="${VIM_Q_HW_ROOT:-$VIM_Q_ROOT/HW}"
export VIM_Q_SW_ROOT="${VIM_Q_SW_ROOT:-$VIM_Q_ROOT/SW}"

# Toolchain paths
# XILINX_PATH should point to the Xilinx version root directory, not settings64.sh.
export XILINX_PATH="/opt/Xilinx/2025.2"
export VIM_Q_VIVADO_SETTINGS="$XILINX_PATH/Vivado/settings64.sh"

# SW data/model paths
export DATA_PATH="${DATA_PATH:-$VIM_Q_ROOT/Data}"
export MODEL_WEIGHTS_BASE="${MODEL_WEIGHTS_BASE:-$VIM_Q_ROOT/Checkpoints}"
export ACT_SCALES_BASE="${ACT_SCALES_BASE:-$VIM_Q_ROOT/Checkpoints/act_scales}"

# Runtime defaults
export NUM_WORKERS="${NUM_WORKERS:-8}"
export CUDA_DEVICES="${CUDA_DEVICES:-0}"
