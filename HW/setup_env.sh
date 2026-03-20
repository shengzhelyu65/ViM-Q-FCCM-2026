#!/bin/bash
# ViM-Q Environment Setup Script

# ============================================================================
# LOAD PROJECT CONFIGURATION
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
. "$PROJECT_ROOT/project_config.sh"

# ============================================================================
# EXPORTS
# ============================================================================

# Export key variables used by HW/SW flows.
export VIM_Q_ROOT
export VIM_Q_HW_ROOT
export VIM_Q_SW_ROOT
export XILINX_PATH
export VIM_Q_VIVADO_SETTINGS
export DATA_PATH
export MODEL_WEIGHTS_BASE
export ACT_SCALES_BASE
export NUM_WORKERS
export CUDA_DEVICES

# Export tool paths so vitis-run/vivado resolve from this shell.
if [ -d "$XILINX_PATH/Vivado/bin" ]; then
    export PATH="$XILINX_PATH/Vivado/bin:$PATH"
fi
if [ -d "$XILINX_PATH/Vitis/bin" ]; then
    export PATH="$XILINX_PATH/Vitis/bin:$PATH"
fi

# Prefer the active conda runtime libs.
if [ -n "$CONDA_PREFIX" ] && [ -d "$CONDA_PREFIX/lib" ]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# ============================================================================
# VALIDATION
# ============================================================================

# Check if VIM_Q_HW_ROOT is valid
if [ ! -d "$VIM_Q_HW_ROOT" ]; then
    echo "WARNING: VIM_Q_HW_ROOT directory does not exist: $VIM_Q_HW_ROOT"
    echo "Please edit project_config.sh and set the correct path."
fi

# Check if VIM_Q_VIVADO_SETTINGS is valid
if [ ! -f "$VIM_Q_VIVADO_SETTINGS" ]; then
    echo "WARNING: VIM_Q_VIVADO_SETTINGS file does not exist: $VIM_Q_VIVADO_SETTINGS"
    echo "Please edit project_config.sh and set the correct path."
fi

echo "=========================================="
echo "ViM-Q Environment Variables Set:"
echo "=========================================="
echo "XILINX_PATH:           $XILINX_PATH"
echo "VIM_Q_HW_ROOT:          $VIM_Q_HW_ROOT"
echo "VIM_Q_VIVADO_SETTINGS: $VIM_Q_VIVADO_SETTINGS"
echo "=========================================="
