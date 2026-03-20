# Environment Setup Guide

This guide walks you through setting up the conda environment for ViM-Q (SW) with all required dependencies.

## Installation Steps

### Step 1: Create Conda Environment

```bash
# Create conda environment with Python 3.10.13
conda create -n vimamba python=3.10.13
conda activate vimamba
```

### Step 2: Install GCC Compiler

```bash
# Install GCC 11 compiler for CUDA compilation
conda install -y gcc_linux-64=11 gxx_linux-64=11
```

### Step 3: Install PyTorch

```bash
# Install PyTorch with CUDA 11.8 support
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
```

### Step 4: Install Mamba Dependencies

```bash
# Install causal-conv1d
pip install causal-conv1d==1.1.3.post1 --no-build-isolation

# Install SSM kernel from source
pip install -e mamba-1p1p1 --no-build-isolation
```

### Step 5: Install VisionMamba Requirements

```bash
# Install all other core requirements
pip install -r vim_requirements.txt
```