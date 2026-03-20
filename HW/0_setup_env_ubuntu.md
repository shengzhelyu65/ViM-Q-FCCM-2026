# Environment Setup Guide for Ubuntu

This guide will help you set up the complete development environment for ViM-Q (HW) on Ubuntu.

## Overview

The ViM-Q hardware development flow requires:
- **Java 17** - For Scala/SBT and SpinalHDL
- **SBT** - Scala build tool
- **Verilator** - Hardware simulation for SpinalHDL
- **Vitis HLS 2025.2** - High-Level Synthesis for C++ to Verilog
- **Vivado 2025.2** - FPGA implementation and bitstream generation

---

## 1. Install Java 17

Java 17 is required for Scala/SBT and SpinalHDL.

```bash
# Update package list
sudo apt-get update

# Install OpenJDK 17
sudo apt-get install openjdk-17-jdk

# Verify installation
java -version
# Should show: openjdk version "17.x.x"

# Set JAVA_HOME environment variable
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# Make it permanent (add to ~/.bashrc)
echo 'export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64' >> ~/.bashrc

# Reload shell configuration
source ~/.bashrc
```

**Verification:**
```bash
java -version          # Should show 17.x.x
echo $JAVA_HOME        # Should show /usr/lib/jvm/java-17-openjdk-amd64
```

---

## 2. Install SBT (Scala Build Tool)

SBT is required to build and run SpinalHDL projects.

### 2.1 Add SBT Repository

```bash
# Add SBT repository
echo "deb https://repo.scala-sbt.org/scalasbt/debian all main" | sudo tee /etc/apt/sources.list.d/sbt.list
echo "deb https://repo.scala-sbt.org/scalasbt/debian /" | sudo tee /etc/apt/sources.list.d/sbt_old.list
```

### 2.2 Add GPG Key

```bash
# Add GPG key for repository verification
curl -sL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x2EE0EA64E40A89B84B2DF73499E82A75642AC823" | sudo apt-key add -
```

### 2.3 Install SBT

```bash
# Update package list
sudo apt-get update

# Install SBT
sudo apt-get install sbt
```

**Verification:**
```bash
sbt --version          # Should show SBT version (e.g., 1.10.0)
```

**Note:** On first run, SBT will download dependencies which may take a few minutes.

---

## 3. Install Verilator

Verilator is required for hardware simulation in SpinalHDL.

### 3.1 Install Prerequisites

```bash
# Install build dependencies
sudo apt-get install git make autoconf g++ flex bison
```

### 3.2 Clone and Build Verilator

```bash
# Clone Verilator repository (only needed first time)
git clone https://github.com/verilator/verilator.git

# Navigate to verilator directory
cd verilator

# Update repository (if already cloned)
git pull

# Checkout stable version
git checkout v5.004

# Generate configure script
autoconf

# Configure build
./configure

# Build Verilator (uses all CPU cores)
make -j$(nproc)

# Install Verilator
sudo make install

# Return to project directory
cd ..
```

**Verification:**
```bash
verilator --version    # Should show: Verilator 5.004
```

**Note:** If you have multiple Verilator installations (e.g., for Chipyard), you can manage them using PATH environment variables or aliases.

---

## 4. Install Xilinx Tools (Vitis HLS and Vivado)

Vitis HLS and Vivado are required for HLS synthesis and FPGA implementation.

### 4.1 Download and Install

1. Download Xilinx Unified Installer 2025.2 from:
   - https://www.xilinx.com/support/download.html

2. Extract and run the installer

3. During installation, select:
   - **Vitis HLS 2025.2**
   - **Vivado 2025.2**

### 4.2 Configure Environment

Add the following to `~/.bashrc`:

```bash
# Xilinx 2025.2
export XILINX_PATH=/opt/Xilinx/2025.2
export PATH=$XILINX_PATH/Vivado/bin:$XILINX_PATH/Vitis_HLS/bin:$PATH
```

Reload shell:
```bash
source ~/.bashrc
```

**Verification:**
```bash
vitis-run --version    # Should show 2025.2
vivado -version       # Should show 2025.2
```

---

## 5. Configure ViM-Q Project

Set up project-specific environment variables.

### 5.1 Edit project_config.sh

Edit `<path-to-repo>/project_config.sh` to set your paths to Xilinx settings.

### 5.2 Source Environment

```bash
cd <path-to-repo>/HW
source setup_env.sh
```

You should see:
```
==========================================
ViM-Q Environment Variables Set:
==========================================
XILINX_PATH:           /opt/Xilinx/2025.2
VIM_Q_HW_ROOT:          <path-to-repo>/HW
VIM_Q_VIVADO_SETTINGS: /opt/Xilinx/2025.2/Vivado/settings64.sh
==========================================
```

---

## 6. Verify Complete Installation

Run these commands to verify all components are installed correctly:

```bash
# Check Java
java -version          # Should show 17.x.x

# Check SBT
sbt --version          # Should show SBT version

# Check Verilator
verilator --version    # Should show 5.004

# Check Vitis HLS
vitis-run --version    # Should show 2025.2

# Check Vivado
vivado -version       # Should show 2025.2

# Check ViM-Q environment
cd <path-to-repo>/HW
source setup_env.sh
```

---

## 7. Verify Test Data

Ensure test data is in the correct location:

```bash
cd $VIM_Q_HW_ROOT

# Required data directories
ls data/
# Should show:
# ├── bin_float32_block/
# ├── image_float32_block/
# ├── ref_float32_block/
# └── ssm_float32/
```

If data is missing, run Python data generation scripts in the SW directory.

---

## Next Steps

After completing environment setup, proceed to `1_run_flow.md` to run the complete development flow.
