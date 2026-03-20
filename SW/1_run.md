# Running Guide

This guide explains how to run ViM-Q inference/quantization and export model data for hardware acceleration.

## Prerequisites

1. Environment setup completed (see `0_setup_env.md`)
2. Model weights downloaded and placed in the correct directory
3. Dataset prepared and accessible

## Setup

### 1. Prepare Data and Checkpoints

By default, ViM-Q uses these paths:

- `Data/` for dataset files
- `Checkpoints/` for model checkpoints
- `Checkpoints/act_scales/` for generated activation scales

Download and prepare them as follows:

```bash
# 1) login once (required for ImageNet-1K on Hugging Face)
huggingface-cli login --token <YOUR_ACCESS_TOKEN>

# 2) download and convert dataset into ImageFolder layout
cd Data
python download_raw_data.py
python convert_raw_data.py

# 3) download pretrained ViM checkpoints
cd ../Checkpoints
python download_vim_checkpoints.py

# 4) return to SW directory
cd ../SW
```

Expected structure:

```text
Data/
  raw_data/            # downloaded HF dataset (intermediate)
  train/
  val/

Checkpoints/
  tiny/vim_t_midclstok_76p1acc.pth
  small/vim_s_midclstok_80p5acc.pth
  base/vim_b_midclstok_81p9acc.pth
  act_scales/          # auto-created when running PTQ scripts
```

### 2. Configure Paths

Edit `project_config.sh` at repo root.

Example:

```bash
export DATA_PATH="<repo>/Data"
export MODEL_WEIGHTS_BASE="<repo>/Checkpoints"
export ACT_SCALES_BASE="<repo>/Checkpoints/act_scales"
```

## Usage

Run the default script:

```bash
cd SW
bash run.sh
```

## Logs and Metrics

PTQ and export scripts write logs to `.txt` files in `SW/scripts/`:

- `SW/scripts/vim-t/ptq-vim-t.txt`
- `SW/scripts/vim-s/ptq-vim-s.txt`
- `SW/scripts/vim-b/ptq-vim-b.txt`
- `SW/scripts/vim-t/export-model-vim-t.txt`

For PTQ results, check `Acc@1` and `Acc@5` in the corresponding `ptq-*.txt` log.

## Output Structure

After running inference, the exported model data will be organized as follows:

```
output/
├── bin_float32_block/      # Binary format model data
├── image_float32_block/    # Image processing data
└── ref_float32_block/       # Reference data
```

## Hardware Integration

### Copy Data to Hardware Folder

After export, copy the model data to the hardware data folder:

```bash
# from repo root
cd ..
cp -r SW/output/* HW/data/
```

This data will be used by the hardware accelerator for inference.
