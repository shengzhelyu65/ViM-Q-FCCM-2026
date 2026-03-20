# ViM-Q

ViM-Q includes:
- software-side PTQ + export flow for Vision Mamba (`SW/`)
- hardware-side HLS/SpinalHDL/Vivado flow (`HW/`)

## Important Notes

This top-level README is a **quick path**. For full details, please read:
- `SW/0_setup_env.md`
- `SW/1_run.md`
- `HW/0_setup_env_ubuntu.md`
- `HW/1_run_flow.md`

Targeted versions in this artifact (tested):
  - **SW:** Python `3.10.13`, GCC/G++ `11`, PyTorch `2.1.1` (CUDA `11.8`)
  - **HW:** Java `17`, SBT `1.10.x`, Verilator `5.004`, Vitis HLS `2025.2`, Vivado `2025.2`
- **Other versions may work (including other HLS/Vivado versions), but they are not systematically tested in this artifact.**
- **PTQ reproducibility:** act-scale generation uses random activation sampling; repeated PTQ runs may not be bitwise-identical, but `Acc@1`/`Acc@5` variation is typically small.

## Repository Layout

- `Data/`: dataset root (`train/`, `val/`) used by SW scripts; includes helper scripts for download/conversion.
- `Checkpoints/`: pretrained ViM checkpoints (`tiny/small/base`) and `act_scales/`.
- `SW/`: PTQ/evaluation/export scripts.
- `HW/`: HLS, SpinalHDL, and Vivado flow.
- `project_config.sh`: central path/tool configuration for SW and HW.

## 1) One-Time Setup

1. Edit `project_config.sh` at repo root.
2. Set up SW environment via `SW/0_setup_env.md`.
3. Set up HW environment via `HW/0_setup_env_ubuntu.md`.

Expected data/checkpoint structure:

```text
Data/
  train/
  val/

Checkpoints/
  tiny/vim_t_midclstok_76p1acc.pth
  small/vim_s_midclstok_80p5acc.pth
  base/vim_b_midclstok_81p9acc.pth
  act_scales/
```

To prepare data/checkpoints, use:
- `Data/download_raw_data.py` + `Data/convert_raw_data.py`
- `Checkpoints/download_vim_checkpoints.py`

`SW/1_run.md` contains the exact command sequence.

## 2) Software Flow (PTQ + Export)

```bash
cd SW
bash run.sh
```

Main outputs:
- `SW/output/bin_float32_block/`
- `SW/output/image_float32_block/`
- `SW/output/ref_float32_block/`

Main logs:
- `SW/scripts/vim-t/ptq-vim-t.txt`
- `SW/scripts/vim-s/ptq-vim-s.txt`
- `SW/scripts/vim-b/ptq-vim-b.txt`
- `SW/scripts/vim-t/export-model-vim-t.txt`

For PTQ metrics, check `Acc@1` / `Acc@5` in `ptq-*.txt`.

## 3) Move SW Output to HW Input

From repo root:

```bash
cp -r SW/output/* HW/data/
```

## 4) Hardware HLS Flow

```bash
cd HW
source setup_env.sh
python3 step1_hls_sim.py
python3 step2_hls_syn.py
python3 step3_hls_cosim.py
```

HLS logs:
- `HW/logs/hls_run_summary_*.log`

## 5) Full Hardware Flow

See `HW/1_run_flow.md` for full details.

Typical continuation:

```bash
cd HW
python3 step5_spinal_flow.py

cd SPINAL
sbt compile
sbt "runMain simulate_vim_full"
sbt "runMain generate_vim_accelerator"

cd ..
python3 run_vivado_flow.py
python3 run_vivado_implementation.py
```

After `sbt "runMain simulate_vim_full"` (Verilator simulation), the overall end-to-end cycle-accurate latency is saved to:
- `HW/SPINAL/latency_report.log`
- `HW/SPINAL/latency_report.csv`

This latency is from full end-to-end Verilator simulation at the default input resolution (`224x224`).

Post-implementation reports:
- `HW/vivado_reports/post_impl/utilization_post_impl.rpt`
- `HW/vivado_reports/post_impl/utilization_hier_post_impl.rpt`
- `HW/vivado_reports/post_impl/power_post_impl.rpt`
