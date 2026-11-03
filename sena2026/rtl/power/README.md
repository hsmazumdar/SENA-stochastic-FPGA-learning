# Artix-7 Vector (SAIF) Power — EnergyIssueFix.docx implementation

This folder implements the recommended **simulation-based (vector) power
estimation** flow: post-route design → MNIST SAIF → Vivado Report Power
(High confidence) → µJ/inference for Table VI.

## Prerequisites

1. Vivado 2020.1+ on PATH  
2. Python deps used by SENA (`numpy`, torchvision or sklearn for MNIST)

## One-time: build MNIST + weight stimuli

```bat
cd D:\_July2026\SENA_Project\sena2026\rtl\power
python prepare_power_stimuli.py --n-images 1000 --quick-train
```

Writes `stimuli/mnist_x.bin`, `w_*.mem`, `stimuli_meta.json`.

## Run the flow

### A) Fully automated

```bat
vivado -mode batch -source power_analysis_full.tcl ^
  -tclargs STAGE=all CLK_MHZ=50 N_IMAGES=500 MEAN_PASSES=2.30 M=3
```

### B) Step-by-step (matches EnergyIssueFix.docx)

| Step | Script | What it does |
|------|--------|----------------|
| 1 | `setup_sena.tcl` | Create proj, synth, P&R on `xc7a100t` |
| 2 | `gen_postroute_netlist.tcl` | Write timesim netlist + SDF |
| 3–4 | `run_simulation_with_saif.tcl` | xsim + SAIF from MNIST TB |
| 5 | `report_power_saif.tcl` | `read_saif` + `report_power` → `power.json` |

Interactive SAIF (Tcl Console) if batch SAIF is empty:

```tcl
open_saif sena_activity.saif
log_saif [get_objects -r /sena_tb/uut/*]
run all
close_saif
```

Requires `-debug typical` (set in `run_simulation_with_saif.tcl`).

## Energy formula (Table VI)

```
µJ/inference = 1000 × Total_mW / (inferences/sec)
inferences/sec = f_clk / (cycles_per_pass × M × mean_passes)
```

| Symbol | Value | Source |
|--------|------:|--------|
| `cycles_per_pass` | **182** | Verilator-validated PAR=8 full net (`ceil(784/8)+ceil(160/2)+4`) |
| `M` | 3 | Sequential ensemble members |
| `mean_passes` | 2.30 | Fixed-N early-exit (primary OP) |
| `f_clk` | 50 MHz | Artix-7 target (20 ns) |

> EnergyIssueFix.docx used **807** cycles — that is the **Papilio 784→16→10
> calibration slice**, not the full 160-hidden net. Do not use 807 for Table VI’s
> full-net row.

## Patch the paper

```bat
python D:\_July2026\SENA_Project\sena2026\patch_table_vi_vivado.py
```

Reads `rtl/_power/power.json` (copied by `report_power_saif.tcl`) and fills the
Vivado Post-Route row in `SENA_2026_Paper_V03.docx`.

## Troubleshooting (from EnergyIssueFix)

| Issue | Fix |
|-------|-----|
| No activity data | `open_saif` before `log_saif`; sim long enough |
| Empty SAIF | Recompile with `-debug typical` |
| Confidence Low | `log_saif [get_objects -r *]` |
| Sim too slow | `N_IMAGES=100..500` |
| Cannot read SAIF | `-instance_name sena_tb/uut` |

## Files vs guide names

| EnergyIssueFix name | This repo |
|---------------------|-----------|
| `sena_top.v` | `power/sena_top.v` (wraps real `sena_tile.v`) |
| `sena_tb.v` | `power/tb/sena_tb.v` |
| `setup_sena.tcl` | `power/setup_sena.tcl` |
| `power_analysis_full.tcl` | `power/power_analysis_full.tcl` |
