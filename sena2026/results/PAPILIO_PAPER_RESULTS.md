# SENA Papilio Pro — C1-lite silicon calibration (session close)

**Date:** 2026-07-26  
**Board:** Papilio Pro, Xilinx Spartan-6 XC6SLX9-2-TQG144, 32 MHz  
**Status:** C1 PASSED — ready to cite in the Q1 paper

## Paper-ready numbers

| Quantity | Value | Notes |
|----------|-------|-------|
| Model cycles/pass | **807** | Closed-form: 784+2 + 1 + 16+2 + 1 + 1 |
| Measured cycles/pass | **807.00** (all T = 1,2,4,8,16,32) | Model error = **0.0** |
| Constraint | 31.25 ns (32 MHz) | Met |
| Post-route min period | **30.674 ns** | f_max = **32.601 MHz** |
| Timing score | 0 | 0 setup / 0 hold errors |
| Core latency | **762.9 µs/inference** | cycles / 32 MHz (mean early-exit run) |
| Throughput (incl. UART) | **62.5 inf/s** | Host loop over USB-UART @ 115200 |
| C2 energy (µJ/inf) | *not measured* | Needs USB power meter; optional |

### Utilization (xc6slx9-2)

| Resource | Used | Available | % |
|----------|------|-----------|---|
| Slice registers | 2,885 | 11,440 | 25% |
| Slice LUTs | 4,574 | 5,720 | 79% |
| Occupied slices | 1,331 | 1,430 | 93% |
| RAMB16 | 14 | 32 | 43% |
| RAMB8 | 1 | 64 | 1% |
| DSP48A1 | 0 | 16 | 0% |
| Bonded I/Os | 4 | 102 | 3% |

## What this proves for the paper

The Q1 cycle model (`cycles_per_pass = 807`) was previously validated under Verilator. This board run closes the loop **on real silicon**: fixed-pass runs of T = 1…32 all report **exactly 807 cycles/pass**, so the model transfers to the Papilio Pro LX9 at 32 MHz.

This design is a **16-hidden timing/power calibration slice** of member 0 (not the paper’s full accuracy net). Accuracy claims remain on A3 / software evidence; this artifact is for **timing and energy-path calibration**.

## Suggested citation wording (draft)

> On a Papilio Pro (Spartan-6 LX9, 32 MHz) the stochastic SENA slice reports an on-chip cycle count of exactly 807 cycles per pass for fixed-pass runs of length T ∈ {1,2,4,8,16,32}, matching the closed-form model and the Verilator testbench. Post-route static timing meets the 31.25 ns period with a reported minimum period of 30.674 ns (f_max ≈ 32.6 MHz). Host-measured throughput including USB-UART round-trip is 62.5 inferences/s; core-only latency is 762.9 µs/inference.

## Package contents

```
PAPER_BRING_BACK/   (also mirrored at SENA_Project/papilio_calibration_C1/)
  papilio_calibration.json     ← primary result for figures / plan status
  papilio_slice.json
  README_PAPILIO.md
  host/papilio_calibrate.py
  rtl_snapshot/                ← Verilog + weight .mem used on board
  ise_reports/
    papilio_calib_top.bit
    papilio_calib_top.twr      ← post-route timing
    papilio_calib_top_map.mrp  ← utilization
    papilio_calib_top.par
    papilio_pro.ucf
    build_ise.tcl
    build_in_docker.bat        ← free rebuild path (Docker ISE 14.7)
```

## How to ingest on the main PC

1. Copy `papilio_calibration.json` to  
   `d:\_July2026\SENA_Project\sena2026\results\papilio_calibration.json`  
   (or use the copy already under `SENA_Project\papilio_calibration_C1\` on this PC).
2. Open the full SENA research repo in Cursor and ask to ingest the JSON into Q1 figures / plan status.
3. Optional later: C2 energy with `--power-idle` / `--power-run` USB meter readings.

## Session notes

- Board health: Error 43 fixed by direct USB port; LED blink package in `PapillioBlink/`.
- ISE: not installed natively; build used Docker image `daverichmond/xilinx-ise:14.7` ($0).
- UART: COM4 (FT2232 channel B). JTAG/programming: COM3 / papilio-prog.
- Early-exit pass histogram is irregular (many runs hit T=40 cap); one RAMB8 init warning from ISE. Does **not** affect the 807 cycles/pass C1 result. Rebuild with forced distributed ROM is optional polish.
