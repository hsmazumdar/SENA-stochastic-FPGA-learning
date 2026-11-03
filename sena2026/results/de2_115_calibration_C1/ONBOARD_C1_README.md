# On-board C1 result note (USB-Blaster bring-up, no RS-232)

Date: 2026-07-28
Board: DE2-115 EP4CE115F29C7
Bitstream: output_files/de2_115_calib.sof
Flow: jtagconfig → quartus compile → quartus_pgm (same as LED blink demo)

## What the FPGA does after programming

Auto-runs C1 cycle calibration on-chip at 25 MHz:
T ∈ {1,2,4,8,16,32}, early-exit OFF, expect cycles == 807×T.

## How to read the board

| Result | Indication |
|--------|------------|
| **PASS** | LEDG[8:1] all ON, HEX0–3 show **0807**, LEDG0 heartbeats |
| **FAIL** | LEDR17 ON; LEDR[5:1]/LEDR6 encode failing T |
| Busy/test | LEDR0 ON while running |
| Re-run | Press **KEY1** (hold **KEY0** = reset) |

## Post-fit (this build)

See `output_files/de2_115_calib.fit.summary` — DSP multipliers must be 0.
