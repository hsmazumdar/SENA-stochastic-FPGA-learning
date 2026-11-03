# SENA DE2-115 — C1-lite silicon calibration (bring-back)

**Date:** 2026-07-28  
**Board:** Terasic DE2-115, Cyclone IV E **EP4CE115F29C7**  
**Tools:** Quartus Prime Lite 25.1std (E:\FPGA_Tools_115\IntelFPGA\25.1)  
**Status:** **C1 PASSED** (on-board self-test via USB-Blaster)

## Paper-ready numbers

| Quantity | Value | Notes |
|----------|-------|-------|
| Model cycles/pass | **807** | Same closed-form as Papilio: 784+2+1+16+2+1+1 |
| Measured cycles/pass | **807** | HEX showed decimal `0807`; LEDG[8:1] PASS |
| Core clock | **25 MHz** | `CLOCK_50 / 2` (50 MHz does not close timing) |
| STA Fmax (slow 85C) | **~31.09 MHz** | Positive slack at 25 MHz |
| DSP / multipliers | **0 / 532** | Multiplier-free datapath |
| Logic elements | **3,749 / 114,480 (~3%)** | Fit summary |
| Memory bits | **203,680 / 3,981,312 (~5%)** | Weight ROMs |
| Slice (hidden) | **16** | Timing/calib slice of member 0 — **not** full accuracy net |

## What this proves

Cross-platform confirmation of the Q1 cycle model on a **second FPGA family** (Intel/Altera Cyclone IV), matching the Papilio Pro (Xilinx Spartan-6) result of exactly 807 cycles/pass. Accuracy claims remain on A3 / software; this artifact is **timing/cycle calibration**.

## Cite carefully

- **Primary C1 metrics** (UART throughput, post-route min period, host JSON): still **Papilio**.
- **DE2-115**: supporting sentence — same model matched on Cyclone IV with 0 DSP.

## Visual PASS criteria (lab)

- HEX0–3: `0807`
- LEDG[8:1] ON; LEDG0 heartbeat
- Optional: photograph this state for the paper supplement

## Bitstream

- `bitstream/de2_115_calib.sof`
- SHA-256: `7E1DA335C903F52A10257154A3B6995EC368EC0038891F7A5FA5A7ED5CE22BB0`

## Not done on DE2-115 (optional later)

- Host UART C1 JSON (needs USB–RS232 to DE9)
- Board rail-power C2 (needs meter)
- Full 784-160-10 folded PoC on Cyclone IV
