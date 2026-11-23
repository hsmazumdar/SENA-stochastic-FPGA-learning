# SENA DE2-115 — UART C1 complete 21 Nov 2026

**Status:** RS-232 host calibration **PASS**. cycles/pass = **807.00** (model error 0).

## Setup
- Board: Terasic DE2-115, EP4CE115F29C7
- USB-Blaster [USB-0] programmed `quartus_calib\output_files\de2_115_calib.sof` (checksum 0x00BD68D7)
- UART: USB–RS232 FTDI on **COM8** → DE9 RS-232 (115200 8N1)
- Host: `python host\de2_115_calibrate.py --port COM8 --runs 20 --loop-seconds 5`

## Results
| Check | Result |
|---|---|
| `V` firmware | version 1 |
| T=1..32 fixed-pass | cycles/pass **807.00** every T |
| model_error_cycles | **0.0** |
| early-exit (random image) | mean passes 40 (ran to T_max; random bits, not labeled comfort demo) |
| throughput incl. UART | 398.8 inf/s |
| core-only latency | 1291.2 µs/inference @ 25 MHz |
| power / energy | not measured (no meter args) |

Artifact: `quartus_calib\host\de2_115_calibration.json`  
Also: `quartus_calib\output_files\de2_115_calibration_21Nov2026.json`  
SSD copy: `<SENA_ROOT>\Revision_V13\DE2_115_UART_C1_21Nov2026\`

## Paper-safe claim (add to C1)
*On Terasic DE2-115 (Cyclone IV EP4CE115) at 25 MHz, the 784→16→10 SENA slice reported exactly 807 cycles per pass over UART for T ∈ {1,2,4,8,16,32}, matching the closed-form model and the Papilio Pro silicon result.*

## Still open (not this run)
- Labeled MNIST UART log (≥100 images) vs Python
- Comfort early-exit at software T (labeled samples)
- VCD / PowerPlay / rail power
- 784→160→10 / Nexys4 Vivado
