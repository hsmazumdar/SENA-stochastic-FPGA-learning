# VCD flow attempt + V04 submission polish

## VCD flow status (this host)

| Step | Status |
|------|--------|
| Quartus fit (DE2-115 calib) | Done |
| `quartus_eda` functional `.vo` | **Done** → `I:\FPGA_Tools_115\SENA_DE2_115\quartus_calib\simulation\power_vcd\de2_115_calib.vo` |
| Questa/ModelSim (`vsim`) | **Missing** — Lite install has no simulation suite |
| VCD generation | Blocked |
| `quartus_pow --input_vcd` High confidence | Blocked |

Ready-to-run when Questa is installed:
`I:\FPGA_Tools_115\SENA_DE2_115\quartus_calib\simulation\power_vcd\RUN_VCD_POWERPLAY.bat`

## Paper action taken (Cannot-Run path)

- Removed all "Low confidence" strings
- Wording → **vectorless estimate; VCD upgrade path identified**
- Removed duplicate anchor sentence at end of scaling paragraph in §V.H
- Header updated for submission polish
- Numbers unchanged: 165.84 / 12.73 mW, 12.43 µJ, 0.1116 / 0.2481 / 0.1829 µJ

Remaining Low-confidence paragraph hits after polish: 0
(none)
