# DE2-115 hardware session — 21 Nov 2026

Canonical archive on SSD: `<SENA_ROOT>/Revision_V13\DE2_115_HW_21Nov2026`

Board tools stay on `E:\FPGA_Tools_115\...` (Quartus/program only). **All experiment results and host logs are written here on K:**

| Folder | Content |
|---|---|
| `calib_uart/` | C1 UART cycles/pass = 807.00 (`de2_115_calibration.json`) |
| `agreement/` | Labeled MNIST board↔RTL agreement (self-contained script + weights + npz) |
| `onchip_learning/` | board_v1 784-160-10 on-chip train/test runs (seeds 0/1/2) |

Do not leave primary results only on E:.
