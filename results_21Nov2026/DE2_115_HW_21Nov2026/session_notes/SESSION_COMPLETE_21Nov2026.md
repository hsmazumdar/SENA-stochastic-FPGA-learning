# DE2-115 hardware — complete 21 Nov 2026

All results under `<SENA_ROOT>\Revision_V13\DE2_115_HW_21Nov2026\`.

## Done on silicon today

### 1. Calib slice UART C1
- `calib_uart/de2_115_calibration.json`
- cycles/pass = **807.00**, model_error = 0 (T∈{1..32})
- HEX PASS confirmed by user

### 2. Calib slice board↔RTL agreement (2500 images)
- `agreement/run_20261121_121846/agreement.json`
- pred/passes/cycles **2500/2500** match; every run 807×passes
- slice accuracy 29.12% (not paper full-net accuracy)

### 3. On-chip learning 784-160-10 (board_v1)
| Seed | Board / twin | BIT_EXACT | Folder |
|---|---|---|---|
| 0 | 2077/2500 = **83.08%** | True | `onchip_learning/board_h160_s0_20261121_122052` |
| 1 | 2074/2500 = **82.96%** | True | `onchip_learning/board_h160_s1_20261121_122509` |
| 2 | 2155/2500 = **86.20%** | True | `onchip_learning/board_h160_s2_20261121_122657` |

Mean accuracy 84.08%. Train ~7.5 s @25 MHz core wall; test ~0.77 s.

## Still not doable without other tools/boards
- Artix-7 / Nexys4 Vivado P&R (Vivado not installed)
- VCD→PowerPlay / rail meter energy
- DWN matched baseline (needs CUDA/Colab)
