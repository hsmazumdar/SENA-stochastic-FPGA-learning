# Hardware results and baselines, 21 Nov 2026

On a Terasic DE2-115 (Cyclone IV E EP4CE115F29C7, 25 MHz) the single-member 784-160-10 SENA learning core trained MNIST **on the chip** for seeds 0, 1 and 2. For every seed, the board matched the bit-exact software twin exactly: the same predictions and the same final weights. Mean board accuracy is **84.08 ± 1.84%** on the 2,500-image binarised test split. The build uses 50,351 logic elements and 0 embedded multipliers. This is a one-member result; ensemble (M = 3) accuracy is not claimed from hardware.

| Seed | Board correct | Board acc. | Twin acc. | Pred. mismatches | W1 mismatches | W2 mismatches | Bit-exact | Train core time | Core cycles / test image |
|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|
| 0 | 2077/2500 | 83.08% | 83.08% | 0 | 0/125,600 | 0/1,610 | yes | 6.00 s | 7,389 |
| 1 | 2074/2500 | 82.96% | 82.96% | 0 | 0/125,600 | 0/1,610 | yes | 5.99 s | 7,389 |
| 2 | 2155/2500 | 86.20% | 86.20% | 0 | 0/125,600 | 0/1,610 | yes | 5.98 s | 7,389 |

Also on the DE2-115:
- The 784-16-10 calibration slice measured **807.00 cycles/pass** over UART (model error 0.0).
- Board versus bit-exact model agreed on **2,500/2,500** test inputs (prediction, pass count and cycle count). This is a 16-hidden slice, so its 29.12% accuracy is not the full-network accuracy.
- The same-split DWN baseline (`sm`, 5 seeds) reaches 95.31 ± 0.19%. On the same EP4CE115 it fits in 8,489 LEs, Fmax 71.38 MHz, 200 ns latency at 25 MHz and ≈3.15 nJ/inference (post-fit vectorless PowerPlay estimate).
- TreeLUT/XGBoost on the same split: 94.68%.

## Folder map

| Path | Contents |
|---|---|
| `DE2_115_HW_21Nov2026/onchip_learning/` | Per-seed board runs: `result.json`, `epochs.csv`, `log.txt` |
| `DE2_115_HW_21Nov2026/agreement/` | 2,500-image board vs model agreement (script, weights, CSVs, `agreement.json`) |
| `DE2_115_HW_21Nov2026/calib_uart/`, `DE2_115_UART_C1_21Nov2026/` | UART C1 cycles/pass calibration |
| `OnChip_Learning/` | Learning-core RTL, testbenches, twin, test vectors, mutants, `board_v1` (DE2-115 board build + host), `quartus_core`, `basys3_v1` (Yosys estimates), results |
| `Baseline_DWN_pkg/` | DWN baseline: training, export, Verilog generator, Quartus DE2-115 project and reports, results |
| `baseline_dwn/` | Earlier DWN Colab script and TreeLUT/XGBoost results (`results_treelut/`) |
| `restored_results/` | Fixed-range checkpoints (`*.npz`) and re-evaluation JSONs (budget / T_min sweeps, 8-bit retrain, selective coverage, online) |
| `analysis_V13b/` | Learning-rule ablation and calibration / risk-coverage analysis scripts and outputs |
| `baselines_V12/` | Tuned same-width FP32 / INT8 / BinaryConnect / XNOR baseline summary |

Notes: absolute local paths in logs and tool reports were replaced by `<SENA_ROOT>`, `<WORK>`, `<HOME>` or `<JULY2026>`. DWN model checkpoints (`model_state.pt`, about 18 MB each) are not included; `Baseline_DWN_pkg/train_dwn.py` regenerates them. The upstream DWN and TreeLUT source trees are not vendored; install them from their own repositories. All power and energy figures are post-fit tool estimates, not rail measurements.
