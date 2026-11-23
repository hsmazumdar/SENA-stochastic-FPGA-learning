# Claim-to-evidence map: SENA manuscript V13d (November 2026)

This file maps every table, figure and headline number in the V13d manuscript and its Supplementary Material to the files in this repository that back it. All paths are relative to the repository root. The manuscript itself is not included.

Paths in the manuscript such as `sim/...` and `rtl/...` are relative to `sena2026/`. On-chip learning paths (`rtl/sena_learn_core.v`, `twin/sena_hw_twin.py`) are relative to `results_21Nov2026/OnChip_Learning/`.

## Main manuscript

| Item | Claim (as in the manuscript) | Evidence files |
|---|---|---|
| Sec. III, Eqs. (1)–(5) | Random-threshold neuron and ±1-LSB update rule (software line references) | `sena2026/sim/stochastic_core.py`, `sena2026/sim/sena_ensemble.py` |
| Sec. III.E | Hardware learning rule v1, bit-exact integer twin | `results_21Nov2026/OnChip_Learning/twin/sena_hw_twin.py`, `results_21Nov2026/OnChip_Learning/rtl/sena_learn_core.v`, `results_21Nov2026/OnChip_Learning/README_OnChip_Learning_v1.md` |
| Fig. 1 | Datapath block diagram | Drawing only; no data |
| Table I | Tuned FP32 / INT8 / BinaryConnect / XNOR-BNN baselines (95.97, 95.98, 95.13, 94.48%) | `results_21Nov2026/baselines_V12/tuned_baselines_merged.json` |
| Table I | SENA M = 3 rows (90.28, 90.76, 84.42, 87.33%), passes, storage | `results_21Nov2026/restored_results/q1_sena_ablations.json`, `results_21Nov2026/restored_results/q1_fixedN_fullT_EE.json`, `sena2026/results/q1_bitwidth.json` |
| Table II, DWN column | 95.31 ± 0.19% (5 seeds), 8,489 LE (core 8,095), Fmax 71.38 MHz, 78.73 / 95.32 mW, ≈3.15 nJ, 200 ns | `results_21Nov2026/Baseline_DWN_pkg/results/dwn_paper_table.csv`, `.../results/dwn_hardware.csv`, `.../runs/sm/accuracy.csv`, `.../quartus/reports/de2_115_dwn/*.rpt`, `results_21Nov2026/Baseline_DWN_pkg/README.md` |
| Table II, slice column | 3,749 LE, 203,680 bits, Fmax ≈31.09 MHz, 12.73 mW, 16.2 µJ, 39.42 passes × 807 cycles | `sena2026/results/de2_115_calibration_C1/reports/`, `sena2026/rtl/_power/power.json`, `results_21Nov2026/DE2_115_HW_21Nov2026/agreement/run_20261121_121846/agreement.json` |
| Table II and Table V, learning core | 50,351 LE, 19,991 registers, 146 M9K (1,017,680 bits), 0/532 multipliers, Fmax 34.77 MHz, +11.24 ns slack, 404.24 / 213.59 mW, 7,389 cycles | `results_21Nov2026/OnChip_Learning/board_v1/quartus/output_files/de2_115_sena.{fit,sta}.rpt`, `results_21Nov2026/OnChip_Learning/board_v1/README_Board_Run.md`, `results_21Nov2026/OnChip_Learning/board_v1/quartus/build_log.txt` |
| Fig. 2, Table III | Risk–coverage, ECE and Brier for SENA full budget, gated and single pass, and DWN (95.26 / 93.10 / 93.31 / 98.73% at 90% coverage) | `results_21Nov2026/analysis_V13b/calibration/` (`V13b_calibration_summary.csv/.json`, `V13b_risk_coverage_curves.csv`, `Fig_V13b_risk_coverage.png`, `analyse_calib.py`, `calib_sena.py`) |
| Table IV | Board on-chip learning 83.08 / 82.96 / 86.20% (2077 / 2074 / 2155 of 2500), BIT_EXACT; twin T = 8 and T = 1 columns | `results_21Nov2026/DE2_115_HW_21Nov2026/onchip_learning/board_h160_s*/{log.txt,result.json,epochs.csv}`, `results_21Nov2026/DE2_115_HW_21Nov2026/TABLE_onchip_mnist_de2.md`, `results_21Nov2026/OnChip_Learning/results/twin_full_h160_s*.json` |
| Table V | Full-epoch RTL vs twin (70.4%, 0 mismatches), stress runs, 5 RTL mutants (4,687–12,088 W1 mismatches), board top-level simulation | `results_21Nov2026/OnChip_Learning/results/e2e_summary.json`, `results_21Nov2026/OnChip_Learning/mutants/`, `results_21Nov2026/OnChip_Learning/vectors/`, `results_21Nov2026/OnChip_Learning/tb/`, `results_21Nov2026/OnChip_Learning/board_v1/sim/` |
| Table VI | Learning-rule ablation: SENA 84.08, FA 78.29, sign-SGD 77.84, DRTP 65.37, output-only 33.99% (FA validation tie 79.53%) | `results_21Nov2026/analysis_V13b/ablation/` (`V13b_ablation_summary.csv`, `V13b_ablation_all_settings.csv`, `V13b_ablation.md`, `ablation_rules.py`, `runs2/`) |
| Table VII | 807.00 cycles/pass on Papilio Pro and DE2-115; 2,500/2,500 board-versus-model agreement | `papilio_calibration_C1/` (`papilio_calibration.json`, `ise_reports/`), `results_21Nov2026/DE2_115_UART_C1_21Nov2026/de2_115_calibration.json`, `results_21Nov2026/DE2_115_HW_21Nov2026/agreement/` |
| Sec. V.F | Artix-7 structural estimate (one member's weights in 55.2 BRAM tiles; three members ≈165.6 tiles at 16 bits, ≈82.8 at 8 bits, derived as 3 × and 1.5 ×) | `sena2026/results/q1_fpga_synth.json` |
| Sec. V.G | PowerPlay slice estimate; operation-energy model (48.54 nJ per ensemble pass; 0.112 / 0.248 / 1.94 µJ) | `sena2026/rtl/_power/power.json`, `sena2026/sim/exp_q1_papilio_model_energy.py`, `Sena_Qi_Paper/results/q1_papilio_model_energy.json` |
| Sec. VI | TreeLUT / XGBoost same-split baseline (94.68 / 93.88%) | `results_21Nov2026/baseline_dwn/results_treelut/treelut_sena_train.json`, `results_21Nov2026/baseline_dwn/train_sena_treelut.py` |
| Sec. VI | Statistical-ADC noise study (clean 85.9%; 74.2% at σ_t = 0.15; 43.0% at σ_t = 0.20; 45.7% at σ_fpn = 0.20) | `sena2026/sim/exp_q1_sensor_adc.py`, `sena2026/results/q1_sensor_adc.{json,csv}` |
| Data split | Binary-pixel MNIST, threshold 0.3, 10,000 / 2,500, seed 2026 | `results_21Nov2026/restored_results/mnist_bin_t0.3_tr10000_te2500_s2026.npz`, `results_21Nov2026/Baseline_DWN_pkg/make_split.py` |

## Supplementary Material

| Item | Evidence files |
|---|---|
| Figs. S1–S4 (analog mismatch, shape IoU, statistical-ADC noise, confusion matrix) | `_media/q1_analog_mc.png`, `_media/q1_shapes.png`, `_media/q1_sensor_adc.png`, `_media/q1_error_analysis.png`; data in `sena2026/results/q1_analog_mc.*`, `q1_shapes.*`, `q1_sensor_adc.*`, `q1_error_analysis.json`; scripts `sena2026/sim/exp_q1_analog_mc.py`, `exp_q1_shapes.py`, `exp_q1_sensor_adc.py` |
| Fig. S5, Table S3 (bit widths) | `results_21Nov2026/baselines_V12/tuned_baselines_merged.json`, `sena2026/results/q1_bitwidth.json` |
| S-V, Tables S5–S6, Fig. S6 (fixed range, pass budget, early exit) | `results_21Nov2026/restored_results/q1_sena_ablations.{json,csv}`, `q1_fixedN_budget_sweep.json`, `q1_fixedN_Tmin_sweep.json`, `q1_fixedN_fullT_EE.json` |
| Fig. S7 (reliability diagrams) | `results_21Nov2026/analysis_V13b/calibration/Fig_V13b_reliability*.png` |
| S-VII, Table S7 (RTL-faithful accuracy) | `Sena_Qi_Paper/results/q1_rtl_mnist_accuracy*.json`, `sena2026/rtl/` |
| S-VIII, Figs. S8–S9 (Fashion-MNIST, unified trade-off) | `Sena_Qi_Paper/results/q1_fashion_mnist.json`, `q1_fashion_mnist_sweep.csv`, `sena2026/results/q1_unified_pareto.json` |
| S-IX, Fig. S10 (online adaptation proxy) | `sena2026/sim/exp_q1_online_train.py`, `results_21Nov2026/restored_results/q1_online_training.json`, `q1_iso_accuracy_online.json` |
| S-X, Figs. S11–S12 (scaling, weight precision) | `sena2026/results/q1_scaling.json`, `sena2026/results/q1_bitwidth.json` |
| Tables S8–S9, Fig. S13 (Papilio validation, Artix-7 estimate) | `papilio_calibration_C1/`, `sena2026/results/q1_fpga_synth.json` |
| Table S10 (absolute energy tiers) | `sena2026/rtl/_power/power.json`, `Sena_Qi_Paper/results/q1_papilio_model_energy.json` |
| S-XIII, Tables S11–S13 (Boolean tile, rate-coded soft output) | `papilio_logic_learn_AiOiXi/Q1_3xNN_TABLES.md`, `sena2026/fpga_papilio_logic_learn/results/`, `comfort_poc/results/_analysis_summary.json`, `comfort_poc/results/comfort_train_results.json` |
| Table S2 (published accelerators) | Values as reported by the cited sources; not reproduced here |

## Not in the repository, by design

- **FPGA bitstreams** (`*.sof`, `*.bit` except the C1 package) and **DWN PyTorch checkpoints** (`model_state.pt`, about 19 MB each): available on request, as the manuscript's Data Availability statement says.
- **Exported MNIST arrays** (`X_*.npy`): regenerate them with `results_21Nov2026/Baseline_DWN_pkg/make_split.py` from the committed `.npz` split.
- **Third-party code:** the DWN library (<https://github.com/alanbacellar/DWN>) and TreeLUT are used as published upstream, not vendored here.
- **Quartus/ISE build databases** (`db/`, `incremental_db/`) and Yosys full logs for the Basys 3 port. The manuscript makes no Basys 3 claim.
- **Other files:**
  - The 5,890-LE figure for the DE2-115 UART calibration build is recorded in `results_21Nov2026/DE2_115_HW_21Nov2026/calib_uart/SESSION_CLOSE_18Nov2026.md` (session note).
  - The corresponding Quartus report for that build is not in this repository.
