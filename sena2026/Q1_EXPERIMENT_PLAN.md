# SENA 2026 Q1 Experimental Plan

This plan separates measured evidence from estimates. Runtime estimates below
refer to the current 16-thread CPU-only PC. Publishable runs use five independent
seeds unless noted.

## Phase A — acceptance blockers

| ID | Experiment | Script / output | Publication figure or table | Runtime |
|---|---|---|---|---|
| A1 | Matched MNIST baselines: FP32 MLP, INT8 PTQ, BinaryConnect, XNOR-BNN | `sim/exp_q1_baselines.py --profile full`; `results/q1_mnist_baselines.{json,csv}` | Fig. B1: accuracy with 5-seed error bars and storage | **3.2 min measured** |
| A2 | SENA ablations: M=1/3, early-exit on/off, adaptive/fixed N | `sim/exp_q1_ablations.py --profile full`; resumable per-repeat NPZ files | Fig. B2 and Table A1: accuracy, passes, relative energy, mean ± SD | **20–30 min/repeat; 1.7–2.5 h total estimated** |
| A3 | Real-weight RTL accuracy with LFSR and power-of-two N | `rtl/exp_rtl_mnist_accuracy.py --profile full`; `results/q1_rtl_mnist_accuracy.json` | Table A2: software vs RTL accuracy and quantisation delta | **5–10 min build + ~45 min/2500 images estimated** |
| A4 | Vendor synthesis and power | Vivado Tcl flow to be added; post-route reports | Table H1: LUT, FF, BRAM, DSP, fmax, W, µJ/inference | 1–3 h per implementation |
| A5 | Fashion-MNIST replication | Extend A1/A2 data loader | Fig. B3: second-dataset accuracy-energy Pareto | 2–3 h |
| A6 | Online multiplier-free training | Streaming trainer + checkpoints | Fig. L1: accuracy vs samples and update energy | 1–2 h |

## Phase B — strengthen claims

| ID | Experiment | Planned script | Deliverable | Runtime |
|---|---|---|---|---|
| B1 | Hidden width × ensemble size × T_max scaling | `sim/exp_q1_scaling.py` | 3-panel Pareto/scalability figure | 4–8 h |
| B2 | Weight precision 8/12/16 bit | `sim/exp_q1_bitwidth.py` | Accuracy/area/energy plot | 2–4 h |
| B3 | Analog noise and mismatch Monte Carlo | extend `sim/analog_neuron.py` | Yield, transfer curves, sensitivity plot | 30–90 min |
| B4 | Shape metrics for circle/square/triangle | extend `sim/pixel_learners.py` | IoU, pixel accuracy, convergence curves | 15–30 min |
| B5 | Error analysis | `sim/analyze_q1_errors.py` | Confusion matrix; early/late-exit examples | 5–15 min |
| B6 | Common energy Pareto | consume A1–A5 JSON files | One fair accuracy–energy comparison figure | <5 min |

## Phase C — hardware proof and reproducibility

| ID | Experiment | Deliverable | Runtime |
|---|---|---|---|
| C1 | FPGA board inference | Board log, photograph/video, measured throughput | 0.5–1 day after board setup |
| C2 | Rail-power measurement | Idle-subtracted W and µJ/inference | 0.5 day |
| C3 | Sensor/statistical-ADC end-to-end test | Analog-input robustness plot | 1–2 h simulation |
| C4 | One-command artifact regeneration | `run_q1_suite.py`, manifest, seeds and environment | 1–2 h engineering |

## Statistical protocol

1. Freeze a dataset split with dataset seed 2026.
2. Use training seeds 0–4 and report mean ± sample standard deviation.
3. Retain per-seed rows, not only aggregate values.
4. Use paired tests where two methods share the same split and seed.
5. Report 95% bootstrap confidence intervals for headline deltas.
6. Label operation-count, model-based energy, post-synthesis power, and
   board-measured power as four different evidence levels.

## Current status

- A1 complete (5-seed matched MNIST baselines).
- A2 ablations **complete** (5 repeats): ensemble early-exit **84.4%±1.4%**
  @ **12.8%±2.8%** relative energy; fixed-N **90.3%±0.5%** @ **5.8%** energy;
  single early-exit **78.4%±4.1%**; ensemble full **87.3%±1.0%**.
- A3 RTL-semantic accuracy on real 784-160-10 weights (**2500-sample full
  set**): software **88.04%** vs RTL-faithful **88.20%** (Δ **+0.16%**).
  Small-tile Verilator remains the bit-exact cosim proof.
- A4 mapping estimate done (no Vivado on host); folded Artix-7 design fits
  at ~8.5% LUT. Vivado Tcl ready when installed.
- A5 Fashion-MNIST **full complete**: FP32 ~84.2%, INT8 ~84.2%, XNOR-BNN
  ~71.8%, BinaryConnect ~59.5%; SENA fixed-budget **75.7%**; early-exit @θ=0.55
  **71.8%** @ ~17% relative energy.
- A6 online training **full complete**: SENA final 59.4% vs RealBP 88.3%,
  with ~34.6× lower cumulative train-energy proxy (no multiplies).
- B6 unified Pareto refreshed (includes Fashion + full A2).
- Phase B **complete**:
  - B1 scaling: H∈{64,160,256} × M∈{1,3,5} × T∈{20,40}, 3 seeds.
  - B2 bitwidth: early-exit acc 8/12/16-bit ≈ **83.9 / 84.4 / 84.4%** (5 seeds).
  - B3 analog MC: match yield drops with mismatch (99%→30%).
  - B4 shapes: Mazumdar IoU ~0.87/0.91/0.81 (cir/sqr/tri).
  - B5 error analysis: 85.9% @ θ=0.55 on A2 r0 ckpt.
  - B6 Pareto refreshed after Phase B.
- Phase C (PC-side) **complete**:
  - C3 statistical-ADC robustness: accuracy holds to σ_t=0.10 (Δ ≤ −0.8%),
    degrades at 0.15 (−11.8%), collapses ≥0.20; early-exit passes rise
    4.3→17 with noise (confidence gate spends compute on hard inputs).
    `sim/exp_q1_sensor_adc.py`, `results/q1_sensor_adc.json`.
  - C4 one-command regeneration: `run_q1_suite.py` (--profile/--check/--only)
    with reproducibility manifest `results/q1_suite_manifest.json`
    (13/13 artifacts, SHA-256, package versions, seeds).
  - **C1 Papilio Pro silicon PASSED** (2026-07-26): measured cycles/pass =
    **807.00** for all T∈{1,2,4,8,16,32} (model error **0.0**). Post-route
    min period **30.674 ns** (f_max **32.601 MHz**), timing score 0.
    Core latency **762.9 µs/inf**; UART-incl. throughput **62.5 inf/s**.
    Utilization: 79% LUT / 93% slices / 43% RAMB16 / 0 DSP on XC6SLX9.
    Artifacts: `results/papilio_calibration.json`,
    `results/PAPILIO_EVIDENCE.md`, `results/PAPILIO_PAPER_RESULTS.md`,
    and full bring-back package `papilio_calibration_C1/` (bitfile + TWR/MRP/PAR
    + RTL snapshot). Bitstream SHA-256
    `AD35F00B8358937FC3FBF920826E02A05D9361DC9A1D5FC18A61B931A5181FC2`.
  - **C2 energy** closed as **model-based** (not rail power):
    `sim/exp_q1_papilio_model_energy.py` → `results/q1_papilio_model_energy.json`.
    Papilio slice ~0.113 µJ/inf (board EE); full PAR8×M3 EE ~0.248 µJ/inf.
  - Manuscript insert: `SENA_2026_Results_v2.docx` (Results + Limitations +
    tables/figures) and `results/RESULTS_V2_README.md`.
  - A4 post-route (Artix-7) still needs Vivado.
- **Q1 manuscript V05 (2026-08-02):** `SENA_2026_Paper_V05.docx` folds good
  Papilio PoCs into §V.I (Tables VII–IX):
  - Ai/Oi/Xi Boolean on-chip Mazumdar — silicon **100%** joint @ mean T=3
  - Analog Comfort soft-p̂ — soft MAE≈0.21 / board MAE≈0.27 @ T=100
  - Folded MNIST-online B+E **excluded** (not a Q1-quality PoC)
  - Regenerator: `sena2026/make_sena_paper_v05.py`; notes `V05_CHANGELOG.md`
