# V13b calibration and selective prediction (20 Nov 2026 IST)

All numbers come from runs made on the box on 20 Nov 2026. Scripts: `calib_sena.py` (inference), `analyse_calib.py` (metrics/figures).

## Data and models
- **SENA**: the five saved fixed-range (N = 64), 16-bit, M = 3 ensembles `q1_ablation_full_r{0..4}_fixedN.npz` (the checkpoints behind the paper's 90.8% / 95.2% numbers; copies in `Revision_V13/restored_results`). Split: binarised MNIST, threshold 0.3, 10k/2.5k, dataset seed 2026. Inference with the original SENA code (`sim/sena_ensemble.py`, NumPy RNG): 40 passes over all 2,500 test images, cumulative votes stored per pass (`sena_vote_trajectories_fixedN_r0-4.npz`).
- **DWN** (sm, our re-implementation, same split): `Baseline_DWN_pkg/runs/sm/seed{0..4}/test_scores.npz` (per-class popcounts and the softmax logits used in training).

## Confidence definitions
- SENA confidence = vote fraction c = max_k V_k / max(Σ_k V_k, 1) (what the hardware computes); probability vector for Brier = V / ΣV.
  - **Full budget**: votes after T = 40 passes (the paper's original 95.2% headline used this).
  - **Gated (deployable)**: state at the first pass t ≤ 40 with c_t ≥ 0.55 (else t = 40), i.e. what the early-exit hardware outputs; ranked by (c at exit, then fewer passes).
  - **Single pass**: votes after T = 1.
  - The gated decision is read from the same vote trajectory as the full-budget one (statistically equivalent to running Algorithm 1; the RNG stream differs from the paper's original evaluation, see note below).
- DWN: ranking by top-1 − top-2 popcount margin (same as `collect_results.py`); ECE/Brier on softmax(logits).
- ECE: 15 equal-width bins on [0, 1]. Brier: multi-class, Σ_k (p_k − 1[y = k])², range 0–2. AURC: area under the risk–coverage curve (mean risk over all coverages).

## Results (mean ± sd; SENA n = 5 repeats, DWN n = 5 seeds)
| Model / confidence | Acc. 100% cov. | Acc. 90% cov. | Acc. 80% cov. | Acc. 70% cov. | ECE₁₅ (%) | Brier | AURC |
|---|---|---|---|---|---|---|---|
| SENA full budget (T = 40, 40 passes) | 90.76 ± 0.39 | **95.26 ± 0.46** | 97.49 ± 0.28 | 98.49 ± 0.22 | 5.69 ± 1.60 | 0.152 ± 0.009 | 0.0161 ± 0.0010 |
| SENA gated θ = 0.55 (2.23 ± 0.31 passes) | 90.40 ± 0.39 | **93.10 ± 0.46** | 94.59 ± 0.50 | 94.94 ± 0.41 | 8.37 ± 0.77 | 0.187 ± 0.010 | 0.0515 ± 0.0044 |
| SENA single pass (T = 1) | 88.54 ± 0.74 | 93.31 ± 0.69 | 94.92 ± 0.45 | 95.11 ± 0.42 | 8.15 ± 0.95 | 0.197 ± 0.013 | 0.0518 ± 0.0050 |
| DWN sm (margin ranking; softmax for ECE/Brier) | 95.31 ± 0.19 | **98.73 ± 0.15** | 99.59 ± 0.13 | 99.87 ± 0.07 | 1.89 ± 0.12 | 0.070 ± 0.002 | 0.0037 ± 0.0003 |

Deployable one-pass abstention rule (answer only if c ≥ 0.55 after the first pass, otherwise abstain): coverage **89.46 ± 1.03%**, accuracy on answered inputs **93.65 ± 0.48%**. The gated decision differs from the full-budget decision on 2.54 ± 0.30% of inputs.

Per-run values: `V13b_calibration_per_run.csv`; summary: `V13b_calibration_summary.csv/.json`; curves on a 1% grid: `V13b_risk_coverage_curves.csv`.
Figures: `Fig_V13b_risk_coverage.png/.pdf`, `Fig_V13b_reliability.png/.pdf` (1×4) and `Fig_V13b_reliability_2x2.png/.pdf` (used in the manuscript); reliability bins pooled over runs.

## Reading
- The 95.2% headline is reproduced (95.26 ± 0.46%) but needs 40 passes. With the confidence the early-exit hardware actually has (≈ 2.2 passes), accuracy at 90% coverage is 93.10 ± 0.46%. The gated curve flattens near 95% below ~70% coverage because most one-pass exits share the same top confidence (c = 1.0), so the ranking has many ties.
- The SENA vote fraction is **under-confident** at full budget (mean confidence 0.853 vs accuracy 0.908; reliability bars above the diagonal) and mis-calibrated for gated/single-pass decisions (top bin c ≈ 1.0 is ≈ 95% accurate). ECE 5.7–8.4%, versus 1.9% for DWN's softmax.
- DWN is better on every selective-prediction metric on this split.

## Note on the earlier 89.2% / 94.05% numbers
The manuscript's earlier figures (89.2 ± 1.2% of inputs exit after one pass, 94.05 ± 0.43% of them correct) came from the original evaluation run, whose result files were not recovered. This re-evaluation of the same checkpoints with a different RNG stream gives 89.46 ± 1.03% and 93.65 ± 0.48%. V13b uses the re-evaluated values.
