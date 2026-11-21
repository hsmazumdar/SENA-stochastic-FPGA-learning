# V13b learning-rule ablation (software, 20 Nov 2026 IST)

All numbers below come from runs made on the box on 20 Nov 2026 (`ablation_rules.py`, results in `runs2/*.json`,
summaries in `V13b_ablation_summary.csv` and `V13b_ablation_all_settings.csv`). **Software only, not hardware.**

## Protocol (identical for every rule)
- Network 784-160-10, one member; SENA binarised MNIST split (threshold 0.3, 10k train / 2.5k test, dataset seed 2026).
- Forward pass exactly as the hardware twin (`sena_hw_twin.py`): per-lane 32-bit Galois LFSR, threshold `LFSR[6:0]−64` (fixed range N = 64).
- Same initial weights (uniform {−2..2}), same sample order, **full budget: 40 epochs × 3,000 online samples (batch 1)** — no epoch reduction was needed (≈ 15–50 s per run).
- 8-bit saturating integer weights (±127), ±1 LSB steps, no multipliers in the update.
- Test: all 2,500 test images, T = 40 stochastic passes, final weights (no snapshot). Seeds 0, 1, 2.
- Only the credit-assignment rule differs:
  - **SENA rule v1** — the hardware rule, unchanged (`HWTwin.train`).
  - **Sign-SGD** — sign of the straight-through gradient: output `ΔW2 = e_k·y_j`, `e_k = t_k − o_k`; hidden `ΔW1[j,i] = sgn(back_j)·x_i` with `back_j = Σ_k e_k W2[k,j]` (pre-update W2).
  - **Feedback alignment (FA)** — as sign-SGD but `back_j = Σ_k e_k B[k,j]`, B a fixed random ±1 matrix.
  - **DRTP** — as FA but `back_j = Σ_k t_k B[k,j]` (one-hot target projected, no error), applied on samples with an output error.
  - **Output-only control** — output layer as sign-SGD, hidden layer frozen at its random initialisation.
- Learning-rate knob for the baselines: each row update is applied with probability p ∈ {1, 1/4, 1/16, 1/64} (sign-SGD also 1/256).
  **p was selected per rule on the 800-image probe set that the schedule holds out of training, never on the test set.**
  SENA was run unchanged (its rule constants were fixed before this study and not tuned here).

## Result (mean ± sd over 3 seeds, test accuracy at T = 40)
| Rule | p (selected on validation) | Test acc. T = 40 (%) | Test acc. T = 1 (%) | Samples with a hidden update (of 120k) | W1 at ±127 (%) |
|---|---|---|---|---|---|
| **SENA rule v1** | – | **84.08 ± 1.84** | 74.39 | 19,779 | 0.00 |
| Feedback alignment | 1/4 | 78.29 ± 1.47 | 67.29 | 59,263 | 2.79 |
| Sign-SGD (STE) | 1/16 | 77.84 ± 2.50 | 55.01 | 75,791 | 1.83 |
| DRTP | 1 | 65.37 ± 8.97 | 54.81 | 70,038 | 34.43 |
| Output layer only (hidden frozen) | 1 | 33.99 ± 8.65 | 18.25 | 0 | 0.00 |

- FA tied on validation between p = 1/4 and p = 1 (both 78.25%); the table uses the first-listed (p = 1/4). At p = 1 FA gives 79.53 ± 3.16% on test. Every setting is listed in `V13b_ablation_all_settings.csv`.
- The SENA row reproduces the twin result reported in `OnChip_Learning/README` exactly (83.08 / 82.96 / 86.20%).
- **Window check (`runs_window_check/`)**: using the exact expected derivative of the noisy threshold (gate |X| < 64, |Z| < 64) instead of the identity STE, sign-SGD and FA collapse to a constant prediction (9.56% on all 3 seeds) because the output sums are driven below −64 and the gate then blocks all further learning; DRTP reaches 18–37%. The identity-STE versions above are therefore the fair baselines.

## Reading
- Under identical 8-bit, ±1-LSB, multiplier-free constraints and the same budget, the SENA rule is 5.8–6.2 pp above tuned FA and sign-SGD and ≈ 19 pp above DRTP; training the hidden layer matters (output-only: 34%).
- SENA updates the hidden layer on far fewer samples (≈ 16% of samples versus 49–63%) and never saturates W1, consistent with its random sensitivity windows acting as an adaptive, sparse step-size.
- Caveat: SENA differs from the baselines in **both** layers (its output update moves every weight of an erring output, including inactive inputs, by ±1 and its hidden update uses ±1 for inactive inputs too). The ablation compares whole rules; it does not isolate which component is responsible. n = 3 seeds; differences of ~6 pp are ≥ 2 sd of each arm.
