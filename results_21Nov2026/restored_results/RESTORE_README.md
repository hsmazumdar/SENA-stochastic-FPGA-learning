# Restored / regenerated V11 pending results (13 Nov 2026)

Original finished JSONs for selective / full-T / T_min / 8-bit-trained were **not** on E: or D:.  
Source checkpoints **were** on `<JULY2026>/SENA_Project\sena2026\results\`. This folder deposits those sources and regenerates the missing re-evals.

## Deposited sources (copied from D:)

| File | Role |
|---|---|
| `q1_ablation_full_r{0..4}_fixedN.npz` | Fixed-range M=3 checkpoints (16-bit) |
| `mnist_bin_t0.3_tr10000_te2500_s2026.npz` | Paper split |
| `q1_sena_ablations.json` | Original July ablation table (EE only for fixed-N) |
| `q1_online_training.json` | Online SENA vs BP energy curves |
| `q1_bitwidth.json` | Adaptive-range bitwidth sweep (not fixed-N 8-bit) |

## Regenerated (this machine, `regen_pending_v11.py`)

| JSON | Pending # | Headline |
|---|---|---|
| `q1_fixedN_fullT_EE.json` | 1 | fixed-N full T=40: **90.76 ± 0.39%** (paper 90.79 ± 0.26) |
| `q1_fixedN_budget_sweep.json` | 2 | T_max ∈ {1,2,3,5,10,20,40} |
| `q1_fixedN_Tmin_sweep.json` | 3 | T_min ∈ {1,2,3,5} at θ=0.55 |
| `q1_selective_coverage.json` | 4 | full-T conf → **95.25 ± 0.45%@90% cov** (paper 95.2%) |
| `q1_iso_accuracy_online.json` | 6 | BP/SENA energy at iso-acc ≈ **6.34×** mean (seed1 = 6.91× ≈ paper 6.9×) |
| `q1_fixedN_requant8.json` | 5b | Requant only — **does not** match ~90.9% (expected) |

See `RESTORE_DIGEST.txt` for the short claim check.

## 8-bit retrain (Pending #5) — done

| JSON / ckpts | Result |
|---|---|
| `q1_fixedN_8bit_retrain.json` + `q1_fixedN_8bit_r{0..4}.npz` | full T=40 **90.99 ± 0.60%**; EE θ=0.55 **90.14 ± 0.61%** (paper ~90.9%) |

Requant of 16-bit weights (`q1_fixedN_requant8.json`, ~83%) does **not** support the claim; use the retrain JSON.

## Reproduce

```text
python regen_pending_v11.py
python retrain_fixedN_8bit.py
```

Requires `<JULY2026>/SENA_Project\sena2026\sim` on `sys.path` (script hard-codes that).
