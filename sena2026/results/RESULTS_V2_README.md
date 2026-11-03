# SENA 2026 — Results + Limitations (v2 draft companion)

Generated alongside `SENA_2026_Results_v2.docx`. Absolute energy is **model-based**;
Papilio cycles are **board-measured**.

## C2 model energy (key numbers)

From `results/q1_papilio_model_energy.json`:

| Config | µJ / inference | Basis |
|---|---|---|
| Papilio 784-16-10, board early-exit | **0.113** | Model × measured 807 cyc/pass × 30.25 passes |
| Papilio slice, fixed T=40 | **0.150** | Model × 40 |
| Full 784-160-10×M=3, PAR=8, EE | **0.248** | Model × A2 mean passes 5.11 |
| Full, PAR=8, T=40 | **1.942** | Model × 40 |
| INT16 MAC forward (ref.) | **0.183** | Same constants |

**Citation sentence (use verbatim):**
> Absolute energy is reported from a cycle×operation energy model (Horowitz-class constants) whose cycle count was validated on a Papilio Pro (Spartan-6 LX9) to be exactly 807 cycles/pass; board rail power was not instrumented on the available setup.

## Primary result tables (summary)

**Table I — MNIST baselines (5 seeds):** FP32 94.81%, INT8 94.83%, BC 81.01%, XNOR 82.20%.

**Table II — Ablations (5 repeats, θ=0.55):**
- Ensemble early-exit **84.42% ± 1.38%** @ **12.8%** relative energy
- Ensemble fixed-N **90.28% ± 0.52%** @ **5.8%** energy
- Ensemble full **87.33%**; single early-exit **78.40%**

**Table III — RTL semantic (n=2500):** software 88.04% vs RTL-faithful 88.20% (Δ +0.16 pp).

**Table IV — Papilio C1:** 807.00 cycles/pass (error 0), f_max 32.601 MHz, 62.5 inf/s UART-incl., 762.9 µs core.

## Figure files (`_media/`)

| Label | File |
|---|---|
| B1 | `q1_mnist_matched_baselines.png` |
| B2 | `q1_sena_ablations.png` |
| B3 | `q1_fashion_mnist_replication.png` |
| B6 | `q1_unified_pareto.png` |
| L1 | `q1_online_training.png` |
| S1–S6 | scaling, bitwidth, analog_mc, shapes, sensor_adc, error_analysis |
| H1 | `q1_fpga_synth_resources.png` |

## Regenerate

```bat
cd sena2026\sim
python exp_q1_papilio_model_energy.py
cd ..
python make_results_v2_docx.py
```
