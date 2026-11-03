# SENA: A Multiplier-Free Stochastic Architecture for On-Chip Learning and Confidence-Gated Inference

Reproducibility package for the manuscript:

**SENA: A Multiplier-Free Stochastic Architecture for On-Chip Learning and Confidence-Gated Inference**

Himanshu S. Mazumdar
R&D Centre, Dharmsinh Desai University, Nadiad, Gujarat, India
`hsmazumdar@ddu.ac.in`

Target venue: *IEEE Transactions on VLSI Systems* (submitted). **The manuscript is not included in this repository.**

## What this is

SENA (Stochastic Ensemble Neural Architecture) is a multiplier-free network that uses
LFSR-driven stochastic passes with integer accumulators instead of multipliers. A small
ensemble votes over repeated stochastic passes, and a confidence gate stops early when
the vote is decisive, so work scales with input difficulty. This package contains the
Python simulators and experiment drivers, the Verilog RTL and Verilator co-simulation,
FPGA projects for a Papilio Pro (Spartan-6 LX9) and a DE2-115 (Cyclone IV E), the result
JSON/CSV files, figures and the vendor timing/map/PAR reports behind every hardware number.

## Repository layout

| Path | Contents |
|---|---|
| `sena2026/sim/` | Python simulators and experiment drivers (`exp_q1_*.py`, `sena_ensemble.py`, `stochastic_core.py`, ...) |
| `sena2026/rtl/` | Verilog RTL, testbenches, Verilator co-simulation, RTL-semantic accuracy, power-stimulus flow |
| `sena2026/fpga_papilio/` | Papilio Pro inference core: RTL, UCF, ISE reports, host calibration script |
| `sena2026/fpga_papilio_logic_learn/`, `sena2026/fpga_de2_logic_learn/` | AND/OR/XOR on-chip learning PoC (Papilio / DE2-115) |
| `sena2026/fpga_papilio_mnist_online/` | On-board MNIST online-learning experiment |
| `sena2026/results/` | All `q1_*` result JSON/CSV/NPZ, Papilio and DE2-115 calibration evidence |
| `sena2026/figures/`, `sena2026/assets/` | Generated figures (PNG/SVG) |
| `sena2026/data/` | Small binarised MNIST/Fashion-MNIST subsets (`*.npz`); raw IDX files are **not** included |
| `sena2026/run_q1_suite.py` | Driver for the full software suite |
| `sena2026/ver01/` | Earlier v0.1 simulator (historical) |
| `Sena_Qi_Paper/` | Curated results, figures, hardware evidence and the experiment plan |
| `papilio_calibration_C1/` | Self-contained C1 package: RTL snapshot, ISE reports (.twr/.mrp/.par), host script |
| `papilio_logic_learn_AiOiXi/` | AND/OR/XOR silicon PoC package with results |
| `comfort_poc/` | Thermal-comfort soft-probability PoC (host training, RTL, ISE, results) |
| `_media/` | Board photographs |

## Headline results and how to reproduce them

Run Python commands from `sena2026/` unless stated otherwise. `sena2026/Q1_EXPERIMENT_PLAN.md`
gives the full plan and the exact profiles used.

| ID | Result (as reported) | Script | Output |
|---|---|---|---|
| A1 baselines (5 seeds) | FP32 94.81%, INT8 94.83%, BinaryConnect 81.01%, XNOR 82.20% | `sim/exp_q1_baselines.py` | `results/q1_baselines*` |
| A2 ablations (5 repeats, T=0.55) | ensemble early-exit 84.42% ± 1.38% @ 12.8% energy; fixed range 90.28% @ 5.8% | `sim/exp_q1_ablations.py` | `results/q1_ablation*` |
| A3 RTL-semantic (n=2500) | software 88.04% vs RTL 88.20% (+0.16 pp) | `rtl/exp_rtl_mnist_accuracy.py` (Verilator) | `results/` |
| A5 Fashion-MNIST | FP32 84.2%; SENA fixed 75.7%; EE@0.55 71.8% @ 17% energy | `sim/exp_q1_fashion.py` | `results/q1_fashion*` |
| A6 online learning | SENA 59.4% vs RealBP 88.3% (train-energy proxy; different accuracy levels) | `sim/exp_q1_online_train.py` | `results/q1_online*` |
| C1 Papilio Pro | 807 cycles/pass exact, f_max 32.601 MHz, 762.9 µs core latency | `fpga_papilio/host/papilio_calibrate.py` + `papilio_calibration_C1/` | `ise_reports/*.twr`, `PAPER_RESULTS.md` |
| C2 energy model | full net PAR8×M3 early-exit ~0.248 µJ/inf; Papilio slice ~0.113 µJ/inf (model-based) | `sim/exp_q1_papilio_model_energy.py` | `results/q1_papilio_model_energy*` |
| PoC AND/OR/XOR | software twin 96.9 ± 1.9% joint; silicon 100% joint @ mean T=3 | `papilio_logic_learn_AiOiXi/host/` | `papilio_logic_learn_AiOiXi/results/` |
| PoC Comfort | soft MAE ≈ 0.21; board MAE ≈ 0.27 @ T=100 | `comfort_poc/host/train_comfort.py`, `papilio_comfort.py` | `comfort_poc/results/` |

Hardware steps need the board connected over USB-serial (`pyserial`), and Xilinx ISE 14.7 /
Intel Quartus to rebuild bitstreams. RTL co-simulation uses the `verilator` pip package.

## Energy-claim wording

> Absolute energy is reported from a cycle×operation energy model (Horowitz-class constants)
> whose cycle count was validated on a Papilio Pro (Spartan-6 LX9) to be exactly 807
> cycles/pass; board rail power was not instrumented on the available setup.

No figure in this repository is a measured rail-power number.

## Requirements

Python 3.10+, see `requirements.txt`. Optional: Verilator (via pip), Xilinx ISE 14.7,
Intel Quartus Prime (DE2-115), a Papilio Pro board.

## Datasets

Raw MNIST and Fashion-MNIST are not redistributed. `torchvision.datasets.MNIST` /
`FashionMNIST` (or `sklearn.datasets.fetch_openml`) download them on first use into
`sena2026/data/`. The small binarised subsets used by the experiments are included as `.npz`.

## Notes

- Some scripts still contain absolute Windows paths (`D:\_July2026\SENA_Project\...`) from
  the original workspace; adjust them or run from the matching layout.
- `results/mnist_sena_checkpoint.npz` is not included; regenerate it with `sim/exp_mnist_sena.py`.

## Citation

```bibtex
@article{mazumdar_sena_2026,
  author  = {Mazumdar, Himanshu S.},
  title   = {SENA: A Multiplier-Free Stochastic Architecture for On-Chip Learning and Confidence-Gated Inference},
  journal = {IEEE Transactions on Very Large Scale Integration (VLSI) Systems},
  year    = {2026},
  note    = {Under review}
}
```

## What is not in this mirror

- The manuscript, drafts, and review notes
- Raw datasets and large build intermediates
- Secrets, credentials, or private correspondence

## License

MIT, see [`LICENSE`](LICENSE). Manuscript text remains the author's scholarly work.
