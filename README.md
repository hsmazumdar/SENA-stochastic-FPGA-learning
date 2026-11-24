# SENA: Multiplier-Free On-Chip Learning and Vote-Confidence Selective Prediction with a Stochastic Ensemble on FPGAs

Reproducibility package for the manuscript:

**SENA: Multiplier-Free On-Chip Learning and Vote-Confidence Selective Prediction with a Stochastic Ensemble on FPGAs**

Himanshu S. Mazumdar
R&D Centre, Dharmsinh Desai University, Nadiad, Gujarat, India
`hsmazumdar@ddu.ac.in`

**The manuscript is not included in this repository.**

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
| `results_21Nov2026/` | DE2-115 on-chip learning runs, board agreement, DWN and TreeLUT baselines, re-evaluation results (21 Nov 2026) |

## Update 21 Nov 2026: DE2-115 on-chip learning and baselines

All files are in [`results_21Nov2026/`](results_21Nov2026/README.md).

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

### Folder map (`results_21Nov2026/`)

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
@misc{mazumdar_sena_2026,
  author       = {Mazumdar, Himanshu S.},
  title        = {SENA: Multiplier-Free On-Chip Learning and Vote-Confidence Selective Prediction with a Stochastic Ensemble on FPGAs},
  year         = {2026},
  howpublished = {\url{https://github.com/hsmazumdar/SENA-stochastic-FPGA-learning}},
  note         = {Reproducibility package}
}
```

## What is not in this mirror

- The manuscript, drafts, and review notes
- Raw datasets and large build intermediates
- Secrets, credentials, or private correspondence

## License

MIT, see [`LICENSE`](LICENSE). Manuscript text remains the author's scholarly work.

## Claim-to-evidence map (manuscript V13d, November 2026)

[`CLAIMS_MAP_V13d.md`](CLAIMS_MAP_V13d.md) lists, for every table and figure of manuscript version V13d and its Supplementary Material, the result files and scripts in this repository that back it. It also lists what is deliberately left out (bitstreams, checkpoints, third-party libraries and build databases).

## Reviewer quick start (no hardware needed)

1. Install Python 3.8 or newer. matplotlib is optional and only needed for the Fig. 2 plot.
2. Run the reproducer:
   - Windows: double-click `reproduce_tables.bat`.
   - Any OS: run `python reproduce_tables.py`.
3. Open `reproduce_output/run_<timestamp>/`:
   - `tables.md`: Tables I–VII and the other numbers in Secs. V.C, V.F, V.G and VI, regenerated from the saved logs and reports in this repository.
   - `check.csv`: every regenerated value next to the number printed in the manuscript, with the source file it came from.
   - `fig2_risk_coverage.png`: Fig. 2, the risk–coverage curves.

The script exits with code 0 only when every value matches the manuscript at the precision printed there. It retrains nothing and does not touch hardware. To rerun the experiments themselves, see the scripts listed in [`CLAIMS_MAP_V13d.md`](CLAIMS_MAP_V13d.md).

### Rebuilding and programming the boards (optional; needs the boards and vendor tools)

| Board / design | Build | Program / run |
|---|---|---|
| DE2-115, on-chip learning core 784-160-10 (Tables II, IV, V) | `results_21Nov2026/OnChip_Learning/board_v1/quartus/BUILD_BOARD.bat` (Quartus Prime Lite 25.1) | `results_21Nov2026/OnChip_Learning/board_v1/quartus/PROGRAM_BOARD.bat`, then the host script in `results_21Nov2026/OnChip_Learning/board_v1/README_Board_Run.md` |
| DE2-115, learning core only (no board I/O) | `results_21Nov2026/OnChip_Learning/quartus_core/BUILD_CORE.bat` | – |
| DE2-115, DWN baseline (Table II) | `results_21Nov2026/Baseline_DWN_pkg/quartus/BUILD_DWN.bat` | – |
| Basys 3 (XC7A35T), 784-16-10 port (not used for any manuscript claim) | `results_21Nov2026/OnChip_Learning/basys3_v1/vivado/BUILD_BASYS3.bat` (Vivado) | `results_21Nov2026/OnChip_Learning/basys3_v1/vivado/PROGRAM_BASYS3.bat` |
| Papilio Pro, C1 cycle calibration (Table VII) | RTL in `papilio_calibration_C1/rtl_snapshot/` (Xilinx ISE; reports in `papilio_calibration_C1/ise_reports/`) | `python papilio_calibration_C1/host/papilio_calibrate.py --port COMx`; the saved run is `papilio_calibration_C1/papilio_calibration.json` |
| Papilio Pro, Boolean tile (Supplement S-XIII) | `sena2026/fpga_papilio_logic_learn/ise/build_in_docker.bat` | `sena2026/fpga_papilio_logic_learn/go_program_and_log.bat` |

Bitstreams are not in the repository. Rebuild them with the scripts above, or ask the author for them.
