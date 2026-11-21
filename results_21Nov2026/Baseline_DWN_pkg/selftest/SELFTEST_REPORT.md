# SELF-TEST REPORT (functional checks only, not results)

These checks were run on 14 Nov 2026 (IST) while the package was being built. They show that the code paths work end to end. **None of the numbers in this folder are baseline results.** The models are either a tiny 3-epoch CPU model (`selftest_tiny`, 120+50 LUTs) or random untrained tables. Do not cite them.

| # | Check | How | Outcome |
|---|---|---|---|
| 1 | SENA split reproduced | `make_split.py` rebuilt the split from raw MNIST (ossci mirror) and compared it with a read-only copy of SENA's `mnist_bin_t0.3_tr10000_te2500_s2026.npz` | all 6 arrays identical (`data/*.sha256.json`) |
| 2 | Pure-PyTorch EFD fallback = official CUDA kernel math | `test_efd_equivalence.py`: compares `EFDTorch` forward and backward with a literal scalar transcription of `efd_cuda_kernel.cu` (n = 2, 4, 6) | forward exact; gradients within 1.5e-7 (float rounding). PASS (`test_efd_equivalence.log`) |
| 3 | Export of **official** `torch_dwn` layers | `test_export_official_layers.py`: builds the model from the DWN repo classes (commit 2b456acf), with learnable and random mapping and a padded GroupSum (47 outputs), and checks that `export_model` + `infer_int` reproduce the official module's predictions and scores | identical. PASS |
| 4 | Training pipeline | `train_dwn.py --arch selftest_tiny`, 3 epochs, CPU, `--efd torch`; run under torch 2.15 and torch 2.7.1 (pinned) | runs; exported integer model agrees with PyTorch on 2500/2500 test inputs |
| 5 | Verilog vs Python, bit-exact | `dwn_to_verilog.py --sim` (Icarus Verilog 13.0) on (a) the tiny trained model, (b) a random full-size 1000+500 LUT-6 model (490 of 2500 inputs have tied top scores, which exercises the tie rule), (c) a random LUT-4/LUT-2 model with GroupSum padding and no pipeline registers | all three: `pred_mismatch=0 count_mismatch=0`, measured latency = expected latency (`gen_*/sim_result.txt`) |
| 6 | Negative control | one expected prediction deliberately corrupted | testbench reports `BIT-EXACT FAIL`; board top shows FAIL (LEDR17) with 1 mismatch |
| 7 | Board self-test wrapper | `quartus/tb_de2_115_dwn_top.v` simulated with (a) and (b) | PASS LEDs on, 0 mismatches, latency 5, 2504 cycles for 2500 inputs (`gen_*/top_sim_result.txt`) |
| 8 | Quartus flow | the tiny model compiled with Quartus Prime Lite 25.1std (EP4CE115F29C7) on the Vivobook, using the portable SSD install `E:\FPGA_Tools_115\IntelFPGA\25.1`, both revisions, via `compile_and_report.tcl` | both compiled; ROMs inferred as M9K; 0 multipliers; board revision met timing at 25 MHz; PowerPlay ran; reports copied. `dwn_core_only` shows the expected "timing requirements not met" against its deliberate 4 ns target. Summaries in `quartus_SELFTEST_tiny/` |
| 9 | Report parser | `collect_results.py` on the reports from item 8 | all fields parsed (`quartus_SELFTEST_tiny/collect_results_parse_test_SELFTEST.csv`) |
| 9b | Checkpoint import | `test_export_checkpoint.py`: a random official-class 1000+500 model is saved in the format of the earlier `baseline_dwn/train_sena_dwn_sm.py` and converted with `export_checkpoint.py` | predictions identical (2500/2500). PASS |
| 10 | CPU timing estimate | one epoch of the full `sm` model (1000+500) on the build box CPU (8 cores), `--efd torch` | about 6 s/epoch, which extrapolates to about 10 min/seed. Its accuracy is not reported (1 epoch, timing only) |

Not tested here:
- The official CUDA extension: the box has no GPU, so run `train_dwn.py --check-efd` on the GPU PC.
- Vivado: not installed on the box.
- Programming the real DE2-115 board.
- The full 100-epoch training of any seed.
