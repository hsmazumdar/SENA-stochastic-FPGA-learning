# SENA on-chip learning: hardware rule v1, bit-exact twin and learning core (steps 1–2)

*14 Nov 2026 (IST). All numbers come from runs made in this session. **None are board measurements.** The twin accuracies are software results from the bit-exact model of the RTL rule. The Quartus figures are a synthesis-only build of the core with a test wrapper, not the DE2-115 board top-level.*

## 1. Hardware rule v1
The full specification is in the header of `twin/sena_hw_twin.py`.
- **Network and weights:** one member, 784-160-10, 8-bit weights saturating at ±127, fixed power-of-two ranges (N = 64).
- **Randomness:** one 32-bit Galois LFSR per lane (taps 0x80200003). Each draw is 32 steps.
  - Bit fields used: [6:0] for the forward threshold, [13:8] for the r1/r2 sensitivity gate, [31:16] for the r3 gate.
- **Back-signal:** computed from the **pre-update** output weights.
- **N_back replacement (shift-only, replaces RMS):** `nb_fx += ((sum|back| >> 7) << 4 − nb_fx) >>> 7`. The gate range is the largest power of two ≤ nb_fx >> 4.
- **Arithmetic:** integer-only. The datapath uses adders, comparators and XORs only, with 0 multipliers.
- **Schedule and initialisation:** the same as the earlier software estimates (40 epochs × 3,000 samples, init uniform {−2..2}). Results use the **final weights** with no snapshot. Test inference uses T = 40 passes; the T = 1 and T = 8 values are read-outs of the same run.

## 2. Twin results (bit-exact software model, SENA 10k/2.5k split; NOT hardware)
| Seed | Test acc, T = 40 | Readout at T = 8 | Readout at T = 1 | W1 update events (of 120k samples) |
|---|---|---|---|---|
| 0 | 83.08% | 82.28% | 73.96% | 20,040 |
| 1 | 82.96% | 82.44% | 73.04% | 19,794 |
| 2 | 86.20% | 84.88% | 76.16% | 19,503 |
| **mean ± sd (n = 3)** | **84.08 ± 1.84%** | 83.20 ± 1.46% | 74.39 ± 1.60% | |
| supplementary, seeds 0–7 (n = 8) | 84.63 ± 2.26% | 83.66 ± 2.18% | 75.64 ± 3.22% | |

- **Run time:** about 25 s per seed (NumPy).
- **Comparison with the earlier estimate:** the earlier NumPy-RNG estimate of this rule gave 86.3 ± 0.7% (final weights). The bit-exact LFSR twin is about 2 pp lower, which is within seed spread at n = 3 but may be systematic.
- **Observation:** nb_fx settles near 0–9. That makes the r3 mask 0, so the back-signal gate reduces to "back ≠ 0". This could be tuned (for example, a smaller SH), but I did not tune it, to avoid overfitting to the test set.

## 3. RTL bit-exact verification (Icarus Verilog 13.0)
| Test | Configuration | W1 mismatches | W2 mismatches | Prediction mismatches | nb_fx / update count | Result |
|---|---|---|---|---|---|---|
| unit_latency | 8-4-10, unique row values, one-hot, all-0 and all-1 inputs | exact sums on all 10 patterns × 4 lanes | – | – | first LFSR draw = twin | PASS |
| smoke_h4 | 784-4-10, 20 train + 5 infer (T = 4) | 0/3,140 | 0/50 | 0/5 | equal | BIT-EXACT |
| e2e_h16_s0 | 784-16-10, 1,000 train + 200 infer (T = 8) | 0/12,560 | 0/170 | 0/200 | equal | BIT-EXACT |
| e2e_h16_extreme_s1 | 784-16-10, init at ±127/±126 (saturation stress; 2,994 weights at ±127), 300 + 50 | 0 | 0 | 0/50 | equal (nb_fx = 395, mask path active) | BIT-EXACT |
| e2e_h160_s0 | **784-160-10**, 300 train + 50 infer (T = 8) | 0/125,600 | 0/1,610 | 0/50 | equal | BIT-EXACT |
| e2e_h160_extreme_s2 | 784-160-10, saturation stress, 100 + 20 | 0 | 0 | 0/20 | equal | BIT-EXACT |
| **e2e_h160_1epoch_s1** | **784-160-10, 3,000 train (one full epoch) + 250 test (T = 40)** | **0/125,600** | **0/1,610** | **0/250** | equal (1,167 W1 updates) | **BIT-EXACT**; the RTL and the twin both score 70.4% on these 250 after 1 epoch |

**Negative controls (mutants must FAIL):**

| Mutant | What it changes | Result |
|---|---|---|
| M1 | Extra address register, i.e. the LX9 bug class | The core's alignment assertion fires (`W1 tag 1 != RAM row 0`) |
| M1b | Same bug with the assertion fooled | Bit-exact comparison catches it (11,687 W1 mismatches). The latency unit test also fails (acc = 58, expected 91). |
| M2 | Back-signal from post-update weights | 11,160 W1 mismatches |
| M3 | Saturating at −128 instead of −127 | 4,687 W1 mismatches |
| M4 | r2 taken from the wrong LFSR bit field | 12,088 W1 mismatches |

**Cycle counts** come from the simulation's cycle counter and match the closed form:
- About 1,119 cycles per training sample when there is no hidden update, and about 1,906 with one.
- 7,387 + 3 cycles per test image at T = 40.
- The full 40 × 3k run plus 2,500 test images comes to about 170 M cycles, which is about 6.8 s at 25 MHz (computed from the cycle counts, not measured).

**Icarus speed** at 784-160-10 is about 10–11k cycles/s, so a full run takes about 4–5 h. **Full-schedule Icarus runs for seeds 0, 1 and 2 were started in the background** (`vectors/full_h160_s{0,1,2}/`); their `result.json` files appear when they finish.

## 4. Quartus synthesis of the core alone (EP4CE115F29C7, Quartus Prime Lite 25.1, run on the Vivobook)
The build used the synthesis-only wrapper `quartus_core/sena_core_synth_top.v` (serial-loaded x/host registers, 22 pins). **It is not the board top-level.**

| Item | Value |
|---|---|
| Logic elements | **44,788 / 114,480 (39%)**; the core itself is 43,853 |
| Registers | 19,162 |
| M9K | **146 / 432 (34%)**: 143 for W1, 3 for W2 |
| Memory bits | 1,017,680 / 3,981,312 (26%) |
| Embedded 9-bit multipliers | **0 / 532** |
| Fmax | **27.95 MHz** (slow 1200 mV 85 °C), 30.89 MHz (slow 0 °C). The 50 MHz constraint fails (slack −15.8 ns); the planned 25 MHz board clock meets timing. |
| Critical path | W2 RAM read → 10-input back-signal sum → abs → `sum_abs` accumulate in one cycle. Adding one pipeline stage is the obvious fix, and it does not change the arithmetic. |

The earlier feasibility estimate (13–22k LE) was too low. The measured logic is 44.8k LE, which still fits comfortably.

## 5. Files
- `twin/sena_hw_twin.py`: the bit-exact twin and rule specification. `twin/run_twin_full.py` runs the 3-seed full schedule.
- `rtl/sena_learn_core.v`: the learning core. `rtl/sdp_ram.v` is the M9K simple-dual-port RAM with 1-cycle registered read, plus simulation assertions.
- `tb/tb_e2e.v` with `tb/run_e2e.py`: twin vector generation, Icarus run and bit-exact comparison.
- `tb/tb_unit_latency.v` with `tb/run_unit.py`: the latency and LFSR unit test.
- `mutants/`: the negative-control RTL variants and their run logs.
- `results/`: twin JSON and weights (`twin_full_h160_s*.json/.npz`), `e2e_summary.json`, unit results and logs.
- `quartus_core/`: the synthesis wrapper, qsf/sdc and `BUILD_CORE.bat`. The Quartus reports are in `quartus_core/output_files/` on the Vivobook copy.

## 6. Still to do
1. **DE2-115 board top-level:**
   - SRAM controller and image streamer: 10k+2.5k images (1.23 MB) go in SRAM, and `x_bits` is filled from SRAM per sample.
   - Host link: a USB-RS232 adapter with `uart_rx/tx`, or Virtual JTAG.
   - Command FSM for the schedule: the sample order comes from the host or a table in SRAM.
   - Weight dump and CRC, HEX/LED status, and a 25 MHz clock.
   - Optionally, pipeline the back-signal sum for more Fmax headroom.
2. **Verilator:** install it on the i9 (MSYS2/MinGW or WSL) and run the full 40 × 3k schedule (about 170 M cycles per seed) for 3 seeds, plus M = 3, bit-exact against the twin. The background Icarus full runs on the box may already give the 1-member full-schedule proof.
3. **Board runs** need the DE2-115, the E: SSD with Quartus, and Himanshu in one place. Programming the board and the host link are needed for these.
4. Optional:
   - Look into the nb_fx collapse and the roughly 2 pp gap against the NumPy-RNG estimate. Any rule change must be applied to the twin and RTL together.
   - M = 3 sequential members and ensemble early exit on the board.
