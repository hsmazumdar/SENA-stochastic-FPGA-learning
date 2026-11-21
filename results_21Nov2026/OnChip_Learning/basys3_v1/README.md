# SENA on-chip learning — Digilent Basys 3 port (basys3_v1)

Artix-7 XC7A35T-1CPG236C (20,800 LUTs, 41,600 FFs, 50 BRAM36 = 1,800 Kb, 100 MHz oscillator, onboard FTDI USB-UART at B18 / A18).

This folder is a **new** port of `OnChip_Learning/board_v1`. The DE2-115 sources were not overwritten. The learning core is board_v1 **v1.1** with one added parameter, `H_LANES`, that can serialise the hidden-lane adders if the parallel design does not fit. When `H_LANES` is 0 or equal to `N_HID`, the cycle schedule matches v1.1 and the twin remains the bit-exact reference.

There is **no external SRAM** on the Basys 3. The 12,500-image MNIST split is about 10 Mbit and cannot live in the 1.8 Mbit of BRAM together with the weights (~1.02 Mbit). Training and test images are therefore **streamed over UART**.

Vivado place-and-route and bitstream generation are to be run later on **HsmPc1**. They were not run here. The LUT/BRAM figures below that come from Yosys `synth_xilinx` are a **rough xc7 estimate**, not a Vivado result. Board accuracies must come from a programmed-board `result.json`; none are invented here.

## 0. What is in this folder

| Path | What it is |
|---|---|
| `rtl/basys3_sena_top.v` | Board top: 50 MHz (clk100/2), stream-UART command FSM, 64-byte RX FIFO, LEDs and muxed 7-seg |
| `rtl/sena_learn_core.v` | Learning core v1.1 + `H_LANES` (0 = full parallel, same schedule as board_v1) |
| `rtl/sdp_ram.v` | Simple dual-port RAM, 1-cycle registered read, `ram_style = "block"` |
| `rtl/uart_rx_s.v`, `uart_tx_s.v`, `uart_rx_fifo.v` | 8N1 UART and a 64-byte first-word-fall-through RX FIFO |
| `vivado/basys3_sena.xdc` | Pins from the official Digilent Basys-3-Master.xdc |
| `vivado/build_basys3.tcl` | Non-project Vivado: synth, impl, bitstream, utilization and timing reports |
| `vivado/BUILD_BASYS3.bat`, `PROGRAM_BASYS3.bat`, `program_basys3.tcl` | Windows helpers for HsmPc1 |
| `host/sena_board_host.py` | Host driver. `--board basys3` (default) streams images; `--board de2` keeps the SRAM protocol |
| `host/sena_hw_twin.py`, `host/mnist_bin_t0.3_tr10000_te2500_s2026.npz` | Twin and dataset (copies) |
| `sim/` | Icarus testbench (`tb_top.v`) and `run_top_sim.sh` |
| `yosys/` | `synth_xc7.ys` plus estimate reports written by the box run |

Ident version is **0x21** (board_v1 was 0x11) so the host can tell the stream port from the SRAM port. Command *headers* (`V S L D T E Q`) are unchanged; `'T'` and `'E'` are followed by 100-byte images instead of an SRAM fetch. `'W'`, `'C'` and `'R'` are not implemented (no SRAM) and return `?`.

## 1. Clock, UART and pins

- Input clock 100 MHz on W5; system clock is a /2 toggle (**50 MHz**).
- UART 8N1, `BAUD_DIV=434` → 50e6/434 ≈ **115,207 baud** (use `--baud 115200`).
- `RsRx` = B18 (FPGA receive), `RsTx` = A18 (FPGA transmit), onboard FTDI — no RS-232 adapter.
- `btnC` (U18, centre, active high) = reset.
- `led[0]` heartbeat (~1.3 s), `led[1]` core busy, `led[2]` training, `led[3]` testing, `led[4]` streaming an image, `led[5]` FIFO overflow (sticky), `led[7]` RX activity, `led[14]` unknown opcode, `led[15]` UART frame error.
- 4-digit 7-seg (muxed, active-low) = samples trained, low 16 bits, hex.

If 50 MHz does not close in Vivado, rebuild with a /4 clock (25 MHz) and `BAUD_DIV=217`, and pass `--f-clk 25e6` to the host. That change is a clock-divider / constraint edit, not a change to the learning rule.

## 2. Image stream format

Each image is **100 bytes** = 50 little-endian 16-bit words, the same packing as `image_words()` in the DE2 host:

- words 0–48: 784 input bits (word *w* bit *b* = pixel 16*w*+*b*);
- word 49, bits 3:0: label.

After `'T' start3 count3` the host sends `count` images in **schedule order** (start3 is accepted and ignored). The board trains each one and, after the last, returns `'F'`, core_cycles, total_cycles, n_w1_updates, as before.

After `'E' first2 count2 T` the host sends `count` test images (first2 ignored). The board returns one pred byte after each image, then `'F'`, correct, core_cycles, total_cycles.

A 64-byte RX FIFO covers a back-to-back stream while the core is busy. At 115200 baud a full 784-160-10 train step is much shorter than a 100-byte UART frame, so the host can stream `'T'` images without a per-sample ACK. For `'E'`, the host waits for the pred before sending the next image.

Rough host wall time at 115200 baud (not a board measurement): 100 bytes × 10 bits × 120,000 train images ≈ 17 minutes of UART for a full 40 × 3,000 run, plus about 22 s for 2,500 test images, plus weight load/dump. This is slower than the DE2-115 path, where images live in SRAM.

## 3. Does it fit?  (BRAM is a calculation; LUTs are a Yosys estimate)

**Block RAM (calculation, independent of the synthesiser).**  
W1 is 785 × (160×8) = 1,004,800 bits; W2 is 161 × (10×8) = 12,880 bits; total **1,017,680 bits**, the same as the DE2-115 compile (146 M9K).  
A Xilinx BRAM36 is 36 Kb = 36,864 bits (or 32 Kb if parity bits are not used as data):

- 1,017,680 / 36,864 ≈ 27.61 → **28 RAMB36** if parity bits are used as data;
- 1,017,680 / 32,768 ≈ 31.06 → **32 RAMB36** if only 32 Kb per block is used.

The XC7A35T has **50 BRAM36 (1,800 Kb)**. Weight memory alone is about 56–64% of that, and the UART FIFO is negligible. **BRAM fits** under either packing. Images do not, which is why they are streamed.

**Logic (Yosys 0.52, `synth_xilinx -family xc7`, 20 Nov 2026).**  
This is a pre-place estimate, typically optimistic or pessimistic versus Vivado, and it is **not** a fit proof. Reports live in `yosys/`. The XC7A35T budget is 20,800 LUTs and 41,600 FFs.

| Build | H_LANES | Yosys LUTs | Yosys FFs | Yosys RAMB36 / RAMB18 | vs 20,800 LUTs | Notes |
|---|---|---|---|---|---|---|
| 784-160-10, full parallel | 0 or 160 | *no count* | — | — | larger than 32-lane; does not fit | Same cycle schedule as DE2-115 v1.1 (50,351 LEs on Cyclone IV). Yosys was **OOM-killed** at 07:57:51 IST on 20 Nov 2026, ~12,907 MB RSS (`dmesg`), during the 9.36 OPT pass. See `yosys/yosys_h160_l0_OOM.txt` and the incomplete `yosys/yosys_h160_l0.log`. No `stat_h160_l0.txt` was written. |
| 784-160-10, serial 32 | 32 | **42,025** | 21,570 | 36 / 3 | 202% — does not fit | 5 beats per row. Peak 10,381 MB, 582.50 s user. `yosys/stat_h160_l32.txt`. |
| 784-160-10, serial 16 | 16 | **37,131** | 21,570 | 36 / 3 | 179% — does not fit | 10 beats per row. Peak 6,550 MB, 379.03 s user. `yosys/stat_h160_l16.txt`. |
| 784-160-10, serial 8 | 8 | **34,348** | 21,570 | 36 / 3 | 165% — does not fit | 20 beats per row. Peak 3,959 MB, 261.45 s user. `yosys/stat_h160_l8.txt`. |
| 784-160-10, serial 4 | 4 | **32,516** | 21,570 | 36 / 3 | 156% — does not fit | 40 beats per row. Peak 2,499 MB, 202.23 s user. `yosys/stat_h160_l4.txt`. |
| 784-160-10, serial 1 | 1 | **29,595** | 21,570 | 36 / 3 | 142% — does not fit | 160 beats per row (fully serial hidden lanes). Peak 1,141 MB, 142.32 s user. `yosys/stat_h160_l1.txt`. This is the smallest 160-hidden build the `H_LANES` parameter can make. |
| 784-16-10, full parallel (smaller member) | 0 | **9,571** | 4,578 | 4 / 3 | 46% — fits in this estimate | Generics `N_HID=16 SH=4 BAUD_DIV=434`. Peak 1,136 MB, 69.60 s user. `yosys/stat_h16_l0.txt`. Useful for a Basys 3 bring-up of the stream-UART path. This is **not** the paper's 784-160-10 result; a 16-hidden member was already known to learn poorly (~65%). |

Sources: Yosys 0.52 (`git sha1 fee39a3284c90249e1d9684cf6944ffbbcbb8f90`), `synth_xilinx -family xc7 -top basys3_sena_top`, 20 Nov 2026 (Asia/Calcutta). Unless noted, generics are `N_HID=160 SH=7 BAUD_DIV=434`. LUT total is LUT1+…+LUT6 from the design-hierarchy `stat` block. Flip-flops are FDCE+FDPE+FDRE. No build used any DSP48E1. For the 160-hidden builds, equivalent BRAM is 37.5 RAMB36 (36×36 Kb + 3×18 Kb), within the 50 BRAM36 budget; Yosys packed more blocks than the 28–32 theoretical minimum. Flip-flop count is identical for every 160-hidden `H_LANES` (the 160-wide LFSR / accumulator state is not reduced by serialising the adder lanes).

**Fit (Yosys, not Vivado):** BRAM and FFs (52% of 41,600) fit for the 784-160-10 member. LUTs do **not**, even at `H_LANES=1` (29,595 vs 20,800, 42% over). Serialising the hidden adders has diminishing returns (32→16 saved 4,894 LUTs; 4→1 saved 2,921). Most of the remaining logic is the 160-wide state, not the adder lanes. A 784-16-10 member *does* fit in this estimate (9,571 LUTs, 4,578 FFs).

Vivado P&R on HsmPc1 is still the real fit test: Yosys can be pessimistic or optimistic versus Vivado. A 42% LUT overrun at the fully serial setting is unlikely to disappear in P&R, so the headline 784-160-10 on-chip learning result should stay on the DE2-115. On HsmPc1, a `H_LANES=1` attempt can still be run to confirm; a `N_HID=16` build is the one Yosys says will close, and is the sensible Basys 3 bring-up. Quote Vivado `util_impl.rpt`, not this table, in the paper.

The Vivado default in `build_basys3.tcl` is **`H_LANES=32`**. Override with `set H_LANES=1` for the smallest 160-hidden attempt, or `set N_HID=16` (and `SH=4`) for the smaller member. `set H_LANES=160` (or 0) is the full-parallel core.

`H_LANES` must be 0 or a positive divisor of `N_HID`. Arithmetic and LFSR sampling stay the same as v1.1; only the number of hidden lanes processed per clock changes.

## 4. Build and program (HsmPc1, later — do not run Vivado on the box)

1. Install / locate Vivado (WebPACK is enough for xc7a35t). Set `VIVADO_PATH` if it is not under `C:\Xilinx\Vivado\<year>\bin\vivado.bat`.
2. From `vivado\`:

```
BUILD_BASYS3.bat
```

Optional: `set H_LANES=160` first. The script writes `out_h160_l<H_LANES>\basys3_sena.bit` plus `util_impl.rpt` and `timing_impl.rpt`. Quote those reports, not the Yosys table, in the paper.

3. Plug the Basys 3 in with the **micro-USB UART/JTAG** cable (the same cable carries programming and the UART). Jumpers on the board should be in the usual QSPI/JTAG configuration used with Vivado Hardware Manager.

```
PROGRAM_BASYS3.bat
```

After programming, `led[0]` should blink about once every 1.3 s, the 7-seg should show `0000`, and `led[15]` / `led[14]` should be off. Press **btnC** to reset.

4. In Device Manager, note the USB-UART COM port (often a Digilent or FTDI COMx).

```
cd host
python sena_board_host.py --board basys3 --port COMx --seed 0 --f-clk 50e6
python sena_board_host.py --board basys3 --port COMx --seed 1 --f-clk 50e6
python sena_board_host.py --board basys3 --port COMx --seed 2 --f-clk 50e6
```

`--skip-load` does nothing useful here (there is no SRAM image to reuse). The twin reference accuracies for a 784-160-10 member, 40 × 3,000, T=40, software Task 2 — the board should match these **exactly** if it is bit-exact — are 2077, 2074 and 2155 out of 2500 for seeds 0, 1 and 2 (83.08%, 82.96%, 86.20%). Those are twin numbers, not Basys 3 measurements.

Outputs go to `host\runs\board_h160_s<seed>_<date_time>\` (`result.json`, `epochs.csv`, `final.npz`, `log.txt`), same layout as board_v1.

## 5. Icarus verification (box, 20 Nov 2026)

Icarus Verilog 13.0, `/workspace/eda/bin`. The host is used in emit/replay mode against `sim/tb_top.v` (UART BFM, no SRAM). Command:

```
cd sim
N_HID=16 SH=4 SEED=0 K=20 SPE=20 NT=4 TT=8 H_LANES=0 NAME=top_h16_l0_s0_k20 ./run_top_sim.sh
N_HID=16 SH=4 SEED=0 K=20 SPE=20 NT=4 TT=8 H_LANES=8 NAME=top_h16_l8_s0_k20 ./run_top_sim.sh
```

Results from `sim/runs/<name>/replay/result.json` and the matching `replay_console.txt` / `sim.log` (20 Nov 2026, Asia/Calcutta). These are Icarus vs twin comparisons on a shortened case (N_HID=16, SH=4, seed 0, 20 train samples, 4 test images), **not** board accuracies and **not** a 784-160-10 result. Emit-mode `result.json` files are not used: the emit pass writes dummy reads, so a mismatch there is expected.

| Case | BIT_EXACT vs twin | Pred / W1 / W2 mismatches | Notes |
|---|---|---|---|
| `top_h16_l0_s0_k20` (full parallel, v1.1 schedule) | **True** | 0 / 0 of 12,560 / 0 of 170 | nupd 17=17, nb_fx 329=329, train core_cyc 29,979. fifo_ovf=0, frame_err=0. Test 0/4 on both board and twin. The `_noread` variant produced the same numbers. |
| `top_h16_l8_s0_k20_noread` (serial, 2 beats/row) | **True** | 0 / 0 of 12,560 / 0 of 170 | nupd 17=17, nb_fx 329=329, train core_cyc 74,960. fifo_ovf=0, frame_err=0. Same 0/4 test. Confirms that `H_LANES < N_HID` stays bit-exact with the non-overlapped serial schedule. |

## 6. Protocol reminder

`V` ident, `S` seed, `L`/`D` load and dump a weight row, `T` train, `E` evaluate, `Q` status. Unknown opcodes (including the DE2 SRAM commands `W`/`C`/`R`) return `?` and light `led[14]`.

Python 3 with `numpy` and, for a real board, `pyserial`.
