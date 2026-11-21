# SENA on-chip learning — DE2-115 board run (board_v1) — Monday runbook

Goal: program the DE2-115 and run the full on-chip training run: 784-160-10, 40 × 3,000 samples, test on 2,500 images at T=40, seeds 0/1/2. Then confirm bit-exactness against the software twin and write JSON/CSV for the paper.

Everything in this folder was built and **simulated** on 14 Nov 2026 (Icarus, full top level: UART + SRAM + core). Nothing here is a board measurement yet. Board numbers come only from `host\runs\...\result.json` after Monday.

## 0. What is in this folder
| Path | What it is |
|---|---|
| `rtl/de2_115_sena_top.v` | Board top: clk25 = CLOCK_50/2, UART protocol FSM, SRAM loader/streamer, training/test sequencer, LEDs/HEX |
| `rtl/sena_learn_core.v` | Learning core **v1.1** (v1 + a pipeline stage in the W2U back-signal sum; bit-exact re-tests listed below) |
| `rtl/sdp_ram.v`, `uart_rx_s.v`, `uart_tx_s.v`, `sram_ctrl.v` | RAM, 8N1 UART (115200 @ 25 MHz, DIV=217), IS61WV102416 controller |
| `quartus/` | `de2_115_sena.qpf/.qsf/.sdc` (EP4CE115F29C7, all pins), `BUILD_BOARD.bat`, `PROGRAM_BOARD.bat`, `output_files/de2_115_sena.sof` (if the compile was done) |
| `host/sena_board_host.py` | Host driver (pyserial). Loads data, trains, tests, dumps, compares with the twin, writes results |
| `host/sena_hw_twin.py`, `host/mnist_bin_t0.3_tr10000_te2500_s2026.npz` | Twin and dataset (copies), so the folder is self-contained |
| `sim/` | Top-level testbench (`tb_top.v`, UART BFM, `is61wv102416_model.v`) and `run_top_sim.sh`, plus `runs/*/replay/result.json` |

## 1. Hardware set-up (office, HsmPc1)
1. The DE2-115 is powered by the 12 V adapter. Turn on with the red power button.
2. Connect a USB cable from HsmPc1 to the **USB BLASTER** port, which is the left USB-B, next to the power jack. Set **SW19 to RUN**.
3. Plug the USB-to-RS-232 adapter straight into the board's **DB9 (RS-232) connector**. Do not use a null-modem adapter at first.
   - Find the COM number in Device Manager → Ports (COM & LPT), e.g. `COM5`.
   - **ESP32 is not supported by this build.** The DB9 carries ±RS-232 levels, which are not 3.3 V-safe. An ESP32 would need different pins on the GPIO header and a rebuilt pin assignment. Use the USB-RS-232 adapter.
4. Plug in the E: SSD. Quartus is at `E:\FPGA_Tools_115\IntelFPGA\25.1\quartus\bin64`.
   - If Windows does not see the USB-Blaster, install its driver from `E:\FPGA_Tools_115\IntelFPGA\25.1\quartus\drivers\usb-blaster` (Device Manager → Update driver → browse).
5. Python on HsmPc1 (once): `pip install pyserial numpy`.

## 2. Program the FPGA (volatile JTAG; redo after every power cycle)
```
cd /d <SENA_ROOT>/Revision_V13\OnChip_Learning\board_v1\quartus
PROGRAM_BOARD.bat
```
Expected: `quartus_pgm -l` lists `USB-Blaster [USB-0]`, then `Info: Configuration succeeded`.
Alternative: Quartus GUI → Tools → Programmer → Add File `output_files\de2_115_sena.sof` → Start.

After programming:
- **LEDG0 blinks** about once every 1.3 s (heartbeat).
- HEX3..0 show `0000`.
- LEDR all off.

**KEY0 = reset.** Reset does not erase SRAM, so loaded images survive; the weights are reloaded by the host anyway.

If there is no .sof (the compile was not done on 14 Nov), run `BUILD_BOARD.bat` first. It takes about 10–20 min; the log is in `build_log.txt`.

## 3. Run (from the `host` folder)
```
cd /d <SENA_ROOT>/Revision_V13\OnChip_Learning\board_v1\host
python sena_board_host.py --port COM5 --seed 0
python sena_board_host.py --port COM5 --seed 1 --skip-load
python sena_board_host.py --port COM5 --seed 2 --skip-load
```
`--skip-load` skips the 1.6 MB image upload; it still verifies the SRAM checksum and always reloads the seed-specific schedule. Use it only if the board was not powered off since the first run.

### Expected console output (seed 0)
```
ident: version 0x11 N_HID=160
  images: 0% ... 98%                                   (about 2.5 min @115200)
images written in ~145 s
image region read-back checksum OK (0x........, 800000 words)
schedule (120000 entries) read-back checksum OK        (about 25 s)
initial weights loaded and read back OK                (about 30 s)
seed 0 -> seed_in 0x1d872b41; status nupd0=0 samples0=0 nb_fx=384
epoch  0: 3000 samples core ... cyc total ... cyc (~... ms @25MHz) nupd board N twin N nb_fx board B twin B OK
...  (40 lines; each epoch should take well under 1 s on the board)
test: board XXXX/2500 = YY.YY%  twin 83.08%  pred mismatches 0
final weights: W1 mismatches 0/125600, W2 0/1610 -> BIT_EXACT=True
```
Twin reference accuracies, software, Task 2 (the board should reproduce these **exactly**):

| Seed | Accuracy | Correct images |
|---|---|---|
| 0 | 83.08% | 2077/2500 |
| 1 | 82.96% | 2074/2500 |
| 2 | 86.20% | 2155/2500 |

Calculated (not measured) board compute time at 25 MHz:
- Training: about 1.1–1.9 k core cycles per sample, plus about 310 cycles of SRAM fetch per sample (measured in the top-level simulation). That is about 7–8 s for 120k samples.
- Test: about 7.4 k cycles per image at T=40, plus fetch. That is about 0.75 s for 2,500 images.
- The host also reports the on-board cycle counters; use those, not these estimates, in the paper.

Total wall time per seed: about 4 min for the first run (dominated by the UART upload), about 1.5 min with `--skip-load`.

### Outputs (per run): `host\runs\board_h160_s<seed>_<date_time>\`
- `result.json`: board accuracy, twin accuracy, pred/W1/W2 mismatch counts, `BIT_EXACT`, train/test cycles (core-busy and total), seconds at 25 MHz, cycles per sample and per image, host wall times.
- `epochs.csv`: per epoch, cycles, the W1-update counter and nb_fx (board versus twin).
- `final.npz`: dumped board weights and predictions, alongside the twin's.
- `log.txt`: the console log.

Exit code is 0 if BIT_EXACT, 2 if there is a mismatch. The host stops at the first protocol or checksum error.

## 4. Protocol (for reference; implemented in RTL and host)
Commands are `V` ident, `W` write SRAM (sum16 echo), `C` SRAM read-back sum32, `R` read SRAM, `S` seed (core init), `L` load weight row (sum16), `D` dump weight row (+sum16), `T` train schedule range, `E` evaluate, and `Q` status. Unknown opcodes get `?` and light LEDR17.

SRAM layout:
- Image n is at word n·64. Words 0–48 hold the 784 bits; word 49 holds the label. Images 0–9999 are train, 10000–12499 are test.
- The schedule is at word 0xD0000.

## 5. Troubleshooting
| Symptom | Check / fix |
|---|---|
| `timeout: wanted 7 bytes, got 0` on ident | Wrong COM port; FPGA not programmed (LEDG0 not blinking); press KEY0 and retry. Next, try a **null-modem** adapter or gender changer (TX/RX swapped). Close PuTTY or any other program holding the port. |
| LEDG7 never flashes while the host sends | Nothing reaches UART_RXD: cable/adapter problem, or TX/RX swapped → null-modem. |
| LEDR0 on (frame error) | Baud mismatch or noise. Use `--baud 115200` (the default); try another adapter (FTDI or Prolific). |
| LEDR17 on | The board received an unknown opcode because the host and board are out of sync. Press KEY0, then restart the host script. |
| `W sum mismatch` / `C ... checksum ... != expected` | UART bytes were lost or SRAM failed. Rerun; if it repeats, try another USB port or adapter. A `C` failure with all `W` OK points to SRAM. |
| `weight read-back mismatch` | The design was programmed with a different N_HID, or there is a host/RTL version mismatch. Rerun `PROGRAM_BOARD.bat`. |
| `nb_fx after seed = ..., expected 384` | 'S' did not reach the core. Press KEY0 and rerun. |
| Epoch line says MISMATCH | Not bit-exact with the twin. **Keep the run folder**; `first_diverged_epoch` in result.json says where. Report it as is and do not tune. |
| Quartus Programmer: no hardware | Install the USB-Blaster driver (see §1.4), replug the cable, SW19=RUN. |

## 6. Verification already done (simulation, 14 Nov 2026)
- **Core v1.1** (pipelined back-signal sum) versus twin, Icarus e2e: all five cases are BIT_EXACT, with 0 W1/W2/prediction mismatches.
  - smoke (4 hidden);
  - 16 hidden for 1000 samples + 200 test;
  - 16 hidden extreme weights;
  - 160 hidden for 300 samples + 50 test;
  - 160 hidden extreme weights.
  - The latency/LFSR unit test passes.
- **Board top level** (UART BFM + IS61WV102416 model + the real host script in emit/replay mode): all BIT_EXACT, including the per-epoch W1-update counter and nb_fx.
  - `top_h16_s0_k60`: 16 hidden, 60 samples, 12 test, T=8.
  - `top_h16_s1_k90`: 16 hidden, 90 samples, 16 test, T=8.
  - `top_h160_s2_k20`: 160 hidden, 20 samples, 4 test, T=8; this loads and dumps all 125,600 W1 weights through the UART.

## 7. Quartus full compile of this board top (Vivobook, Quartus Prime Lite 25.1std, 14 Nov 2026, 22:28 IST)
The compile succeeded (0 errors) and produced `quartus/output_files/de2_115_sena.sof`. Device EP4CE115F29C7.

| Metric | Value |
|---|---|
| Logic elements | 50,351 / 114,480 (44%) |
| Registers | 19,991 |
| Memory | 146 M9K (34%), 1,017,680 bits |
| Multipliers / PLLs | 0 / 0 |
| Pins | 103 |
| **Fmax clk25, slow 85 °C** | **34.77 MHz** |
| Fmax clk25, slow 0 °C | 38.09 MHz |
| Setup slack at 25 MHz | +11.24 ns (85 °C) |
| Hold, recovery and removal slack | All positive |
| PowerPlay total | 404.24 mW |
| PowerPlay breakdown | core dynamic 213.59 mW, core static 102.45 mW, I/O 88.19 mW |

PowerPlay confidence is **Low** because it uses the default 12.5% toggle rate rather than simulation activity. Quote it as an estimate.

The v1 core alone reached 27.95 MHz at slow 85 °C. The v1.1 pipeline stage lifts the critical path, so 25 MHz now closes with margin.

Warnings reviewed and benign:
- truncation warnings;
- constant outputs: unused LEDs and SRAM CE/UB/LB tied low;
- unused KEY[3:1];
- a loop-variable latch note.
