# SENA Papilio Pro Calibration Package (C1-lite / C2-lite)

Self-contained package to run the **PC–FPGA calibration experiment** on a
Papilio Pro board (Xilinx **Spartan-6 LX9**, 32 MHz oscillator, FT2232
USB-UART). Copy this whole `fpga_papilio/` folder to the FPGA PC — nothing
else from the repository is needed.

## What this experiment proves

The Q1 paper's FPGA speed/energy numbers rest on a cycle model
(`cycles_per_pass = N_IN reads + N_HID reads + fixed FSM overhead`) that was
validated under Verilator on the dev PC. This board run closes the loop **on
real silicon**:

1. **Cycle calibration** — the board reports an on-chip cycle counter for
   runs of 1–32 fixed passes. Expected: **exactly 807 cycles/pass**
   (784+2 hidden ROM reads + 1 fire + 16+2 output reads + 1 fire + 1 vote).
   The Verilator testbench already measures exactly 807; the board must
   match, proving the model transfers to hardware.
2. **Wall-clock / throughput** — inference latency at the real 32 MHz clock,
   including USB round-trip.
3. **C2-lite energy** — idle-subtracted board power from a USB power meter
   divided by inference rate gives a measured µJ/inference **upper bound**.
4. **Early-exit on silicon** — the confidence gate (θ = 9/16 ≈ 0.5625) runs
   in hardware with real trained weights (a 16-hidden slice of the A2 MNIST
   checkpoint, member 0).

The 16-hidden slice is NOT the paper's accuracy configuration — accuracy
evidence comes from A3 (RTL-semantic, 2500 samples). This design exists to
calibrate **timing and power**.

## Folder contents

```
fpga_papilio/
  rtl/
    papilio_calib_top.v   top: UART protocol + core (115200 8N1)
    calib_core.v          784->16->10 stochastic slice, LFSRs, early exit
    uart_rx.v, uart_tx.v  minimal UART (Verilog-2001)
    w_hid.mem, w_out.mem  REAL trained weights (from A2 checkpoint r0, m0)
  ise/
    papilio_pro.ucf       pins: clk P94, rx P101, tx P105, led P134
    build_ise.tcl         command-line ISE build (xtclsh)
  host/
    papilio_calibrate.py  calibration driver (needs pyserial + numpy)
    make_papilio_mems.py  regenerates the .mem files (dev PC only)
  tb/
    tb_papilio.cpp        Verilator TB (dev PC only) — PASSED, 807 cyc/pass
    build_and_run_tb.py
  papilio_slice.json      n_log2 register values + slice metadata
```

## FPGA-PC setup (one-time)

1. **ISE 14.7 WebPack** (free): download from AMD/Xilinx archive
   ("ISE Design Suite 14.7", WebPack license covers the LX9).
   On Windows 10/11 use the "ISE 14.7 Windows 10" VM-free release or run the
   64-bit tools; only XST/ngdbuild/map/par/bitgen + xtclsh are needed.
2. **Papilio Loader** (GadgetFactory) to program the bitfile over USB.
3. **Python 3** with `pip install pyserial numpy`.
4. Plug in the board; note the **two** COM ports the FT2232 creates.
   The UART is **channel B — usually the higher-numbered port**.

## Build the bitfile

Command line (recommended):

```bat
cd fpga_papilio\ise
xtclsh build_ise.tcl
```

Or in the ISE GUI: new project, family Spartan6, device xc6slx9, package
tqg144, speed -2; add the four `rtl/*.v` files and `ise/papilio_pro.ucf`;
copy `rtl/*.mem` into the project directory; set `papilio_calib_top` as top;
Generate Programming File.

Record from the post-route timing report: the design must meet the 31.25 ns
(32 MHz) constraint with large margin; note the reported minimum period —
that is the silicon f_max data point for the paper.

If XST complains about `$readmemh` with a parameter filename, hard-code the
two filenames in `calib_core.v` (two-line change, marked by the parameters
`W_HID_FILE` / `W_OUT_FILE`).

## Program the board

Papilio Loader GUI → select `papilio_calib.bit` → Write to FPGA (SRAM load is
enough; SPI-flash write survives power cycles).

## Run the calibration

```bat
cd fpga_papilio\host
python papilio_calibrate.py --port COM5
```

Expected console output:

- `board alive, fw version 1`
- a cycle table where **cycles/pass = 807.00 for every T** (deterministic;
  any other value means clock or protocol trouble),
- early-exit pass statistics over 20 runs,
- a 10 s inference loop for the power reading.

### C2 power measurement

1. Read the USB power meter with the board idle (bitfile loaded, no traffic):
   `P_idle` watts.
2. Re-run with a longer loop and read the meter during the loop phase:
   `python papilio_calibrate.py --port COM5 --loop-seconds 30` → `P_run`.
3. Final run recording both:
   `python papilio_calibrate.py --port COM5 --power-idle P_idle --power-run P_run`

No USB meter? A 0.1 Ω shunt in a sacrificial USB cable's 5 V line + DMM
works; P = 5 V × I. The FT2232 dominates idle draw, which is exactly why the
idle-subtracted number is the honest one.

## Bring the results back

Copy `host/papilio_calibration.json` to `sena2026/results/` on the main PC.
It contains: measured cycles/pass vs model, silicon f_clk, throughput,
early-exit pass histogram, and (if provided) the µJ/inference figure — cite
it as *board-measured* evidence level, distinct from the model-based numbers
in `fpga_speed_model.json`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| No reply to 'V' | Wrong COM port — try the other FT2232 channel |
| Garbage bytes | Check UCF pins against your board revision (GadgetFactory Papilio-Pro repo); rx=P101, tx=P105 assumes the stock wiring of FT2232 channel B |
| cycles/pass ≠ 807 | Should be impossible if the bitfile is this RTL; re-program |
| XST BRAM warnings | 785×256b ROM ⇒ ~16 RAMB16s of 32 — expected |
| Timing failure | Should not happen at 32 MHz; check the PERIOD constraint survived |
