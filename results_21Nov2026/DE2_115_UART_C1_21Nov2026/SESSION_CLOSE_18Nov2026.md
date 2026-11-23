# SENA DE2-115 — Session close 18 Nov 2026

**Status:** CLOSED for today. Results saved under `quartus_calib/output_files/`.

## Board / tools
- Board: Terasic DE2-115, Cyclone IV E **EP4CE115F29C7**
- Power: 12V DC; PC link: **USB-Blaster** (JTAG)
- Quartus: 25.1std Lite at `<TOOLS>/IntelFPGA\25.1\quartus\bin64`
- Project: `<TOOLS>/SENA_DE2_115\quartus_calib\`

## Done today (silicon)

| Item | Result |
|------|--------|
| USB-Blaster detect / program | OK (earlier today) |
| Bitstream | `output_files/de2_115_calib.sof` (11:57 IST, checksum programmed successfully) |
| Design | 784→**16**→10 SENA slice, trained weights (`w_hid.mem` / `w_out.mem` from A2 checkpoint member 0) |
| Embedded MNIST BIST | **HEX showed PASS** (user confirmed) |
| Images (PASS) | MNIST te idx 1→label **1**, 18→**0**, 33→**4** (`embed_vectors.json`) |
| DSP multipliers | **0** |
| Fit | 5,890 LEs (5%), 2,895 regs, 203,680 memory bits |
| Timing @ 25 MHz core | Setup slack **+1.075 ns** (clk_25), TNS 0 |

### Paper-safe claim (draft)
*A 784→16→10 SENA slice with trained weights was demonstrated on Terasic DE2-115 (Cyclone IV EP4CE115), correctly classifying three embedded MNIST samples on-chip (HEX PASS), using zero DSP blocks.*

### Do **not** claim from today alone
- Full paper MNIST accuracy / 784→160→10 ensemble
- Host UART C1 cycles/pass = 807
- Board power / energy / throughput over UART
- Nexys4 / Artix-7 / Vivado results

## Artifacts to keep
- `output_files/de2_115_calib.sof`
- `output_files/de2_115_calib.fit.summary` (+ `fit_summary_snapshot.txt`)
- `output_files/de2_115_calib.sta.summary` (+ `sta_summary_snapshot.txt`)
- `output_files/embed_vectors.json`
- `output_files/program_18Nov2026.log` (earlier program)
- `output_files/NN_EMBED_BIST_18Nov2026.md`
- `output_files/VALIDATION_18Nov2026.md`
- `rtl/embed_vectors.vh`, `rtl/calib_core.v` (LFSR re-seed on start)
- `de2_115_calib_top.v` (LOAD→SETTLE→START BIST)
- `host/gen_embed_mnist.py`, `host/de2_115_calibrate.py`
- This file: `SESSION_CLOSE_18Nov2026.md`

## Pending next session
1. **Friday:** USB–RS232 on DE9 → `python host/de2_115_calibrate.py --port COMx` (expect cycles/pass **807**)
2. Optional: more embedded images; on-chip 807 cycle BIST; **784→160→10** port
3. Nexys4 / Vivado: still blocked (Vivado not installed; see `<HOME>/vivado_auto_install\PAUSE_16Nov2026.md`)
4. Copy this note to `<SENA_ROOT>\...` when K: is mounted; update `CLAIM_GAP_CLOSURE.md`

## Resume commands
```bat
<TOOLS>/IntelFPGA\25.1\quartus\bin64\jtagconfig.exe
<TOOLS>/IntelFPGA\25.1\quartus\bin64\quartus_pgm.exe -c "USB-Blaster [USB-0]" -m jtag -o "p;<TOOLS>/SENA_DE2_115\quartus_calib\output_files\de2_115_calib.sof"
```
On PASS: HEX **PASS**, LEDG on. KEY1 re-runs. KEY2 on FAIL shows idx|expect|got.
