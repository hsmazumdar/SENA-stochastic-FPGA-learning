# B+E MNIST online PoC — status 2026-08-01

## Delivered
Package: `SENA_Project\sena2026\fpga_papilio_mnist_online\`

| Item | Result |
|------|--------|
| Bitstream | `ise/papilio_mnist_online_top.bit` (340993 B) |
| Util | **90% LUT / 97% slices / 18% BRAM16 / 0 DSP** |
| IDENT | `A5 03` on COM11 |
| Train | 400 samples (2×200), 4014 update events |
| Eval | 10/100 (**10%**) with EE hist populated |
| Telemetry E | `I` / `H` OK; per-class `C`/`S` still needs a mux audit |

## Silicon limits that forced scope cuts
1. Full **784→16→10 + reverse-gate hidden updates** → ~214% LX9 (LUT RAM / carry).
2. Fit revision: **784→8→10**, **output-layer Mazumdar ±1 only**, hidden = frozen int8 BRAM features from A2 ckpt.
3. Timing report f_max ≈ **26.5 MHz** vs 32 MHz board clock — paths are marginal (next: /2 clock or pipeline).

## How to re-run
```bat
go_program_and_run.bat COM11
```

## Next upgrades (still on LX9)
- Clock /2 (16 MHz) to clear timing
- Fix `C`/`S` tele mux if class tables look inconsistent
- Folded reverse-gate on **one** hidden unit/sample (re-add limited B depth)
- Longer train (full 800) once timing solid
