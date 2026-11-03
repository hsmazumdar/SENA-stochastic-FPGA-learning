# Analog Comfort PoC — GO status (2026-08-01)

## From `AnalogComfortPoC.docx` last chapter — implemented

| Step | Status |
|------|--------|
| Dataset (NBC/TSI soft \(p^*\)) | **DONE** — synthetic `data/train.csv` + `test.csv` (IEEE DataPort optional later) |
| Software Mazumdar train M=3 | **DONE** — test MAE ≈ **0.21**, RMSE ≈ **0.25** @ T=80 |
| Weight export BRAM `.mem` | **DONE** — `weights/w_*_m{0,1,2}.mem` |
| RTL 16→16→1 + M=3 + \(p̂=ones/T\) | **DONE** — `rtl/comfort_*.v` |
| ISE bitfile | **DONE** — `ise/comfort_sena_top.bit` (**35% LUT / 40% slices**, f_max **120 MHz**) |
| UART `A5 04` host | **DONE** — `host/papilio_comfort.py` |
| Board program + measure | **BLOCKED** — FTDI `0403:6010` Access Denied / papilio-prog cannot claim (hung COM after prior run). **Unplug Papilio 5 s, replug rear USB, then `go_program_and_run.bat COM11`** |

## Concept locked (chapter + chat)
- Comfort from **T + RH** (Indian NBC/TSI-style), not hard 0% error
- \(\hat{p}=ones/T\); T=100 → 1%, T=1000 → **0.1%**
- Ensemble M=3; anytime improve with more passes

## Next physical step
```bat
D:\_July2026\SENA_Project\comfort_poc\go_program_and_run.bat COM11
```
Then optional: `--t-passes 1000` for 0.1% resolution demo.
