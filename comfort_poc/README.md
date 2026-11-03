# Analog Comfort PoC (from AnalogComfortPoC.docx last chapter)

Papilio Pro — stochastic Mazumdar **comfort probability** from T + RH (Indian NBC/TSI-style soft targets).

| Item | Value |
|------|--------|
| Net | 16 bit-features → 16 hid → 1 out × M=3 |
| Output | \(\hat{p}=ones/T\) (T=100 → 1%, T=1000 → 0.1%) |
| IDENT | `A5 04` |
| Train | Software Mazumdar ±1 (`host/train_comfort.py`) |
| Board | Inference + UART (`host/papilio_comfort.py`) |

```bat
python host\train_comfort.py
ise\build_in_docker.bat
go_program_and_run.bat COM11
```
