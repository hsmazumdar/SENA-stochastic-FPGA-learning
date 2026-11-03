# Papilio Pro — folded 784→16→10 Mazumdar **online** slice (B + E)

## Goal
Richer LX9 PoC than Boolean Ai/Oi/Xi:
- **B:** weights in BRAM, host-streamed MNIST, on-chip Mazumdar ±1 on the
  **output layer** (serialized; hidden = frozen int8 features from A2 ckpt slice)
- **E:** per-class correct/seen, EE histogram, train/update counters

> Note: full reverse-gate hidden updates overflowed LX9 LUT/carry budget in the
> first cut; this revision keeps online learning + telemetry inside silicon limits.

## Topology
| Item | Value |
|------|--------|
| Net | 784 → **8** → 10 (int8; LX9 fit; output-layer online) |
| Device | XC6SLX9-2-TQG144 @ 32 MHz |
| Init | A2 ckpt member0 first-16 hiddens, clipped int8 |
| UART | COM11 (channel B), IDENT `A5 03` |

## Build
```bat
cd host
python make_int8_mems.py
cd ..\ise
build_in_docker.bat
```

## Run
```bat
papilio-prog -v -f ise\papilio_mnist_online_top.bit
python host\papilio_mnist_online.py --port COM11 --train 200 --test 100 --epochs 2
```

## Protocol (summary)
`V X L N R T Z P I C S H` — see `rtl/papilio_mnist_online_top.v` header.
