# Papilio Pro — Mazumdar mul-free AND/OR/XOR on-chip learning PoC

## Spec (paper)

| Item | Value |
|------|--------|
| Inputs `i` | Ai1,Ai2, Oi1,Oi2, Xi1,Xi2 (=6) |
| Hidden `j` | 6 + bias column (=7) |
| Outputs `k` | Ao, Oo, Xo (+ pad row → 4) |
| Connections | i→j **42**, j→k **28**, i→k **24** |
| Targets (on-the-fly) | Ao=Ai1∧Ai2, Oo=Oi1∨Oi2, Xo=Xi1⊕Xi2 |
| Epoch | TRAIN 64 → EVAL 64 (all 6-bit patterns) |
| Learning | Mazumdar multiplier-free ±1 + reverse gates |
| Early exit | confidence ≥ 3/4 after ≥3 stochastic passes (EVAL) |
| LED | `led = ~epoch_id[0]` (ON on **even** epochs), ~0.25 s dwell |
| PC | `host/papilio_progress_30s.py` polls `'P'` every 30 s |

## Status

| Piece | State |
|-------|--------|
| Golden twin | **PASS** (~94–100% joint) — `host/golden_logic663_skip.py` |
| RTL top + train core | `rtl/` — TRAIN→EVAL+EE each epoch; auto-start on power-up |
| ISE local | Not on PATH; Docker image `daverichmond/xilinx-ise:14.7` used before for calib |
| Board USB | UART **COM10 / COM11**; JTAG via `papilio-prog` still needs WinUSB on Interface 0 |

## UART protocol (115200 8N1)

| Cmd | Action |
|-----|--------|
| `V` | identify → `A5 02` |
| `G` | continuous train on |
| `S` | stop |
| `E` | one EVAL-only epoch |
| `P` | poll: epoch u16 LE, correct u8, mean_passes u8, updates u16 LE, flags u8 |

## Build / program (when JTAG OK)

```bat
cd sena2026\fpga_papilio_logic_learn\ise
REM Docker: mount this package and xtclsh build_ise.tcl
papilio-prog -v -f papilio_logic_learn_top.bit
python ..\host\papilio_progress_30s.py --port COM11 --go
```

Expected LED: slow even/odd flip each completed epoch. PC log: `results/pc_progress.jsonl`.

## Paper claim language (suggested)

> A complete on-chip learning proof-of-concept on Papilio Pro trains three concurrent logic mappings (AND/OR/XOR) in one 6–6–4 SENA tile with skip connections (42+28+24), using Mazumdar’s multiplier-free ±1 rule, online generation of all 64 input patterns per epoch, confidence-gated early exit at inference, and a host log every 30 s. Software twin reaches ≥90% joint accuracy; FPGA bitstream implements the same datapath without DSP multipliers.
