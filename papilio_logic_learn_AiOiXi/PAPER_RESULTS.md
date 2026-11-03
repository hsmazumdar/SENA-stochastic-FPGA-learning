# SENA Papilio Pro — Ai/Oi/Xi (AND/OR/XOR) on-chip learning PoC

## Verdict
**Silicon PASS.** Continuous Mazumdar training on Papilio Pro reached and held **100% joint accuracy (64/64)** across all host polls, with early-exit mean depth **3** stochastic passes. Logger stopped; board train disabled (`S`).

## What was measured
| Item | Value |
|------|--------|
| Board | Papilio Pro, XC6SLX9-2-TQG144, 32 MHz |
| UART | COM11 @ 115200, IDENT `A5 02` |
| Bitstream | `papilio_logic_learn_top.bit` (programmed 2026-08-01, ~594 ms) |
| Inputs | Ai1,Ai2, Oi1,Oi2, Xi1,Xi2 |
| Targets | Ao=AND, Oo=OR, Xo=XOR |
| Fabric | 6–6–4 with skip (42+28+24 connections) |
| Host polls | 30 s interval, epochs **78 → 1164** at stop |
| Joint accuracy | **100.0%** on every poll |
| Mean T (EE) | **3** |

## Suggested manuscript sentence
> On Papilio Pro, a 6–6–4 Mazumdar multiplier-free tile trained concurrent AND/OR/XOR mappings (Ai/Oi/Xi) on-chip; a host UART log at 30 s intervals recorded 100% joint accuracy (64/64 patterns) with early-exit mean depth 3 across the soak (epochs 78–1164).

## Package contents
- `papilio_logic_learn_top.bit` — bitstream
- `papilio_logic_learn_silicon.json` — machine-readable silicon summary
- `pc_progress.jsonl` / `results/pc_progress.jsonl` — raw host log
- `Q1_3xNN_TABLES.md` — software twin tables + hardware status
- `rtl_snapshot/` — Verilog used for the PoC
- `ise_reports/` — TWR / MRP / PAR
- `host/` — UART logger + golden twin scripts
- `go_program_and_log.bat` — reproducer

## Reproduce
```bat
go_program_and_log.bat COM11
REM later: send S via host or stop python logger
```
