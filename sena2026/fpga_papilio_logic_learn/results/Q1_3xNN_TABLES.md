# Q1 paper tables — 3×NN (AND/OR/XOR) Mazumdar PoC
## Topology
| Layer | Size | Notes |
| --- | --- | --- |
| i (inputs) | 6 | Ai1,Ai2, Oi1,Oi2, Xi1,Xi2 |
| j (hidden) | 6 + bias | bias column → 7 |
| k (outputs) | 3 + pad | Ao, Oo, Xo (+ unused row) |
| i→j | 42 | 6×7 |
| j→k | 28 | 4×7 |
| i→k skip | 24 | 4×6 |
| Epoch | 64 | all 6-bit patterns, on-the-fly targets |

## Table A — Final accuracy (5 seeds, mean ± std)
| Config | AND % | OR % | XOR % | Joint % | Mean T | Epochs |
| --- | --- | --- | --- | --- | --- | --- |
| skip_ee | 100.0±0.0 | 100.0±0.0 | 96.9±1.9 | 96.9±1.9 | 3.00±0.00 | 24±7 |
| skip_fixedT | 100.0±0.0 | 100.0±0.0 | 99.1±2.1 | 99.1±2.1 | 32.00±0.00 | 22±10 |
| noskip_ee | 99.1±2.1 | 99.7±0.7 | 96.6±4.5 | 95.6±5.6 | 3.00±0.00 | 79±2 |

**Config key:** `skip_ee` = skip + early-exit (paper default); `skip_fixedT` = skip, fixed T=16; `noskip_ee` = no i→k skip.

## Table B — Learning curve (skip_ee, seed mean at selected epochs)
| Epoch | AND % | OR % | XOR % | Joint % | Mean T |
| --- | --- | --- | --- | --- | --- |
| 1 | 82.5 | 89.1 | 51.6 | 38.8 | 3.00 |
| 5 | 99.4 | 99.4 | 57.2 | 56.9 | 3.00 |
| 10 | 99.4 | 99.4 | 78.1 | 77.2 | 3.00 |
| 15 | 99.7 | 100.0 | 92.8 | 92.5 | 3.00 |
| 20 | 99.6 | 100.0 | 98.0 | 97.7 | 3.00 |
| 25 | 100.0 | 100.0 | 99.5 | 99.5 | 3.00 |
| 30 | 100.0 | 100.0 | 98.4 | 98.4 | 3.00 |
| 35 | 100.0 | 100.0 | 100.0 | 100.0 | 3.00 |

## Table C — Hardware status (2026-08-01 IST) — SILICON CLOSED
| Item | Status |
| --- | --- |
| Golden twin (this table) | DONE — software Mazumdar, 5 seeds |
| Papilio RTL (42+28+24, EE, LED, UART) | DONE — ISE XST-clean |
| Bitstream | **DONE** — `ise/papilio_logic_learn_top.bit` (340712 B) via Docker ISE 14.7 |
| On-board program | **DONE** — `papilio-prog` 2026-08-01 (~594 ms) |
| UART COM11 | **DONE** — IDENT `A5 02`; train stopped with `S` |
| PC 30 s progress logger | **DONE** — 100% joint (64/64), mean T=3, epochs 78→1164 |
| Bring-back | `SENA_Project\papilio_logic_learn_AiOiXi\` + `papilio_logic_learn_silicon.json` |

## Suggested manuscript sentence
> On the 6–6–4 AND/OR/XOR PoC (42+28+24 connections), Mazumdar multiplier-free online training reaches **96.9±1.9%** joint accuracy across 5 seeds, with early-exit mean depth **3.00** stochastic passes (vs fixed T=16).
