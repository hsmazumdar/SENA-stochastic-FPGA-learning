# DE2-115 on-chip MNIST learning — summary

On-chip single-member 784-160-10 training on Terasic DE2-115 (Cyclone IV E EP4CE115F29C7, 25 MHz). Board matches software twin bit-exactly. Ensemble M = 3 accuracy (≈90.8%) is not claimed from this table.

| Seed | Board correct / 2500 | Board acc. | Twin acc. | Pred mismatches | W1 / W2 mismatches | BIT_EXACT |
|---|---:|---:|---:|---:|---:|:---:|
| 0 | 2077 | 83.08% | 83.08% | 0 | 0 / 0 | Yes |
| 1 | 2074 | 82.96% | 82.96% | 0 | 0 / 0 | Yes |
| 2 | 2155 | 86.20% | 86.20% | 0 | 0 / 0 | Yes |
| Mean | — | 84.08% | 84.08% | 0 | 0 / 0 | Yes |

**Resources (Quartus 25.1 fit):** 50,351 / 114,480 LEs (44%); 1,017,680 memory bits (26%); 0 / 532 multipliers.  
**Logs:** `Revision_V13/DE2_115_HW_21Nov2026/onchip_learning/board_h160_s*_*/result.json`
