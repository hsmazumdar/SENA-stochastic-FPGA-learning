# Papilio Pro hardware evidence

## Citable result

On 2026-07-26, a Papilio Pro (Xilinx Spartan-6 XC6SLX9-2) implementation of
the 784→16→10 SENA calibration slice was operated at the board's 32 MHz
clock. The on-chip cycle counter reported exactly 807 cycles per stochastic
pass for fixed run lengths T={1,2,4,8,16,32}, matching the closed-form model
with zero cycle error.

At 32 MHz, one pass takes 25.21875 µs. The retained early-exit sample averaged
30.25 passes over 20 runs, corresponding to 762.867 µs mean core latency.
Host-observed throughput, including UART protocol overhead, was 62.512
inferences/s.

Post-route timing analysis reported a 30.674 ns minimum period (32.601 MHz)
and 0.576 ns setup slack against the 31.25 ns constraint. All constraints
were met and all signals were routed.

## Evidence files

- `papilio_calibration.json`: board measurements and run metadata.
- FPGA build reports:
  `SENA_Papilio_FPGA/fpga_papilio/ise/papilio_calib_top.twr`,
  `papilio_calib_top.par`, and `papilio_calib_top.drc`.
- RTL and testbench:
  `SENA_Papilio_FPGA/fpga_papilio/rtl/` and
  `SENA_Papilio_FPGA/fpga_papilio/tb/`.
- Retained bitstream SHA-256:
  `AD35F00B8358937FC3FBF920826E02A05D9361DC9A1D5FC18A61B931A5181FC2`.
- Source calibration JSON SHA-256:
  `2996E3BBA89558D76AC05E242F9BD5E78EA842E2B2517F4025E3EFD009EDC610`.

## Claim boundaries

- This experiment validates the cycle model on physical FPGA hardware.
- It does not validate paper accuracy: the design is a 16-hidden calibration
  slice and the retained run used a deterministic random image.
- It does not yet provide measured energy: the JSON `energy` field is null.
- The post-route DRC warns that RAMB8BWER initialization may be incorrect
  (Xilinx Answer Record 39999). A software-golden prediction check or a
  RAMB16-only rebuild is required before claiming weight-faithful FPGA
  inference.
- The checkpoint path recorded in the JSON is not currently present in this
  workspace. Preserve or restore the exact checkpoint before claiming that
  the packaged ROMs can be regenerated independently.
- The measured 32 MHz timing margin is narrow (1.9%), not large.

## Remaining C2 measurement

Record idle and active USB power, in watts, then rerun:

```text
python papilio_calibrate.py --port COM5 --loop-seconds 30 --power-idle P_IDLE --power-run P_RUN
```

Preserve both raw meter readings. The resulting idle-subtracted
µJ/inference value includes UART overhead and should be reported as an upper
bound for this calibration slice.
