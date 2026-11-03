# SENA C1 calibration on DE2-115

Port of the Papilio Pro 784→16→10 silicon calibration slice to
Terasic DE2-115 (Cyclone IV E EP4CE115F29C7).

## Clock

- Board oscillator: 50 MHz (`CLOCK_50`)
- Core / UART clock: **25 MHz** (`CLOCK_50 / 2`)
- Reason: Papilio post-route f_max ≈ 32.6 MHz; 50 MHz does not close timing

## Build / program

```bat
BUILD_AND_PROGRAM.bat
```

Or Quartus: open `de2_115_calib.qpf`, compile, program `.sof` via USB-Blaster.

## Visual check (no UART)

- **LEDG0** heartbeats ~1 Hz when bitstream is alive
- **LEDR0** lights while the core is busy (inference)

## Host C1 calibration (needs RS-232)

DE2-115 UART is the **DE9 RS-232** connector (`UART_TXD`/`UART_RXD`), not the
USB-Blaster cable. Use a USB–RS232 adapter (null-modem if needed).

```bat
cd host
python de2_115_calibrate.py --list-ports
python de2_115_calibrate.py --port COMx
```

Expect **cycles/pass = 807.00** for every fixed T (same model as Papilio).

## Protocol (unchanged)

115200 8N1: `V` / `X`+98 / `N`+2 / `R`+2 → same as Papilio README.
