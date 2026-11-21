#!/bin/bash
# Rough xc7 estimates for several H_LANES. Not a Vivado P&R result.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
RTL=$HERE/../rtl
run_one () {
  local L=$1
  local OUT=$HERE/stat_h160_l${L}.txt
  local LOG=$HERE/yosys_h160_l${L}.log
  echo "=== Yosys synth_xilinx H_LANES=$L ==="
  yosys -l "$LOG" -p "
    read_verilog -sv $RTL/sdp_ram.v
    read_verilog -sv $RTL/sena_learn_core.v
    read_verilog -sv $RTL/uart_rx_s.v
    read_verilog -sv $RTL/uart_tx_s.v
    read_verilog -sv $RTL/uart_rx_fifo.v
    read_verilog -sv $RTL/basys3_sena_top.v
    chparam -set N_HID 160 -set SH 7 -set BAUD_DIV 434 -set H_LANES $L basys3_sena_top
    hierarchy -top basys3_sena_top
    synth_xilinx -family xc7 -top basys3_sena_top
    tee -o $OUT stat -width
  "
  echo "wrote $OUT"
  tail -40 "$OUT"
}
run_one 32
run_one 16
run_one 0
