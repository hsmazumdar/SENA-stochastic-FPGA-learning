#!/bin/bash
# Reduced-case top-level simulation: host emits a byte script, Icarus runs the full board top
# (UART + SRAM + core) against it, host replays the received bytes and checks vs the twin.
set -e
HERE=$(cd "$(dirname "$0")" && pwd); RTL=$HERE/../rtl; HOST=$HERE/../host
N_HID=${N_HID:-16}; SH=${SH:-4}; SEED=${SEED:-0}; K=${K:-60}; SPE=${SPE:-30}; NT=${NT:-12}; TT=${TT:-8}; DIV=${DIV:-8}
NAME=${NAME:-top_h${N_HID}_s${SEED}_k${K}}; OUT=$HERE/runs/$NAME; mkdir -p $OUT
ARGS="--seed $SEED --n-hid $N_HID --sh $SH --spe $SPE --sched-limit $K --n-test $NT --T $TT --sparse-load"
python3 $HOST/sena_board_host.py --mode emit --script $OUT/script.txt --out $OUT/emit $ARGS > $OUT/emit.log
iverilog -g2012 -DN_HID=$N_HID -DSH=$SH -DBAUD_DIV=$DIV -DSCRIPT=\"$OUT/script.txt\" -DRXLOG=\"$OUT/rx.txt\" \
  -o $OUT/tb.vvp $HERE/tb_top.v $HERE/is61wv102416_model.v $RTL/de2_115_sena_top.v $RTL/sena_learn_core.v \
  $RTL/sdp_ram.v $RTL/uart_rx_s.v $RTL/uart_tx_s.v $RTL/sram_ctrl.v
( time vvp -n $OUT/tb.vvp ) > $OUT/sim.log 2>&1
tail -3 $OUT/sim.log
python3 $HOST/sena_board_host.py --mode replay --rxlog $OUT/rx.txt --out $OUT/replay $ARGS | tail -8
