#!/bin/bash
# Reduced-case top-level simulation: host emits a byte script, Icarus runs the
# Basys 3 top (UART stream + core, no SRAM) against it, host replays vs the twin.
set -e
HERE=$(cd "$(dirname "$0")" && pwd); RTL=$HERE/../rtl; HOST=$HERE/../host
IVERILOG=${IVERILOG:-/workspace/eda/bin/iverilog}
VVP=${VVP:-/workspace/eda/bin/vvp}
N_HID=${N_HID:-16}; SH=${SH:-4}; SEED=${SEED:-0}; K=${K:-60}; SPE=${SPE:-30}; NT=${NT:-12}; TT=${TT:-8}
DIV=${DIV:-8}; H_LANES=${H_LANES:-0}
NAME=${NAME:-top_h${N_HID}_l${H_LANES}_s${SEED}_k${K}}; OUT=$HERE/runs/$NAME; mkdir -p $OUT
ARGS="--board basys3 --seed $SEED --n-hid $N_HID --sh $SH --spe $SPE --sched-limit $K --n-test $NT --T $TT --f-clk 50e6 ${EXTRA_ARGS:-}"
python3 $HOST/sena_board_host.py --mode emit --script $OUT/script.txt --out $OUT/emit $ARGS > $OUT/emit.log
$IVERILOG -g2012 -DN_HID=$N_HID -DSH=$SH -DBAUD_DIV=$DIV -DH_LANES=$H_LANES \
  -DSCRIPT=\"$OUT/script.txt\" -DRXLOG=\"$OUT/rx.txt\" \
  -o $OUT/tb.vvp $HERE/tb_top.v $RTL/basys3_sena_top.v $RTL/sena_learn_core.v \
  $RTL/sdp_ram.v $RTL/uart_rx_s.v $RTL/uart_tx_s.v $RTL/uart_rx_fifo.v
( time $VVP -n $OUT/tb.vvp ) > $OUT/sim.log 2>&1
tail -5 $OUT/sim.log
python3 $HOST/sena_board_host.py --mode replay --rxlog $OUT/rx.txt --out $OUT/replay $ARGS | tee $OUT/replay_console.txt | tail -12
