# Non-project Vivado build for basys3_sena_top (Basys 3, XC7A35T-1CPG236C).
# Run on HsmPc1 (do not run on the box).  From this directory:
#   vivado -mode batch -source build_basys3.tcl
# Optional defines before sourcing:
#   set H_LANES 32     ;# default 32 (serial) — use 160 or 0 for full-parallel
#   set N_HID   160
#   set SH      7
#   set BAUD_DIV 434   ;# 50 MHz / 434 ~= 115207 baud

if {![info exists H_LANES]}  { set H_LANES  32 }
if {![info exists N_HID]}    { set N_HID    160 }
if {![info exists SH]}       { set SH       7 }
if {![info exists BAUD_DIV]} { set BAUD_DIV 434 }
if {![info exists OUTDIR]}   { set OUTDIR   [file normalize [file join [file dirname [info script]] "out_h${N_HID}_l${H_LANES}"]] }

set ROOT [file normalize [file join [file dirname [info script]] ".."]]
set RTL  [file join $ROOT rtl]
file mkdir $OUTDIR
cd $OUTDIR

set_part xc7a35tcpg236-1

read_verilog -sv [file join $RTL sdp_ram.v]
read_verilog -sv [file join $RTL sena_learn_core.v]
read_verilog -sv [file join $RTL uart_rx_s.v]
read_verilog -sv [file join $RTL uart_tx_s.v]
read_verilog -sv [file join $RTL uart_rx_fifo.v]
read_verilog -sv [file join $RTL basys3_sena_top.v]
read_xdc [file join $ROOT vivado basys3_sena.xdc]

synth_design -top basys3_sena_top -part xc7a35tcpg236-1 \
    -generic N_HID=$N_HID -generic SH=$SH -generic BAUD_DIV=$BAUD_DIV -generic H_LANES=$H_LANES
write_checkpoint -force synth.dcp
report_utilization -file util_synth.rpt
report_timing_summary -file timing_synth.rpt

opt_design
place_design
route_design
write_checkpoint -force impl.dcp
report_utilization -file util_impl.rpt
report_timing_summary -file timing_impl.rpt
report_utilization -hierarchical -file util_hier.rpt
report_drc -file drc.rpt

write_bitstream -force basys3_sena.bit
puts "BUILD_DONE H_LANES=$H_LANES N_HID=$N_HID bitstream=$OUTDIR/basys3_sena.bit"
