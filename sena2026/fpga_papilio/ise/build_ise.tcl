# =============================================================================
# build_ise.tcl — command-line ISE 14.7 build for the Papilio calibration top
#
# Usage (from this ise/ directory, with ISE environment sourced):
#   xtclsh build_ise.tcl
#
# Produces papilio_calib.bit for the Papilio Loader.
# =============================================================================

project new papilio_calib.xise
project set family "Spartan6"
project set device "xc6slx9"
project set package "tqg144"
project set speed "-2"

xfile add ../rtl/uart_rx.v
xfile add ../rtl/uart_tx.v
xfile add ../rtl/calib_core.v
xfile add ../rtl/papilio_calib_top.v
xfile add papilio_pro.ucf

# .mem files must sit next to the synthesis work dir; copy them in
file copy -force ../rtl/w_hid.mem .
file copy -force ../rtl/w_out.mem .

project set top "papilio_calib_top"
process run "Generate Programming File"

set status [process get "Generate Programming File" status]
puts "Generate Programming File: $status"
if {$status != "up_to_date"} { exit 1 }
project close
