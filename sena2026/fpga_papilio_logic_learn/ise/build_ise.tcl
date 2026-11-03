# build_ise.tcl — Papilio Pro logic-learn PoC bitstream
# From this ise/ directory (ISE 14.7 env):  xtclsh build_ise.tcl
if {[file exists papilio_logic_learn.xise]} {
    file delete -force papilio_logic_learn.xise
}
# wipe stale run products so XST re-runs cleanly
foreach f [glob -nocomplain *.ngc *.ngd *.ncd *.pcf *.bit *.bgn *.drc *.pad *.par *.unroutes *.xpi *.map *.mrp *.psr *.syr *.twr *.twx *.xrpt *_summary.html *_usage.xml] {
    catch {file delete -force $f}
}
project new papilio_logic_learn.xise
project set family "Spartan6"
project set device "xc6slx9"
project set package "tqg144"
project set speed "-2"

xfile add ../rtl/uart_rx.v
xfile add ../rtl/uart_tx.v
xfile add ../rtl/logic663_train_core.v
xfile add ../rtl/papilio_logic_learn_top.v
xfile add papilio_pro.ucf

project set top "papilio_logic_learn_top"
process run "Generate Programming File"

set status [process get "Generate Programming File" status]
puts "Generate Programming File: $status"
if {![file exists papilio_logic_learn_top.bit]} {
    puts "ERROR: bitstream missing"
    project close
    exit 1
}
project close
