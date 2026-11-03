# =============================================================================
# build_ise.tcl — Papilio MNIST online (B+E) slice
# =============================================================================
project new papilio_mnist_online.xise
project set family "Spartan6"
project set device "xc6slx9"
project set package "tqg144"
project set speed "-2"

xfile add ../rtl/uart_rx.v
xfile add ../rtl/uart_tx.v
xfile add ../rtl/mnist_online_core.v
xfile add ../rtl/papilio_mnist_online_top.v
xfile add papilio_pro.ucf

file copy -force ../rtl/w_hid.mem .
file copy -force ../rtl/w_out.mem .

project set top "papilio_mnist_online_top"
process run "Generate Programming File"

set status [process get "Generate Programming File" status]
puts "Generate Programming File: $status"
if {($status ne "up_to_date") && ($status ne "warnings")} { exit 1 }
project close
