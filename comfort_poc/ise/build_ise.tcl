# Comfort PoC ISE build
project new comfort_sena.xise
project set family "Spartan6"
project set device "xc6slx9"
project set package "tqg144"
project set speed "-2"

xfile add ../rtl/uart_rx.v
xfile add ../rtl/uart_tx.v
xfile add ../rtl/comfort_core.v
xfile add ../rtl/comfort_sena_top.v
xfile add papilio_pro.ucf

file copy -force ../weights/w_hid_m0.mem .
file copy -force ../weights/w_hid_m1.mem .
file copy -force ../weights/w_hid_m2.mem .
file copy -force ../weights/w_out_m0.mem .
file copy -force ../weights/w_out_m1.mem .
file copy -force ../weights/w_out_m2.mem .

project set top "comfort_sena_top"
process run "Generate Programming File"
set status [process get "Generate Programming File" status]
puts "Generate Programming File: $status"
if {($status ne "up_to_date") && ($status ne "warnings")} { exit 1 }
project close
