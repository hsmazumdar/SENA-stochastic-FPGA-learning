# Program the Basys 3 over Digilent USB-JTAG.  Volatile (SRAM) configuration.
# Looks for basys3_sena.bit in the same folder as this script, then in out_*/.
set here [file dirname [file normalize [info script]]]
set bit ""
foreach cand [list \
    [file join $here basys3_sena.bit] \
    [file join $here out_h160_l32 basys3_sena.bit] \
    [file join $here out_h160_l160 basys3_sena.bit] \
    [file join $here out_h160_l0 basys3_sena.bit]] {
    if {[file exists $cand]} { set bit $cand; break }
}
if {$bit eq ""} {
    puts "ERROR: basys3_sena.bit not found. Run build_basys3.tcl first."
    exit 1
}
puts "Programming $bit"
open_hw_manager
connect_hw_server
open_hw_target
current_hw_device [lindex [get_hw_devices] 0]
set_property PROGRAM.FILE $bit [current_hw_device]
program_hw_devices [current_hw_device]
refresh_hw_device [current_hw_device]
puts "PROGRAM_DONE"
