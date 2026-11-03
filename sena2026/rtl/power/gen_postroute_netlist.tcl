# ============================================================
# Generate Post-Route Simulation Netlist (EnergyIssueFix Step 2)
# ============================================================
# Run after setup_sena.tcl (implemented design open), or standalone:
#   vivado -mode batch -source gen_postroute_netlist.tcl
# ============================================================

set root [file normalize [file dirname [info script]]]
set out  [file join $root sena_power]
set dcp  [file join $out impl.dcp]

if {[file exists $dcp]} {
    open_checkpoint $dcp
} else {
    # Assume project already open with impl_1
    open_run impl_1
}

write_verilog -mode timesim -sdf_anno true \
    -sdf_file [file join $out sena_postroute.sdf] \
    -file [file join $out sena_postroute_netlist.v]

write_sdf -force [file join $out sena_postroute.sdf]

puts "Post-route netlist: $out/sena_postroute_netlist.v"
puts "SDF:                $out/sena_postroute.sdf"
