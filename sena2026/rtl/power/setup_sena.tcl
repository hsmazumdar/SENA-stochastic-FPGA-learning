# ============================================================
# SENA Power Analysis - Project Setup (EnergyIssueFix Step 1)
# ============================================================
# Usage:
#   vivado -mode batch -source setup_sena.tcl
#   vivado -mode batch -source setup_sena.tcl -tclargs CLK_NS=20 N_IMAGES=1000
# ============================================================

set PART    xc7a100tcsg324-1
set CLK_NS  20.0
set N_IMAGES 1000
set PASSES   3
set CLK_MHZ  50

foreach arg $argv {
    if {[regexp {^([A-Za-z0-9_]+)=(.+)$} $arg -> k v]} { set $k $v }
}
set CLK_NS [expr {1000.0 / double($CLK_MHZ)}]

set root [file normalize [file dirname [info script]]]
set rtl  [file normalize [file join $root ..]]
set out  [file join $root sena_power]
file mkdir $out

# Ensure stimuli exist (caller should run prepare_power_stimuli.py first)
set stim [file join $root stimuli]
if {![file exists [file join $stim mnist_x.bin]]} {
    puts "ERROR: missing $stim/mnist_x.bin — run:"
    puts "  python prepare_power_stimuli.py --n-images $N_IMAGES --quick-train"
    exit 1
}

create_project -force sena_power $out -part $PART
set_property target_language Verilog [current_project]

# RTL sources (core + power wrapper)
add_files -norecurse [list \
    [file join $rtl lfsr32.v] \
    [file join $rtl stochastic_neuron.v] \
    [file join $rtl sena_layer.v] \
    [file join $rtl sena_early_exit.v] \
    [file join $rtl sena_tile.v] \
    [file join $root sena_top.v]]

# Testbench
add_files -fileset sim_1 -norecurse [file join $root tb sena_tb.v]
# Stimuli + weight mems on sim include path
set_property include_dirs [list $stim] [get_filesets sim_1]
set_property verilog_define [list \
    "N_IMAGES_DEFAULT=$N_IMAGES"] [get_filesets sim_1]

# Copy stimuli into project sim dir so $fopen / $readmemh resolve
foreach f {w_hid.mem b_hid.mem w_out.mem b_out.mem mnist_x.bin mnist_x.mem} {
    file copy -force [file join $stim $f] [file join $out $f]
}

set_property top sena_top [current_fileset]
set_property top sena_tb [get_filesets sim_1]

# Force full-net generics on the wrapper's tile via Verilog parameters in sena_top
create_fileset -constrset constrs_1
set xdc [file join $out sena.xdc]
set fh [open $xdc w]
puts $fh "create_clock -period $CLK_NS -name clk \[get_ports clk\]"
close $fh
add_files -fileset constrs_1 $xdc

# Synth + implement
launch_runs synth_1 -jobs 4
wait_on_run synth_1
launch_runs impl_1 -jobs 4
wait_on_run impl_1

open_run impl_1
report_utilization -file [file join $out utilization.rpt]
report_timing_summary -file [file join $out timing.rpt]
report_power -file [file join $out baseline_power_no_activity.rpt]
write_checkpoint -force [file join $out impl.dcp]

puts "Setup complete. Project: $out"
puts "Next: source gen_postroute_netlist.tcl  OR  power_analysis_full.tcl STAGE=saif"
