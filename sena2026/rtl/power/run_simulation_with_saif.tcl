# ============================================================
# SAIF-Generating Simulation (EnergyIssueFix Steps 3–4)
# ============================================================
# Behavioral xsim of sena_tb with MNIST vectors + SAIF dump.
# Prefer SAIF over VCD for Vivado Report Power (High confidence).
#
# Usage (from Vivado Tcl, project already created by setup_sena.tcl):
#   source run_simulation_with_saif.tcl
#
# Or batch:
#   vivado -mode batch -source run_simulation_with_saif.tcl \
#     -tclargs N_IMAGES=500 PASSES=3 CLK_MHZ=50
# ============================================================

set N_IMAGES 1000
set PASSES   3
set CLK_MHZ  50

foreach arg $argv {
    if {[regexp {^([A-Za-z0-9_]+)=(.+)$} $arg -> k v]} { set $k $v }
}

set root [file normalize [file dirname [info script]]]
set out  [file join $root sena_power]
set saif [file join $out sena_activity.saif]
set xsim_tcl [file join $out xsim_saif.tcl]

# Open project if needed
set proj [file join $out sena_power.xpr]
if {[file exists $proj]} {
    catch {close_project}
    open_project $proj
}

set_property -name {xsim.simulate.runtime} -value {all} [get_filesets sim_1]
set_property -name {xsim.simulate.saif} -value $saif [get_filesets sim_1]
set_property -name {xsim.simulate.saif_all_signals} -value {true} [get_filesets sim_1]
set_property -name {xsim.elaborate.debug_level} -value {typical} [get_filesets sim_1]

# Plusargs for TB
set_property -name {xsim.simulate.xsim.more_options} \
    -value "-testplusarg N_IMAGES=$N_IMAGES -testplusarg PASSES=$PASSES -testplusarg CLK_MHZ=$CLK_MHZ" \
    [get_filesets sim_1]

# Working directory must see mnist_x.bin / weight .mem
set_property DIRECTORY $out [get_filesets sim_1]

puts "Launching xsim with SAIF → $saif ..."
launch_simulation -mode behavioral -simset sim_1

# Interactive-equivalent SAIF commands (batch-safe via sim Tcl)
# When launch_simulation returns in GUI, user can also paste:
#   open_saif sena_activity.saif
#   log_saif [get_objects -r /sena_tb/uut/*]
#   run all
#   close_saif

# In batch, xsim.simulate.saif property should auto-collect; also force:
relaunch_sim
# Give the sim a chance; if still running use:
if {[catch {run all} err]} {
    puts "run all: $err"
}
if {[catch {close_saif} err]} {
    puts "close_saif (may already be closed): $err"
}

puts "SAIF simulation complete."
puts "SAIF file: $saif"
puts "Next: source report_power_saif.tcl"
