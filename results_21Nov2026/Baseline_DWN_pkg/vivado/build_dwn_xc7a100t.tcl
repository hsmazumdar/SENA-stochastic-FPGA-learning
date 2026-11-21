# build_dwn_xc7a100t.tcl - OPTIONAL Artix-7 build of the generated DWN core.
# Same part and flow style as SENA's rtl/synth_full_net.tcl (xc7a100tcsg324-1,
# out-of-context synthesis + implementation).  Vivado ML Standard (free) supports it.
# Usage (from Baseline_DWN/):
#   vivado -mode batch -source vivado/build_dwn_xc7a100t.tcl -tclargs GEN=quartus/gen CLK_NS=4.0 OUT=vivado/out_sm_seed0
# Read: utilization_impl.rpt (LUT/FF/BRAM/DSP), timing_impl.rpt (WNS -> Fmax = 1000/(CLK_NS-WNS)),
#       power_impl.rpt (vectorless default activity; label as "Vivado estimate").
set GEN    quartus/gen
set CLK_NS 4.0
set OUT    vivado/out
set PART   xc7a100tcsg324-1
foreach a $argv {
    set kv [split $a =]
    set [lindex $kv 0] [lindex $kv 1]
}
if {[file exists $OUT]} { set OUT "${OUT}_[clock format [clock seconds] -format %Y%m%d_%H%M%S]" }
file mkdir $OUT
create_project -in_memory -part $PART
read_verilog [file join $GEN dwn_core.v]
synth_design -mode out_of_context -top dwn_core -part $PART -flatten_hierarchy rebuilt
create_clock -name clk -period $CLK_NS [get_ports clk]
report_utilization -file [file join $OUT utilization_synth.rpt]
opt_design
place_design
phys_opt_design
route_design
report_utilization      -file [file join $OUT utilization_impl.rpt]
report_timing_summary   -file [file join $OUT timing_impl.rpt]
report_power            -file [file join $OUT power_impl.rpt]
set wns [get_property SLACK [get_timing_paths -max_paths 1 -setup]]
set fp [open [file join $OUT summary.json] w]
puts $fp "{\n  \"part\": \"$PART\",\n  \"clk_ns\": $CLK_NS,\n  \"wns_ns\": $wns,\n  \"fmax_mhz\": [expr {1000.0/($CLK_NS - $wns)}]\n}"
close $fp
file copy -force [file join $GEN dwn_core_info.json] $OUT
puts "VIVADO DONE -> $OUT  (WNS $wns ns)"
