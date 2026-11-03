# ============================================================
# Import SAIF + Report Power (EnergyIssueFix Step 5) — THE PAYOFF
# ============================================================
# Usage:
#   vivado -mode batch -source report_power_saif.tcl \
#     -tclargs CLK_MHZ=50 MEAN_PASSES=2.30 M=3
# ============================================================

set CLK_MHZ     50
set MEAN_PASSES 2.30
set M           3
# Full-net PAR8 cycles/member/pass (Verilator-validated). NOT Papilio's 807.
set CYCLES_PER_PASS 182

foreach arg $argv {
    if {[regexp {^([A-Za-z0-9_]+)=(.+)$} $arg -> k v]} { set $k $v }
}

set root [file normalize [file dirname [info script]]]
set out  [file join $root sena_power]
set saif [file join $out sena_activity.saif]
set dcp  [file join $out impl.dcp]
set proj [file join $out sena_power.xpr]

if {[file exists $proj]} {
    catch {close_project}
    open_project $proj
}
if {[file exists $dcp]} {
    open_checkpoint $dcp
} else {
    open_run impl_1
}

if {![file exists $saif]} {
    puts "ERROR: missing SAIF $saif — run run_simulation_with_saif.tcl first"
    exit 1
}

puts "Reading SAIF $saif ..."
# Match TB hierarchy: sena_tb/uut → sena_top
if {[catch {read_saif -i $saif -instance_name sena_tb/uut} err]} {
    puts "WARN read_saif -instance_name: $err"
    if {[catch {read_saif $saif} err2]} {
        puts "ERROR read_saif: $err2"
        exit 1
    }
}

report_power -verbose -file [file join $out sena_postroute_power.rpt]
report_power -file [file join $out sena_postroute_power_summary.rpt]

set txt [read [open [file join $out sena_postroute_power.rpt] r]]
set total_w 0.0
set static_w 0.0
set dynamic_w 0.0
regexp -nocase {Total On-Chip Power \(W\)\s*\|\s*([0-9.]+)} $txt -> total_w
regexp -nocase {Dynamic \(W\)\s*\|\s*([0-9.]+)} $txt -> dynamic_w
regexp -nocase {Device Static \(W\)\s*\|\s*([0-9.]+)} $txt -> static_w
if {$total_w == 0.0} {
    regexp -nocase {Total\s+([0-9.]+)\s+W} $txt -> total_w
}

set total_mw   [expr {$total_w * 1000.0}]
set static_mw  [expr {$static_w * 1000.0}]
set dynamic_mw [expr {$dynamic_w * 1000.0}]

# Sequential ensemble: M member forwards per stochastic pass
set cycles_inf [expr {$CYCLES_PER_PASS * double($M) * double($MEAN_PASSES)}]
set thruput    [expr {($CLK_MHZ * 1.0e6) / $cycles_inf}]
# µJ/inf = 1000 * mW / (inf/s)
set uj [expr {1000.0 * $total_mw / $thruput}]

set wnx [open [file join $out power.json] w]
puts $wnx "\{"
puts $wnx "  \"tool\": \"vivado_postroute_saif\","
puts $wnx "  \"part\": \"xc7a100tcsg324-1\","
puts $wnx "  \"clk_mhz\": $CLK_MHZ,"
puts $wnx "  \"cycles_per_pass_member\": $CYCLES_PER_PASS,"
puts $wnx "  \"M\": $M,"
puts $wnx "  \"mean_passes\": $MEAN_PASSES,"
puts $wnx "  \"static_mw\": $static_mw,"
puts $wnx "  \"dynamic_mw\": $dynamic_mw,"
puts $wnx "  \"total_mw\": $total_mw,"
puts $wnx "  \"throughput_inf_s\": $thruput,"
puts $wnx "  \"energy_uj_per_inference\": $uj,"
puts $wnx "  \"saif\": \"$saif\","
puts $wnx "  \"confidence\": \"High (SAIF vector activity)\","
puts $wnx "  \"table_vi_row\": \"Full 784-160-10×3, PAR8, fixed-N EE (Vivado Post-Route, Artix-7)\","
puts $wnx "  \"note\": \"EnergyIssueFix used 807 (16-hid Papilio); this flow uses Verilator-validated 182 for PAR8 full net.\""
puts $wnx "\}"
close $wnx

# Also mirror for patch_table_vi_vivado.py default path
set rtl [file normalize [file join $root ..]]
file mkdir [file join $rtl _power]
file copy -force [file join $out power.json] [file join $rtl _power power.json]

puts "============================================================"
puts [format "Static  = %.3f mW" $static_mw]
puts [format "Dynamic = %.3f mW" $dynamic_mw]
puts [format "Total   = %.3f mW" $total_mw]
puts [format "Thruput = %.1f inf/s  (f=%g MHz, cyc/pass=%d, M=%d, passes=%.2f)" \
    $thruput $CLK_MHZ $CYCLES_PER_PASS $M $MEAN_PASSES]
puts [format "Energy  = %.4f µJ/inference  << paste into Table VI" $uj]
puts "Wrote $out/power.json"
puts "============================================================"
