# ============================================================
# SENA Power Analysis - Complete Automated Flow (EnergyIssueFix)
# ============================================================
# One-click entry point. Stages: all | setup | netlist | saif | power
#
#   vivado -mode batch -source power_analysis_full.tcl \
#     -tclargs STAGE=all CLK_MHZ=50 N_IMAGES=500 MEAN_PASSES=2.30
# ============================================================

set STAGE       all
set CLK_MHZ     50
set N_IMAGES    500
set PASSES      3
set MEAN_PASSES 2.30
set M           3

foreach arg $argv {
    if {[regexp {^([A-Za-z0-9_]+)=(.+)$} $arg -> k v]} { set $k $v }
}

set root [file normalize [file dirname [info script]]]

proc run_src {root name args_list} {
    set cmd [list source [file join $root $name]]
    # Re-exec via nested vivado is heavy; just source with argv override
    set ::argv $args_list
    uplevel #0 source [file join $root $name]
}

set common [list CLK_MHZ=$CLK_MHZ N_IMAGES=$N_IMAGES PASSES=$PASSES \
                 MEAN_PASSES=$MEAN_PASSES M=$M]

if {$STAGE eq "all" || $STAGE eq "setup"} {
    puts "===== STAGE setup ====="
    set ::argv $common
    source [file join $root setup_sena.tcl]
}
if {$STAGE eq "all" || $STAGE eq "netlist"} {
    puts "===== STAGE netlist ====="
    set ::argv $common
    source [file join $root gen_postroute_netlist.tcl]
}
if {$STAGE eq "all" || $STAGE eq "saif"} {
    puts "===== STAGE saif ====="
    set ::argv $common
    source [file join $root run_simulation_with_saif.tcl]
}
if {$STAGE eq "all" || $STAGE eq "power"} {
    puts "===== STAGE power ====="
    set ::argv $common
    source [file join $root report_power_saif.tcl]
}

puts "=== SENA Power Analysis Complete (STAGE=$STAGE) ==="
puts "Check sena_power/sena_postroute_power.rpt and sena_power/power.json"
