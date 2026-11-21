# compile_and_report.tcl - full Quartus flow + timing + PowerPlay for one revision,
# then copies the summaries into reports/<revision>/ (never overwrites: a new
# time-stamped folder is used if one exists).
# Usage (from this quartus/ folder):
#   quartus_sh -t compile_and_report.tcl de2_115_dwn
#   quartus_sh -t compile_and_report.tcl dwn_core_only
load_package flow
set rev [lindex $argv 0]
if {$rev eq ""} { set rev de2_115_dwn }
project_open dwn_baseline -revision $rev
if {[catch {execute_flow -compile} msg]} {
    puts "COMPILE FAILED: $msg"; project_close; exit 1
}
# PowerPlay: same command line as SENA's DE2-115 run (vectorless, 12.5% default toggle)
if {[catch {execute_module -tool pow -args "--use_vectorless_estimation=on --estimate_power=on"} msg]} {
    puts "POWER ANALYSIS FAILED: $msg"
}
project_close
set dst [file join reports $rev]
if {[file exists $dst]} { set dst "${dst}_[clock format [clock seconds] -format %Y%m%d_%H%M%S]" }
file mkdir $dst
foreach ext {fit.summary fit.rpt sta.summary sta.rpt pow.summary pow.rpt map.rpt flow.rpt} {
    set f [file join output_files "$rev.$ext"]
    if {[file exists $f]} { file copy $f $dst }
}
foreach f {gen/dwn_core_info.json} { if {[file exists $f]} { file copy $f $dst } }
puts "REPORTS COPIED TO $dst"
