# =============================================================================
# synth_full_net.tcl — Vivado OOC synth of full folded 784→160→10 (PAR=8)
# -----------------------------------------------------------------------------
# Answers: "Does the 160-hidden net fit on Artix-7 xc7a100t?"
#
# Usage:
#   vivado -mode batch -source synth_full_net.tcl
#   vivado -mode batch -source synth_full_net.tcl -tclargs PART=xc7a100tcsg324-1 CLK_NS=10.0
#
# Writes rtl/_synth/full_784_160_10_par8/{utilization.rpt,timing.rpt,synth.json}
# =============================================================================

set N_IN       784
set N_HID_BANK 160
set N_OUT      10
set PAR        8
set PAR_OUT    2
set T_MAX      40
set PART       xc7a100tcsg324-1
set CLK_NS     10.0

foreach arg $argv {
    if {[regexp {^([A-Za-z0-9_]+)=(.+)$} $arg -> k v]} {
        set $k $v
    }
}

set root [file normalize [file dirname [info script]]]
set tag  "full_784_160_10_par${PAR}"
set outdir [file join $root _synth $tag]
file mkdir $outdir

create_project -force sena_full_$tag $outdir -part $PART
set_property target_language Verilog [current_project]

add_files [list \
    [file join $root lfsr32.v] \
    [file join $root stochastic_neuron.v] \
    [file join $root sena_layer.v] \
    [file join $root sena_early_exit.v] \
    [file join $root sena_tile.v]]
set_property top sena_tile [current_fileset]

set_property generic [list \
    N_IN=$N_IN \
    N_HID_BANK=$N_HID_BANK \
    N_OUT=$N_OUT \
    PAR=$PAR \
    PAR_OUT=$PAR_OUT \
    T_MAX=$T_MAX] [get_filesets sources_1]

create_clock -period $CLK_NS [get_ports clk]
synth_design -mode out_of_context -top sena_tile -part $PART \
    -generic N_IN=$N_IN -generic N_HID_BANK=$N_HID_BANK -generic N_OUT=$N_OUT \
    -generic PAR=$PAR -generic PAR_OUT=$PAR_OUT -generic T_MAX=$T_MAX
opt_design
report_utilization -file [file join $outdir utilization.rpt]
report_timing_summary -file [file join $outdir timing.rpt]

set util_txt [read [open [file join $outdir utilization.rpt] r]]
regexp {Slice LUTs\*?[^\|]*\|\s*([0-9]+)} $util_txt -> lut_n
regexp {Slice Registers[^\|]*\|\s*([0-9]+)} $util_txt -> ff_n
regexp {Block RAM Tile[^\|]*\|\s*([0-9.]+)} $util_txt -> bram_n
regexp {DSPs[^\|]*\|\s*([0-9]+)} $util_txt -> dsp_n
if {![info exists lut_n]} { set lut_n 0 }
if {![info exists ff_n]}  { set ff_n 0 }
if {![info exists bram_n]} { set bram_n 0 }
if {![info exists dsp_n]} { set dsp_n 0 }

# xc7a100t capacities
set CAP_LUT  63400
set CAP_FF   126800
set CAP_BRAM 135
set CAP_DSP  240

set wnx [open [file join $outdir synth.json] w]
puts $wnx "\{"
puts $wnx "  \"tool\": \"vivado\","
puts $wnx "  \"part\": \"$PART\","
puts $wnx "  \"design\": \"full_784_160_10_folded_PAR${PAR}\","
puts $wnx "  \"n_in\": $N_IN,"
puts $wnx "  \"n_hid_bank\": $N_HID_BANK,"
puts $wnx "  \"n_out\": $N_OUT,"
puts $wnx "  \"par\": $PAR,"
puts $wnx "  \"par_out\": $PAR_OUT,"
puts $wnx "  \"clock_ns\": $CLK_NS,"
puts $wnx "  \"lut\": $lut_n,"
puts $wnx "  \"ff\": $ff_n,"
puts $wnx "  \"bram\": $bram_n,"
puts $wnx "  \"dsp\": $dsp_n,"
puts $wnx "  \"lut_pct\": [expr {100.0 * $lut_n / $CAP_LUT}],"
puts $wnx "  \"ff_pct\": [expr {100.0 * $ff_n / $CAP_FF}],"
puts $wnx "  \"bram_pct\": [expr {100.0 * $bram_n / double($CAP_BRAM)}],"
puts $wnx "  \"dsp_pct\": [expr {100.0 * $dsp_n / $CAP_DSP}],"
puts $wnx "  \"fits\": [expr {($lut_n < $CAP_LUT) && ($bram_n < $CAP_BRAM)}]"
puts $wnx "\}"
close $wnx

puts "============================================================"
puts "Full-net synth complete: $outdir"
puts "LUT=$lut_n ([format %.1f [expr {100.0*$lut_n/$CAP_LUT}]]%)  FF=$ff_n  BRAM=$bram_n  DSP=$dsp_n"
puts "============================================================"
close_project
exit 0
