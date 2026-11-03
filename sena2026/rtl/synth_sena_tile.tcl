# =============================================================================
# synth_sena_tile.tcl — Vivado out-of-context synthesis for one SENA tile
# -----------------------------------------------------------------------------
# Usage (batch):
#   vivado -mode batch -source synth_sena_tile.tcl \
#     -tclargs N_HID_BANK=16 PAR=8 PAR_OUT=2 PART=xc7a100tcsg324-1
#
# Produces under rtl/_synth/<tag>/:
#   utilization.rpt, timing.rpt, power.rpt, synth.json
# =============================================================================

set N_IN       784
set N_HID_BANK 16
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
set tag  "hid${N_HID_BANK}_par${PAR}_${PAR_OUT}"
set outdir [file join $root _synth $tag]
file mkdir $outdir

create_project -force sena_tile_$tag $outdir -part $PART
set_property target_language Verilog [current_project]

add_files [list \
    [file join $root lfsr32.v] \
    [file join $root stochastic_neuron.v] \
    [file join $root sena_layer.v] \
    [file join $root sena_early_exit.v] \
    [file join $root sena_tile.v]]
set_property top sena_tile [current_fileset]

# Force the tile parameters for out-of-context synth.
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
report_power -file [file join $outdir power.rpt]

# Machine-readable util summary
set lut  [get_property SLICE_LUTS [report_utilization -return_string]]
set ff   [get_property SLICE_REGISTERS [report_utilization -return_string]]
# Fallback parse from the written report when properties are unavailable.
set util_txt [read [open [file join $outdir utilization.rpt] r]]
regexp {Slice LUTs\*?[^\|]*\|\s*([0-9]+)} $util_txt -> lut_n
regexp {Slice Registers[^\|]*\|\s*([0-9]+)} $util_txt -> ff_n
regexp {Block RAM Tile[^\|]*\|\s*([0-9.]+)} $util_txt -> bram_n
regexp {DSPs[^\|]*\|\s*([0-9]+)} $util_txt -> dsp_n
if {![info exists lut_n]} { set lut_n 0 }
if {![info exists ff_n]}  { set ff_n 0 }
if {![info exists bram_n]} { set bram_n 0 }
if {![info exists dsp_n]} { set dsp_n 0 }

set wnx [open [file join $outdir synth.json] w]
puts $wnx "\{"
puts $wnx "  \"tool\": \"vivado\","
puts $wnx "  \"part\": \"$PART\","
puts $wnx "  \"n_in\": $N_IN,"
puts $wnx "  \"n_hid_bank\": $N_HID_BANK,"
puts $wnx "  \"n_out\": $N_OUT,"
puts $wnx "  \"par\": $PAR,"
puts $wnx "  \"par_out\": $PAR_OUT,"
puts $wnx "  \"clock_ns\": $CLK_NS,"
puts $wnx "  \"lut\": $lut_n,"
puts $wnx "  \"ff\": $ff_n,"
puts $wnx "  \"bram\": $bram_n,"
puts $wnx "  \"dsp\": $dsp_n"
puts $wnx "\}"
close $wnx

puts "Wrote $outdir"
close_project
exit 0
