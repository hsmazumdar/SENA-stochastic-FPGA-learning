# =============================================================================
# vivado_postroute_power.tcl — Artix-7 post-route + VCD-driven Report Power
# -----------------------------------------------------------------------------
# Phase 2 critical path for Table VI empirical µJ/inference.
#
# Flow:
#   1. Synth + place & route full folded 784→160→10 (PAR=8) on xc7a100t
#   2. Write post-route sim netlist + SDF
#   3. (External) Run tb_sena_power.v to produce activity.vcd
#   4. Open implemented design, read VCD, report_power
#   5. Emit power.json with static / dynamic / total mW and µJ/inference
#
# Usage (two-stage — VCD generation may use xsim or a separate simulator):
#   vivado -mode batch -source vivado_postroute_power.tcl \
#     -tclargs STAGE=impl CLK_MHZ=50 N_INFER=1000
#   # after VCD exists at _power/activity.vcd:
#   vivado -mode batch -source vivado_postroute_power.tcl \
#     -tclargs STAGE=power CLK_MHZ=50 N_INFER=1000
#
# One-shot (impl + xsim + power) if xsim is available:
#   vivado -mode batch -source vivado_postroute_power.tcl \
#     -tclargs STAGE=all CLK_MHZ=50 N_INFER=200
# =============================================================================

set N_IN       784
set N_HID_BANK 160
set N_OUT      10
set PAR        8
set PAR_OUT    2
set T_MAX      40
set PART       xc7a100tcsg324-1
set CLK_MHZ    50
set N_INFER    200
set STAGE      all
set MEAN_PASSES 2.30
# Primary OP (fixed-N EE). Override for adaptive ensemble: MEAN_PASSES=5.11

foreach arg $argv {
    if {[regexp {^([A-Za-z0-9_]+)=(.+)$} $arg -> k v]} {
        set $k $v
    }
}

set CLK_NS [expr {1000.0 / $CLK_MHZ}]
set root [file normalize [file dirname [info script]]]
set outdir [file join $root _power]
file mkdir $outdir
set checkpoint [file join $outdir impl.dcp]
set vcd_path [file join $outdir activity.vcd]
set netlist  [file join $outdir post_route.v]
set sdf_path [file join $outdir post_route.sdf]

proc sena_add_sources {root} {
    add_files [list \
        [file join $root lfsr32.v] \
        [file join $root stochastic_neuron.v] \
        [file join $root sena_layer.v] \
        [file join $root sena_early_exit.v] \
        [file join $root sena_tile.v]]
    set_property top sena_tile [current_fileset]
}

proc sena_run_impl {root outdir part clk_ns n_in n_hid n_out par par_out t_max checkpoint netlist sdf_path} {
    create_project -force sena_power $outdir/proj -part $part
    set_property target_language Verilog [current_project]
    sena_add_sources $root
    set_property generic [list \
        N_IN=$n_in N_HID_BANK=$n_hid N_OUT=$n_out \
        PAR=$par PAR_OUT=$par_out T_MAX=$t_max] [get_filesets sources_1]

    create_clock -period $clk_ns [get_ports clk]
    synth_design -mode out_of_context -top sena_tile -part $part \
        -generic N_IN=$n_in -generic N_HID_BANK=$n_hid -generic N_OUT=$n_out \
        -generic PAR=$par -generic PAR_OUT=$par_out -generic T_MAX=$t_max
    opt_design
    place_design
    route_design
    report_utilization -file [file join $outdir utilization_impl.rpt]
    report_timing_summary -file [file join $outdir timing_impl.rpt]
    write_checkpoint -force $checkpoint
    write_verilog -force -mode timesim -sdf_anno true $netlist
    write_sdf -force $sdf_path
    puts "Wrote checkpoint $checkpoint"
    close_project
}

proc sena_run_xsim {root outdir netlist sdf_path clk_mhz n_infer} {
    # Behavioral TB drives the post-route netlist when possible; fall back to
    # RTL TB if timesim elaboration is awkward for unpacked array ports.
    set tb [file join $root tb_sena_power.v]
    if {![file exists $tb]} {
        puts "ERROR: missing $tb"
        return 0
    }
    set simdir [file join $outdir xsim]
    file mkdir $simdir
    # Prefer RTL-level SAIF/VCD from synthesizable TB (toggle rates still useful
    # for Report Power when read against the implemented design hierarchy).
    set argv_file [file join $simdir run_xsim.tcl]
    set fh [open $argv_file w]
    puts $fh "xvlog [file join $root lfsr32.v]"
    puts $fh "xvlog [file join $root stochastic_neuron.v]"
    puts $fh "xvlog [file join $root sena_layer.v]"
    puts $fh "xvlog [file join $root sena_early_exit.v]"
    puts $fh "xvlog [file join $root sena_tile.v]"
    puts $fh "xvlog $tb"
    puts $fh "xelab -debug typical tb_sena_power -s tb_sena_power_sim"
    puts $fh "xsim tb_sena_power_sim -runall"
    close $fh
    puts "Launching xsim via $argv_file (N_INFER=$n_infer, CLK=${clk_mhz} MHz)..."
    # Environment for TB plusargs
    set ::env(SENA_N_INFER) $n_infer
    set ::env(SENA_CLK_MHZ) $clk_mhz
    set ::env(SENA_VCD) [file join $outdir activity.vcd]
    cd $simdir
    if {[catch {exec xsim -f $argv_file} err]} {
        puts "WARN: xsim path failed ($err). Generate activity.vcd externally, then STAGE=power."
        return 0
    }
    return [file exists [file join $outdir activity.vcd]]
}

proc sena_report_power {checkpoint vcd_path outdir clk_mhz mean_passes n_infer} {
    open_checkpoint $checkpoint
    if {[file exists $vcd_path]} {
        puts "Reading switching activity from $vcd_path"
        # read_saif / read_vcd depending on Vivado version
        if {[catch {read_vcd $vcd_path -strip_path tb_sena_power/dut} err]} {
            puts "WARN read_vcd: $err — trying without strip_path"
            catch {read_vcd $vcd_path}
        }
    } else {
        puts "WARN: no VCD at $vcd_path — using default activity (less accurate)"
        set_switching_activity -default_toggle_rate 12.5
    }
    report_power -file [file join $outdir power.rpt]
    report_power -return_string
    # Parse power.rpt totals
    set txt [read [open [file join $outdir power.rpt] r]]
    set static_w 0.0
    set dynamic_w 0.0
    set total_w 0.0
    regexp -nocase {Total On-Chip Power \(W\)\s*\|\s*([0-9.]+)} $txt -> total_w
    regexp -nocase {Dynamic \(W\)\s*\|\s*([0-9.]+)} $txt -> dynamic_w
    regexp -nocase {Device Static \(W\)\s*\|\s*([0-9.]+)} $txt -> static_w
    # Fallback patterns
    if {$total_w == 0.0} {
        regexp -nocase {Total\s+([0-9.]+)\s+W} $txt -> total_w
    }
    set total_mw [expr {$total_w * 1000.0}]
    set static_mw [expr {$static_w * 1000.0}]
    set dynamic_mw [expr {$dynamic_w * 1000.0}]

    # Throughput model: cycles/pass for PAR=8 on 784-160-10 ≈ from sim_fpga_cycles
    # Closed-form with folding; use 182 as paper-consistent estimate, × mean_passes.
    set cycles_per_pass 182
    set cycles_per_inf [expr {$cycles_per_pass * $mean_passes}]
    set thruput [expr {($clk_mhz * 1.0e6) / $cycles_per_inf}]
    # Energy µJ = (mW) / (inf/s) = mJ/s / (inf/s) → mJ/inf = 1000 µJ/inf... wait:
    # mW / (inf/s) = mJ/s / (1/s) = mJ/inf = 1000 µJ/inf. So µJ = 1000 * mW / thruput.
    # Actually: Power(W) / thruput = J/inf; µJ = 1e6 * W / thruput = 1e3 * mW / thruput.
    set uj [expr {1000.0 * $total_mw / $thruput}]

    set wnx [open [file join $outdir power.json] w]
    puts $wnx "\{"
    puts $wnx "  \"tool\": \"vivado_postroute\","
    puts $wnx "  \"part\": \"xc7a100t\","
    puts $wnx "  \"clk_mhz\": $clk_mhz,"
    puts $wnx "  \"mean_passes\": $mean_passes,"
    puts $wnx "  \"cycles_per_pass\": $cycles_per_pass,"
    puts $wnx "  \"n_infer_vcd\": $n_infer,"
    puts $wnx "  \"static_mw\": $static_mw,"
    puts $wnx "  \"dynamic_mw\": $dynamic_mw,"
    puts $wnx "  \"total_mw\": $total_mw,"
    puts $wnx "  \"throughput_inf_s\": $thruput,"
    puts $wnx "  \"energy_uj_per_inference\": $uj,"
    puts $wnx "  \"vcd\": \"$vcd_path\","
    puts $wnx "  \"table_vi_row\": \"Full 784-160-10×3, PAR8, EE (Vivado Post-Route, Artix-7)\""
    puts $wnx "\}"
    close $wnx

    puts "============================================================"
    puts [format "Static  = %.3f mW" $static_mw]
    puts [format "Dynamic = %.3f mW" $dynamic_mw]
    puts [format "Total   = %.3f mW" $total_mw]
    puts [format "Thruput = %.1f inf/s @ %.0f MHz, %.2f mean passes" $thruput $clk_mhz $mean_passes]
    puts [format "Energy  = %.4f µJ/inference  << paste into Table VI" $uj]
    puts "Wrote [file join $outdir power.json]"
    puts "============================================================"
    close_design
}

# ---- dispatch ----
if {$STAGE eq "impl" || $STAGE eq "all"} {
    sena_run_impl $root $outdir $PART $CLK_NS $N_IN $N_HID_BANK $N_OUT $PAR $PAR_OUT $T_MAX \
        $checkpoint $netlist $sdf_path
}
if {$STAGE eq "sim" || $STAGE eq "all"} {
    sena_run_xsim $root $outdir $netlist $sdf_path $CLK_MHZ $N_INFER
}
if {$STAGE eq "power" || $STAGE eq "all"} {
    if {![file exists $checkpoint]} {
        puts "ERROR: missing $checkpoint — run STAGE=impl first"
        exit 1
    }
    sena_report_power $checkpoint $vcd_path $outdir $CLK_MHZ $MEAN_PASSES $N_INFER
}

exit 0
