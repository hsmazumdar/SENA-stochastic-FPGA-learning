# Raise DE2-115 PowerPlay from Low → High (post-fit VCD)
#
# Prerequisite: ModelSim/Questa (comes with some Quartus installs) OR another
# IEEE simulator that can run the post-fit netlist.
#
# 1) Write post-fit simulation netlist:
#    quartus_eda de2_115_calib --simulation --tool=modelsim --format=verilog
#
# 2) Simulate with 1000 MNIST vectors; dump VCD of the DUT hierarchy.
#
# 3) Import into PowerPlay:
#    quartus_pow de2_115_calib --input_vcd=postfit_mnist.vcd --estimate_power=on
#
# 4) Refresh JSON + regenerate V04 with High-confidence claims:
#    python collect_de2_energy_params.py
#    python ..\..\make_sena_paper_v04.py --high-confidence
#
# Until step 3 reports "Power Estimation Confidence : High", do NOT claim
# High confidence in the manuscript.
