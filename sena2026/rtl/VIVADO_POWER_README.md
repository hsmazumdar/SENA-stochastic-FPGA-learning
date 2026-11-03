# DEPRECATED for SAIF vector power — use rtl/power/ instead (EnergyIssueFix.docx).
#
# Kept for the older VCD-oriented path:
#   vivado_postroute_power.tcl + tb_sena_power.v
#
# Preferred flow:
#   cd rtl/power
#   python prepare_power_stimuli.py --n-images 1000
#   vivado -mode batch -source power_analysis_full.tcl -tclargs STAGE=all
#   python ../../patch_table_vi_vivado.py
#
# See rtl/power/README.md
