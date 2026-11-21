create_clock -name CLOCK_50 -period 20.000 [get_ports CLOCK_50]
derive_clock_uncertainty
set_false_path -from [all_inputs]
set_false_path -to [all_outputs]
