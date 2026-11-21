# DE2-115 SENA board_v1 timing constraints
create_clock -name CLOCK_50 -period 20.000 [get_ports CLOCK_50]
# clk25 = toggle register on CLOCK_50 (40 ns)
create_generated_clock -name clk25 -source [get_ports CLOCK_50] -divide_by 2 [get_registers {clk25}]
derive_clock_uncertainty
# Asynchronous / quasi-static I/O: UART (2-FF synchroniser), KEY (synchronised), LEDs/HEX.
# SRAM: all control/address/data are registered in clk25 and held >= 1 full 40 ns cycle around
# each access (10 ns SRAM); DQ is captured one full cycle after OE/address -> cut I/O paths.
set_false_path -from [get_ports {UART_RXD KEY[*] SRAM_DQ[*]}]
set_false_path -to   [get_ports {UART_TXD LEDR[*] LEDG[*] HEX*[*] SRAM_ADDR[*] SRAM_DQ[*] SRAM_*_N}]
