# Board oscillator 50 MHz; SENA core runs on divide-by-2 = 25 MHz
create_clock -name CLOCK_50 -period 20.000 [get_ports {CLOCK_50}]
create_generated_clock -name clk_25 -source [get_ports {CLOCK_50}] -divide_by 2 [get_registers {clk_25}]
derive_clock_uncertainty

# UART / async I/O are false paths relative to core timing proof
set_false_path -from [get_ports {UART_RXD KEY*}]
set_false_path -to   [get_ports {UART_TXD LEDR* LEDG* HEX*}]
