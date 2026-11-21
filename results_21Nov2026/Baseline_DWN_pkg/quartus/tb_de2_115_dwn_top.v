// tb_de2_115_dwn_top.v - simulation of the on-board self-test wrapper.
// Run from the folder that holds the generated files (gen/):
//   iverilog -g2005 -I gen -o top.vvp quartus/tb_de2_115_dwn_top.v quartus/de2_115_dwn_top.v gen/dwn_core.v
`timescale 1ns/1ps
module tb_de2_115_dwn_top;
    reg CLOCK_50 = 0; reg [3:0] KEY = 4'hF;
    wire UART_TXD; wire [17:0] LEDR; wire [8:0] LEDG; wire [6:0] HEX0, HEX1, HEX2, HEX3;
    de2_115_dwn_top #(.X_FILE("test_x.hex"), .E_FILE("test_exp.hex")) dut (
        .CLOCK_50(CLOCK_50), .KEY(KEY), .UART_RXD(1'b1), .UART_TXD(UART_TXD),
        .LEDR(LEDR), .LEDG(LEDG), .HEX0(HEX0), .HEX1(HEX1), .HEX2(HEX2), .HEX3(HEX3));
    always #10 CLOCK_50 = ~CLOCK_50;
    initial begin
        wait (dut.st == 2'd3);
        repeat (4) @(posedge dut.clk);
        $display("TOP RESULT done=%0d pass(LEDG[8:1])=%0d fail(LEDR17)=%0d correct=%0d mismatches=%0d latency=%0d total_cycles=%0d HEX0=%b",
                 dut.done, &LEDG[8:1], LEDR[17], dut.n_correct, dut.n_mism, dut.t_first_out, dut.total_cycles, HEX0);
        $finish;
    end
    initial begin #50_000_000; $display("TIMEOUT"); $finish; end
endmodule
