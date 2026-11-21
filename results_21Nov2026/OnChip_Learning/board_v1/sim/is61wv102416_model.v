// Behavioural IS61WV102416 (1M x 16 async SRAM, 10 ns class) for board_v1 simulation.
// Read: DQ driven tAA=8 ns after address/OE change while CE_N=OE_N=0 and WE_N=1.
// Write: data latched at the rising edge of WE_N (CE_N=0); byte enables honoured.
// Flags bus contention (OE_N and WE_N both low) and X on address during a write.
`timescale 1ns/1ps
module is61wv102416_model (
    input  wire [19:0] A, inout wire [15:0] DQ,
    input  wire CE_N, OE_N, WE_N, UB_N, LB_N
);
    reg [15:0] mem [0:1048575];
    integer i, n_err = 0;
    initial for (i = 0; i < 1048576; i = i + 1) mem[i] = 16'hDEAD;   // "random" power-up content
    wire rd = !CE_N && !OE_N && WE_N;
    wire [15:0] rv = mem[A];
    assign #8 DQ = rd ? rv : 16'hzzzz;
    always @(posedge WE_N) if (!CE_N) begin
        if (^A === 1'bx) begin n_err = n_err + 1; $display("SRAM: X address on write @%t", $time); end
        if (!LB_N) mem[A][7:0]  <= DQ[7:0];
        if (!UB_N) mem[A][15:8] <= DQ[15:8];
    end
    always @(*) if (!CE_N && !OE_N && !WE_N) begin n_err = n_err + 1; $display("SRAM: OE_N and WE_N both low @%t", $time); end
endmodule
