// =============================================================================
// lfsr32.v — 32-bit Galois LFSR (on-chip RNG for random threshold)
// =============================================================================
`timescale 1ns/1ps
module lfsr32 #(
    parameter [31:0] SEED = 32'hACE1_0001
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        en,
    output reg  [31:0] r
);
    // taps: x^32 + x^22 + x^2 + x^1 + 1  (common Galois poly)
    localparam [31:0] TAPS = 32'h8020_0003;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            r <= SEED;
        else if (en)
            r <= {1'b0, r[31:1]} ^ (r[0] ? TAPS : 32'b0);
    end
endmodule
