// =============================================================================
// sena_layer.v — bank of N_OUT parallel stochastic neurons (one layer slice)
// =============================================================================
`timescale 1ns/1ps
module sena_layer #(
    parameter integer N_IN   = 784,
    parameter integer N_OUT  = 16,     // neurons in this parallel bank
    parameter integer W_BITS = 16,
    parameter integer PAR    = 1       // inputs folded per clock, per neuron
) (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  start,
    input  wire [N_IN-1:0]       bits,
    // flattened weights: neuron-major  [n*N_IN + i]
    input  wire signed [W_BITS-1:0] weights [0:N_OUT*N_IN-1],
    input  wire signed [W_BITS-1:0] biases  [0:N_OUT-1],
    input  wire signed [W_BITS-1:0] K,
    input  wire [3:0]            n_log2,
    output wire                  busy,
    output wire                  done,
    output wire [N_OUT-1:0]      y_bits
);
    wire [N_OUT-1:0] busy_i, done_i;

    genvar n, i;
    generate
        for (n = 0; n < N_OUT; n = n + 1) begin : GEN_NEURON
            wire signed [W_BITS-1:0] w_local [0:N_IN-1];
            for (i = 0; i < N_IN; i = i + 1) begin : GEN_W
                assign w_local[i] = weights[n*N_IN + i];
            end
            stochastic_neuron #(
                .N_IN(N_IN), .W_BITS(W_BITS), .PAR(PAR),
                .LFSR_SEED(32'hA5A50001 + n*32'h01010001)
            ) u_n (
                .clk(clk), .rst_n(rst_n), .start(start),
                .bits(bits), .weights(w_local), .bias(biases[n]),
                .K(K), .n_log2(n_log2),
                .busy(busy_i[n]), .done(done_i[n]), .y(y_bits[n])
            );
        end
    endgenerate

    assign busy = |busy_i;
    assign done = &done_i;
endmodule
