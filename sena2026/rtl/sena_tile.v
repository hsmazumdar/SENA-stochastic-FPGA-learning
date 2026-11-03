// =============================================================================
// sena_tile.v — one SENA learner tile (hidden bank + output bank stub)
// -----------------------------------------------------------------------------
// Configurable small tile for synthesis experiments.  Default matches a
// *slice* of the MNIST net (not the full 160 hidden in one shot) so that FPGA
// place-and-route stays tractable; scale N_HID_BANK / N_OUT for full mapping.
//
// Full MNIST 784-160-10 is obtained by instantiating multiple tiles or
// increasing N_HID_BANK (see estimate_resources.py).
// =============================================================================
`timescale 1ns/1ps
module sena_tile #(
    parameter integer N_IN       = 784,
    parameter integer N_HID_BANK = 16,   // parallel hidden neurons in this tile
    parameter integer N_OUT      = 10,
    parameter integer W_BITS     = 16,
    parameter integer PAR        = 1,   // input-layer folding
    parameter integer PAR_OUT    = 1,   // output-layer folding
    parameter integer T_MAX      = 40,
    parameter integer TH_NUM     = 3,
    parameter integer TH_DEN_LOG2 = 2
) (
    input  wire                  clk,
    input  wire                  rst_n,
    // sample control
    input  wire                  sample_start,   // load new input, clear votes
    input  wire                  pass_start,     // run one stochastic pass
    input  wire [N_IN-1:0]       x_bits,
    // weight memories (host/DMA write path omitted for clarity)
    input  wire signed [W_BITS-1:0] w_hid  [0:N_HID_BANK*N_IN-1],
    input  wire signed [W_BITS-1:0] b_hid  [0:N_HID_BANK-1],
    input  wire signed [W_BITS-1:0] w_out  [0:N_OUT*N_HID_BANK-1],
    input  wire signed [W_BITS-1:0] b_out  [0:N_OUT-1],
    input  wire signed [W_BITS-1:0] K_hid,
    input  wire signed [W_BITS-1:0] K_out,
    input  wire [3:0]            n_log2_hid,
    input  wire [3:0]            n_log2_out,
    // status / result
    output wire [11:0]           votes_dbg [0:N_OUT-1],
    output wire                  busy,
    output wire                  exit_ready,
    output wire                  done_force,
    output wire [$clog2(N_OUT)-1:0] pred,
    output wire [$clog2(T_MAX+1)-1:0] passes_used
);
    localparam S_IDLE=0, S_HID=1, S_OUT=2, S_VOTE=3;
    reg [1:0] state;
    reg clear_votes;

    wire hid_busy, hid_done, out_busy, out_done;
    wire [N_HID_BANK-1:0] h_bits;
    wire [N_OUT-1:0]      o_bits;

    sena_layer #(
        .N_IN(N_IN), .N_OUT(N_HID_BANK), .W_BITS(W_BITS), .PAR(PAR)
    ) u_hid (
        .clk(clk), .rst_n(rst_n),
        .start(pass_start && (state==S_IDLE)),
        .bits(x_bits), .weights(w_hid), .biases(b_hid),
        .K(K_hid), .n_log2(n_log2_hid),
        .busy(hid_busy), .done(hid_done), .y_bits(h_bits)
    );

    sena_layer #(
        .N_IN(N_HID_BANK), .N_OUT(N_OUT), .W_BITS(W_BITS), .PAR(PAR_OUT)
    ) u_out (
        .clk(clk), .rst_n(rst_n),
        .start(hid_done),
        .bits(h_bits), .weights(w_out), .biases(b_out),
        .K(K_out), .n_log2(n_log2_out),
        .busy(out_busy), .done(out_done), .y_bits(o_bits)
    );

    sena_early_exit #(
        .N_CLASS(N_OUT), .T_MAX(T_MAX),
        .TH_NUM(TH_NUM), .TH_DEN_LOG2(TH_DEN_LOG2)
    ) u_ee (
        .clk(clk), .rst_n(rst_n),
        .clear(sample_start | clear_votes),
        .pass_valid(out_done),
        .class_bits(o_bits),
        .votes(votes_dbg), .passes(passes_used),
        .exit_ready(exit_ready), .pred(pred), .done_force(done_force)
    );

    assign busy = hid_busy | out_busy | (state != S_IDLE);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            clear_votes <= 1'b0;
        end else begin
            clear_votes <= 1'b0;
            case (state)
                S_IDLE: if (pass_start) state <= S_HID;
                S_HID:  if (hid_done)   state <= S_OUT;
                S_OUT:  if (out_done)   state <= S_VOTE;
                S_VOTE: state <= S_IDLE;
                default: state <= S_IDLE;
            endcase
            if (sample_start) begin
                state <= S_IDLE;
                clear_votes <= 1'b1;
            end
        end
    end
endmodule
