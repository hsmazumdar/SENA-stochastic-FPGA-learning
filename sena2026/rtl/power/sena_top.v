// =============================================================================
// sena_top.v — Artix-7 power-analysis wrapper around sena_tile
// -----------------------------------------------------------------------------
// Clean host ports + on-chip weight memories ($readmemh from stimuli/*.mem).
// Instantiates the full folded 784→160→10 tile (PAR=8 / PAR_OUT=2).
// =============================================================================
`timescale 1ns/1ps

module sena_top #(
    parameter integer N_IN       = 784,
    parameter integer N_HID      = 160,
    parameter integer N_OUT      = 10,
    parameter integer W_BITS     = 16,
    parameter integer PAR        = 8,
    parameter integer PAR_OUT    = 2,
    parameter integer T_MAX      = 40,
    parameter         W_HID_FILE = "w_hid.mem",
    parameter         B_HID_FILE = "b_hid.mem",
    parameter         W_OUT_FILE = "w_out.mem",
    parameter         B_OUT_FILE = "b_out.mem"
) (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  sample_start,
    input  wire                  pass_start,
    input  wire [N_IN-1:0]       x_bits,
    input  wire [3:0]            n_log2_hid,
    input  wire [3:0]            n_log2_out,
    output wire [$clog2(N_OUT)-1:0] prediction,
    output wire                  busy,
    output wire                  exit_ready,
    output wire                  done_force,
    output wire [$clog2(T_MAX+1)-1:0] passes_used
);
    localparam integer W_HID_DEPTH = N_HID * N_IN;
    localparam integer W_OUT_DEPTH = N_OUT * N_HID;

    (* ram_style = "block" *) reg signed [W_BITS-1:0] w_hid_mem [0:W_HID_DEPTH-1];
    (* ram_style = "block" *) reg signed [W_BITS-1:0] b_hid_mem [0:N_HID-1];
    (* ram_style = "block" *) reg signed [W_BITS-1:0] w_out_mem [0:W_OUT_DEPTH-1];
    (* ram_style = "block" *) reg signed [W_BITS-1:0] b_out_mem [0:N_OUT-1];

    // Flat → unpacked ports expected by sena_tile
    wire signed [W_BITS-1:0] w_hid [0:W_HID_DEPTH-1];
    wire signed [W_BITS-1:0] b_hid [0:N_HID-1];
    wire signed [W_BITS-1:0] w_out [0:W_OUT_DEPTH-1];
    wire signed [W_BITS-1:0] b_out [0:N_OUT-1];

    genvar gi;
    generate
        for (gi = 0; gi < W_HID_DEPTH; gi = gi + 1) begin : g_wh
            assign w_hid[gi] = w_hid_mem[gi];
        end
        for (gi = 0; gi < N_HID; gi = gi + 1) begin : g_bh
            assign b_hid[gi] = b_hid_mem[gi];
        end
        for (gi = 0; gi < W_OUT_DEPTH; gi = gi + 1) begin : g_wo
            assign w_out[gi] = w_out_mem[gi];
        end
        for (gi = 0; gi < N_OUT; gi = gi + 1) begin : g_bo
            assign b_out[gi] = b_out_mem[gi];
        end
    endgenerate

    wire [11:0] votes_dbg [0:N_OUT-1];
    wire signed [W_BITS-1:0] K_hid = '0;
    wire signed [W_BITS-1:0] K_out = '0;

    initial begin
        $readmemh(W_HID_FILE, w_hid_mem);
        $readmemh(B_HID_FILE, b_hid_mem);
        $readmemh(W_OUT_FILE, w_out_mem);
        $readmemh(B_OUT_FILE, b_out_mem);
    end

    sena_tile #(
        .N_IN(N_IN), .N_HID_BANK(N_HID), .N_OUT(N_OUT),
        .W_BITS(W_BITS), .PAR(PAR), .PAR_OUT(PAR_OUT), .T_MAX(T_MAX),
        .TH_NUM(3), .TH_DEN_LOG2(2)
    ) uut_tile (
        .clk(clk), .rst_n(rst_n),
        .sample_start(sample_start), .pass_start(pass_start),
        .x_bits(x_bits),
        .w_hid(w_hid), .b_hid(b_hid), .w_out(w_out), .b_out(b_out),
        .K_hid(K_hid), .K_out(K_out),
        .n_log2_hid(n_log2_hid), .n_log2_out(n_log2_out),
        .votes_dbg(votes_dbg),
        .busy(busy), .exit_ready(exit_ready), .done_force(done_force),
        .pred(prediction), .passes_used(passes_used)
    );
endmodule
