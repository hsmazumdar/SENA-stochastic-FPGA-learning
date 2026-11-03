// =============================================================================
// sena_early_exit.v
// -----------------------------------------------------------------------------
// Confidence-gated early exit over temporal stochastic passes.
//   votes[c] incremented when class bit fires
//   conf = max(votes) / sum(votes)
//   EXIT when  max * 2^TH_DEN_LOG2 >= sum * TH_NUM
//
// The threshold test is evaluated on votes_next (this pass INCLUDED), and
// sum*TH_NUM is built from an explicit shift-add network, so the whole tile
// contains no multiplier of any kind.
// =============================================================================
`timescale 1ns/1ps
module sena_early_exit #(
    parameter integer N_CLASS = 10,
    parameter integer VOTE_BITS = 12,
    parameter integer T_MAX = 40,
    // theta = TH_NUM / 2^TH_DEN_LOG2   (e.g. 3/4 = 0.75, 9/16 = 0.5625)
    parameter integer TH_NUM = 3,
    parameter integer TH_DEN_LOG2 = 2,
    // derived widths (do not override)
    parameter integer PASS_BITS = $clog2(T_MAX + 1),
    parameter integer CLS_BITS  = (N_CLASS > 1) ? $clog2(N_CLASS) : 1
) (
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     clear,       // new sample
    input  wire                     pass_valid,  // one pass result ready
    input  wire [N_CLASS-1:0]       class_bits,  // one-hot or multi-fire OK
    output reg  [VOTE_BITS-1:0]     votes [0:N_CLASS-1],
    output reg  [PASS_BITS-1:0]     passes,
    output reg                      exit_ready,
    output reg  [CLS_BITS-1:0]      pred,
    output reg                      done_force   // hit T_MAX without theta
);
    localparam integer ACCW      = VOTE_BITS + 8;
    localparam [PASS_BITS-1:0] TMAX_C = T_MAX[PASS_BITS-1:0];

    integer c, b;

    // ---- votes_next: current votes plus this pass -----------------------
    reg [VOTE_BITS-1:0] votes_next [0:N_CLASS-1];
    always @* begin
        for (c = 0; c < N_CLASS; c = c + 1)
            votes_next[c] = votes[c] + {{(VOTE_BITS-1){1'b0}}, class_bits[c]};
    end

    // ---- sum / max / argmax over votes_next -----------------------------
    reg [ACCW-1:0]      sum_v;
    reg [VOTE_BITS-1:0] max_v;
    reg [CLS_BITS-1:0]  argmax;
    always @* begin
        sum_v  = {ACCW{1'b0}};
        max_v  = {VOTE_BITS{1'b0}};
        argmax = {CLS_BITS{1'b0}};
        for (c = 0; c < N_CLASS; c = c + 1) begin
            sum_v = sum_v + {{(ACCW-VOTE_BITS){1'b0}}, votes_next[c]};
            if (votes_next[c] > max_v) begin
                max_v  = votes_next[c];
                argmax = c[CLS_BITS-1:0];
            end
        end
    end

    // ---- sum * TH_NUM as an explicit shift-add network (no multiplier) ---
    reg [ACCW-1:0] sum_num;
    always @* begin
        sum_num = {ACCW{1'b0}};
        for (b = 0; b < 8; b = b + 1)
            if (TH_NUM[b]) sum_num = sum_num + (sum_v << b);
    end

    wire [ACCW-1:0] max_shift = {{(ACCW-VOTE_BITS){1'b0}}, max_v} << TH_DEN_LOG2;
    wire conf_ok = (sum_v != {ACCW{1'b0}}) && (max_shift >= sum_num);
    wire at_tmax = ((passes + {{(PASS_BITS-1){1'b0}}, 1'b1}) >= TMAX_C);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n || clear) begin
            for (c = 0; c < N_CLASS; c = c + 1) votes[c] <= {VOTE_BITS{1'b0}};
            passes     <= {PASS_BITS{1'b0}};
            exit_ready <= 1'b0;
            pred       <= {CLS_BITS{1'b0}};
            done_force <= 1'b0;
        end else if (pass_valid && !exit_ready && !done_force) begin
            for (c = 0; c < N_CLASS; c = c + 1) votes[c] <= votes_next[c];
            passes <= passes + {{(PASS_BITS-1){1'b0}}, 1'b1};
            pred   <= argmax;
            if (conf_ok)      exit_ready <= 1'b1;
            else if (at_tmax) done_force <= 1'b1;
        end
    end
endmodule
