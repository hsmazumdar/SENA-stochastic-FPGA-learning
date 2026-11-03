// =============================================================================
// logic663_core.v — On-chip SENA learning demo: 6→6→3 AND / OR / XOR
// -----------------------------------------------------------------------------
// Inputs : {a0,b0, a1,b1, a2,b2}
// Targets: {a0&b0, a1|b1, a2^b2}
//
// Modes (cmd):
//   0 = idle
//   1 = evaluate all 64 patterns (report correct count) with early-exit inference
//   2 = train one epoch over 64 patterns (integer ±1 updates, no multipliers)
//
// Weights live in writable regs (FPGA BRAM inference). Optional Flash save is
// a board wrapper concern — this core is the learning fabric.
// =============================================================================
`timescale 1ns/1ps
module logic663_core #(
    parameter integer N_IN     = 6,
    parameter integer N_HID    = 6,
    parameter integer N_OUT    = 3,
    parameter integer W_BITS   = 8,
    parameter integer ACC_BITS = 16,
    parameter integer T_MAX    = 16,
    parameter integer TH_NUM   = 3,   // early-exit: max_votes >= (TH_NUM*T)/4
    parameter integer TH_DEN   = 4
) (
    input  wire                clk,
    input  wire                rst_n,
    input  wire                go,           // pulse: start cmd
    input  wire [1:0]          cmd,          // 1=eval, 2=train
    output reg                 busy,
    output reg                 done,         // 1-cycle pulse
    output reg  [6:0]          correct_cnt,  // 0..64 after eval
    output reg  [15:0]         epoch_updates,// train steps committed
    output reg  [7:0]          last_passes,  // mean-ish: passes of last pattern
    output reg  [2:0]          last_pred
);
    // ---- writable weights (bias column = index N_IN / N_HID) ----
    reg signed [W_BITS-1:0] w_hid [0:N_HID*(N_IN+1)-1];
    reg signed [W_BITS-1:0] w_out [0:N_OUT*(N_HID+1)-1];

    // LFSR bank
    localparam [31:0] TAPS = 32'h8020_0003;
    reg [31:0] lfsr_h [0:N_HID-1];
    reg [31:0] lfsr_o [0:N_OUT-1];

    integer i, j, k;

    // init small random-ish weights
    initial begin
        for (i = 0; i < N_HID*(N_IN+1); i = i + 1)
            w_hid[i] = ((i * 3) ^ 8'h5A) - 8'sd20;
        for (i = 0; i < N_OUT*(N_HID+1); i = i + 1)
            w_out[i] = ((i * 5) ^ 8'h3C) - 8'sd10;
    end

    // pattern index 0..63 packs the 6 input bits
    reg [5:0] pat;
    wire [N_IN-1:0] x_bits = pat;
    wire [N_OUT-1:0] target = {
        x_bits[5] ^ x_bits[4],   // XOR of pair2
        x_bits[3] | x_bits[2],   // OR  of pair1
        x_bits[1] & x_bits[0]    // AND of pair0
    };

    // ---- forward / train FSM ----
    localparam S_IDLE=0, S_LOAD=1, S_PASS=2, S_FIRE_H=3, S_FIRE_O=4,
               S_VOTE=5, S_TRAIN=6, S_NEXT=7, S_DONE=8;
    reg [3:0] state;
    reg [1:0] mode;          // latched cmd
    reg [4:0] t_idx;         // pass index
    reg [3:0] in_idx;
    reg signed [ACC_BITS-1:0] acc_h [0:N_HID-1];
    reg signed [ACC_BITS-1:0] acc_o [0:N_OUT-1];
    reg [N_HID-1:0] h_bits;
    reg [N_OUT-1:0] o_bits;
    reg [4:0] votes1 [0:N_OUT-1];  // count of 1s over passes
    reg [4:0] passes_used;
    reg [N_OUT-1:0] pred_bits;
    reg signed [W_BITS-1:0] d_out [0:N_OUT-1]; // {-1,0,+1}

    function signed [ACC_BITS-1:0] rnd8;
        input [31:0] r;
        begin
            // uniform-ish in [-8,7]
            rnd8 = $signed({{(ACC_BITS-4){r[3]}}, r[3:0]}) - 5'sd8;
        end
    endfunction

    function signed [W_BITS-1:0] sat8;
        input signed [ACC_BITS-1:0] v;
        begin
            if (v > 127) sat8 = 8'sd127;
            else if (v < -128) sat8 = -8'sd128;
            else sat8 = v[W_BITS-1:0];
        end
    endfunction

    // early-exit: each head has clear majority
    function automatic ee_ok;
        input [4:0] v1;
        input [4:0] t;
        reg [4:0] v0, maj;
        begin
            v0 = t - v1;
            maj = (v1 > v0) ? v1 : v0;
            ee_ok = (maj * TH_DEN >= TH_NUM * t) && (t >= 4);
        end
    endfunction

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            busy <= 0; done <= 0;
            correct_cnt <= 0; epoch_updates <= 0;
            last_passes <= 0; last_pred <= 0;
            pat <= 0; mode <= 0; t_idx <= 0;
            for (k = 0; k < N_HID; k = k + 1) lfsr_h[k] <= 32'hACE1_0000 + k;
            for (k = 0; k < N_OUT; k = k + 1) lfsr_o[k] <= 32'hBEEF_0000 + k;
        end else begin
            done <= 0;
            case (state)
                S_IDLE: begin
                    busy <= 0;
                    if (go) begin
                        mode <= cmd;
                        busy <= 1;
                        pat <= 0;
                        correct_cnt <= 0;
                        epoch_updates <= 0;
                        state <= S_LOAD;
                    end
                end

                S_LOAD: begin
                    t_idx <= 0;
                    passes_used <= 0;
                    for (k = 0; k < N_OUT; k = k + 1) votes1[k] <= 0;
                    for (k = 0; k < N_HID; k = k + 1) acc_h[k] <= 0;
                    in_idx <= 0;
                    state <= S_PASS;
                end

                // Accumulate hidden: for each input bit / bias
                S_PASS: begin
                    if (in_idx <= N_IN) begin
                        for (k = 0; k < N_HID; k = k + 1) begin
                            if (in_idx == N_IN || x_bits[in_idx])
                                acc_h[k] <= acc_h[k] +
                                    {{(ACC_BITS-W_BITS){w_hid[k*(N_IN+1)+in_idx][W_BITS-1]}},
                                     w_hid[k*(N_IN+1)+in_idx]};
                        end
                        in_idx <= in_idx + 1;
                    end else begin
                        state <= S_FIRE_H;
                    end
                end

                S_FIRE_H: begin
                    for (k = 0; k < N_HID; k = k + 1) begin
                        lfsr_h[k] <= {lfsr_h[k][30:0],
                            ^(lfsr_h[k] & TAPS)};
                        h_bits[k] <= (acc_h[k] > rnd8(lfsr_h[k]));
                    end
                    for (k = 0; k < N_OUT; k = k + 1) acc_o[k] <= 0;
                    in_idx <= 0;
                    state <= S_FIRE_O;
                end

                // reuse S_FIRE_O as output accumulate over hidden+bias
                S_FIRE_O: begin
                    if (in_idx <= N_HID) begin
                        for (k = 0; k < N_OUT; k = k + 1) begin
                            if (in_idx == N_HID || h_bits[in_idx])
                                acc_o[k] <= acc_o[k] +
                                    {{(ACC_BITS-W_BITS){w_out[k*(N_HID+1)+in_idx][W_BITS-1]}},
                                     w_out[k*(N_HID+1)+in_idx]};
                        end
                        in_idx <= in_idx + 1;
                    end else begin
                        for (k = 0; k < N_OUT; k = k + 1) begin
                            lfsr_o[k] <= {lfsr_o[k][30:0],
                                ^(lfsr_o[k] & TAPS)};
                            o_bits[k] <= (acc_o[k] > rnd8(lfsr_o[k]));
                        end
                        state <= S_VOTE;
                    end
                end

                S_VOTE: begin
                    for (k = 0; k < N_OUT; k = k + 1)
                        if (o_bits[k]) votes1[k] <= votes1[k] + 1;
                    passes_used <= t_idx + 1;
                    // majority prediction so far
                    for (k = 0; k < N_OUT; k = k + 1)
                        pred_bits[k] <= (votes1[k] + (o_bits[k]?1:0))
                                        >= ((t_idx + 1) >> 1);
                    // fix: votes already updated above — recompute cleanly next
                    if ((mode == 2 && t_idx == 0) ||
                        (t_idx + 1 >= T_MAX) ||
                        (ee_ok(votes1[0], t_idx+1) &&
                         ee_ok(votes1[1], t_idx+1) &&
                         ee_ok(votes1[2], t_idx+1))) begin
                        // finalize pred from votes1
                        for (k = 0; k < N_OUT; k = k + 1)
                            pred_bits[k] <= (votes1[k] >= ((t_idx + 1) >> 1));
                        last_passes <= t_idx + 1;
                        last_pred <= pred_bits;
                        if (mode == 2)
                            state <= S_TRAIN;
                        else
                            state <= S_NEXT;
                    end else begin
                        t_idx <= t_idx + 1;
                        for (k = 0; k < N_HID; k = k + 1) acc_h[k] <= 0;
                        in_idx <= 0;
                        state <= S_PASS;
                    end
                end

                S_TRAIN: begin
                    // Integer ±1 update (multiplier-free), using last o_bits/h_bits
                    // Output layer: d = target - pred (here use single-pass o_bits)
                    for (k = 0; k < N_OUT; k = k + 1) begin
                        if (target[k] && !o_bits[k]) d_out[k] = 8'sd1;
                        else if (!target[k] && o_bits[k]) d_out[k] = -8'sd1;
                        else d_out[k] = 8'sd0;
                    end
                    for (k = 0; k < N_OUT; k = k + 1) begin
                        for (j = 0; j < N_HID; j = j + 1) begin
                            if (d_out[k] != 0) begin
                                if (h_bits[j])
                                    w_out[k*(N_HID+1)+j] <= sat8(
                                        w_out[k*(N_HID+1)+j] + d_out[k]);
                                else
                                    w_out[k*(N_HID+1)+j] <= sat8(
                                        w_out[k*(N_HID+1)+j] - d_out[k]);
                            end
                        end
                        // bias
                        if (d_out[k] != 0)
                            w_out[k*(N_HID+1)+N_HID] <= sat8(
                                w_out[k*(N_HID+1)+N_HID] + d_out[k]);
                    end
                    // Hidden: simple sign backprop proxy (no multiply)
                    for (j = 0; j < N_HID; j = j + 1) begin
                        // back_j ≈ sum_k sign(d_k)*sign(w_out)
                        // use OR of error heads as enable
                        if ((d_out[0] != 0) || (d_out[1] != 0) || (d_out[2] != 0)) begin
                            for (i = 0; i < N_IN; i = i + 1) begin
                                if (x_bits[i])
                                    w_hid[j*(N_IN+1)+i] <= sat8(
                                        w_hid[j*(N_IN+1)+i] +
                                        ((d_out[0]|d_out[1]|d_out[2]) ? 8'sd1 : -8'sd1));
                            end
                            w_hid[j*(N_IN+1)+N_IN] <= sat8(
                                w_hid[j*(N_IN+1)+N_IN] + 8'sd1);
                        end
                    end
                    epoch_updates <= epoch_updates + 1;
                    state <= S_NEXT;
                end

                S_NEXT: begin
                    if (mode == 1) begin
                        if (pred_bits == target)
                            correct_cnt <= correct_cnt + 1;
                    end
                    if (pat == 6'd63) begin
                        state <= S_DONE;
                    end else begin
                        pat <= pat + 1;
                        state <= S_LOAD;
                    end
                end

                S_DONE: begin
                    busy <= 0;
                    done <= 1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
