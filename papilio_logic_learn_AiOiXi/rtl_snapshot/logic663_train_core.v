// =============================================================================
// logic663_train_core.v — Papilio Pro on-chip learning PoC
// -----------------------------------------------------------------------------
// Topology (paper PoC):
//   i = 6  Ai1,Ai2,Oi1,Oi2,Xi1,Xi2
//   j = 6 + bias column  → i→j = 42
//   k = Ao,Oo,Xo (+ pad row) → j→k = 28, i→k skip = 24
//
// Epoch loop (64 patterns each phase, targets on-the-fly):
//   TRAIN 64 (Mazumdar mul-free ±1 + reverse gates)
//   EVAL  64 (stochastic majority + early-exit ≥75% after EE_MIN)
//   dwell so LED polarity (~epoch_id[0]) is visible
// =============================================================================
`timescale 1ns/1ps
module logic663_train_core #(
    parameter integer W_BITS      = 8,
    parameter integer ACC_W       = 16,
    parameter integer T_MAX       = 12,
    parameter integer EE_MIN      = 3,
    parameter integer N_FWD       = 12,
    parameter integer EPOCH_DWELL = 8000000  // ~0.25 s @ 32 MHz
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        enable_train,
    input  wire        eval_pulse,     // one EVAL-only epoch when idle
    output reg         busy,
    output reg         epoch_done,     // 1-cycle after TRAIN+EVAL
    output reg         eval_done,
    output reg  [15:0] epoch_id,
    output reg  [6:0]  correct_cnt,    // EVAL joint correct / 64
    output reg  [7:0]  last_mean_passes,
    output reg  [15:0] update_count
);
    localparam integer N_IN  = 6;
    localparam integer N_HID = 6;
    localparam integer N_OUT = 3;
    localparam integer JCOL  = 7;
    localparam integer N_IJ  = 42;
    localparam integer N_JK  = 28;
    localparam integer N_IK  = 24;

    reg signed [W_BITS-1:0] w_ij [0:N_IJ-1];
    reg signed [W_BITS-1:0] w_jk [0:N_JK-1];
    reg signed [W_BITS-1:0] w_ik [0:N_IK-1];

    localparam [31:0] TAPS = 32'h80200003;
    reg [31:0] lfsr;
    integer gi;

    initial begin
        for (gi = 0; gi < N_IJ; gi = gi + 1)
            w_ij[gi] = $signed({1'b0, gi[6:0]}) - 8'sd20;
        for (gi = 0; gi < N_JK; gi = gi + 1)
            w_jk[gi] = $signed({1'b0, gi[6:0]}) - 8'sd14;
        for (gi = 0; gi < N_IK; gi = gi + 1)
            w_ik[gi] = $signed({1'b0, gi[6:0]}) - 8'sd10;
        lfsr = 32'hA5A5_1234;
    end

    reg [5:0] pat;
    wire [N_IN-1:0] x = pat;
    // Ai1=x0 Ai2=x1 Oi1=x2 Oi2=x3 Xi1=x4 Xi2=x5
    wire Ao = x[0] & x[1];
    wire Oo = x[2] | x[3];
    wire Xo = x[4] ^ x[5];
    wire [N_OUT-1:0] tgt = {Xo, Oo, Ao};

    localparam S_IDLE=0, S_CLR=1, S_HID=2, S_HOUT=3, S_OUT=4, S_SKIP=5,
               S_FIRE=6, S_DEC=7, S_UPD_O=8, S_BACK=9, S_UPD_H=10,
               S_NEXT=11, S_EPOCH=12, S_DWELL=13;
    reg [3:0] st;
    reg [31:0] dwell_cnt;

    // phase: 0=TRAIN, 1=EVAL (full epoch = train then eval)
    reg       phase_eval;
    reg       eval_only;
    reg [3:0] tpass;
    reg [2:0] idx;
    reg signed [ACC_W-1:0] acc_h [0:N_HID-1];
    reg signed [ACC_W-1:0] acc_o [0:N_OUT-1];
    reg signed [ACC_W-1:0] pre_h [0:N_HID-1];
    reg signed [ACC_W-1:0] pre_o [0:N_OUT-1];
    // snapshot of w_jk used for backprop (pre-update)
    reg signed [W_BITS-1:0] wjk_snap [0:N_JK-1];
    reg [N_HID-1:0] hbits;
    reg [N_OUT-1:0] obits;
    reg [3:0] votes1 [0:N_OUT-1];
    reg [N_OUT-1:0] pred;
    reg [11:0] pass_sum;
    reg [6:0]  corr_acc;
    reg [3:0]  passes_used;
    reg signed [1:0] d_k [0:N_OUT-1];
    reg [N_OUT-1:0] gate_p;
    reg signed [ACC_W-1:0] back_j [0:N_HID-1];
    reg [2:0] upd_k;
    reg [2:0] upd_j;
    reg signed [1:0] dir_j;
    reg signed [ACC_W-1:0] abs_tmp, r_gate;
    reg [31:0] lfsr_x;
    reg signed [1:0] d_cur;
    reg d_neg;

    function signed [ACC_W-1:0] rnd_n;
        input [31:0] r;
        begin
            rnd_n = $signed({1'b0, r[3:0]}) - (N_FWD / 2);
        end
    endfunction

    function signed [W_BITS-1:0] sat8;
        input signed [ACC_W-1:0] v;
        begin
            if (v > 127) sat8 = 8'sd127;
            else if (v < -128) sat8 = -8'sd128;
            else sat8 = v[W_BITS-1:0];
        end
    endfunction

    function ee_ready;
        input [3:0] v0, v1, v2, tp;
        reg [3:0] z0, z1, z2, m0, m1, m2;
        begin
            z0 = tp - v0; z1 = tp - v1; z2 = tp - v2;
            m0 = (v0 > z0) ? v0 : z0;
            m1 = (v1 > z1) ? v1 : z1;
            m2 = (v2 > z2) ? v2 : z2;
            ee_ready = (tp >= EE_MIN) &&
                       (m0 >= ((tp * 3) / 4)) &&
                       (m1 >= ((tp * 3) / 4)) &&
                       (m2 >= ((tp * 3) / 4));
        end
    endfunction

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st <= S_IDLE;
            busy <= 1'b0;
            epoch_done <= 1'b0;
            eval_done <= 1'b0;
            epoch_id <= 16'd0;
            correct_cnt <= 7'd0;
            last_mean_passes <= 8'd0;
            update_count <= 16'd0;
            pat <= 6'd0;
            phase_eval <= 1'b0;
            eval_only <= 1'b0;
            lfsr <= 32'hA5A5_1234;
            dwell_cnt <= 32'd0;
            upd_k <= 3'd0;
            upd_j <= 3'd0;
        end else begin
            epoch_done <= 1'b0;
            eval_done <= 1'b0;
            lfsr <= {lfsr[30:0], ^(lfsr & TAPS)};

            case (st)
                S_IDLE: begin
                    busy <= 1'b0;
                    if (enable_train || eval_pulse) begin
                        busy <= 1'b1;
                        eval_only <= eval_pulse && !enable_train;
                        phase_eval <= (eval_pulse && !enable_train);
                        pat <= 6'd0;
                        corr_acc <= 7'd0;
                        pass_sum <= 12'd0;
                        update_count <= 16'd0;
                        st <= S_CLR;
                    end
                end

                S_CLR: begin
                    tpass <= 4'd0;
                    votes1[0] <= 4'd0;
                    votes1[1] <= 4'd0;
                    votes1[2] <= 4'd0;
                    for (gi = 0; gi < N_HID; gi = gi + 1)
                        acc_h[gi] <= {ACC_W{1'b0}};
                    idx <= 3'd0;
                    st <= S_HID;
                end

                S_HID: begin
                    if (idx <= N_IN) begin
                        for (gi = 0; gi < N_HID; gi = gi + 1) begin
                            if (idx == N_IN || x[idx])
                                acc_h[gi] <= acc_h[gi] +
                                    {{(ACC_W-W_BITS){w_ij[gi*JCOL+idx][W_BITS-1]}},
                                     w_ij[gi*JCOL+idx]};
                        end
                        idx <= idx + 3'd1;
                    end else begin
                        for (gi = 0; gi < N_HID; gi = gi + 1)
                            pre_h[gi] <= acc_h[gi];
                        st <= S_HOUT;
                    end
                end

                S_HOUT: begin
                    for (gi = 0; gi < N_HID; gi = gi + 1)
                        hbits[gi] <= (acc_h[gi] > rnd_n(lfsr ^ (gi * 32'h11)));
                    for (gi = 0; gi < N_OUT; gi = gi + 1)
                        acc_o[gi] <= {ACC_W{1'b0}};
                    idx <= 3'd0;
                    st <= S_OUT;
                end

                S_OUT: begin
                    if (idx <= N_HID) begin
                        for (gi = 0; gi < N_OUT; gi = gi + 1) begin
                            if (idx == N_HID || hbits[idx])
                                acc_o[gi] <= acc_o[gi] +
                                    {{(ACC_W-W_BITS){w_jk[gi*JCOL+idx][W_BITS-1]}},
                                     w_jk[gi*JCOL+idx]};
                        end
                        idx <= idx + 3'd1;
                    end else
                        st <= S_SKIP;
                end

                S_SKIP: begin
                    for (gi = 0; gi < N_OUT; gi = gi + 1) begin
                        acc_o[gi] <= acc_o[gi]
                          + (x[0] ? {{(ACC_W-W_BITS){w_ik[gi*6+0][W_BITS-1]}}, w_ik[gi*6+0]} : {ACC_W{1'b0}})
                          + (x[1] ? {{(ACC_W-W_BITS){w_ik[gi*6+1][W_BITS-1]}}, w_ik[gi*6+1]} : {ACC_W{1'b0}})
                          + (x[2] ? {{(ACC_W-W_BITS){w_ik[gi*6+2][W_BITS-1]}}, w_ik[gi*6+2]} : {ACC_W{1'b0}})
                          + (x[3] ? {{(ACC_W-W_BITS){w_ik[gi*6+3][W_BITS-1]}}, w_ik[gi*6+3]} : {ACC_W{1'b0}})
                          + (x[4] ? {{(ACC_W-W_BITS){w_ik[gi*6+4][W_BITS-1]}}, w_ik[gi*6+4]} : {ACC_W{1'b0}})
                          + (x[5] ? {{(ACC_W-W_BITS){w_ik[gi*6+5][W_BITS-1]}}, w_ik[gi*6+5]} : {ACC_W{1'b0}});
                    end
                    st <= S_FIRE;
                end

                S_FIRE: begin
                    for (gi = 0; gi < N_OUT; gi = gi + 1) begin
                        pre_o[gi] <= acc_o[gi];
                        if (acc_o[gi] > rnd_n(lfsr ^ (gi * 32'h101))) begin
                            obits[gi] <= 1'b1;
                            votes1[gi] <= votes1[gi] + 4'd1;
                        end else
                            obits[gi] <= 1'b0;
                    end
                    tpass <= tpass + 4'd1;
                    st <= S_DEC;
                end

                S_DEC: begin
                    if (!phase_eval) begin
                        // TRAIN: one stochastic pass → update
                        passes_used <= 4'd1;
                        pred <= obits;
                        for (gi = 0; gi < N_OUT; gi = gi + 1) begin
                            if (tgt[gi] && !obits[gi]) d_k[gi] <= 2'sd1;
                            else if (!tgt[gi] && obits[gi]) d_k[gi] <= -2'sd1;
                            else d_k[gi] <= 2'sd0;
                        end
                        for (gi = 0; gi < N_JK; gi = gi + 1)
                            wjk_snap[gi] <= w_jk[gi];
                        upd_k <= 3'd0;
                        st <= S_UPD_O;
                    end else if (ee_ready(votes1[0], votes1[1], votes1[2], tpass) ||
                                 tpass >= T_MAX) begin
                        passes_used <= tpass;
                        pred[0] <= (votes1[0] >= (tpass >> 1));
                        pred[1] <= (votes1[1] >= (tpass >> 1));
                        pred[2] <= (votes1[2] >= (tpass >> 1));
                        st <= S_NEXT;
                    end else begin
                        for (gi = 0; gi < N_HID; gi = gi + 1)
                            acc_h[gi] <= {ACC_W{1'b0}};
                        idx <= 3'd0;
                        st <= S_HID;
                    end
                end

                // Output + skip Mazumdar ±1 (sign of active feature)
                S_UPD_O: begin
                    if (upd_k < N_OUT) begin
                        d_cur = d_k[upd_k];
                        d_neg = d_cur[1];
                        if (d_cur != 2'sd0) begin
                            w_jk[upd_k*JCOL+6] <= sat8(
                                {{(ACC_W-W_BITS){w_jk[upd_k*JCOL+6][W_BITS-1]}}, w_jk[upd_k*JCOL+6]}
                                + {{(ACC_W-2){d_neg}}, d_cur});
                            for (gi = 0; gi < N_HID; gi = gi + 1) begin
                                if (hbits[gi])
                                    w_jk[upd_k*JCOL+gi] <= sat8(
                                        {{(ACC_W-W_BITS){w_jk[upd_k*JCOL+gi][W_BITS-1]}}, w_jk[upd_k*JCOL+gi]}
                                        + {{(ACC_W-2){d_neg}}, d_cur});
                                else
                                    w_jk[upd_k*JCOL+gi] <= sat8(
                                        {{(ACC_W-W_BITS){w_jk[upd_k*JCOL+gi][W_BITS-1]}}, w_jk[upd_k*JCOL+gi]}
                                        - {{(ACC_W-2){d_neg}}, d_cur});
                            end
                            for (gi = 0; gi < N_IN; gi = gi + 1) begin
                                if (x[gi])
                                    w_ik[upd_k*N_IN+gi] <= sat8(
                                        {{(ACC_W-W_BITS){w_ik[upd_k*N_IN+gi][W_BITS-1]}}, w_ik[upd_k*N_IN+gi]}
                                        + {{(ACC_W-2){d_neg}}, d_cur});
                                else
                                    w_ik[upd_k*N_IN+gi] <= sat8(
                                        {{(ACC_W-W_BITS){w_ik[upd_k*N_IN+gi][W_BITS-1]}}, w_ik[upd_k*N_IN+gi]}
                                        - {{(ACC_W-2){d_neg}}, d_cur});
                            end
                            update_count <= update_count + 16'd1;
                            abs_tmp = pre_o[upd_k][ACC_W-1] ? -pre_o[upd_k] : pre_o[upd_k];
                            lfsr_x = lfsr ^ (upd_k * 32'h55);
                            r_gate = {12'd0, lfsr_x[3:0]};
                            gate_p[upd_k] <= (abs_tmp <= r_gate) ? 1'b1 : 1'b0;
                        end else
                            gate_p[upd_k] <= 1'b0;
                        upd_k <= upd_k + 3'd1;
                    end else begin
                        st <= S_BACK;
                    end
                end

                // back_j from snapshotted W_jk and reverse gates
                S_BACK: begin
                    for (gi = 0; gi < N_HID; gi = gi + 1) begin
                        back_j[gi] <=
                            ((gate_p[0] && d_k[0] != 2'sd0) ?
                                (d_k[0][1] ?
                                    -{{(ACC_W-W_BITS){wjk_snap[0*JCOL+gi][W_BITS-1]}}, wjk_snap[0*JCOL+gi]} :
                                     {{(ACC_W-W_BITS){wjk_snap[0*JCOL+gi][W_BITS-1]}}, wjk_snap[0*JCOL+gi]})
                                : {ACC_W{1'b0}})
                          + ((gate_p[1] && d_k[1] != 2'sd0) ?
                                (d_k[1][1] ?
                                    -{{(ACC_W-W_BITS){wjk_snap[1*JCOL+gi][W_BITS-1]}}, wjk_snap[1*JCOL+gi]} :
                                     {{(ACC_W-W_BITS){wjk_snap[1*JCOL+gi][W_BITS-1]}}, wjk_snap[1*JCOL+gi]})
                                : {ACC_W{1'b0}})
                          + ((gate_p[2] && d_k[2] != 2'sd0) ?
                                (d_k[2][1] ?
                                    -{{(ACC_W-W_BITS){wjk_snap[2*JCOL+gi][W_BITS-1]}}, wjk_snap[2*JCOL+gi]} :
                                     {{(ACC_W-W_BITS){wjk_snap[2*JCOL+gi][W_BITS-1]}}, wjk_snap[2*JCOL+gi]})
                                : {ACC_W{1'b0}});
                    end
                    upd_j <= 3'd0;
                    st <= S_UPD_H;
                end

                // One hidden unit per cycle: reverse-gate then ±1 on all 7 cols
                S_UPD_H: begin
                    if (upd_j < N_HID) begin
                        abs_tmp = pre_h[upd_j][ACC_W-1] ? -pre_h[upd_j] : pre_h[upd_j];
                        lfsr_x = lfsr ^ (upd_j * 32'h21);
                        if (abs_tmp <= {12'd0, lfsr_x[3:0]}) begin
                            abs_tmp = back_j[upd_j][ACC_W-1] ? -back_j[upd_j] : back_j[upd_j];
                            lfsr_x = lfsr ^ (upd_j * 32'h43);
                            if (abs_tmp > {13'd0, lfsr_x[2:0]}) begin
                                dir_j = back_j[upd_j][ACC_W-1] ? -2'sd1 : 2'sd1;
                                w_ij[upd_j*JCOL+6] <= sat8(
                                    {{(ACC_W-W_BITS){w_ij[upd_j*JCOL+6][W_BITS-1]}}, w_ij[upd_j*JCOL+6]}
                                    + {{(ACC_W-2){dir_j[1]}}, dir_j});
                                for (gi = 0; gi < N_IN; gi = gi + 1) begin
                                    if (x[gi])
                                        w_ij[upd_j*JCOL+gi] <= sat8(
                                            {{(ACC_W-W_BITS){w_ij[upd_j*JCOL+gi][W_BITS-1]}},
                                             w_ij[upd_j*JCOL+gi]}
                                            + {{(ACC_W-2){dir_j[1]}}, dir_j});
                                    else
                                        w_ij[upd_j*JCOL+gi] <= sat8(
                                            {{(ACC_W-W_BITS){w_ij[upd_j*JCOL+gi][W_BITS-1]}},
                                             w_ij[upd_j*JCOL+gi]}
                                            - {{(ACC_W-2){dir_j[1]}}, dir_j});
                                end
                            end
                        end
                        upd_j <= upd_j + 3'd1;
                    end else
                        st <= S_NEXT;
                end

                S_NEXT: begin
                    if (phase_eval && (pred == tgt))
                        corr_acc <= corr_acc + 7'd1;
                    if (phase_eval)
                        pass_sum <= pass_sum + {8'd0, passes_used};

                    if (pat == 6'd63) begin
                        if (!phase_eval && !eval_only) begin
                            // TRAIN done → EVAL
                            phase_eval <= 1'b1;
                            pat <= 6'd0;
                            corr_acc <= 7'd0;
                            pass_sum <= 12'd0;
                            st <= S_CLR;
                        end else
                            st <= S_EPOCH;
                    end else begin
                        pat <= pat + 6'd1;
                        st <= S_CLR;
                    end
                end

                S_EPOCH: begin
                    correct_cnt <= corr_acc;
                    // mean passes ≈ pass_sum/64 → take [11:6]
                    last_mean_passes <= {2'b0, pass_sum[11:6]};
                    epoch_id <= epoch_id + 16'd1;
                    epoch_done <= 1'b1;
                    eval_done <= 1'b1;
                    dwell_cnt <= EPOCH_DWELL;
                    st <= S_DWELL;
                end

                S_DWELL: begin
                    if (dwell_cnt != 32'd0)
                        dwell_cnt <= dwell_cnt - 32'd1;
                    else if (enable_train) begin
                        eval_only <= 1'b0;
                        phase_eval <= 1'b0;
                        pat <= 6'd0;
                        corr_acc <= 7'd0;
                        pass_sum <= 12'd0;
                        update_count <= 16'd0;
                        st <= S_CLR;
                    end else begin
                        busy <= 1'b0;
                        st <= S_IDLE;
                    end
                end

                default: st <= S_IDLE;
            endcase
        end
    end
endmodule
