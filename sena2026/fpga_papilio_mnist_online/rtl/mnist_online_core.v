// =============================================================================
// mnist_online_core.v — folded 784→16→10 online slice (LX9 fit revision)
// Sync BRAM weights. Inference = calib-style vote/EE.
// Train = Mazumdar ±1 on OUTPUT layer only (serialized one lane/cycle) so the
// design fits XC6SLX9; hidden weights stay as BRAM-init features from ckpt.
// Telemetry E: per-class + EE hist via tele mux.
// =============================================================================
`timescale 1ns/1ps
module mnist_online_core #(
    parameter integer N_IN = 784,
    parameter integer N_HID = 8,
    parameter integer N_OUT = 10,
    parameter integer W_BITS = 8,
    parameter integer ACC_BITS = 20,
    parameter integer VOTE_BITS = 12,
    parameter integer TH_NUM = 9,
    parameter integer TH_DEN_LOG2 = 4,
    parameter W_HID_FILE = "w_hid.mem",
    parameter W_OUT_FILE = "w_out.mem"
) (
    input  wire clk,
    input  wire rst_n,
    input  wire start_infer,
    input  wire start_train,
    input  wire [5:0] t_max,
    input  wire early_exit_en,
    input  wire [3:0] n_log2_hid,
    input  wire [3:0] n_log2_out,
    input  wire [N_IN-1:0] x_bits,
    input  wire [3:0] label,
    input  wire score_en,
    input  wire clear_stats,
    input  wire [2:0] tele_kind,
    input  wire [3:0] tele_idx,
    output reg  [15:0] tele_data,
    output reg busy,
    output reg done,
    output reg [3:0] pred,
    output reg [5:0] passes,
    output reg [31:0] run_cycles,
    output reg [31:0] samples_trained,
    output reg [31:0] update_events
);
    (* ram_style = "block" *) reg [N_HID*W_BITS-1:0] ram_hid [0:N_IN];
    initial $readmemh(W_HID_FILE, ram_hid);
    reg [9:0] hid_addr;
    reg [N_HID*W_BITS-1:0] hid_do;
    always @(posedge clk) begin
        hid_do <= ram_hid[hid_addr];
    end

    (* ram_style = "block" *) reg [N_OUT*W_BITS-1:0] ram_out [0:N_HID];
    initial $readmemh(W_OUT_FILE, ram_out);
    reg [4:0] out_addr;
    reg out_we;
    reg [N_OUT*W_BITS-1:0] out_di, out_do;
    always @(posedge clk) begin
        if (out_we) ram_out[out_addr] <= out_di;
        out_do <= ram_out[out_addr];
    end

    localparam [31:0] TAPS = 32'h8020_0003;
    reg [31:0] lfsr_hid [0:N_HID-1];
    reg [31:0] lfsr_out [0:N_OUT-1];

    function signed [ACC_BITS-1:0] sext_w;
        input signed [W_BITS-1:0] w;
        begin sext_w = {{(ACC_BITS-W_BITS){w[W_BITS-1]}}, w}; end
    endfunction

    function signed [W_BITS-1:0] sat8;
        input signed [ACC_BITS-1:0] v;
        begin
            if (v > 20'sd127) sat8 = 8'sd127;
            else if (v < -20'sd128) sat8 = -8'sd128;
            else sat8 = v[W_BITS-1:0];
        end
    endfunction

    function signed [ACC_BITS-1:0] rnd_of;
        input [31:0] r; input [3:0] nl2;
        reg [16:0] half, mask, bits;
        begin
            half = 17'd1 << nl2;
            mask = (half << 1) - 17'd1;
            bits = r[16:0] & mask;
            rnd_of = $signed({{(ACC_BITS-18){1'b0}}, 1'b0, bits})
                   - $signed({{(ACC_BITS-18){1'b0}}, 1'b0, half});
        end
    endfunction

    function [2:0] ee_bin;
        input [5:0] p;
        begin
            if (p <= 6'd4) ee_bin = 3'd0;
            else if (p <= 6'd8) ee_bin = 3'd1;
            else if (p <= 6'd12) ee_bin = 3'd2;
            else if (p <= 6'd16) ee_bin = 3'd3;
            else if (p <= 6'd24) ee_bin = 3'd4;
            else if (p <= 6'd32) ee_bin = 3'd5;
            else if (p <= 6'd40) ee_bin = 3'd6;
            else ee_bin = 3'd7;
        end
    endfunction

    localparam S_IDLE=4'd0, S_HID_A=4'd1, S_HID_B=4'd2, S_HFIRE=4'd3,
               S_OUT_A=4'd4, S_OUT_B=4'd5, S_OFIRE=4'd6, S_VOTE=4'd7,
               S_DO_D=4'd8, S_UO_A=4'd9, S_UO_B=4'd10;

    reg [3:0] state;
    reg mode_train, do_score;
    reg [9:0] idx;
    reg [3:0] k_idx;
    reg bit_d;
    reg [N_HID:0] h_ext;
    reg [N_OUT-1:0] o_bits;
    reg signed [ACC_BITS-1:0] acc_hid [0:N_HID-1];
    reg signed [ACC_BITS-1:0] acc_out [0:N_OUT-1];
    reg signed [ACC_BITS-1:0] pre_o [0:N_OUT-1];
    reg [VOTE_BITS-1:0] votes [0:N_OUT-1];
    reg [31:0] cyc_free, cyc_start;
    reg signed [1:0] d_k [0:N_OUT-1];

    reg [15:0] class_correct [0:9];
    reg [15:0] class_seen [0:9];
    reg [15:0] ee_hist [0:7];
    reg [15:0] infer_total, infer_correct;

    integer k, b, c;
    reg signed [ACC_BITS-1:0] lane_acc;
    reg signed [W_BITS-1:0] lane_w;
    reg signed [1:0] d_cur;
    reg [N_OUT*W_BITS-1:0] new_out;
    reg [2:0] bin;
    reg [3:0] best;

    wire x_at = (idx == N_IN) ? 1'b1 : ((idx < N_IN) ? x_bits[idx] : 1'b0);

    reg [VOTE_BITS-1:0] votes_next [0:N_OUT-1];
    reg [VOTE_BITS+7:0] sum_v, sum_num;
    reg [VOTE_BITS-1:0] max_v;
    reg [3:0] argmax;
    always @* begin
        sum_v = 0; max_v = 0; argmax = 4'd0;
        for (c = 0; c < N_OUT; c = c + 1) begin
            votes_next[c] = votes[c] + {{(VOTE_BITS-1){1'b0}}, o_bits[c]};
            sum_v = sum_v + {8'd0, votes_next[c]};
            if (votes_next[c] > max_v) begin max_v = votes_next[c]; argmax = c[3:0]; end
        end
        sum_num = 0;
        for (b = 0; b < 8; b = b + 1)
            if (TH_NUM[b]) sum_num = sum_num + (sum_v << b);
    end
    wire conf_ok = (sum_v != 0) && (({8'd0, max_v} << TH_DEN_LOG2) >= sum_num);

    always @* begin
        case (tele_kind)
            3'd0: tele_data = class_correct[tele_idx];
            3'd1: tele_data = class_seen[tele_idx];
            3'd2: tele_data = ee_hist[tele_idx[2:0]];
            3'd3: tele_data = infer_total;
            3'd4: tele_data = infer_correct;
            default: tele_data = 16'd0;
        endcase
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            busy <= 0; done <= 0; pred <= 0; passes <= 0;
            run_cycles <= 0; samples_trained <= 0; update_events <= 0;
            infer_total <= 0; infer_correct <= 0;
            cyc_free <= 0; cyc_start <= 0;
            idx <= 0; k_idx <= 0; bit_d <= 0;
            h_ext <= 0; o_bits <= 0; mode_train <= 0; do_score <= 0;
            out_we <= 0; hid_addr <= 0; out_addr <= 0;
            out_di <= 0;
            for (k = 0; k < N_HID; k = k + 1) begin
                lfsr_hid[k] <= 32'hA5A5_0001 + k * 32'h0101_0001;
                acc_hid[k] <= 0;
            end
            for (k = 0; k < N_OUT; k = k + 1) begin
                lfsr_out[k] <= 32'h5A5A_1001 + k * 32'h0202_0001;
                acc_out[k] <= 0; pre_o[k] <= 0; votes[k] <= 0; d_k[k] <= 0;
                class_correct[k] <= 0; class_seen[k] <= 0;
            end
            for (k = 0; k < 8; k = k + 1) ee_hist[k] <= 0;
        end else begin
            done <= 0; out_we <= 0;
            cyc_free <= cyc_free + 1;
            if (clear_stats) begin
                infer_total <= 0; infer_correct <= 0;
                for (k = 0; k < N_OUT; k = k + 1) begin
                    class_correct[k] <= 0; class_seen[k] <= 0;
                end
                for (k = 0; k < 8; k = k + 1) ee_hist[k] <= 0;
            end
            if (busy) begin
                for (k = 0; k < N_HID; k = k + 1)
                    lfsr_hid[k] <= {1'b0, lfsr_hid[k][31:1]}
                                   ^ (lfsr_hid[k][0] ? TAPS : 32'b0);
                for (k = 0; k < N_OUT; k = k + 1)
                    lfsr_out[k] <= {1'b0, lfsr_out[k][31:1]}
                                   ^ (lfsr_out[k][0] ? TAPS : 32'b0);
            end

            case (state)
                S_IDLE: if (start_infer || start_train) begin
                    busy <= 1; mode_train <= start_train;
                    do_score <= score_en & ~start_train;
                    passes <= 0;
                    for (k = 0; k < N_OUT; k = k + 1) votes[k] <= 0;
                    for (k = 0; k < N_HID; k = k + 1) acc_hid[k] <= 0;
                    idx <= 0; cyc_start <= cyc_free;
                    hid_addr <= 0; state <= S_HID_A;
                end

                S_HID_A: begin
                    hid_addr <= idx; bit_d <= x_at; state <= S_HID_B;
                end
                S_HID_B: begin
                    if (bit_d) begin
                        for (k = 0; k < N_HID; k = k + 1)
                            acc_hid[k] <= acc_hid[k]
                                + sext_w(hid_do[k*W_BITS +: W_BITS]);
                    end
                    if (idx == N_IN) begin state <= S_HFIRE; idx <= 0; end
                    else begin idx <= idx + 1; state <= S_HID_A; end
                end

                S_HFIRE: begin
                    for (k = 0; k < N_HID; k = k + 1)
                        h_ext[k] <= (acc_hid[k] > rnd_of(lfsr_hid[k], n_log2_hid));
                    h_ext[N_HID] <= 1'b1;
                    for (k = 0; k < N_OUT; k = k + 1) acc_out[k] <= 0;
                    idx <= 0; state <= S_OUT_A;
                end

                S_OUT_A: begin
                    out_addr <= idx[4:0]; bit_d <= h_ext[idx[4:0]]; state <= S_OUT_B;
                end
                S_OUT_B: begin
                    if (bit_d) begin
                        for (k = 0; k < N_OUT; k = k + 1)
                            acc_out[k] <= acc_out[k]
                                + sext_w(out_do[k*W_BITS +: W_BITS]);
                    end
                    if (idx == N_HID) begin state <= S_OFIRE; idx <= 0; end
                    else begin idx <= idx + 1; state <= S_OUT_A; end
                end

                S_OFIRE: begin
                    for (k = 0; k < N_OUT; k = k + 1) begin
                        pre_o[k] <= acc_out[k];
                        o_bits[k] <= (acc_out[k] > rnd_of(lfsr_out[k], n_log2_out));
                    end
                    state <= mode_train ? S_DO_D : S_VOTE;
                end

                S_VOTE: begin
                    for (k = 0; k < N_OUT; k = k + 1) votes[k] <= votes_next[k];
                    passes <= passes + 1; pred <= argmax;
                    if ((early_exit_en && conf_ok) || (passes + 1 >= t_max)) begin
                        busy <= 0; done <= 1;
                        run_cycles <= cyc_free - cyc_start;
                        if (do_score) begin
                            infer_total <= infer_total + 1;
                            class_seen[label] <= class_seen[label] + 1;
                            if (argmax == label) begin
                                infer_correct <= infer_correct + 1;
                                class_correct[label] <= class_correct[label] + 1;
                            end
                            bin = ee_bin(passes + 1);
                            ee_hist[bin] <= ee_hist[bin] + 1;
                        end
                        state <= S_IDLE;
                    end else begin
                        for (k = 0; k < N_HID; k = k + 1) acc_hid[k] <= 0;
                        idx <= 0; state <= S_HID_A;
                    end
                end

                S_DO_D: begin
                    for (k = 0; k < N_OUT; k = k + 1) begin
                        if ((label == k[3:0]) && !o_bits[k]) d_k[k] <= 2'sd1;
                        else if ((label != k[3:0]) && o_bits[k]) d_k[k] <= -2'sd1;
                        else d_k[k] <= 2'sd0;
                    end
                    best = 0;
                    for (k = 1; k < N_OUT; k = k + 1)
                        if (pre_o[k] > pre_o[best]) best = k[3:0];
                    pred <= best; passes <= 1;
                    idx <= 0; k_idx <= 0; state <= S_UO_A;
                end

                // Serialize: for each feature idx, for each class k_idx, RMW one lane
                S_UO_A: begin
                    out_addr <= idx[4:0];
                    state <= S_UO_B;
                end
                S_UO_B: begin
                    d_cur = d_k[k_idx];
                    new_out = out_do;
                    if (d_cur != 2'sd0) begin
                        lane_w = out_do[k_idx*W_BITS +: W_BITS];
                        if (h_ext[idx[4:0]])
                            lane_acc = sext_w(lane_w) + {{(ACC_BITS-2){d_cur[1]}}, d_cur};
                        else
                            lane_acc = sext_w(lane_w) - {{(ACC_BITS-2){d_cur[1]}}, d_cur};
                        new_out[k_idx*W_BITS +: W_BITS] = sat8(lane_acc);
                        update_events <= update_events + 1;
                        out_we <= 1'b1; out_di <= new_out; out_addr <= idx[4:0];
                    end
                    if (k_idx == (N_OUT-1)) begin
                        k_idx <= 0;
                        if (idx == N_HID) begin
                            busy <= 0; done <= 1;
                            run_cycles <= cyc_free - cyc_start;
                            samples_trained <= samples_trained + 1;
                            state <= S_IDLE;
                        end else begin
                            idx <= idx + 1; state <= S_UO_A;
                        end
                    end else begin
                        k_idx <= k_idx + 1;
                        // re-read same addr for next lane (data may be written)
                        state <= S_UO_A;
                    end
                end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
