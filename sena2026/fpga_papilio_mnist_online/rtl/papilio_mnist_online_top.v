// =============================================================================
// papilio_mnist_online_top.v — UART front-end for folded MNIST online PoC
// -----------------------------------------------------------------------------
// 115200 8N1 LE replies:
//   V              -> A5 03
//   X + 98B        -> X
//   L + label      -> L
//   N + nh + no    -> N
//   R + tmax + mode-> pred, passes, cycles[4]   mode bit0=EE bit1=score
//   T              -> pred, passes, cycles[4]   train one
//   Z              -> Z                         clear telemetry
//   P              -> samples_trained[4], update_events[4], busy
//   I              -> infer_correct[2], infer_total[2]
//   C + idx        -> class_correct[2]  (idx 0..9)
//   S + idx        -> class_seen[2]
//   H + idx        -> ee_hist[2]        (idx 0..7)
// =============================================================================
`timescale 1ns/1ps
module papilio_mnist_online_top #(
    parameter integer CLKS_PER_BIT = 278,
    parameter W_HID_FILE = "w_hid.mem",
    parameter W_OUT_FILE = "w_out.mem"
) (
    input  wire clk,
    input  wire rx,
    output wire tx,
    output wire led
);
    localparam integer N_IN = 784;
    localparam integer N_X_BYTES = (N_IN + 7) / 8;
    localparam integer REPLY_PAUSE = CLKS_PER_BIT * 12;

    reg [7:0] por = 8'd0;
    wire rst_n = por[7];
    always @(posedge clk)
        if (!por[7]) por <= por + 8'd1;

    wire [7:0] rx_data;
    wire       rx_valid;
    reg  [7:0] tx_data;
    reg        tx_send;
    wire       tx_busy;

    uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_rx (
        .clk(clk), .rst_n(rst_n), .rx(rx),
        .data(rx_data), .valid(rx_valid));
    uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_tx (
        .clk(clk), .rst_n(rst_n), .data(tx_data), .send(tx_send),
        .tx(tx), .busy(tx_busy));

    reg  [N_IN-1:0] x_bits;
    reg  [3:0]      label;
    reg  [5:0]      t_max;
    reg             ee_en, score_en;
    reg  [3:0]      n_log2_hid, n_log2_out;
    reg             start_infer, start_train, clear_stats;
    reg  [2:0]      tele_kind;
    reg  [3:0]      tele_idx;
    wire [15:0]     tele_data;
    wire            core_busy, core_done;
    wire [3:0]      core_pred;
    wire [5:0]      core_passes;
    wire [31:0]     core_cycles, samples_trained, update_events;

    mnist_online_core #(
        .W_HID_FILE(W_HID_FILE), .W_OUT_FILE(W_OUT_FILE)
    ) u_core (
        .clk(clk), .rst_n(rst_n),
        .start_infer(start_infer), .start_train(start_train),
        .t_max(t_max), .early_exit_en(ee_en),
        .n_log2_hid(n_log2_hid), .n_log2_out(n_log2_out),
        .x_bits(x_bits), .label(label), .score_en(score_en),
        .clear_stats(clear_stats),
        .tele_kind(tele_kind), .tele_idx(tele_idx), .tele_data(tele_data),
        .busy(core_busy), .done(core_done),
        .pred(core_pred), .passes(core_passes), .run_cycles(core_cycles),
        .samples_trained(samples_trained), .update_events(update_events));

    assign led = core_busy;

    localparam P_CMD=4'd0, P_XLOAD=4'd1, P_NARG=4'd2, P_RARG=4'd3,
               P_WAIT=4'd4, P_REPLY=4'd5, P_LARG=4'd6, P_IDX=4'd7,
               P_INF=4'd8;
    reg [3:0] pstate;
    reg [6:0] byte_cnt;
    reg [1:0] arg_cnt;
    reg [71:0] reply;   // up to 9 bytes
    reg [3:0]  reply_len, reply_idx;
    reg [15:0] pause;
    reg [2:0]  pending_kind;
    reg        inf_phase;
    reg        reply_from_tele;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pstate <= P_CMD;
            byte_cnt <= 7'd0;
            arg_cnt <= 2'd0;
            x_bits <= {N_IN{1'b0}};
            label <= 4'd0;
            t_max <= 6'd16;
            ee_en <= 1'b1;
            score_en <= 1'b0;
            n_log2_hid <= 4'd8;
            n_log2_out <= 4'd8;
            start_infer <= 1'b0;
            start_train <= 1'b0;
            clear_stats <= 1'b0;
            tele_kind <= 3'd0;
            tele_idx <= 4'd0;
            tx_send <= 1'b0;
            tx_data <= 8'd0;
            reply <= 72'd0;
            reply_len <= 4'd0;
            reply_idx <= 4'd0;
            pause <= 16'd0;
            pending_kind <= 3'd0;
            inf_phase <= 1'b0;
            reply_from_tele <= 1'b0;
        end else begin
            start_infer <= 1'b0;
            start_train <= 1'b0;
            clear_stats <= 1'b0;
            tx_send <= 1'b0;

            case (pstate)
                P_CMD: if (rx_valid) begin
                    case (rx_data)
                        8'h56: begin
                            reply <= {56'd0, 8'h03, 8'hA5};
                            reply_len <= 4'd2; reply_idx <= 4'd0;
                            reply_from_tele <= 1'b0;
                            pause <= REPLY_PAUSE[15:0]; pstate <= P_REPLY;
                        end
                        8'h58: begin byte_cnt <= 7'd0; pstate <= P_XLOAD; end
                        8'h4C: pstate <= P_LARG;
                        8'h4E: begin arg_cnt <= 2'd0; pstate <= P_NARG; end
                        8'h52: begin arg_cnt <= 2'd0; pstate <= P_RARG; end
                        8'h54: begin start_train <= 1'b1; pstate <= P_WAIT; end
                        8'h5A: begin
                            clear_stats <= 1'b1;
                            reply <= {64'd0, 8'h5A};
                            reply_len <= 4'd1; reply_idx <= 4'd0;
                            reply_from_tele <= 1'b0;
                            pause <= REPLY_PAUSE[15:0]; pstate <= P_REPLY;
                        end
                        8'h50: begin // P
                            reply <= {{7'b0, core_busy},
                                      update_events,
                                      samples_trained};
                            reply_len <= 4'd9; reply_idx <= 4'd0;
                            reply_from_tele <= 1'b0;
                            pause <= REPLY_PAUSE[15:0]; pstate <= P_REPLY;
                        end
                        8'h49: begin // I
                            tele_kind <= 3'd4;
                            tele_idx <= 4'd0;
                            inf_phase <= 1'b0;
                            pstate <= P_INF;
                        end
                        8'h43: begin pending_kind <= 3'd0; pstate <= P_IDX; end
                        8'h53: begin pending_kind <= 3'd1; pstate <= P_IDX; end
                        8'h48: begin pending_kind <= 3'd2; pstate <= P_IDX; end
                        default: ;
                    endcase
                end

                P_XLOAD: if (rx_valid) begin
                    x_bits[{byte_cnt, 3'b000} +: 8] <= rx_data;
                    if (byte_cnt == N_X_BYTES-1) begin
                        reply <= {64'd0, 8'h58};
                        reply_len <= 4'd1; reply_idx <= 4'd0;
                        reply_from_tele <= 1'b0;
                        pause <= REPLY_PAUSE[15:0]; pstate <= P_REPLY;
                    end else
                        byte_cnt <= byte_cnt + 7'd1;
                end

                P_LARG: if (rx_valid) begin
                    label <= rx_data[3:0];
                    reply <= {64'd0, 8'h4C};
                    reply_len <= 4'd1; reply_idx <= 4'd0;
                    reply_from_tele <= 1'b0;
                    pause <= REPLY_PAUSE[15:0]; pstate <= P_REPLY;
                end

                P_NARG: if (rx_valid) begin
                    if (arg_cnt == 2'd0) begin
                        n_log2_hid <= rx_data[3:0];
                        arg_cnt <= 2'd1;
                    end else begin
                        n_log2_out <= rx_data[3:0];
                        reply <= {64'd0, 8'h4E};
                        reply_len <= 4'd1; reply_idx <= 4'd0;
                        reply_from_tele <= 1'b0;
                        pause <= REPLY_PAUSE[15:0]; pstate <= P_REPLY;
                    end
                end

                P_RARG: if (rx_valid) begin
                    if (arg_cnt == 2'd0) begin
                        t_max <= rx_data[5:0];
                        arg_cnt <= 2'd1;
                    end else begin
                        ee_en <= rx_data[0];
                        score_en <= rx_data[1];
                        start_infer <= 1'b1;
                        pstate <= P_WAIT;
                    end
                end

                P_IDX: if (rx_valid) begin
                    tele_kind <= pending_kind;
                    tele_idx <= rx_data[3:0];
                    pause <= REPLY_PAUSE[15:0];
                    reply_len <= 4'd2; reply_idx <= 4'd0;
                    reply_from_tele <= 1'b1;
                    pstate <= P_REPLY;
                end

                P_INF: begin
                    if (!inf_phase) begin
                        reply[15:0] <= tele_data;
                        tele_kind <= 3'd3;
                        inf_phase <= 1'b1;
                        pause <= 16'd2;
                    end else if (pause != 16'd0) begin
                        pause <= pause - 16'd1;
                    end else begin
                        reply[31:16] <= tele_data;
                        reply_len <= 4'd4; reply_idx <= 4'd0;
                        reply_from_tele <= 1'b0;
                        pause <= REPLY_PAUSE[15:0];
                        pstate <= P_REPLY;
                        inf_phase <= 1'b0;
                    end
                end

                P_WAIT: if (core_done) begin
                    reply <= {24'd0, core_cycles, {2'b00, core_passes},
                              {4'b0000, core_pred}};
                    reply_len <= 4'd6; reply_idx <= 4'd0;
                    reply_from_tele <= 1'b0;
                    pause <= REPLY_PAUSE[15:0]; pstate <= P_REPLY;
                end

                P_REPLY: begin
                    if (pause != 16'd0)
                        pause <= pause - 16'd1;
                    else if (!tx_busy && !tx_send) begin
                        if (reply_idx == 4'd0 && reply_from_tele)
                            reply[15:0] <= tele_data;
                        tx_data <= reply[8*reply_idx +: 8];
                        tx_send <= 1'b1;
                        if (reply_idx + 4'd1 >= reply_len)
                            pstate <= P_CMD;
                        else
                            reply_idx <= reply_idx + 4'd1;
                    end
                end
                default: pstate <= P_CMD;
            endcase
        end
    end
endmodule
