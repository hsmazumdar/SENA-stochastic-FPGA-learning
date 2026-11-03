// =============================================================================
// papilio_logic_learn_top.v — Papilio Pro AND/OR/XOR on-chip learning PoC
// -----------------------------------------------------------------------------
// Board: Papilio Pro (Spartan-6 LX9, 32 MHz, FT2232 UART)
// LED: flips every epoch using even-epoch polarity: led = ~epoch_id[0]
//      (ON on even epochs 0,2,4,... after each completed epoch_id increment)
// UART 115200 8N1:
//   'V' -> A5 02
//   'G' -> start continuous train
//   'S' -> stop train
//   'E' -> one eval epoch (no weight update)
//   'P' -> poll: epoch_id[15:0] LE, correct[7:0], mean_passes[7:0],
//                updates[15:0] LE, flags[7:0] (bit0=busy, bit1=train_en)
// =============================================================================
`timescale 1ns/1ps
module papilio_logic_learn_top #(
    parameter integer CLKS_PER_BIT = 278   // 32e6 / 115200
) (
    input  wire clk,
    input  wire rx,
    output wire tx,
    output wire led
);
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

    reg  enable_train;
    reg  eval_pulse;
    wire busy, epoch_done, eval_done;
    wire [15:0] epoch_id;
    wire [6:0]  correct_cnt;
    wire [7:0]  last_mean_passes;
    wire [15:0] update_count;

    logic663_train_core u_core (
        .clk(clk), .rst_n(rst_n),
        .enable_train(enable_train),
        .eval_pulse(eval_pulse),
        .busy(busy),
        .epoch_done(epoch_done),
        .eval_done(eval_done),
        .epoch_id(epoch_id),
        .correct_cnt(correct_cnt),
        .last_mean_passes(last_mean_passes),
        .update_count(update_count)
    );

    // Flip LED every epoch; ON when epoch_id is even (bit0 == 0).
    assign led = ~epoch_id[0];

    localparam P_CMD = 2'd0, P_REPLY = 2'd1;
    reg [1:0] pstate;
    reg [63:0] reply;
    reg [2:0]  reply_len, reply_idx;
    localparam integer REPLY_PAUSE = CLKS_PER_BIT * 12;
    reg [15:0] pause;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            enable_train <= 1'b1;   // auto-start learning on power-up
            eval_pulse <= 1'b0;
            tx_send <= 1'b0;
            tx_data <= 8'd0;
            pstate <= P_CMD;
            reply <= 0;
            reply_len <= 0;
            reply_idx <= 0;
            pause <= 0;
        end else begin
            eval_pulse <= 1'b0;
            tx_send <= 1'b0;

            case (pstate)
                P_CMD: if (rx_valid) begin
                    case (rx_data)
                        8'h56: begin // 'V'
                            reply <= {48'd0, 8'h02, 8'hA5};
                            reply_len <= 3'd2;
                            reply_idx <= 3'd0;
                            pause <= REPLY_PAUSE[15:0];
                            pstate <= P_REPLY;
                        end
                        8'h47: enable_train <= 1'b1; // 'G'
                        8'h53: enable_train <= 1'b0; // 'S'
                        8'h45: begin                 // 'E'
                            enable_train <= 1'b0;
                            eval_pulse <= 1'b1;
                        end
                        8'h50: begin // 'P' poll
                            reply <= {
                                8'd0,
                                {7'd0, enable_train} | {6'd0, busy, 1'b0},
                                update_count,
                                last_mean_passes,
                                {1'b0, correct_cnt},
                                epoch_id
                            };
                            // pack LE bytes manually below
                            reply[7:0]   <= epoch_id[7:0];
                            reply[15:8]  <= epoch_id[15:8];
                            reply[23:16] <= {1'b0, correct_cnt};
                            reply[31:24] <= last_mean_passes;
                            reply[39:32] <= update_count[7:0];
                            reply[47:40] <= update_count[15:8];
                            reply[55:48] <= {6'd0, busy, enable_train};
                            reply_len <= 3'd7;
                            reply_idx <= 3'd0;
                            pause <= REPLY_PAUSE[15:0];
                            pstate <= P_REPLY;
                        end
                        default: ;
                    endcase
                end

                P_REPLY: begin
                    if (pause != 0) pause <= pause - 1'b1;
                    else if (!tx_busy && !tx_send) begin
                        if (reply_idx < reply_len) begin
                            tx_data <= reply[8*reply_idx +: 8];
                            tx_send <= 1'b1;
                            reply_idx <= reply_idx + 1'b1;
                        end else pstate <= P_CMD;
                    end
                end

                default: pstate <= P_CMD;
            endcase
        end
    end
endmodule
