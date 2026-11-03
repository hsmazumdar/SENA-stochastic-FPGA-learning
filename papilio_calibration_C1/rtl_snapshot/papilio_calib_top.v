// =============================================================================
// papilio_calib_top.v — UART front-end for the SENA calibration slice
// -----------------------------------------------------------------------------
// Board: Papilio Pro (Spartan-6 LX9, 32 MHz oscillator, FT2232 USB-UART)
//
// UART protocol, 115200 8N1 (all replies little-endian):
//   'V' (0x56)                      -> 0xA5, 0x01            (id, version)
//   'X' (0x58) + 98 bytes           -> 0x58                  (784 input bits,
//                                        byte 0 bit 0 = pixel 0, LSB first)
//   'N' (0x4E) + n_hid + n_out      -> 0x4E                  (LFSR range log2)
//   'R' (0x52) + t_max + mode       -> pred, passes, cycles[4]
//                                        mode bit0 = early-exit enable
// =============================================================================
`timescale 1ns/1ps
module papilio_calib_top #(
    parameter integer CLKS_PER_BIT = 278,   // 32 MHz / 115200
    parameter W_HID_FILE = "w_hid.mem",
    parameter W_OUT_FILE = "w_out.mem"
) (
    input  wire clk,
    input  wire rx,
    output wire tx,
    output wire led
);
    localparam integer N_IN = 784;
    localparam integer N_X_BYTES = (N_IN + 7) / 8;   // 98

    // power-on reset (no button needed on Papilio)
    reg [7:0] por = 8'd0;
    wire rst_n = por[7];
    always @(posedge clk)
        if (!por[7]) por <= por + 8'd1;

    // ------------------------------------------------------------- UART
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

    // ------------------------------------------------------------- core
    reg  [N_IN-1:0] x_bits;
    reg  [5:0] t_max;
    reg        ee_en;
    reg  [3:0] n_log2_hid, n_log2_out;
    reg        core_start;
    wire       core_busy, core_done;
    wire [3:0] core_pred;
    wire [5:0] core_passes;
    wire [31:0] core_cycles;

    calib_core #(
        .W_HID_FILE(W_HID_FILE), .W_OUT_FILE(W_OUT_FILE)
    ) u_core (
        .clk(clk), .rst_n(rst_n),
        .start(core_start), .t_max(t_max), .early_exit_en(ee_en),
        .n_log2_hid(n_log2_hid), .n_log2_out(n_log2_out),
        .x_bits(x_bits),
        .busy(core_busy), .done(core_done),
        .pred(core_pred), .passes(core_passes), .run_cycles(core_cycles));

    assign led = core_busy;

    // ------------------------------------------------------------- protocol
    localparam P_CMD = 3'd0, P_XLOAD = 3'd1, P_NARG = 3'd2,
               P_RARG = 3'd3, P_WAIT = 3'd4, P_REPLY = 3'd5;
    reg [2:0] pstate;
    reg [6:0] byte_cnt;
    reg [1:0] arg_cnt;
    reg [47:0] reply;       // up to 6 bytes, LSB sent first
    reg [2:0]  reply_len;
    reg [2:0]  reply_idx;
    // settle time before the first reply byte so the reply never overlaps
    // the tail of the host's stop bit (~1.2 byte-times)
    localparam integer REPLY_PAUSE = CLKS_PER_BIT * 12;
    reg [15:0] pause;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pstate <= P_CMD;
            byte_cnt <= 7'd0;
            arg_cnt <= 2'd0;
            x_bits <= {N_IN{1'b0}};
            t_max <= 6'd40;
            ee_en <= 1'b1;
            n_log2_hid <= 4'd8;
            n_log2_out <= 4'd8;
            core_start <= 1'b0;
            tx_send <= 1'b0;
            tx_data <= 8'd0;
            reply <= 48'd0;
            reply_len <= 3'd0;
            reply_idx <= 3'd0;
            pause <= 16'd0;
        end else begin
            core_start <= 1'b0;
            tx_send <= 1'b0;

            case (pstate)
                P_CMD: if (rx_valid) begin
                    case (rx_data)
                        8'h56: begin                       // 'V'
                            reply <= {32'd0, 8'h01, 8'hA5};
                            reply_len <= 3'd2;
                            reply_idx <= 3'd0;
                            pause <= REPLY_PAUSE[15:0];
                            pstate <= P_REPLY;
                        end
                        8'h58: begin                       // 'X'
                            byte_cnt <= 7'd0;
                            pstate <= P_XLOAD;
                        end
                        8'h4E: begin                       // 'N'
                            arg_cnt <= 2'd0;
                            pstate <= P_NARG;
                        end
                        8'h52: begin                       // 'R'
                            arg_cnt <= 2'd0;
                            pstate <= P_RARG;
                        end
                        default: ;                         // ignore
                    endcase
                end

                P_XLOAD: if (rx_valid) begin
                    // byte b covers bits [8b+7 : 8b]; last byte partial
                    x_bits[{byte_cnt, 3'b000} +: 8] <= rx_data;
                    if (byte_cnt == N_X_BYTES-1) begin
                        reply <= {40'd0, 8'h58};
                        reply_len <= 3'd1;
                        reply_idx <= 3'd0;
                        pause <= REPLY_PAUSE[15:0];
                        pstate <= P_REPLY;
                    end else
                        byte_cnt <= byte_cnt + 7'd1;
                end

                P_NARG: if (rx_valid) begin
                    if (arg_cnt == 2'd0) begin
                        n_log2_hid <= rx_data[3:0];
                        arg_cnt <= 2'd1;
                    end else begin
                        n_log2_out <= rx_data[3:0];
                        reply <= {40'd0, 8'h4E};
                        reply_len <= 3'd1;
                        reply_idx <= 3'd0;
                        pause <= REPLY_PAUSE[15:0];
                        pstate <= P_REPLY;
                    end
                end

                P_RARG: if (rx_valid) begin
                    if (arg_cnt == 2'd0) begin
                        t_max <= rx_data[5:0];
                        arg_cnt <= 2'd1;
                    end else begin
                        ee_en <= rx_data[0];
                        core_start <= 1'b1;
                        pstate <= P_WAIT;
                    end
                end

                P_WAIT: if (core_done) begin
                    reply <= {core_cycles,
                              {2'b00, core_passes},
                              {4'b0000, core_pred}};
                    reply_len <= 3'd6;
                    reply_idx <= 3'd0;
                    pause <= REPLY_PAUSE[15:0];
                    pstate <= P_REPLY;
                end

                P_REPLY: begin
                    if (pause != 16'd0)
                        pause <= pause - 16'd1;
                    else if (!tx_busy && !tx_send) begin
                        if (reply_idx == reply_len) begin
                            pstate <= P_CMD;
                        end else begin
                            tx_data <= reply[{reply_idx, 3'b000} +: 8];
                            tx_send <= 1'b1;
                            reply_idx <= reply_idx + 3'd1;
                        end
                    end
                end
                default: pstate <= P_CMD;
            endcase
        end
    end
endmodule
