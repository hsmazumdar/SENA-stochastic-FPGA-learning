// =============================================================================
// comfort_sena_top.v — UART front-end for Analog Comfort PoC (Papilio Pro)
// -----------------------------------------------------------------------------
// 115200 8N1 LE:
//   V              -> A5 04
//   B + member     -> B          select ensemble member 0..2
//   S + t_u8 + rh  -> S          set sensors (normalized 0..255)
//   N + n_log2     -> N
//   R + T_lo+T_hi  -> ones[2], passes[2], cycles[4]
//                     p_comfort = ones/passes ; resolution = 1/passes
// =============================================================================
`timescale 1ns/1ps
module comfort_sena_top #(
    parameter integer CLKS_PER_BIT = 278
) (
    input  wire clk,
    input  wire rx,
    output wire tx,
    output wire led
);
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

    reg [7:0]  t_u8, rh_u8;
    reg [1:0]  member;
    reg [3:0]  n_log2;
    reg [15:0] t_passes;
    reg        start;
    wire       busy, done;
    wire [15:0] ones_count, passes_done;
    wire [31:0] run_cycles;

    comfort_core u_core (
        .clk(clk), .rst_n(rst_n), .start(start),
        .member(member), .t_passes(t_passes), .n_log2(n_log2),
        .t_u8(t_u8), .rh_u8(rh_u8),
        .busy(busy), .done(done),
        .ones_count(ones_count), .passes_done(passes_done),
        .run_cycles(run_cycles));

    assign led = busy;

    localparam P_CMD=3'd0, P_B=3'd1, P_S=3'd2, P_N=3'd3, P_R=3'd4,
               P_WAIT=3'd5, P_REPLY=3'd6;
    reg [2:0] pstate;
    reg [1:0] arg;
    reg [63:0] reply;
    reg [3:0] reply_len, reply_idx;
    reg [15:0] pause;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pstate <= P_CMD;
            arg <= 0;
            t_u8 <= 8'd128; rh_u8 <= 8'd128;
            member <= 0; n_log2 <= 4'd6; t_passes <= 16'd100;
            start <= 0; tx_send <= 0; tx_data <= 0;
            reply <= 0; reply_len <= 0; reply_idx <= 0; pause <= 0;
        end else begin
            start <= 0; tx_send <= 0;
            case (pstate)
                P_CMD: if (rx_valid) begin
                    case (rx_data)
                        8'h56: begin
                            reply <= {48'd0, 8'h04, 8'hA5};
                            reply_len <= 3'd2; reply_idx <= 3'd0;
                            pause <= REPLY_PAUSE[15:0]; pstate <= P_REPLY;
                        end
                        8'h42: pstate <= P_B; // B
                        8'h53: begin arg <= 0; pstate <= P_S; end
                        8'h4E: pstate <= P_N;
                        8'h52: begin arg <= 0; pstate <= P_R; end
                        default: ;
                    endcase
                end
                P_B: if (rx_valid) begin
                    member <= rx_data[1:0];
                    reply <= {56'd0, 8'h42};
                    reply_len <= 3'd1; reply_idx <= 3'd0;
                    pause <= REPLY_PAUSE[15:0]; pstate <= P_REPLY;
                end
                P_S: if (rx_valid) begin
                    if (arg == 0) begin
                        t_u8 <= rx_data; arg <= 1;
                    end else begin
                        rh_u8 <= rx_data;
                        reply <= {56'd0, 8'h53};
                        reply_len <= 3'd1; reply_idx <= 3'd0;
                        pause <= REPLY_PAUSE[15:0]; pstate <= P_REPLY;
                    end
                end
                P_N: if (rx_valid) begin
                    n_log2 <= rx_data[3:0];
                    reply <= {56'd0, 8'h4E};
                    reply_len <= 3'd1; reply_idx <= 3'd0;
                    pause <= REPLY_PAUSE[15:0]; pstate <= P_REPLY;
                end
                P_R: if (rx_valid) begin
                    if (arg == 0) begin
                        t_passes[7:0] <= rx_data; arg <= 1;
                    end else begin
                        t_passes[15:8] <= rx_data;
                        start <= 1;
                        pstate <= P_WAIT;
                    end
                end
                P_WAIT: if (done) begin
                    reply <= {run_cycles, passes_done, ones_count};
                    reply_len <= 4'd8; reply_idx <= 4'd0;
                    pause <= REPLY_PAUSE[15:0]; pstate <= P_REPLY;
                end
                P_REPLY: begin
                    if (pause != 0) pause <= pause - 1;
                    else if (!tx_busy && !tx_send) begin
                        tx_data <= reply[8*reply_idx +: 8];
                        tx_send <= 1;
                        if (reply_idx + 1 >= reply_len) pstate <= P_CMD;
                        else reply_idx <= reply_idx + 1;
                    end
                end
                default: pstate <= P_CMD;
            endcase
        end
    end
endmodule
