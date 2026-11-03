// =============================================================================
// uart_rx.v — minimal 8N1 UART receiver (Verilog-2001, ISE 14.7 compatible)
// Samples mid-bit. CLKS_PER_BIT = round(F_CLK / BAUD), e.g. 32e6/115200 = 278.
// =============================================================================
`timescale 1ns/1ps
module uart_rx #(
    parameter integer CLKS_PER_BIT = 278
) (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       rx,
    output reg  [7:0] data,
    output reg        valid          // 1-cycle pulse when data is ready
);
    localparam S_IDLE  = 3'd0,
               S_START = 3'd1,
               S_DATA  = 3'd2,
               S_STOP  = 3'd3;

    reg [2:0]  state;
    reg [15:0] cnt;
    reg [2:0]  bit_idx;
    reg        rx_d, rx_q;           // 2-FF synchroniser

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rx_d <= 1'b1;
            rx_q <= 1'b1;
        end else begin
            rx_d <= rx;
            rx_q <= rx_d;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state   <= S_IDLE;
            cnt     <= 16'd0;
            bit_idx <= 3'd0;
            data    <= 8'd0;
            valid   <= 1'b0;
        end else begin
            valid <= 1'b0;
            case (state)
                S_IDLE: begin
                    cnt     <= 16'd0;
                    bit_idx <= 3'd0;
                    if (rx_q == 1'b0)
                        state <= S_START;
                end
                S_START: begin
                    if (cnt == (CLKS_PER_BIT/2)) begin
                        if (rx_q == 1'b0) begin
                            cnt   <= 16'd0;
                            state <= S_DATA;
                        end else
                            state <= S_IDLE;   // glitch
                    end else
                        cnt <= cnt + 16'd1;
                end
                S_DATA: begin
                    if (cnt == (CLKS_PER_BIT-1)) begin
                        cnt           <= 16'd0;
                        data[bit_idx] <= rx_q;
                        if (bit_idx == 3'd7) begin
                            bit_idx <= 3'd0;
                            state   <= S_STOP;
                        end else
                            bit_idx <= bit_idx + 3'd1;
                    end else
                        cnt <= cnt + 16'd1;
                end
                S_STOP: begin
                    if (cnt == (CLKS_PER_BIT-1)) begin
                        cnt   <= 16'd0;
                        valid <= 1'b1;
                        state <= S_IDLE;
                    end else
                        cnt <= cnt + 16'd1;
                end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
