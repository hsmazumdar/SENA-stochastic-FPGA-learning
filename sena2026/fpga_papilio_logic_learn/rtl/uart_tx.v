// =============================================================================
// uart_tx.v — minimal 8N1 UART transmitter (Verilog-2001, ISE 14.7 compatible)
// =============================================================================
`timescale 1ns/1ps
module uart_tx #(
    parameter integer CLKS_PER_BIT = 278
) (
    input  wire       clk,
    input  wire       rst_n,
    input  wire [7:0] data,
    input  wire       send,           // pulse; ignored while busy
    output reg        tx,
    output reg        busy
);
    localparam S_IDLE  = 2'd0,
               S_START = 2'd1,
               S_DATA  = 2'd2,
               S_STOP  = 2'd3;

    reg [1:0]  state;
    reg [15:0] cnt;
    reg [2:0]  bit_idx;
    reg [7:0]  sh;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state   <= S_IDLE;
            cnt     <= 16'd0;
            bit_idx <= 3'd0;
            sh      <= 8'd0;
            tx      <= 1'b1;
            busy    <= 1'b0;
        end else begin
            case (state)
                S_IDLE: begin
                    tx   <= 1'b1;
                    busy <= 1'b0;
                    cnt  <= 16'd0;
                    if (send) begin
                        sh    <= data;
                        busy  <= 1'b1;
                        state <= S_START;
                    end
                end
                S_START: begin
                    tx <= 1'b0;
                    if (cnt == (CLKS_PER_BIT-1)) begin
                        cnt     <= 16'd0;
                        bit_idx <= 3'd0;
                        state   <= S_DATA;
                    end else
                        cnt <= cnt + 16'd1;
                end
                S_DATA: begin
                    tx <= sh[bit_idx];
                    if (cnt == (CLKS_PER_BIT-1)) begin
                        cnt <= 16'd0;
                        if (bit_idx == 3'd7)
                            state <= S_STOP;
                        else
                            bit_idx <= bit_idx + 3'd1;
                    end else
                        cnt <= cnt + 16'd1;
                end
                S_STOP: begin
                    tx <= 1'b1;
                    if (cnt == (CLKS_PER_BIT-1)) begin
                        cnt   <= 16'd0;
                        state <= S_IDLE;
                        busy  <= 1'b0;
                    end else
                        cnt <= cnt + 16'd1;
                end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
