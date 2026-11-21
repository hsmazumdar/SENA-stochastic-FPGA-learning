// uart_tx_s.v -- 8N1 UART transmitter. `start` is sampled when !busy; busy rises at that edge.
`timescale 1ns/1ps
module uart_tx_s #(parameter integer DIV = 217) (
    input wire clk, input wire rst_n, input wire start, input wire [7:0] data,
    output reg tx, output reg busy);
    reg [15:0] c; reg [3:0] b; reg [9:0] sh;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin tx <= 1; busy <= 0; c <= 0; b <= 0; sh <= 10'h3FF; end
        else if (!busy) begin
            if (start) begin sh <= {1'b1, data, 1'b0}; busy <= 1; c <= DIV - 1; b <= 0; tx <= 1'b0; end
        end else begin
            if (c != 0) c <= c - 1'b1;
            else begin
                c <= DIV - 1;
                if (b == 4'd9) begin busy <= 1'b0; tx <= 1'b1; end
                else begin b <= b + 1'b1; tx <= sh[b + 1]; end
            end
        end
    end
endmodule
