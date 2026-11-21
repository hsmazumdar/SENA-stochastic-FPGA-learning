// uart_rx_s.v -- 8N1 UART receiver, DIV clocks per bit, 2-FF synchronizer, mid-bit sampling.
`timescale 1ns/1ps
module uart_rx_s #(parameter integer DIV = 217) (
    input  wire clk, input wire rst_n, input wire rx,
    output reg  valid, output reg [7:0] data, output reg frame_err);
    reg r1, r2; reg [15:0] c; reg [3:0] b; reg [1:0] st; reg [7:0] sh;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin r1 <= 1; r2 <= 1; c <= 0; b <= 0; st <= 0; sh <= 0; valid <= 0; data <= 0; frame_err <= 0; end
        else begin
            r1 <= rx; r2 <= r1; valid <= 1'b0;
            case (st)
            2'd0: if (!r2) begin c <= DIV/2; st <= 2'd1; end
            2'd1: if (c != 0) c <= c - 1'b1;
                  else if (!r2) begin c <= DIV - 1; b <= 0; st <= 2'd2; end
                  else st <= 2'd0;                                   // glitch
            2'd2: if (c != 0) c <= c - 1'b1;
                  else begin
                      sh <= {r2, sh[7:1]}; c <= DIV - 1;
                      if (b == 4'd7) st <= 2'd3; b <= b + 1'b1;
                  end
            2'd3: if (c != 0) c <= c - 1'b1;
                  else begin
                      if (r2) begin data <= sh; valid <= 1'b1; end else frame_err <= 1'b1;
                      st <= 2'd0;
                  end
            endcase
        end
    end
endmodule
