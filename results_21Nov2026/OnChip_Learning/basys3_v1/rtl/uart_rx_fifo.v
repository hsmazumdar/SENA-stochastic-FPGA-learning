// uart_rx_fifo.v -- small FWFT byte FIFO so a streamed image is not lost
// while the core is busy. DEPTH must be a power of two.
`timescale 1ns/1ps
module uart_rx_fifo #(
    parameter integer DEPTH = 64,
    parameter integer AW    = 6
) (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       wr_en,
    input  wire [7:0] wr_data,
    input  wire       rd_en,
    output wire [7:0] rd_data,
    output wire       empty,
    output wire       full,
    output reg        overflow
);
    reg [7:0]     mem [0:DEPTH-1];
    reg [AW:0]    wptr, rptr;   // extra bit distinguishes full vs empty
    wire [AW:0]   wptr_n = wptr + {{AW{1'b0}}, 1'b1};
    assign empty = (wptr == rptr);
    assign full  = (wptr_n[AW-1:0] == rptr[AW-1:0]) && (wptr_n[AW] != rptr[AW]);
    assign rd_data = mem[rptr[AW-1:0]];
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wptr <= 0; rptr <= 0; overflow <= 1'b0;
        end else begin
            if (wr_en) begin
                if (full) overflow <= 1'b1;
                else begin
                    mem[wptr[AW-1:0]] <= wr_data;
                    wptr <= wptr_n;
                end
            end
            if (rd_en && !empty) rptr <= rptr + {{AW{1'b0}}, 1'b1};
        end
    end
endmodule
