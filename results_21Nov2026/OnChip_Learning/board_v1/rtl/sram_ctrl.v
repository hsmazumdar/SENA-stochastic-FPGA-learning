// sram_ctrl.v -- IS61WV102416 (DE2-115, 1M x 16, 10 ns) controller at 25 MHz (40 ns cycle).
// write: 3 cycles (addr+data, WE_N low for one full 40 ns cycle, data held 1 cycle after WE_N rises)
// read : 2 cycles (addr + OE_N low, data captured 40 ns later at the end of the 2nd cycle)
// CE_N, UB_N, LB_N tied low.  DQ is only driven during writes; OE_N high during writes.
`timescale 1ns/1ps
module sram_ctrl (
    input  wire        clk, input wire rst_n,
    input  wire        req, input wire we, input wire [19:0] addr, input wire [15:0] wdata,
    output reg         done, output reg [15:0] rdata, output wire ready,
    output reg  [19:0] SRAM_ADDR, inout wire [15:0] SRAM_DQ,
    output wire SRAM_CE_N, output reg SRAM_OE_N, output reg SRAM_WE_N,
    output wire SRAM_UB_N, output wire SRAM_LB_N);
    reg [1:0] st; reg dq_oe; reg [15:0] dq_out;
    assign SRAM_DQ = dq_oe ? dq_out : 16'hzzzz;
    assign SRAM_CE_N = 1'b0; assign SRAM_UB_N = 1'b0; assign SRAM_LB_N = 1'b0;
    assign ready = (st == 2'd0) && !req;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin st <= 0; dq_oe <= 0; dq_out <= 0; SRAM_ADDR <= 0; SRAM_OE_N <= 1; SRAM_WE_N <= 1; done <= 0; rdata <= 0; end
        else begin
            done <= 1'b0;
            case (st)
            2'd0: if (req) begin
                      SRAM_ADDR <= addr;
                      if (we) begin dq_out <= wdata; dq_oe <= 1'b1; SRAM_WE_N <= 1'b0; SRAM_OE_N <= 1'b1; st <= 2'd1; end
                      else begin SRAM_OE_N <= 1'b0; st <= 2'd3; end
                  end
            2'd1: begin SRAM_WE_N <= 1'b1; st <= 2'd2; end                 // write: WE_N rises, data held
            2'd2: begin dq_oe <= 1'b0; done <= 1'b1; st <= 2'd0; end
            2'd3: begin rdata <= SRAM_DQ; SRAM_OE_N <= 1'b1; done <= 1'b1; st <= 2'd0; end
            endcase
        end
    end
endmodule
