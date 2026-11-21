// =============================================================================
// sdp_ram.v -- simple-dual-port RAM (1 write port, 1 read port) for Cyclone IV M9K.
// READ LATENCY = 1 CLOCK, REGISTERED OUTPUT:
//     the address presented on `raddr` before edge e is returned on `rdata` after edge e.
// The consumer (sena_learn_core) registers its tag (row index, gating bit) at the SAME edge
// that captures the read, so tag and data are always aligned.  `raddr_q` is the address that
// produced the current `rdata`; the core checks tag == raddr_q in simulation.
// Same-address read-during-write never happens by construction (checked in simulation), so
// the RAM is declared no_rw_check (no bypass logic needed).
// =============================================================================
`timescale 1ns/1ps
module sdp_ram #(
    parameter integer DW    = 8,
    parameter integer DEPTH = 16,
    parameter integer AW    = 4,
    parameter         INIT  = ""
) (
    input  wire          clk,
    input  wire          we,
    input  wire [AW-1:0] waddr,
    input  wire [DW-1:0] wdata,
    input  wire [AW-1:0] raddr,
    output reg  [DW-1:0] rdata,
    output reg  [AW-1:0] raddr_q
);
    (* ramstyle = "M9K, no_rw_check" *) reg [DW-1:0] mem [0:DEPTH-1];
    initial begin
        if (INIT != "") $readmemh(INIT, mem);
    end
    always @(posedge clk) begin
        if (we) mem[waddr] <= wdata;
        rdata   <= mem[raddr];
        raddr_q <= raddr;
    end
`ifndef SYNTHESIS
    always @(posedge clk) begin
        if (we && (waddr == raddr)) begin
            $display("SDP_RAM ERROR %m: same-address read-during-write at addr %0d, t=%0t", waddr, $time);
            $fatal(1);
        end
        if (we && (waddr >= DEPTH)) begin
            $display("SDP_RAM ERROR %m: write address %0d out of range", waddr);
            $fatal(1);
        end
    end
`endif
endmodule
