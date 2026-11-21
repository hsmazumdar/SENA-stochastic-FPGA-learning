// tb_dwn_core.v - AUTO-GENERATED bit-exact check: RTL vs Python integer model
`timescale 1ns/1ps
module tb_dwn_core;
    `include "dwn_params.vh"
    reg clk = 0, rst_n = 0, in_valid = 0;
    reg  [DWN_N_IN-1:0] x;
    wire out_valid;
    wire [DWN_IW-1:0] pred;
    wire [DWN_K*DWN_CW-1:0] counts;
    reg [DWN_N_IN-1:0]      xm [0:DWN_N_TEST-1];
    reg [7:0]               em [0:DWN_N_TEST-1];
    reg [DWN_K*DWN_CW-1:0]  cm [0:DWN_N_TEST-1];
    integer i, k, pm, cmis, correct, t_first_in, t_first_out, cyc;
    dwn_core dut (.clk(clk), .rst_n(rst_n), .in_valid(in_valid), .x(x),
                  .out_valid(out_valid), .pred(pred), .counts(counts));
    always #5 clk = ~clk;
    always @(posedge clk) cyc <= cyc + 1;
    initial begin
        $readmemh("test_x.hex", xm);
        $readmemh("test_exp.hex", em);
        $readmemh("test_counts.hex", cm);
        cyc = 0; k = 0; pm = 0; cmis = 0; correct = 0; t_first_in = -1; t_first_out = -1;
        x = 0;
        repeat (3) @(posedge clk);
        rst_n = 1;
        @(posedge clk);
        for (i = 0; i < DWN_N_TEST; i = i + 1) begin
            @(negedge clk); in_valid = 1; x = xm[i];
            if (t_first_in < 0) t_first_in = cyc;
        end
        @(negedge clk); in_valid = 0;
        wait (k == DWN_N_TEST);
        $display("RESULT n=%0d pred_mismatch=%0d count_mismatch=%0d rtl_correct=%0d latency=%0d expected_latency=%0d",
                 DWN_N_TEST, pm, cmis, correct, t_first_out - t_first_in, DWN_LATENCY);
        if (pm == 0 && cmis == 0 && (t_first_out - t_first_in) == DWN_LATENCY)
            $display("BIT-EXACT PASS");
        else
            $display("BIT-EXACT FAIL");
        $finish;
    end
    always @(posedge clk) if (out_valid) begin
        if (t_first_out < 0) t_first_out = cyc;
        if (pred !== em[k][3:0]) pm = pm + 1;
        if (counts !== cm[k]) cmis = cmis + 1;
        if (pred == em[k][7:4]) correct = correct + 1;
        k = k + 1;
    end
endmodule
