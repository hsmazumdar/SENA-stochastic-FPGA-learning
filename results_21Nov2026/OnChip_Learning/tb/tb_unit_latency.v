// tb_unit_latency.v -- direct guard against the LX9 off-by-one class.
// W1 row i, lane j holds a unique value (i*8 + j + 1); bias row holds 90+j.
// For one-hot, all-zero and all-one inputs the hidden accumulators after the
// sweep must equal exactly sum(selected rows) + bias, for every lane.
// Also prints the first LFSR draw of lanes 0..3 for comparison with the twin.
`timescale 1ns/1ps
module tb_unit_latency;
    localparam integer N_IN = 8, N_HID = 4, N_OUT = 10, W = 8;
    reg clk = 0; always #5 clk = ~clk;
    reg rst_n = 0, init = 0, si = 0;
    reg [N_IN-1:0] x = 0;
    wire busy, done; wire [3:0] pred; wire [31:0] nupd;
    wire [N_HID*W-1:0] hr1; wire [N_OUT*W-1:0] hr2;
    sena_learn_core #(.N_IN(N_IN), .N_HID(N_HID), .N_OUT(N_OUT), .W(W), .SH(2),
                      .W1_INIT("unit_w1.hex"), .W2_INIT("unit_w2.hex")) dut (
        .clk(clk), .rst_n(rst_n), .init(init), .seed_in(32'h1D872B41),
        .start_train(1'b0), .start_infer(si), .x_bits(x), .label(4'd0), .t_passes(6'd1),
        .busy(busy), .done(done), .pred(pred), .n_w1_updates(nupd),
        .host_we1(1'b0), .host_we2(1'b0), .host_addr1('0), .host_addr2('0),
        .host_wdata1('0), .host_wdata2('0), .host_rdata1(hr1), .host_rdata2(hr2));
    integer t, i, j, exp, errs, first;
    initial begin
        errs = 0; first = 1;
        #23 rst_n = 1;
        @(negedge clk) init = 1; @(negedge clk) init = 0;
        for (t = 0; t < N_IN + 2; t = t + 1) begin
            x = (t < N_IN) ? (8'd1 << t) : ((t == N_IN) ? 8'h00 : 8'hFF);
            @(negedge clk) si = 1; @(negedge clk) si = 0;
            wait (dut.state == 4'd2);   // S_HFIRE: sweep finished
            #1;
            for (j = 0; j < N_HID; j = j + 1) begin
                exp = 90 + j;
                for (i = 0; i < N_IN; i = i + 1) if (x[i]) exp = exp + i*8 + j + 1;
                if (dut.acc_h[j] !== exp) begin
                    errs = errs + 1;
                    $display("MISMATCH x=%b lane %0d: acc=%0d expected=%0d", x, j, dut.acc_h[j], exp);
                end
            end
            if (first) begin
                wait (dut.hstep == 6'd32); #1;
                $display("LFSR_FIRST_DRAW %08x %08x %08x %08x", dut.lh[0], dut.lh[1], dut.lh[2], dut.lh[3]);
                first = 0;
            end
            wait (done); @(negedge clk);
        end
        if (errs == 0) $display("UNIT_LATENCY PASS (%0d input patterns x %0d lanes)", N_IN + 2, N_HID);
        else           $display("UNIT_LATENCY FAIL errors=%0d", errs);
        $finish;
    end
endmodule
