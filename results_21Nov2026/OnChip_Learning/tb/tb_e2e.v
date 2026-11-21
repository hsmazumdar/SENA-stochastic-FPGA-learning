// tb_e2e.v -- end-to-end bit-exact test of sena_learn_core against twin vectors.
// Vector set paths/sizes come from tb_params.vh (written by tb/run_e2e.py).
`timescale 1ns/1ps
`include "tb_params.vh"
module tb_e2e;
    localparam integer N_IN = `N_IN, N_HID = `N_HID, N_OUT = 10, W = 8;
    reg clk = 1'b0; always #5 clk = ~clk;
    reg rst_n = 1'b0, init = 1'b0, st = 1'b0, si = 1'b0;
    reg [N_IN-1:0] x = 0; reg [3:0] lab = 0;
    reg [N_IN-1:0] xmem [0:`NX-1];
    reg [3:0]      labmem [0:`NX-1];
    reg [15:0]     cmd [0:`NCMD-1];
    wire busy, done; wire [3:0] pred; wire [31:0] nupd;
    wire [N_HID*W-1:0] hr1; wire [N_OUT*W-1:0] hr2;

    sena_learn_core #(.N_IN(N_IN), .N_HID(N_HID), .N_OUT(N_OUT), .W(W), .SH(`SH),
                      .W1_INIT(`W1_INIT), .W2_INIT(`W2_INIT)) dut (
        .clk(clk), .rst_n(rst_n), .init(init), .seed_in(`SEED_IN),
        .start_train(st), .start_infer(si), .x_bits(x), .label(lab), .t_passes(6'd`T_PASSES),
        .busy(busy), .done(done), .pred(pred), .n_w1_updates(nupd),
        .host_we1(1'b0), .host_we2(1'b0), .host_addr1('0), .host_addr2('0),
        .host_wdata1({N_HID*W{1'b0}}), .host_wdata2({N_OUT*W{1'b0}}),
        .host_rdata1(hr1), .host_rdata2(hr2));

    integer c, fp, t0, cyc;
    always @(posedge clk) cyc <= cyc + 1;
    initial begin
        cyc = 0;
        $readmemh(`X_FILE, xmem);
        $readmemh(`LAB_FILE, labmem);
        $readmemh(`CMD_FILE, cmd);
        fp = $fopen(`OUT_PRED, "w");
        #23 rst_n = 1'b1;
        @(negedge clk) init = 1'b1;
        @(negedge clk) init = 1'b0;
        t0 = cyc;
        for (c = 0; c < `NCMD; c = c + 1) begin
            x   = xmem[cmd[c][14:0]];
            lab = labmem[cmd[c][14:0]];
            @(negedge clk);
            if (cmd[c][15]) si = 1'b1; else st = 1'b1;
            @(negedge clk); si = 1'b0; st = 1'b0;
            @(posedge clk); #1;
            while (!done) begin @(posedge clk); #1; end
            if (cmd[c][15]) $fdisplay(fp, "%0d %0d", cmd[c][14:0], pred);
            if ((c % 1000) == 999) $display("  cmd %0d / %0d  cycles=%0d", c + 1, `NCMD, cyc - t0);
        end
        $fclose(fp);
        $writememh(`OUT_W1, dut.u_w1.mem);
        $writememh(`OUT_W2, dut.u_w2.mem);
        fp = $fopen(`OUT_STATE, "w");
        $fdisplay(fp, "nb_fx %0d", dut.nb_fx);
        $fdisplay(fp, "n_w1_updates %0d", nupd);
        $fdisplay(fp, "cycles %0d", cyc - t0);
        $fclose(fp);
        $display("TB DONE cmds=%0d cycles=%0d", `NCMD, cyc - t0);
        $finish;
    end
endmodule
