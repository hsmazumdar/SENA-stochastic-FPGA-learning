// =============================================================================
// tb_sena_power.v — representative activity dump for Vivado Report Power
// -----------------------------------------------------------------------------
// Drives N_INFER random "MNIST-like" bit-vectors through sena_tile and writes
// a VCD of internal toggles. Weights are constant (nonzero pseudo-random) so
// the datapath actually switches; accuracy is irrelevant for power.
//
// Plusargs / env (set by vivado_postroute_power.tcl):
//   +N_INFER=200   +CLK_MHZ=50   +VCD=activity.vcd
// =============================================================================
`timescale 1ns/1ps

module tb_sena_power;
    localparam integer N_IN       = 784;
    localparam integer N_HID_BANK = 160;
    localparam integer N_OUT      = 10;
    localparam integer W_BITS     = 16;
    localparam integer PAR        = 8;
    localparam integer PAR_OUT    = 2;
    localparam integer T_MAX      = 40;

    integer n_infer;
    integer clk_mhz;
    real    clk_half_ns;
    reg [1023:0] vcd_path;

    reg clk, rst_n;
    reg sample_start, pass_start;
    reg [N_IN-1:0] x_bits;

    reg signed [W_BITS-1:0] w_hid [0:N_HID_BANK*N_IN-1];
    reg signed [W_BITS-1:0] b_hid [0:N_HID_BANK-1];
    reg signed [W_BITS-1:0] w_out [0:N_OUT*N_HID_BANK-1];
    reg signed [W_BITS-1:0] b_out [0:N_OUT-1];
    reg signed [W_BITS-1:0] K_hid, K_out;
    reg [3:0] n_log2_hid, n_log2_out;

    wire [11:0] votes_dbg [0:N_OUT-1];
    wire busy, exit_ready, done_force;
    wire [$clog2(N_OUT)-1:0] pred;
    wire [$clog2(T_MAX+1)-1:0] passes_used;

    integer i, k, seed;
    integer passes_this;

    sena_tile #(
        .N_IN(N_IN), .N_HID_BANK(N_HID_BANK), .N_OUT(N_OUT),
        .W_BITS(W_BITS), .PAR(PAR), .PAR_OUT(PAR_OUT), .T_MAX(T_MAX)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .sample_start(sample_start), .pass_start(pass_start),
        .x_bits(x_bits),
        .w_hid(w_hid), .b_hid(b_hid), .w_out(w_out), .b_out(b_out),
        .K_hid(K_hid), .K_out(K_out),
        .n_log2_hid(n_log2_hid), .n_log2_out(n_log2_out),
        .votes_dbg(votes_dbg), .busy(busy),
        .exit_ready(exit_ready), .done_force(done_force),
        .pred(pred), .passes_used(passes_used)
    );

    initial begin
        n_infer = 200;
        clk_mhz = 50;
        if ($value$plusargs("N_INFER=%d", n_infer)) ;
        if ($value$plusargs("CLK_MHZ=%d", clk_mhz)) ;
        if (!$value$plusargs("VCD=%s", vcd_path))
            vcd_path = "activity.vcd";
        clk_half_ns = 500.0 / clk_mhz;
    end

    initial begin
        clk = 0;
        forever #(clk_half_ns) clk = ~clk;
    end

    // Deterministic nonzero weights (LFSR-ish) so MACs/adds toggle.
    function automatic signed [W_BITS-1:0] winit(input integer idx);
        begin
            winit = $signed({1'b0, idx[14:0]} ^ {idx[7:0], idx[14:8]});
            if (winit == 0) winit = 1;
        end
    endfunction

    initial begin
        $dumpfile(vcd_path);
        $dumpvars(0, tb_sena_power);

        seed = 2026;
        for (i = 0; i < N_HID_BANK*N_IN; i = i + 1) w_hid[i] = winit(i);
        for (i = 0; i < N_HID_BANK; i = i + 1) b_hid[i] = winit(100000 + i);
        for (i = 0; i < N_OUT*N_HID_BANK; i = i + 1) w_out[i] = winit(200000 + i);
        for (i = 0; i < N_OUT; i = i + 1) b_out[i] = winit(300000 + i);
        K_hid = 0; K_out = 0;
        n_log2_hid = 4'd3; // N≈8
        n_log2_out = 4'd3;

        rst_n = 0;
        sample_start = 0;
        pass_start = 0;
        x_bits = {N_IN{1'b0}};
        repeat (10) @(posedge clk);
        rst_n = 1;
        repeat (5) @(posedge clk);

        for (k = 0; k < n_infer; k = k + 1) begin
            // Random sparse-ish input (~25% ones, MNIST-like after thresh)
            for (i = 0; i < N_IN; i = i + 1) begin
                seed = seed * 1103515245 + 12345;
                x_bits[i] = (seed[31:24] < 64);
            end

            @(posedge clk);
            sample_start <= 1'b1;
            @(posedge clk);
            sample_start <= 1'b0;

            // Fixed-N early-exit style: run a few passes (mean ~2–3)
            passes_this = 2 + (k % 3);
            for (i = 0; i < passes_this; i = i + 1) begin
                @(posedge clk);
                while (busy) @(posedge clk);
                pass_start <= 1'b1;
                @(posedge clk);
                pass_start <= 1'b0;
                @(posedge clk);
                while (busy) @(posedge clk);
            end

            if (k % 50 == 0)
                $display("[%0t] inference %0d/%0d pred=%0d passes=%0d",
                         $time, k, n_infer, pred, passes_used);
        end

        repeat (20) @(posedge clk);
        $display("VCD complete: %0d inferences → %0s", n_infer, vcd_path);
        $finish;
    end
endmodule
