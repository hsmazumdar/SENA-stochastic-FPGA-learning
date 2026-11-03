// =============================================================================
// sena_tb.v — MNIST vector TB for SAIF/VCD switching-activity capture
// -----------------------------------------------------------------------------
// Loads stimuli/mnist_x.bin (98 bytes = 784 bits per image, MSB-first) and
// runs PASSES_PER_IMAGE stochastic passes per sample (default 3 ≈ fixed-N EE).
//
// Plusargs: +N_IMAGES=1000  +PASSES=3  +CLK_MHZ=50
// =============================================================================
`timescale 1ns/1ps

module sena_tb;
    localparam integer N_IN    = 784;
    localparam integer N_OUT   = 10;
    localparam integer T_MAX   = 40;
    localparam integer BYTES_PER_IMG = 98; // 784/8
    localparam integer MAX_IMG = 2048;

    integer n_images, passes_per, clk_mhz, i, j, b, bit_i;
    integer fd, nbytes;
    real    half_ns;

    reg clk, rst_n;
    reg sample_start, pass_start;
    reg [N_IN-1:0] x_bits;
    reg [3:0] n_log2_hid, n_log2_out;

    wire [$clog2(N_OUT)-1:0] prediction;
    wire busy, exit_ready, done_force;
    wire [$clog2(T_MAX+1)-1:0] passes_used;

    reg [N_IN-1:0] mnist_mem [0:MAX_IMG-1];
    reg [7:0]      raw [0:MAX_IMG*BYTES_PER_IMG-1];
    integer        load_ok, seed;

    sena_top uut (
        .clk(clk), .rst_n(rst_n),
        .sample_start(sample_start), .pass_start(pass_start),
        .x_bits(x_bits),
        .n_log2_hid(n_log2_hid), .n_log2_out(n_log2_out),
        .prediction(prediction),
        .busy(busy), .exit_ready(exit_ready), .done_force(done_force),
        .passes_used(passes_used)
    );

    initial begin
        n_images = 1000;
        passes_per = 3;
        clk_mhz = 50;
        if ($value$plusargs("N_IMAGES=%d", n_images)) ;
        if ($value$plusargs("PASSES=%d", passes_per)) ;
        if ($value$plusargs("CLK_MHZ=%d", clk_mhz)) ;
        if (n_images > MAX_IMG) n_images = MAX_IMG;
        half_ns = 500.0 / clk_mhz;
        n_log2_hid = 4'd9;  // matches A2 adaptive checkpoint N_fwd
        n_log2_out = 4'd8;
        if ($value$plusargs("N_LOG2_HID=%d", n_log2_hid)) ;
        if ($value$plusargs("N_LOG2_OUT=%d", n_log2_out)) ;
    end

    initial begin
        clk = 0;
        forever #(half_ns) clk = ~clk;
    end

    initial begin
        load_ok = 0;
        fd = $fopen("mnist_x.bin", "rb");
        if (fd == 0) begin
            $display("WARN: mnist_x.bin not found — seeded pseudo-MNIST");
            seed = 2026;
            for (i = 0; i < MAX_IMG; i = i + 1)
                for (j = 0; j < N_IN; j = j + 1) begin
                    seed = seed * 1103515245 + 12345;
                    mnist_mem[i][j] = (seed[31:24] < 64);
                end
        end else begin
            nbytes = $fread(raw, fd);
            $fclose(fd);
            $display("fread %0d bytes from mnist_x.bin", nbytes);
            for (i = 0; i < MAX_IMG; i = i + 1) begin
                for (b = 0; b < BYTES_PER_IMG; b = b + 1) begin
                    for (bit_i = 0; bit_i < 8; bit_i = bit_i + 1) begin
                        j = b * 8 + bit_i;
                        if (j < N_IN)
                            mnist_mem[i][j] =
                                raw[i*BYTES_PER_IMG + b][7-bit_i];
                    end
                end
            end
            load_ok = 1;
        end
    end

    initial begin
        sample_start = 0;
        pass_start = 0;
        x_bits = {N_IN{1'b0}};
        rst_n = 0;
        repeat (20) @(posedge clk);
        rst_n = 1;
        repeat (10) @(posedge clk);

        for (i = 0; i < n_images; i = i + 1) begin
            x_bits = mnist_mem[i];
            @(posedge clk);
            sample_start <= 1'b1;
            @(posedge clk);
            sample_start <= 1'b0;

            for (j = 0; j < passes_per; j = j + 1) begin
                @(posedge clk);
                while (busy) @(posedge clk);
                pass_start <= 1'b1;
                @(posedge clk);
                pass_start <= 1'b0;
                @(posedge clk);
                while (busy) @(posedge clk);
            end

            if (i % 100 == 0)
                $display("[%0t] img %0d/%0d pred=%0d passes=%0d",
                         $time, i, n_images, prediction, passes_used);
        end

        repeat (50) @(posedge clk);
        $display("SENA TB done: %0d images × %0d passes (load_ok=%0d)",
                 n_images, passes_per, load_ok);
        $finish;
    end
endmodule
