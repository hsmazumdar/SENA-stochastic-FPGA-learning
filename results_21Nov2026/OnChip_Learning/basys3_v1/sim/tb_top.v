// tb_top.v -- Basys 3 top-level testbench: UART BFM driven by a host-generated
// byte script.  No SRAM model; images are streamed over UART.
// Script lines: "S hh" send byte, "R n" wait until n bytes received in total, "X 0" end.
`timescale 1ns/1ps
`ifndef N_HID
 `define N_HID 16
`endif
`ifndef SH
 `define SH 4
`endif
`ifndef BAUD_DIV
 `define BAUD_DIV 8
`endif
`ifndef H_LANES
 `define H_LANES 0
`endif
module tb_top;
    // 100 MHz board oscillator; the DUT divides by 2, so the UART bit period is
    // BAUD_DIV cycles of the 50 MHz internal clock = BAUD_DIV * 20 ns.
    reg clk100 = 0; always #5 clk100 = ~clk100;
    reg btnC = 1'b1;
    reg rxd = 1'b1; wire txd;
    wire [15:0] led; wire [6:0] seg; wire dp; wire [3:0] an;
    basys3_sena_top #(.N_HID(`N_HID), .SH(`SH), .BAUD_DIV(`BAUD_DIV), .H_LANES(`H_LANES)) dut (
        .clk100(clk100), .btnC(btnC), .RsRx(rxd), .RsTx(txd),
        .led(led), .seg(seg), .dp(dp), .an(an));

    localparam real BIT = `BAUD_DIV * 20.0;
    integer nrx = 0, flog, fs, code, val, nsent = 0, nferr = 0;
    reg [8*8:1] tok;
    reg [7:0] rb; integer k;
    always @(negedge txd) begin : rxmon
        #(BIT * 1.5);
        for (k = 0; k < 8; k = k + 1) begin rb[k] = txd; #(BIT); end
        if (txd !== 1'b1) nferr = nferr + 1;
        $fwrite(flog, "%02x\n", rb); nrx = nrx + 1; $fflush(flog);
    end
    task send(input [7:0] b);
        integer j;
        begin
            rxd = 0; #(BIT);
            for (j = 0; j < 8; j = j + 1) begin rxd = b[j]; #(BIT); end
            rxd = 1; #(BIT); nsent = nsent + 1;
        end
    endtask
    initial begin
        flog = $fopen(`RXLOG, "w");
        fs = $fopen(`SCRIPT, "r");
        if (fs == 0) begin $display("cannot open script"); $finish; end
        #200 btnC = 1'b0; #2000;
        while (!$feof(fs)) begin
            code = $fscanf(fs, "%s", tok);
            if (code != 1) begin end
            else if (tok == "S") begin code = $fscanf(fs, "%h", val); send(val[7:0]); end
            else if (tok == "R") begin code = $fscanf(fs, "%d", val); wait (nrx >= val); end
            else if (tok == "X") begin code = $fscanf(fs, "%d", val);
                #(BIT * 20);
                $display("TB_DONE sent=%0d received=%0d frame_err_rx_side=%0d overflow=%0d led=%h t=%0t ns",
                         nsent, nrx, nferr, dut.fifo_ovf, led, $time);
                $fclose(flog); $finish;
            end
        end
        $display("TB_EOF without X"); $fclose(flog); $finish;
    end
    initial begin : prog
        forever begin
            #100_000_000; // 100 ms
            $display("t=%0d ms sent=%0d rx=%0d samples=%0d st=%0d fifo_ovf=%0d", $time / 1_000_000, nsent, nrx, dut.samples, dut.st, dut.fifo_ovf);
        end
    end
endmodule
