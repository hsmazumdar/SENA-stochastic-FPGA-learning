// sena_core_synth_top.v -- synthesis-only wrapper to measure sena_learn_core (784-160-10, 8-bit)
// on EP4CE115.  NOT the DE2-115 board top: x_bits and a host weight row are loaded through
// serial shift registers so the wide ports stay real logic without needing ~2,000 pins.
`timescale 1ns/1ps
module sena_core_synth_top (
    input  wire       CLOCK_50,
    input  wire       rst_n,
    input  wire       sh_x,       // shift one input bit into x register
    input  wire       sh_h,       // shift one bit into host row/address register
    input  wire       sdi,
    input  wire       init,
    input  wire       start_train,
    input  wire       start_infer,
    input  wire       host_we1,
    input  wire       host_we2,
    input  wire [3:0] label,
    output wire [3:0] pred,
    output wire       busy,
    output wire       done,
    output wire       rd_parity,
    output wire       upd_parity
);
    localparam integer N_IN = 784, N_HID = 160, N_OUT = 10, W = 8;
    reg [N_IN-1:0]     xr;
    reg [N_HID*W-1:0]  hsr;
    reg [9:0]          har;
    reg [31:0]         seed_r;
    always @(posedge CLOCK_50) begin
        if (sh_x) xr <= {xr[N_IN-2:0], sdi};
        if (sh_h) begin hsr <= {hsr[N_HID*W-2:0], sdi}; har <= {har[8:0], hsr[N_HID*W-1]}; seed_r <= {seed_r[30:0], har[9]}; end
    end
    wire [N_HID*W-1:0] hr1; wire [N_OUT*W-1:0] hr2; wire [31:0] nupd;
    sena_learn_core #(.N_IN(N_IN), .N_HID(N_HID), .N_OUT(N_OUT), .W(W), .SH(7),
                      .W1_INIT("w1_init.hex"), .W2_INIT("w2_init.hex")) core (
        .clk(CLOCK_50), .rst_n(rst_n), .init(init), .seed_in(seed_r),
        .start_train(start_train), .start_infer(start_infer), .x_bits(xr), .label(label),
        .t_passes(6'd40), .busy(busy), .done(done), .pred(pred), .n_w1_updates(nupd),
        .host_we1(host_we1), .host_we2(host_we2), .host_addr1(har), .host_addr2(har[7:0]),
        .host_wdata1(hsr), .host_wdata2(hsr[N_OUT*W-1:0]), .host_rdata1(hr1), .host_rdata2(hr2));
    assign rd_parity  = ^{hr1, hr2};
    assign upd_parity = ^nupd;
endmodule
