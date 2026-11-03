// =============================================================================
// de2_115_calib_top.v — SENA C1 on-board self-test (USB-Blaster bring-up)
// PASS: LEDG[8:1] ON + HEX decimal 0807. FAIL: LEDR17 ON.
// KEY0=reset, KEY1=re-run.
// =============================================================================
`timescale 1ns/1ps
module de2_115_calib_top (
    input  wire        CLOCK_50,
    input  wire [3:0]  KEY,
    input  wire        UART_RXD,
    output wire        UART_TXD,
    output wire [17:0] LEDR,
    output wire [8:0]  LEDG,
    output wire [6:0]  HEX0,
    output wire [6:0]  HEX1,
    output wire [6:0]  HEX2,
    output wire [6:0]  HEX3
);
    localparam integer CLKS_PER_BIT = 217;
    localparam integer N_IN = 784;

    reg clk_25 = 1'b0;
    always @(posedge CLOCK_50) clk_25 <= ~clk_25;
    wire clk = clk_25;

    reg [7:0] por = 8'd0;
    wire rst_n = por[7] & KEY[0];
    always @(posedge clk)
        if (!por[7]) por <= por + 8'd1;

    reg  [N_IN-1:0] x_bits;
    reg  [5:0]      t_max;
    reg             ee_en;
    reg  [3:0]      n_log2_hid, n_log2_out;
    reg             core_start;
    wire            core_busy, core_done;
    wire [3:0]      core_pred;
    wire [5:0]      core_passes;
    wire [31:0]     core_cycles;

    calib_core #(
        .W_HID_FILE("w_hid.mem"),
        .W_OUT_FILE("w_out.mem")
    ) u_core (
        .clk(clk), .rst_n(rst_n),
        .start(core_start), .t_max(t_max), .early_exit_en(ee_en),
        .n_log2_hid(n_log2_hid), .n_log2_out(n_log2_out),
        .x_bits(x_bits),
        .busy(core_busy), .done(core_done),
        .pred(core_pred), .passes(core_passes), .run_cycles(core_cycles)
    );

    localparam ST_ARM=0, ST_RUN=1, ST_CAP=2, ST_CHK=3, ST_PASS=4, ST_FAIL=5;
    reg [2:0]  st, t_idx;
    reg [5:0]  fail_t, cur_t, cap_passes;
    reg [31:0] cap_cycles;
    reg        bist_active, bist_done_pass, bist_done_fail;
    reg [15:0] arm_wait;
    reg [3:0]  hex_d0, hex_d1, hex_d2, hex_d3;

    function [5:0] t_of;
        input [2:0] idx;
        case (idx)
            3'd0: t_of = 6'd1;
            3'd1: t_of = 6'd2;
            3'd2: t_of = 6'd4;
            3'd3: t_of = 6'd8;
            3'd4: t_of = 6'd16;
            default: t_of = 6'd32;
        endcase
    endfunction

    function [31:0] expect_cycles;
        input [5:0] t;
        case (t)
            6'd1:  expect_cycles = 32'd807;
            6'd2:  expect_cycles = 32'd1614;
            6'd4:  expect_cycles = 32'd3228;
            6'd8:  expect_cycles = 32'd6456;
            6'd16: expect_cycles = 32'd12912;
            6'd32: expect_cycles = 32'd25824;
            default: expect_cycles = 32'd807;
        endcase
    endfunction

    reg key1_d;
    wire key1_rise = key1_d & ~KEY[1];
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) key1_d <= 1'b1;
        else key1_d <= KEY[1];
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st <= ST_ARM; t_idx <= 3'd0; cur_t <= 6'd1;
            fail_t <= 6'd0; cap_cycles <= 0; cap_passes <= 0;
            bist_active <= 1'b1; bist_done_pass <= 0; bist_done_fail <= 0;
            arm_wait <= 0; core_start <= 0;
            t_max <= 6'd1; ee_en <= 0;
            n_log2_hid <= 4'd9; n_log2_out <= 4'd8;
            x_bits <= {N_IN{1'b0}};
            x_bits[0] <= 1'b1; x_bits[100] <= 1'b1;
            x_bits[200] <= 1'b1; x_bits[400] <= 1'b1; x_bits[700] <= 1'b1;
            hex_d0 <= 0; hex_d1 <= 0; hex_d2 <= 0; hex_d3 <= 0;
        end else begin
            core_start <= 1'b0;
            if (key1_rise) begin
                st <= ST_ARM; t_idx <= 0; bist_active <= 1;
                bist_done_pass <= 0; bist_done_fail <= 0; arm_wait <= 0;
                hex_d0 <= 0; hex_d1 <= 0; hex_d2 <= 0; hex_d3 <= 0;
            end else case (st)
                ST_ARM: begin
                    bist_active <= 1'b1;
                    if (arm_wait == 16'd20000) begin
                        cur_t <= t_of(0); t_max <= t_of(0); ee_en <= 0;
                        t_idx <= 0; st <= ST_RUN; core_start <= 1'b1;
                    end else arm_wait <= arm_wait + 16'd1;
                end
                ST_RUN: if (core_done) st <= ST_CAP;
                ST_CAP: begin
                    cap_cycles <= core_cycles;
                    cap_passes <= core_passes;
                    st <= ST_CHK;
                end
                ST_CHK: begin
                    if (cap_passes == cur_t && cap_cycles == expect_cycles(cur_t)) begin
                        // decimal 0807
                        hex_d3 <= 4'd0; hex_d2 <= 4'd8; hex_d1 <= 4'd0; hex_d0 <= 4'd7;
                        if (t_idx == 3'd5) begin
                            st <= ST_PASS; bist_done_pass <= 1; bist_active <= 0;
                        end else begin
                            t_idx <= t_idx + 3'd1;
                            cur_t <= t_of(t_idx + 3'd1);
                            t_max <= t_of(t_idx + 3'd1);
                            st <= ST_RUN; core_start <= 1'b1;
                        end
                    end else begin
                        // Debug: show measured cycles for T=1 (e.g. 0806/0808)
                        if (cur_t == 6'd1) begin
                            hex_d3 <= 4'd0;
                            hex_d2 <= cap_cycles[9:0] / 10'd100;
                            hex_d1 <= (cap_cycles[9:0] / 10'd10) % 10'd10;
                            hex_d0 <= cap_cycles[9:0] % 10'd10;
                        end else begin
                            hex_d3 <= 4'd0; hex_d2 <= 4'd0;
                            hex_d1 <= cur_t / 6'd10;
                            hex_d0 <= cur_t % 6'd10;
                        end
                        fail_t <= cur_t;
                        st <= ST_FAIL; bist_done_fail <= 1; bist_active <= 0;
                    end
                end
                ST_PASS: bist_active <= 0;
                ST_FAIL: bist_active <= 0;
                default: st <= ST_ARM;
            endcase
        end
    end

    wire [7:0] rx_data; wire rx_valid, tx_busy;
    reg [7:0] tx_data; reg tx_send;
    uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_rx (
        .clk(clk), .rst_n(rst_n), .rx(UART_RXD), .data(rx_data), .valid(rx_valid));
    uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_tx (
        .clk(clk), .rst_n(rst_n), .data(tx_data), .send(tx_send),
        .tx(UART_TXD), .busy(tx_busy));
    reg p_reply; reg [15:0] pause, reply; reg [1:0] reply_idx;
    localparam integer REPLY_PAUSE = CLKS_PER_BIT * 12;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            p_reply <= 0; tx_send <= 0; tx_data <= 0;
            pause <= 0; reply <= 0; reply_idx <= 0;
        end else begin
            tx_send <= 0;
            if (bist_active) p_reply <= 0;
            else if (!p_reply && rx_valid && rx_data == 8'h56) begin
                reply <= {8'h01, 8'hA5}; reply_idx <= 0;
                pause <= REPLY_PAUSE[15:0]; p_reply <= 1;
            end else if (p_reply) begin
                if (pause != 0) pause <= pause - 1;
                else if (!tx_busy && !tx_send) begin
                    if (reply_idx == 2) p_reply <= 0;
                    else begin
                        tx_data <= reply[{reply_idx, 3'b000} +: 8];
                        tx_send <= 1; reply_idx <= reply_idx + 1;
                    end
                end
            end
        end
    end

    reg [24:0] hb_div; reg hb;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin hb_div <= 0; hb <= 0; end
        else if (hb_div == 25'd12_499_999) begin hb_div <= 0; hb <= ~hb; end
        else hb_div <= hb_div + 1;
    end

    assign LEDR[0]     = core_busy | bist_active;
    assign LEDR[4:1]   = bist_done_fail ? fail_t[4:1] : core_pred;
    assign LEDR[5]     = bist_done_fail & fail_t[5];
    assign LEDR[16:6]  = 11'd0;
    assign LEDR[17]    = bist_done_fail;

    assign LEDG[0]   = hb;
    assign LEDG[7:1] = bist_done_pass ? 7'b1111111 : 7'd0;
    assign LEDG[8]   = bist_done_pass;

    seg7 s0 (.d(hex_d0), .seg(HEX0));
    seg7 s1 (.d(hex_d1), .seg(HEX1));
    seg7 s2 (.d(hex_d2), .seg(HEX2));
    seg7 s3 (.d(hex_d3), .seg(HEX3));
endmodule

module seg7 (input wire [3:0] d, output reg [6:0] seg);
    always @* case (d)
        4'd0: seg = 7'b1000000; 4'd1: seg = 7'b1111001;
        4'd2: seg = 7'b0100100; 4'd3: seg = 7'b0110000;
        4'd4: seg = 7'b0011001; 4'd5: seg = 7'b0010010;
        4'd6: seg = 7'b0000010; 4'd7: seg = 7'b1111000;
        4'd8: seg = 7'b0000000; 4'd9: seg = 7'b0010000;
        default: seg = 7'b1111111;
    endcase
endmodule
