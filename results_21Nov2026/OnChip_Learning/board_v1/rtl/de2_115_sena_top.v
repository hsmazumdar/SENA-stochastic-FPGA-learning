// =============================================================================
// de2_115_sena_top.v -- DE2-115 board top for SENA on-chip MNIST learning (board_v1)
// -----------------------------------------------------------------------------
// clk25 = CLOCK_50 / 2 (toggle register, global clock).  KEY[0] = reset (active low).
// RS-232 (DB9) 8N1 at 25e6/BAUD_DIV baud (BAUD_DIV=217 -> 115200).
// SRAM (IS61WV102416) layout, 16-bit words:
//   image n (0..12499): words n*64 + 0..48 = 784 input bits (word w bit b = pixel 16w+b),
//                       word n*64 + 49 = label (bits 3:0).  n<10000 train, n>=10000 test.
//   schedule entry s  : word SCHED_BASE + s = image index to train on (twin order).
// Host protocol (all multi-byte fields little-endian), see host/sena_board_host.py:
//   'V'                          -> 'S','E','N','A', ver, N_HID lo, hi              (7 B)
//   'W' addr3 n2 data[2n]        -> 'K', sum16 of data bytes                        (3 B)
//   'C' addr3 n3                 -> 'K', sum32 of the n words read back from SRAM   (5 B)
//   'R' addr3 n2                 -> 2n bytes (words read back, LE)
//   'S' seed4                    -> 'K'   (core init: LFSR seeds, nb_fx)            (1 B)
//   'L' sel1 row2 bytes[N]       -> 'K', sum16   (sel 1: W1 row, N=N_HID; sel 2: W2 row, N=10)
//   'D' sel1 row2                -> bytes[N], sum16
//   'T' start3 count3            -> 'F', core_cycles4, total_cycles4, n_w1_updates4 (13 B)
//   'E' first2 count2 T1         -> pred[count] then 'F', correct2, core_cycles4, total_cycles4
//   'Q'                          -> 'Q', n_w1_updates4, samples_trained4, nb_fx4     (13 B)
//   other                        -> '?'
// LEDG0 heartbeat, LEDG1 core busy, LEDG2 training, LEDG3 testing, LEDG4 loading,
// LEDG7 UART RX activity, LEDR0 UART frame error seen, LEDR17 unknown command seen.
// HEX3..HEX0 = samples trained (low 16 bits, hex).
// =============================================================================
`timescale 1ns/1ps
module de2_115_sena_top #(
    parameter integer N_HID    = 160,
    parameter integer SH       = 7,
    parameter integer BAUD_DIV = 217,
    parameter [19:0]  SCHED_BASE = 20'hD0000
) (
    input  wire        CLOCK_50,
    input  wire [3:0]  KEY,
    input  wire        UART_RXD,
    output wire        UART_TXD,
    output wire [19:0] SRAM_ADDR,
    inout  wire [15:0] SRAM_DQ,
    output wire        SRAM_CE_N, SRAM_OE_N, SRAM_WE_N, SRAM_UB_N, SRAM_LB_N,
    output wire [8:0]  LEDG,
    output wire [17:0] LEDR,
    output wire [6:0]  HEX0, HEX1, HEX2, HEX3
);
    localparam integer N_IN = 784, N_OUT = 10, W = 8, RB = N_HID * W;
    localparam [7:0] VER = 8'h11;
    localparam integer AW1 = $clog2(N_IN + 1), AW2 = $clog2(N_HID + 1);

    // ---------------------------------------------------------------- clock / reset
    reg clk25 = 1'b0;
    always @(posedge CLOCK_50) clk25 <= ~clk25;
    reg [2:0] rs = 3'b000;
    always @(posedge clk25) rs <= {rs[1:0], KEY[0]};
    wire rst_n = rs[2];

    // ---------------------------------------------------------------- UART
    wire rx_valid, rx_ferr; wire [7:0] rx_data;
    reg tx_start; reg [7:0] tx_data; wire tx_busy;
    uart_rx_s #(.DIV(BAUD_DIV)) u_rx (.clk(clk25), .rst_n(rst_n), .rx(UART_RXD), .valid(rx_valid), .data(rx_data), .frame_err(rx_ferr));
    uart_tx_s #(.DIV(BAUD_DIV)) u_tx (.clk(clk25), .rst_n(rst_n), .start(tx_start), .data(tx_data), .tx(UART_TXD), .busy(tx_busy));

    // ---------------------------------------------------------------- SRAM
    reg sr_req, sr_we; reg [19:0] sr_addr; reg [15:0] sr_wdata; wire sr_done, sr_ready; wire [15:0] sr_rdata;
    sram_ctrl u_sram (.clk(clk25), .rst_n(rst_n), .req(sr_req), .we(sr_we), .addr(sr_addr), .wdata(sr_wdata),
        .done(sr_done), .rdata(sr_rdata), .ready(sr_ready),
        .SRAM_ADDR(SRAM_ADDR), .SRAM_DQ(SRAM_DQ), .SRAM_CE_N(SRAM_CE_N), .SRAM_OE_N(SRAM_OE_N),
        .SRAM_WE_N(SRAM_WE_N), .SRAM_UB_N(SRAM_UB_N), .SRAM_LB_N(SRAM_LB_N));

    // ---------------------------------------------------------------- core
    reg core_init, core_st, core_si; reg [31:0] seed_r; reg [N_IN-1:0] x_reg; reg [3:0] lab_r; reg [5:0] t_r;
    reg h_we1, h_we2; reg [AW1-1:0] h_addr1; reg [AW2-1:0] h_addr2; reg [RB-1:0] rowbuf;
    wire core_busy, core_done; wire [3:0] core_pred; wire [31:0] nupd, nb_fx;
    wire [RB-1:0] h_rd1; wire [N_OUT*W-1:0] h_rd2;
    sena_learn_core #(.N_IN(N_IN), .N_HID(N_HID), .N_OUT(N_OUT), .W(W), .SH(SH)) u_core (
        .clk(clk25), .rst_n(rst_n), .init(core_init), .seed_in(seed_r),
        .start_train(core_st), .start_infer(core_si), .x_bits(x_reg), .label(lab_r), .t_passes(t_r),
        .busy(core_busy), .done(core_done), .pred(core_pred), .n_w1_updates(nupd), .nb_fx_out(nb_fx),
        .host_we1(h_we1), .host_we2(h_we2), .host_addr1(h_addr1), .host_addr2(h_addr2),
        .host_wdata1(rowbuf), .host_wdata2(rowbuf[RB-1 -: N_OUT*W]),
        .host_rdata1(h_rd1), .host_rdata2(h_rd2));

    // ---------------------------------------------------------------- command FSM
    localparam [5:0]
        S_IDLE=0, S_HDR=1, S_DISPATCH=2, S_TX=3, S_TXW=4, S_RESP=5, S_SR=6, S_SRW=7,
        S_W_DATA=8, S_C_LOOP=9, S_C_ACC=10, S_R_LOOP=11, S_R_HI=12, S_R_NEXT=13,
        S_L_DATA=14, S_L_WR=15, S_D_RD=16, S_D_LATCH=17, S_D_SEND=18, S_D_SUM=19, S_D_SUM2=20,
        S_T_LOOP=21, S_T_SCHED=22, S_FETCH=23, S_FETCH_ST=24, S_T_RUN=25, S_T_WAIT=26,
        S_E_LOOP=27, S_E_RUN=28, S_E_WAIT=29, S_E_SEND=30, S_DONE_T=31, S_DONE_E=32, S_D_WAIT=33,
        S_C_WAIT=34, S_FETCH_W=35, S_R_WAIT=36;

    reg [5:0]  st, ret_tx, ret_sr, ret_fetch;
    reg [7:0]  op;
    reg [7:0]  hb [0:5];
    reg [2:0]  hcnt, hneed;
    reg [7:0]  resp [0:12];
    reg [3:0]  ridx, rlen;
    reg [7:0]  snd;
    reg [23:0] n_left;
    reg [19:0] a_cur;
    reg [15:0] sum16;
    reg [31:0] sum32;
    reg        lo_have; reg [7:0] lo_byte;
    reg [7:0]  sel; reg [7:0] nbytes, bcnt;
    reg [23:0] s_cur, cnt_left;
    reg [15:0] img, correct;
    reg [5:0]  fw;
    reg [31:0] core_cyc, tot_cyc, samples;
    reg        mode_T, mode_E, mode_W, unk;
    integer i;

    wire [23:0] f_a3 = {hb[2], hb[1], hb[0]};

    always @(posedge clk25 or negedge rst_n) begin
        if (!rst_n) begin
            st <= S_IDLE; tx_start <= 0; tx_data <= 0; sr_req <= 0; sr_we <= 0; sr_addr <= 0; sr_wdata <= 0;
            core_init <= 0; core_st <= 0; core_si <= 0; seed_r <= 0; x_reg <= 0; lab_r <= 0; t_r <= 6'd40;
            h_we1 <= 0; h_we2 <= 0; h_addr1 <= 0; h_addr2 <= 0; rowbuf <= 0;
            hcnt <= 0; hneed <= 0; ridx <= 0; rlen <= 0; snd <= 0; n_left <= 0; a_cur <= 0; sum16 <= 0; sum32 <= 0;
            lo_have <= 0; lo_byte <= 0; sel <= 0; nbytes <= 0; bcnt <= 0; s_cur <= 0; cnt_left <= 0; img <= 0;
            correct <= 0; fw <= 0; core_cyc <= 0; tot_cyc <= 0; samples <= 0; mode_T <= 0; mode_E <= 0; mode_W <= 0;
            unk <= 0; op <= 0; ret_tx <= 0; ret_sr <= 0; ret_fetch <= 0;
            for (i = 0; i < 6; i = i + 1) hb[i] <= 0;
            for (i = 0; i < 13; i = i + 1) resp[i] <= 0;
        end else begin
            tx_start <= 1'b0; sr_req <= 1'b0; core_init <= 1'b0; core_st <= 1'b0; core_si <= 1'b0;
            h_we1 <= 1'b0; h_we2 <= 1'b0;
            if (mode_T || mode_E) begin
                tot_cyc <= tot_cyc + 1;
                if (core_busy) core_cyc <= core_cyc + 1;
            end
            case (st)
            // ---------------------------------------------------- command entry
            S_IDLE: if (rx_valid) begin
                op <= rx_data; hcnt <= 0;
                case (rx_data)
                    "W": begin hneed <= 5; st <= S_HDR; end
                    "C": begin hneed <= 6; st <= S_HDR; end
                    "R": begin hneed <= 5; st <= S_HDR; end
                    "S": begin hneed <= 4; st <= S_HDR; end
                    "L": begin hneed <= 3; st <= S_HDR; end
                    "D": begin hneed <= 3; st <= S_HDR; end
                    "T": begin hneed <= 6; st <= S_HDR; end
                    "E": begin hneed <= 5; st <= S_HDR; end
                    "V": begin resp[0] <= "S"; resp[1] <= "E"; resp[2] <= "N"; resp[3] <= "A"; resp[4] <= VER;
                               resp[5] <= N_HID[7:0]; resp[6] <= N_HID[15:8]; rlen <= 7; ridx <= 0; st <= S_RESP; end
                    "Q": begin resp[0] <= "Q";
                               {resp[4], resp[3], resp[2], resp[1]} <= nupd;
                               {resp[8], resp[7], resp[6], resp[5]} <= samples;
                               {resp[12], resp[11], resp[10], resp[9]} <= nb_fx;
                               rlen <= 13; ridx <= 0; st <= S_RESP; end
                    default: begin unk <= 1'b1; resp[0] <= "?"; rlen <= 1; ridx <= 0; st <= S_RESP; end
                endcase
            end
            S_HDR: if (rx_valid) begin
                hb[hcnt] <= rx_data; hcnt <= hcnt + 1'b1;
                if (hcnt + 1'b1 == hneed) st <= S_DISPATCH;
            end
            S_DISPATCH: begin
                case (op)
                "W": begin a_cur <= f_a3[19:0]; n_left <= {8'd0, hb[4], hb[3]}; sum16 <= 0; lo_have <= 0; mode_W <= 1; st <= S_W_DATA; end
                "C": begin a_cur <= f_a3[19:0]; n_left <= {hb[5], hb[4], hb[3]}; sum32 <= 0; st <= S_C_LOOP; end
                "R": begin a_cur <= f_a3[19:0]; n_left <= {8'd0, hb[4], hb[3]}; st <= S_R_LOOP; end
                "S": begin seed_r <= {hb[3], hb[2], hb[1], hb[0]}; core_init <= 1'b1; resp[0] <= "K"; rlen <= 1; ridx <= 0; st <= S_RESP; end
                "L": begin sel <= hb[0]; h_addr1 <= {hb[2], hb[1]}; h_addr2 <= {hb[2], hb[1]};
                           nbytes <= (hb[0] == 8'd1) ? N_HID[7:0] : 8'd10; bcnt <= 0; sum16 <= 0; st <= S_L_DATA; end
                "D": begin sel <= hb[0]; h_addr1 <= {hb[2], hb[1]}; h_addr2 <= {hb[2], hb[1]};
                           nbytes <= (hb[0] == 8'd1) ? N_HID[7:0] : 8'd10; bcnt <= 0; sum16 <= 0; st <= S_D_WAIT; end
                "T": begin s_cur <= f_a3; cnt_left <= {hb[5], hb[4], hb[3]}; core_cyc <= 0; tot_cyc <= 0; mode_T <= 1; st <= S_T_LOOP; end
                "E": begin img <= {hb[1], hb[0]}; cnt_left <= {8'd0, hb[3], hb[2]}; t_r <= hb[4][5:0]; correct <= 0;
                           core_cyc <= 0; tot_cyc <= 0; mode_E <= 1; st <= S_E_LOOP; end
                default: st <= S_IDLE;
                endcase
            end
            // ---------------------------------------------------- generic: send one byte, response buffer, SRAM op
            S_TX:  if (!tx_busy && !tx_start) begin tx_data <= snd; tx_start <= 1'b1; st <= S_TXW; end
            S_TXW: st <= ret_tx;
            S_RESP: if (ridx < rlen) begin snd <= resp[ridx]; ridx <= ridx + 1'b1; ret_tx <= S_RESP; st <= S_TX; end
                    else st <= S_IDLE;
            S_SR:  if (sr_ready) begin sr_req <= 1'b1; st <= S_SRW; end
            S_SRW: if (sr_done) st <= ret_sr;
            // ---------------------------------------------------- 'W' write SRAM block
            S_W_DATA: begin
                if (n_left == 0) begin
                    mode_W <= 0; resp[0] <= "K"; resp[1] <= sum16[7:0]; resp[2] <= sum16[15:8]; rlen <= 3; ridx <= 0; st <= S_RESP;
                end else if (rx_valid) begin
                    sum16 <= sum16 + rx_data;
                    if (!lo_have) begin lo_byte <= rx_data; lo_have <= 1'b1; end
                    else begin
                        lo_have <= 1'b0; sr_we <= 1'b1; sr_addr <= a_cur; sr_wdata <= {rx_data, lo_byte};
                        a_cur <= a_cur + 1'b1; n_left <= n_left - 1'b1; ret_sr <= S_W_DATA; st <= S_SR;
                    end
                end
            end
            // ---------------------------------------------------- 'C' checksum of SRAM region
            S_C_LOOP: begin
                if (n_left == 0) begin
                    resp[0] <= "K"; {resp[4], resp[3], resp[2], resp[1]} <= sum32; rlen <= 5; ridx <= 0; st <= S_RESP;
                end else begin
                    sr_we <= 1'b0; sr_addr <= a_cur; ret_sr <= S_C_ACC; st <= S_SR;
                end
            end
            S_C_ACC: begin sum32 <= sum32 + sr_rdata; a_cur <= a_cur + 1'b1; n_left <= n_left - 1'b1; st <= S_C_LOOP; end
            // ---------------------------------------------------- 'R' read SRAM block
            S_R_LOOP: begin
                if (n_left == 0) st <= S_IDLE;
                else begin sr_we <= 1'b0; sr_addr <= a_cur; ret_sr <= S_R_WAIT; st <= S_SR; end
            end
            S_R_WAIT: begin snd <= sr_rdata[7:0]; ret_tx <= S_R_HI; st <= S_TX; end
            S_R_HI:   begin snd <= sr_rdata[15:8]; ret_tx <= S_R_NEXT; st <= S_TX; end
            S_R_NEXT: begin a_cur <= a_cur + 1'b1; n_left <= n_left - 1'b1; st <= S_R_LOOP; end
            // ---------------------------------------------------- 'L' load weight row
            S_L_DATA: if (bcnt == nbytes) st <= S_L_WR;
                      else if (rx_valid) begin
                          rowbuf <= {rx_data, rowbuf[RB-1:8]}; sum16 <= sum16 + rx_data; bcnt <= bcnt + 1'b1;
                      end
            S_L_WR: begin
                if (sel == 8'd1) h_we1 <= 1'b1; else h_we2 <= 1'b1;
                resp[0] <= "K"; resp[1] <= sum16[7:0]; resp[2] <= sum16[15:8]; rlen <= 3; ridx <= 0; st <= S_RESP;
            end
            // ---------------------------------------------------- 'D' dump weight row
            S_D_WAIT:  st <= S_D_RD;                                   // address settles / RAM read
            S_D_RD:    st <= S_D_LATCH;
            S_D_LATCH: begin
                if (sel == 8'd1) rowbuf <= h_rd1; else rowbuf <= {{(RB-N_OUT*W){1'b0}}, h_rd2};
                st <= S_D_SEND;
            end
            S_D_SEND: if (bcnt == nbytes) st <= S_D_SUM;
                      else begin
                          snd <= rowbuf[7:0]; sum16 <= sum16 + rowbuf[7:0]; rowbuf <= {8'd0, rowbuf[RB-1:8]};
                          bcnt <= bcnt + 1'b1; ret_tx <= S_D_SEND; st <= S_TX;
                      end
            S_D_SUM:  begin snd <= sum16[7:0];  ret_tx <= S_D_SUM2; st <= S_TX; end
            S_D_SUM2: begin snd <= sum16[15:8]; ret_tx <= S_IDLE;   st <= S_TX; end
            // ---------------------------------------------------- image fetch subroutine (img -> x_reg, lab_r)
            S_FETCH:   begin fw <= 0; st <= S_FETCH_ST; end
            S_FETCH_ST: begin sr_we <= 1'b0; sr_addr <= {img[13:0], fw}; ret_sr <= S_FETCH_W; st <= S_SR; end
            S_FETCH_W: begin
                if (fw == 6'd49) begin lab_r <= sr_rdata[3:0]; st <= ret_fetch; end
                else begin x_reg[fw*16 +: 16] <= sr_rdata; fw <= fw + 1'b1; st <= S_FETCH_ST; end
            end
            // ---------------------------------------------------- 'T' train schedule entries
            S_T_LOOP: begin
                if (cnt_left == 0) st <= S_DONE_T;
                else begin sr_we <= 1'b0; sr_addr <= SCHED_BASE + s_cur[19:0]; ret_sr <= S_T_SCHED; st <= S_SR; end
            end
            S_T_SCHED: begin img <= sr_rdata; ret_fetch <= S_T_RUN; st <= S_FETCH; end
            S_T_RUN:   begin core_st <= 1'b1; st <= S_T_WAIT; end
            S_T_WAIT:  if (core_done) begin
                           samples <= samples + 1; s_cur <= s_cur + 1'b1; cnt_left <= cnt_left - 1'b1; st <= S_T_LOOP;
                       end
            S_DONE_T: begin
                mode_T <= 0;
                resp[0] <= "F"; {resp[4], resp[3], resp[2], resp[1]} <= core_cyc;
                {resp[8], resp[7], resp[6], resp[5]} <= tot_cyc; {resp[12], resp[11], resp[10], resp[9]} <= nupd;
                rlen <= 13; ridx <= 0; st <= S_RESP;
            end
            // ---------------------------------------------------- 'E' evaluate test images
            S_E_LOOP: begin
                if (cnt_left == 0) st <= S_DONE_E;
                else begin ret_fetch <= S_E_RUN; st <= S_FETCH; end
            end
            S_E_RUN:  begin core_si <= 1'b1; st <= S_E_WAIT; end
            S_E_WAIT: if (core_done) begin
                          if (core_pred == lab_r) correct <= correct + 1'b1;
                          snd <= {4'd0, core_pred}; ret_tx <= S_E_SEND; st <= S_TX;
                      end
            S_E_SEND: begin img <= img + 1'b1; cnt_left <= cnt_left - 1'b1; st <= S_E_LOOP; end
            S_DONE_E: begin
                mode_E <= 0;
                resp[0] <= "F"; resp[1] <= correct[7:0]; resp[2] <= correct[15:8];
                {resp[6], resp[5], resp[4], resp[3]} <= core_cyc; {resp[10], resp[9], resp[8], resp[7]} <= tot_cyc;
                rlen <= 11; ridx <= 0; st <= S_RESP;
            end
            default: st <= S_IDLE;
            endcase
        end
    end

    // ---------------------------------------------------------------- status
    reg [24:0] hb_cnt; reg [21:0] rx_led; reg ferr_seen;
    always @(posedge clk25 or negedge rst_n) begin
        if (!rst_n) begin hb_cnt <= 0; rx_led <= 0; ferr_seen <= 0; end
        else begin
            hb_cnt <= hb_cnt + 1'b1;
            if (rx_valid) rx_led <= {22{1'b1}}; else if (rx_led != 0) rx_led <= rx_led - 1'b1;
            if (rx_ferr) ferr_seen <= 1'b1;
        end
    end
    function [6:0] seg7;   // active-low, segments g..a
        input [3:0] v;
        case (v)
            4'h0: seg7 = 7'b1000000; 4'h1: seg7 = 7'b1111001; 4'h2: seg7 = 7'b0100100; 4'h3: seg7 = 7'b0110000;
            4'h4: seg7 = 7'b0011001; 4'h5: seg7 = 7'b0010010; 4'h6: seg7 = 7'b0000010; 4'h7: seg7 = 7'b1111000;
            4'h8: seg7 = 7'b0000000; 4'h9: seg7 = 7'b0010000; 4'hA: seg7 = 7'b0001000; 4'hB: seg7 = 7'b0000011;
            4'hC: seg7 = 7'b1000110; 4'hD: seg7 = 7'b0100001; 4'hE: seg7 = 7'b0000110; default: seg7 = 7'b0001110;
        endcase
    endfunction
    assign HEX0 = seg7(samples[3:0]);  assign HEX1 = seg7(samples[7:4]);
    assign HEX2 = seg7(samples[11:8]); assign HEX3 = seg7(samples[15:12]);
    assign LEDG = {1'b0, (rx_led != 0), 2'b00, mode_W, mode_E, mode_T, core_busy, hb_cnt[24]};
    assign LEDR = {unk, 16'd0, ferr_seen};
endmodule
