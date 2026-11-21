// =============================================================================
// basys3_sena_top.v -- Digilent Basys 3 (XC7A35T-1CPG236C) board top for SENA
// on-chip MNIST learning (basys3_v1).  Core is sena_learn_core v1.1 plus an
// optional H_LANES serialisation parameter (0 or N_HID = same schedule as
// board_v1).  There is no external SRAM: training and test images are streamed
// over the onboard FTDI USB-UART after each 'T' / 'E' header.
// -----------------------------------------------------------------------------
// clk = clk100 / 2 (toggle + BUFG in Vivado via XDC generated clock), default
// 50 MHz.  btnC (centre, active high) = reset after a 3-FF synchroniser.
// UART 8N1 at clk/BAUD_DIV baud (BAUD_DIV=434 -> 115207 baud at 50 MHz).
//
// Host protocol (multi-byte fields little-endian), see host/sena_board_host.py.
// Command headers are the same as board_v1; VER = 0x21 marks the stream port.
//   'V'                          -> 'S','E','N','A', ver, N_HID lo, hi              (7 B)
//   'S' seed4                    -> 'K'
//   'L' sel1 row2 bytes[N]       -> 'K', sum16
//   'D' sel1 row2                -> bytes[N], sum16
//   'T' start3 count3            -> then count images of 100 B each (50 LE words:
//                                   words 0..48 = 784 input bits, word 49[3:0] =
//                                   label).  start3 is accepted and ignored.
//                                   Then 'F', core_cycles4, total_cycles4,
//                                   n_w1_updates4                                 (13 B)
//   'E' first2 count2 T1         -> then count images of 100 B each (first2
//                                   ignored).  After each image the board sends
//                                   one pred byte.  Then 'F', correct2,
//                                   core_cycles4, total_cycles4                    (11 B)
//   'Q'                          -> 'Q', n_w1_updates4, samples_trained4, nb_fx4   (13 B)
//   'W','C','R' (SRAM, no device) and any other opcode -> '?'
// led[0] heartbeat, led[1] core busy, led[2] training, led[3] testing,
// led[4] streaming an image, led[5] UART FIFO overflow (sticky),
// led[7] UART RX activity, led[14] unknown command, led[15] frame error seen.
// 4-digit 7-seg (muxed, active-low) = samples trained (low 16 bits, hex).
// =============================================================================
`timescale 1ns/1ps
module basys3_sena_top #(
    parameter integer N_HID    = 160,
    parameter integer SH       = 7,
    parameter integer BAUD_DIV = 434,
    parameter integer H_LANES  = 0,     // 0 => N_HID (full parallel)
    parameter integer FIFO_D   = 64
) (
    input  wire        clk100,
    input  wire        btnC,
    input  wire        RsRx,
    output wire        RsTx,
    output wire [15:0] led,
    output wire [6:0]  seg,
    output wire        dp,
    output wire [3:0]  an
);
    localparam integer N_IN = 784, N_OUT = 10, W = 8, RB = N_HID * W;
    localparam [7:0] VER = 8'h21;
    localparam integer AW1 = $clog2(N_IN + 1), AW2 = $clog2(N_HID + 1);
    localparam integer IMG_BYTES = 100;   // 50 x 16-bit words

    // ---------------------------------------------------------------- clock / reset
    // 100 MHz -> 50 MHz.  Icarus uses the toggle net as the clock; Vivado
    // should buffer it (BUFG inferred, or the XDC generated-clock constraint).
    reg clk_div = 1'b0;
    always @(posedge clk100) clk_div <= ~clk_div;
    wire clk = clk_div;

    reg [2:0] rs = 3'b000;
    always @(posedge clk) rs <= {rs[1:0], btnC};
    wire rst_n = ~rs[2];          // btnC pressed => reset

    // ---------------------------------------------------------------- UART + RX FIFO
    wire rx_valid, rx_ferr; wire [7:0] rx_raw;
    reg  tx_start; reg [7:0] tx_data; wire tx_busy;
    uart_rx_s #(.DIV(BAUD_DIV)) u_rx (.clk(clk), .rst_n(rst_n), .rx(RsRx),
        .valid(rx_valid), .data(rx_raw), .frame_err(rx_ferr));
    uart_tx_s #(.DIV(BAUD_DIV)) u_tx (.clk(clk), .rst_n(rst_n),
        .start(tx_start), .data(tx_data), .tx(RsTx), .busy(tx_busy));

    wire [7:0] rx_data; wire fifo_empty, fifo_full, fifo_ovf;
    reg  fifo_rd;
    uart_rx_fifo #(.DEPTH(FIFO_D), .AW($clog2(FIFO_D))) u_fifo (
        .clk(clk), .rst_n(rst_n),
        .wr_en(rx_valid), .wr_data(rx_raw),
        .rd_en(fifo_rd), .rd_data(rx_data),
        .empty(fifo_empty), .full(fifo_full), .overflow(fifo_ovf));
    wire rx_take = !fifo_empty && fifo_rd; // data presented this cycle; rd_en registered in FIFO

    // ---------------------------------------------------------------- core
    reg core_init, core_st, core_si; reg [31:0] seed_r; reg [N_IN-1:0] x_reg; reg [3:0] lab_r; reg [5:0] t_r;
    reg h_we1, h_we2; reg [AW1-1:0] h_addr1; reg [AW2-1:0] h_addr2; reg [RB-1:0] rowbuf;
    wire core_busy, core_done; wire [3:0] core_pred; wire [31:0] nupd, nb_fx;
    wire [RB-1:0] h_rd1; wire [N_OUT*W-1:0] h_rd2;
    sena_learn_core #(.N_IN(N_IN), .N_HID(N_HID), .N_OUT(N_OUT), .W(W), .SH(SH), .H_LANES(H_LANES)) u_core (
        .clk(clk), .rst_n(rst_n), .init(core_init), .seed_in(seed_r),
        .start_train(core_st), .start_infer(core_si), .x_bits(x_reg), .label(lab_r), .t_passes(t_r),
        .busy(core_busy), .done(core_done), .pred(core_pred), .n_w1_updates(nupd), .nb_fx_out(nb_fx),
        .host_we1(h_we1), .host_we2(h_we2), .host_addr1(h_addr1), .host_addr2(h_addr2),
        .host_wdata1(rowbuf), .host_wdata2(rowbuf[RB-1 -: N_OUT*W]),
        .host_rdata1(h_rd1), .host_rdata2(h_rd2));

    // ---------------------------------------------------------------- command FSM
    localparam [5:0]
        S_IDLE=0, S_HDR=1, S_DISPATCH=2, S_TX=3, S_TXW=4, S_RESP=5,
        S_L_DATA=14, S_L_WR=15, S_D_RD=16, S_D_LATCH=17, S_D_SEND=18, S_D_SUM=19, S_D_SUM2=20,
        S_T_LOOP=21, S_STREAM=22, S_STREAM_B=23, S_T_RUN=25, S_T_WAIT=26,
        S_E_LOOP=27, S_E_RUN=28, S_E_WAIT=29, S_E_SEND=30, S_DONE_T=31, S_DONE_E=32, S_D_WAIT=33;

    reg [5:0]  st, ret_tx, ret_fetch;
    reg [7:0]  op;
    reg [7:0]  hb [0:5];
    reg [2:0]  hcnt, hneed;
    reg [7:0]  resp [0:12];
    reg [3:0]  ridx, rlen;
    reg [7:0]  snd;
    reg [15:0] sum16;
    reg [7:0]  sel; reg [7:0] nbytes, bcnt;
    reg [23:0] s_cur, cnt_left;
    reg [15:0] correct;
    reg [31:0] core_cyc, tot_cyc, samples;
    reg        mode_T, mode_E, mode_W, unk;
    reg [7:0]  ibyte;                 // 0..99 during image stream
    reg        lo_have; reg [7:0] lo_byte;
    integer i;

    wire [23:0] f_a3 = {hb[2], hb[1], hb[0]};
    // FWFT FIFO: rd_data is combinational from rptr.  fifo_rd is registered and
    // is observed by the FIFO on the NEXT edge, which also increments rptr.
    // Consume only when !fifo_rd so the same byte cannot be sampled twice on
    // the edge that pops the FIFO (empty and rptr update with nonblocking assigns).
    wire take = !fifo_empty && !fifo_rd;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st <= S_IDLE; tx_start <= 0; tx_data <= 0; fifo_rd <= 0;
            core_init <= 0; core_st <= 0; core_si <= 0; seed_r <= 0; x_reg <= 0; lab_r <= 0; t_r <= 6'd40;
            h_we1 <= 0; h_we2 <= 0; h_addr1 <= 0; h_addr2 <= 0; rowbuf <= 0;
            hcnt <= 0; hneed <= 0; ridx <= 0; rlen <= 0; snd <= 0; sum16 <= 0;
            sel <= 0; nbytes <= 0; bcnt <= 0; s_cur <= 0; cnt_left <= 0;
            correct <= 0; core_cyc <= 0; tot_cyc <= 0; samples <= 0; mode_T <= 0; mode_E <= 0; mode_W <= 0;
            unk <= 0; op <= 0; ret_tx <= 0; ret_fetch <= 0;
            ibyte <= 0; lo_have <= 0; lo_byte <= 0;
            for (i = 0; i < 6; i = i + 1) hb[i] <= 0;
            for (i = 0; i < 13; i = i + 1) resp[i] <= 0;
        end else begin
            tx_start <= 1'b0; core_init <= 1'b0; core_st <= 1'b0; core_si <= 1'b0;
            h_we1 <= 1'b0; h_we2 <= 1'b0;
            fifo_rd <= 1'b0;
            if (mode_T || mode_E) begin
                tot_cyc <= tot_cyc + 1;
                if (core_busy) core_cyc <= core_cyc + 1;
            end
            case (st)
            S_IDLE: if (take) begin
                fifo_rd <= 1'b1;
                op <= rx_data; hcnt <= 0;
                case (rx_data)
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
            S_HDR: if (take) begin
                fifo_rd <= 1'b1;
                hb[hcnt] <= rx_data; hcnt <= hcnt + 1'b1;
                if (hcnt + 1'b1 == hneed) st <= S_DISPATCH;
            end
            S_DISPATCH: begin
                case (op)
                "S": begin seed_r <= {hb[3], hb[2], hb[1], hb[0]}; core_init <= 1'b1; resp[0] <= "K"; rlen <= 1; ridx <= 0; st <= S_RESP; end
                "L": begin sel <= hb[0]; h_addr1 <= {hb[2], hb[1]}; h_addr2 <= {hb[2], hb[1]};
                           nbytes <= (hb[0] == 8'd1) ? N_HID[7:0] : 8'd10; bcnt <= 0; sum16 <= 0; st <= S_L_DATA; end
                "D": begin sel <= hb[0]; h_addr1 <= {hb[2], hb[1]}; h_addr2 <= {hb[2], hb[1]};
                           nbytes <= (hb[0] == 8'd1) ? N_HID[7:0] : 8'd10; bcnt <= 0; sum16 <= 0; st <= S_D_WAIT; end
                "T": begin s_cur <= f_a3; cnt_left <= {hb[5], hb[4], hb[3]}; core_cyc <= 0; tot_cyc <= 0; mode_T <= 1; st <= S_T_LOOP; end
                "E": begin cnt_left <= {8'd0, hb[3], hb[2]}; t_r <= hb[4][5:0]; correct <= 0;
                           core_cyc <= 0; tot_cyc <= 0; mode_E <= 1; st <= S_E_LOOP; end
                default: st <= S_IDLE;
                endcase
            end
            S_TX:  if (!tx_busy && !tx_start) begin tx_data <= snd; tx_start <= 1'b1; st <= S_TXW; end
            S_TXW: st <= ret_tx;
            S_RESP: if (ridx < rlen) begin snd <= resp[ridx]; ridx <= ridx + 1'b1; ret_tx <= S_RESP; st <= S_TX; end
                    else st <= S_IDLE;
            S_L_DATA: if (bcnt == nbytes) st <= S_L_WR;
                      else if (take) begin
                          fifo_rd <= 1'b1;
                          rowbuf <= {rx_data, rowbuf[RB-1:8]}; sum16 <= sum16 + rx_data; bcnt <= bcnt + 1'b1;
                      end
            S_L_WR: begin
                if (sel == 8'd1) h_we1 <= 1'b1; else h_we2 <= 1'b1;
                resp[0] <= "K"; resp[1] <= sum16[7:0]; resp[2] <= sum16[15:8]; rlen <= 3; ridx <= 0; st <= S_RESP;
            end
            S_D_WAIT:  st <= S_D_RD;
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

            // ---- stream one 100-byte image into x_reg / lab_r, then ret_fetch
            S_STREAM: begin ibyte <= 0; lo_have <= 0; mode_W <= 1; st <= S_STREAM_B; end
            S_STREAM_B: begin
                if (ibyte == IMG_BYTES[7:0]) begin
                    mode_W <= 0; st <= ret_fetch;
                end else if (take) begin
                    fifo_rd <= 1'b1;
                    if (!lo_have) begin lo_byte <= rx_data; lo_have <= 1'b1; ibyte <= ibyte + 1'b1; end
                    else begin
                        lo_have <= 1'b0;
                        // word index = ibyte[7:1]; words 0..48 -> x_reg; word 49 label is the LE low byte
                        if (ibyte == 8'd99) lab_r <= lo_byte[3:0];
                        else                 x_reg[{ibyte[7:1], 4'b0000} +: 16] <= {rx_data, lo_byte};
                        ibyte <= ibyte + 1'b1;
                    end
                end
            end

            S_T_LOOP: begin
                if (cnt_left == 0) st <= S_DONE_T;
                else begin ret_fetch <= S_T_RUN; st <= S_STREAM; end
            end
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
            S_E_LOOP: begin
                if (cnt_left == 0) st <= S_DONE_E;
                else begin ret_fetch <= S_E_RUN; st <= S_STREAM; end
            end
            S_E_RUN:  begin core_si <= 1'b1; st <= S_E_WAIT; end
            S_E_WAIT: if (core_done) begin
                          if (core_pred == lab_r) correct <= correct + 1'b1;
                          snd <= {4'd0, core_pred}; ret_tx <= S_E_SEND; st <= S_TX;
                      end
            S_E_SEND: begin cnt_left <= cnt_left - 1'b1; st <= S_E_LOOP; end
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
    reg [25:0] hb_cnt; reg [21:0] rx_led; reg ferr_seen;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin hb_cnt <= 0; rx_led <= 0; ferr_seen <= 0; end
        else begin
            hb_cnt <= hb_cnt + 1'b1;
            if (rx_valid) rx_led <= {22{1'b1}}; else if (rx_led != 0) rx_led <= rx_led - 1'b1;
            if (rx_ferr) ferr_seen <= 1'b1;
        end
    end

    // 4-digit muxed 7-seg, active-low anodes and segments, ~763 Hz at 50 MHz
    reg [15:0] seg_div; reg [1:0] dig;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin seg_div <= 0; dig <= 0; end
        else begin
            seg_div <= seg_div + 1'b1;
            if (seg_div == 16'd0) dig <= dig + 2'd1;
        end
    end
    function [6:0] seg7;
        input [3:0] v;
        case (v)
            4'h0: seg7 = 7'b1000000; 4'h1: seg7 = 7'b1111001; 4'h2: seg7 = 7'b0100100; 4'h3: seg7 = 7'b0110000;
            4'h4: seg7 = 7'b0011001; 4'h5: seg7 = 7'b0010010; 4'h6: seg7 = 7'b0000010; 4'h7: seg7 = 7'b1111000;
            4'h8: seg7 = 7'b0000000; 4'h9: seg7 = 7'b0010000; 4'hA: seg7 = 7'b0001000; 4'hB: seg7 = 7'b0000011;
            4'hC: seg7 = 7'b1000110; 4'hD: seg7 = 7'b0100001; 4'hE: seg7 = 7'b0000110; default: seg7 = 7'b0001110;
        endcase
    endfunction
    wire [3:0] nib0 = samples[3:0], nib1 = samples[7:4], nib2 = samples[11:8], nib3 = samples[15:12];
    assign seg = (dig == 2'd0) ? seg7(nib0) :
                 (dig == 2'd1) ? seg7(nib1) :
                 (dig == 2'd2) ? seg7(nib2) : seg7(nib3);
    assign an  = (dig == 2'd0) ? 4'b1110 :
                 (dig == 2'd1) ? 4'b1101 :
                 (dig == 2'd2) ? 4'b1011 : 4'b0111;
    assign dp  = 1'b1;

    assign led = {ferr_seen, unk, 6'd0, (rx_led != 0), 1'b0, fifo_ovf, mode_W, mode_E, mode_T, core_busy, hb_cnt[25]};
endmodule
