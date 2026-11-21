// =============================================================================
// sena_learn_core.v -- SENA on-chip learning core, hardware rule v1 (one member)
// -----------------------------------------------------------------------------
// N_IN -> N_HID -> N_OUT stochastic network, W-bit saturating weights in two
// simple-dual-port M9K RAMs, multiplier-free (adders, comparators, XOR LFSRs).
//   W1 RAM : depth N_IN+1  (row i = input i, row N_IN = bias), width N_HID*W
//   W2 RAM : depth N_HID+1 (row j = hidden j, row N_HID = bias), width N_OUT*W
// All N_HID (resp. N_OUT) lanes are processed in parallel; rows are swept one per
// clock with a 1-cycle registered read.  The rule is specified in
// twin/sena_hw_twin.py, which is the bit-exact reference for this RTL.
//
// OFF-BY-ONE GUARD (the LX9 mnist_online_core bug class): the RAM read address
// is driven COMBINATIONALLY from the sweep counter `cnt`; the RAM captures
// mem[cnt] at edge e and, at the SAME edge e, the core registers the tag
// (tag_d = cnt, gb_d = gating bit of row cnt, v_d = 1).  Data and tag are
// therefore consumed together at edge e+1.  In simulation every consumed row is
// checked (tag_d == raddr_q of the RAM), and the RAM aborts on any same-address
// read-during-write.
//
// Randomness: one 32-bit Galois LFSR per lane (taps 0x80200003).  A draw is 32
// single steps, executed one step per clock in the background (step counter);
// a FIRE state waits until all 32 steps are done, uses the state, captures the
// gate fields, and restarts the counter.  Fields: [6:0] forward threshold,
// [13:8] r1/r2 sensitivity gate, [31:16] r3 back-signal gate.
//
// Interface rules: x_bits and label must be held stable while busy.  Host weight
// port is honoured only in S_IDLE.  `init` (in S_IDLE) loads LFSR seeds from
// seed_in and resets nb_fx.
// =============================================================================
`timescale 1ns/1ps
module sena_learn_core #(
    parameter integer N_IN   = 784,
    parameter integer N_HID  = 160,
    parameter integer N_OUT  = 10,
    parameter integer W      = 8,
    parameter integer ACC    = 20,
    parameter integer BK     = 16,
    parameter integer SH     = 7,      // mabs = sum|back| >> SH
    parameter integer T_BITS = 6,
    parameter         W1_INIT = "",
    parameter         W2_INIT = "",
    parameter integer AW1    = $clog2(N_IN + 1),   // derived: do not override
    parameter integer AW2    = $clog2(N_HID + 1)   // derived: do not override
) (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  init,
    input  wire [31:0]           seed_in,
    input  wire                  start_train,
    input  wire                  start_infer,
    input  wire [N_IN-1:0]       x_bits,
    input  wire [3:0]            label,
    input  wire [T_BITS-1:0]     t_passes,
    output reg                   busy,
    output reg                   done,
    output reg  [3:0]            pred,
    output reg  [31:0]           n_w1_updates,
    // host weight port (S_IDLE only)
    input  wire                  host_we1,
    input  wire                  host_we2,
    input  wire [AW1-1:0]        host_addr1,
    input  wire [AW2-1:0]        host_addr2,
    input  wire [N_HID*W-1:0]    host_wdata1,
    input  wire [N_OUT*W-1:0]    host_wdata2,
    output wire [N_HID*W-1:0]    host_rdata1,
    output wire [N_OUT*W-1:0]    host_rdata2
);
    localparam integer CW  = ((AW1 > AW2) ? AW1 : AW2) + 1;
    localparam integer WMAX = (1 << (W - 1)) - 1;
    localparam [31:0] TAPS = 32'h8020_0003;
    localparam [31:0] C_H_MUL = 32'h9E37_79B9;
    localparam [31:0] C_O_MUL = 32'h7F4A_7C15;
    localparam [31:0] C_O_XOR = 32'hA5A5_A5A5;

    localparam [3:0] S_IDLE = 4'd0, S_HSW = 4'd1, S_HFIRE = 4'd2, S_OSW = 4'd3,
                     S_OFIRE = 4'd4, S_W2U = 4'd5, S_NB = 4'd6, S_GATE = 4'd7,
                     S_W1U = 4'd8, S_ARGMAX = 4'd9, S_DONE = 4'd10;

    // ------------------------------------------------------------------ state
    reg [3:0]    state;
    reg          mode_train;
    reg [CW-1:0] cnt;
    reg          v_d;
    reg [CW-1:0] tag_d;
    reg          gb_d;

    // ------------------------------------------------------------------ RAMs
    wire          idle = (state == S_IDLE);
    reg           upd_we1, upd_we2;
    reg [AW1-1:0] upd_waddr1;
    reg [AW2-1:0] upd_waddr2;
    reg [N_HID*W-1:0] upd_wdata1;
    reg [N_OUT*W-1:0] upd_wdata2;

    wire sweep1 = (state == S_HSW) || (state == S_W1U);
    wire sweep2 = (state == S_OSW) || (state == S_W2U);
    wire [AW1-1:0] raddr1 = sweep1 ? ((cnt <= N_IN)  ? cnt[AW1-1:0] : {AW1{1'b0}}) : host_addr1;
    wire [AW2-1:0] raddr2 = sweep2 ? ((cnt <= N_HID) ? cnt[AW2-1:0] : {AW2{1'b0}}) : host_addr2;
    wire           we1    = upd_we1 | (idle & host_we1);
    wire           we2    = upd_we2 | (idle & host_we2);
    wire [AW1-1:0] waddr1 = upd_we1 ? upd_waddr1 : host_addr1;
    wire [AW2-1:0] waddr2 = upd_we2 ? upd_waddr2 : host_addr2;
    wire [N_HID*W-1:0] wdata1 = upd_we1 ? upd_wdata1 : host_wdata1;
    wire [N_OUT*W-1:0] wdata2 = upd_we2 ? upd_wdata2 : host_wdata2;
    wire [N_HID*W-1:0] rdata1;
    wire [N_OUT*W-1:0] rdata2;
    wire [AW1-1:0] raddr1_q;
    wire [AW2-1:0] raddr2_q;
    assign host_rdata1 = rdata1;
    assign host_rdata2 = rdata2;

    sdp_ram #(.DW(N_HID*W), .DEPTH(N_IN+1), .AW(AW1), .INIT(W1_INIT)) u_w1 (
        .clk(clk), .we(we1), .waddr(waddr1), .wdata(wdata1),
        .raddr(raddr1), .rdata(rdata1), .raddr_q(raddr1_q));
    sdp_ram #(.DW(N_OUT*W), .DEPTH(N_HID+1), .AW(AW2), .INIT(W2_INIT)) u_w2 (
        .clk(clk), .we(we2), .waddr(waddr2), .wdata(wdata2),
        .raddr(raddr2), .rdata(rdata2), .raddr_q(raddr2_q));

    // ------------------------------------------------------------------ lanes
    reg signed [ACC-1:0] acc_h [0:N_HID-1];
    reg [31:0]           lh    [0:N_HID-1];
    reg [5:0]            r2_c  [0:N_HID-1];
    reg [15:0]           r3_c  [0:N_HID-1];
    reg signed [BK-1:0]  back  [0:N_HID-1];
    reg [N_HID-1:0]      y, dj, dirp;
    reg [5:0]            hstep;

    reg signed [ACC-1:0] acc_o [0:N_OUT-1];
    reg [31:0]           lo    [0:N_OUT-1];
    reg signed [1:0]     dk    [0:N_OUT-1];
    reg signed [1:0]     gs    [0:N_OUT-1];
    reg [7:0]            votes [0:N_OUT-1];
    reg [5:0]            ostep;

    reg [31:0]           sum_abs;
    reg signed [31:0]    nb_fx;
    reg [T_BITS-1:0]     pass;

    // ------------------------------------------------------------------ helpers
    function [31:0] lfsr_step;
        input [31:0] s;
        begin lfsr_step = {1'b0, s[31:1]} ^ (s[0] ? TAPS : 32'd0); end
    endfunction

    function [31:0] seed_of;
        input [31:0] s;
        begin seed_of = (s == 32'd0) ? 32'd1 : s; end
    endfunction

    // y = (a + (b - 64)) > 0, b = 7-bit LFSR field
    function fire_bit;
        input signed [ACC-1:0] a;
        input [6:0] b;
        reg signed [ACC+1:0] s;
        begin
            s = a + $signed({1'b0, b}) - 64;
            fire_bit = (s > 0);
        end
    endfunction

    function [ACC-1:0] abs_acc;
        input signed [ACC-1:0] a;
        begin abs_acc = a[ACC-1] ? -a : a; end
    endfunction

    function [BK-1:0] abs_bk;
        input signed [BK-1:0] a;
        begin abs_bk = a[BK-1] ? -a : a; end
    endfunction

    // saturating w + dl, dl in {-1,0,+1}, range [-WMAX, WMAX]
    function [W-1:0] sat_add;
        input signed [W-1:0] w;
        input signed [1:0]   dl;
        reg signed [W+1:0] t;
        begin
            t = w + dl;
            if (t > WMAX)       sat_add = WMAX;
            else if (t < -WMAX-1) sat_add = -WMAX-1;   // MUTANT
            else                sat_add = t[W-1:0];
        end
    endfunction

    function [15:0] pow2_mask;   // largest power of two <= v, minus 1 (v=0 -> 0)
        input [31:0] v;
        reg [31:0] m;
        begin
            m = v;
            m = m | (m >> 1);  m = m | (m >> 2);  m = m | (m >> 4);
            m = m | (m >> 8);  m = m | (m >> 16);
            pow2_mask = m[16:1];
        end
    endfunction

    wire x_at_cnt = (cnt == N_IN) ? 1'b1 : ((cnt < N_IN) ? x_bits[cnt] : 1'b0);
    wire y_at_cnt = (cnt == N_HID) ? 1'b1 : ((cnt < N_HID) ? y[cnt] : 1'b0);

    // temporaries (blocking, local to the clocked block)
    integer j, k;
    reg signed [W-1:0]  wv;
    reg signed [1:0]    dl;
    reg signed [BK-1:0] bsum;
    reg [N_HID*W-1:0]   row1;
    reg [N_OUT*W-1:0]   row2;
    reg                 ob, anyd, av, bv;
    reg signed [31:0]   mabs;
    reg [15:0]          msk;
    reg [3:0]           best;
    reg [7:0]           bestv;

    // ------------------------------------------------------------------ main
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; mode_train <= 1'b0;
            cnt <= 0; v_d <= 1'b0; tag_d <= 0; gb_d <= 1'b0;
            busy <= 1'b0; done <= 1'b0; pred <= 4'd0; n_w1_updates <= 32'd0;
            upd_we1 <= 1'b0; upd_we2 <= 1'b0;
            upd_waddr1 <= 0; upd_waddr2 <= 0; upd_wdata1 <= 0; upd_wdata2 <= 0;
            y <= 0; dj <= 0; dirp <= 0; hstep <= 6'd32; ostep <= 6'd32;
            sum_abs <= 32'd0; nb_fx <= 32'sd384; pass <= 0;
            for (j = 0; j < N_HID; j = j + 1) begin
                acc_h[j] <= 0; lh[j] <= 32'd1; r2_c[j] <= 0; r3_c[j] <= 0; back[j] <= 0;
            end
            for (k = 0; k < N_OUT; k = k + 1) begin
                acc_o[k] <= 0; lo[k] <= 32'd1; dk[k] <= 0; gs[k] <= 0; votes[k] <= 0;
            end
        end else begin
            done <= 1'b0;
            upd_we1 <= 1'b0;
            upd_we2 <= 1'b0;

            // ---- background LFSR stepping: 32 steps per draw
            if (hstep != 6'd32) begin
                for (j = 0; j < N_HID; j = j + 1) lh[j] <= lfsr_step(lh[j]);
                hstep <= hstep + 6'd1;
            end
            if (ostep != 6'd32) begin
                for (k = 0; k < N_OUT; k = k + 1) lo[k] <= lfsr_step(lo[k]);
                ostep <= ostep + 6'd1;
            end

            case (state)
            // ------------------------------------------------------------
            S_IDLE: begin
                if (init) begin
                    for (j = 0; j < N_HID; j = j + 1)
                        lh[j] <= seed_of(seed_in ^ (C_H_MUL * (j + 1)));
                    for (k = 0; k < N_OUT; k = k + 1)
                        lo[k] <= seed_of(seed_in ^ (C_O_MUL * (k + 1)) ^ C_O_XOR);
                    hstep <= 6'd0; ostep <= 6'd0;
                    nb_fx <= 32'sd384;
                end else if (start_train || start_infer) begin
                    busy <= 1'b1;
                    mode_train <= start_train;
                    for (j = 0; j < N_HID; j = j + 1) acc_h[j] <= 0;
                    for (k = 0; k < N_OUT; k = k + 1) votes[k] <= 0;
                    pass <= 0;
                    cnt <= 0; v_d <= 1'b0;
                    state <= S_HSW;
                end
            end
            // ------------------------------------------------------------ hidden sweep
            S_HSW: begin
                if (cnt <= N_IN) begin
                    v_d <= 1'b1; tag_d <= cnt; gb_d <= x_at_cnt; cnt <= cnt + 1'b1;
                end else v_d <= 1'b0;
                if (v_d && gb_d)
                    for (j = 0; j < N_HID; j = j + 1)
                        acc_h[j] <= acc_h[j] + $signed(rdata1[j*W +: W]);
                if (cnt > N_IN && !v_d) state <= S_HFIRE;
            end
            // ------------------------------------------------------------ hidden fire
            S_HFIRE: begin
                if (hstep == 6'd32) begin
                    for (j = 0; j < N_HID; j = j + 1) begin
                        y[j]    <= fire_bit(acc_h[j], lh[j][6:0]);
                        r2_c[j] <= lh[j][13:8];
                        r3_c[j] <= lh[j][31:16];
                    end
                    hstep <= 6'd0;
                    for (k = 0; k < N_OUT; k = k + 1) acc_o[k] <= 0;
                    cnt <= 0; v_d <= 1'b0;
                    state <= S_OSW;
                end
            end
            // ------------------------------------------------------------ output sweep
            S_OSW: begin
                if (cnt <= N_HID) begin
                    v_d <= 1'b1; tag_d <= cnt; gb_d <= y_at_cnt; cnt <= cnt + 1'b1;
                end else v_d <= 1'b0;
                if (v_d && gb_d)
                    for (k = 0; k < N_OUT; k = k + 1)
                        acc_o[k] <= acc_o[k] + $signed(rdata2[k*W +: W]);
                if (cnt > N_HID && !v_d) state <= S_OFIRE;
            end
            // ------------------------------------------------------------ output fire
            S_OFIRE: begin
                if (ostep == 6'd32) begin
                    ostep <= 6'd0;
                    for (k = 0; k < N_OUT; k = k + 1) begin
                        ob = fire_bit(acc_o[k], lo[k][6:0]);
                        if (mode_train) begin
                            // d = t - o
                            if ((label == k) && !ob)      dl = 2'sd1;
                            else if ((label != k) && ob)  dl = -2'sd1;
                            else                          dl = 2'sd0;
                            dk[k] <= dl;
                            gs[k] <= ((dl != 0) && (abs_acc(acc_o[k]) <= lo[k][13:8])) ? dl : 2'sd0;
                        end else begin
                            votes[k] <= votes[k] + {7'd0, ob};
                        end
                    end
                    if (mode_train) begin
                        cnt <= 0; v_d <= 1'b0; sum_abs <= 32'd0;
                        state <= S_W2U;
                    end else begin
                        pass <= pass + 1'b1;
                        if (pass + 1'b1 == t_passes) state <= S_ARGMAX;
                        else                         state <= S_HFIRE;
                    end
                end
            end
            // ------------------------------------------------------------ W2 update + back-signal
            S_W2U: begin
                if (cnt <= N_HID) begin
                    v_d <= 1'b1; tag_d <= cnt; gb_d <= y_at_cnt; cnt <= cnt + 1'b1;
                end else v_d <= 1'b0;
                if (v_d) begin
                    bsum = 0;
                    for (k = 0; k < N_OUT; k = k + 1) begin
                        wv = rdata2[k*W +: W];
                        if (gs[k] == 2'sd1)       bsum = bsum + wv;   // pre-update weight
                        else if (gs[k] == -2'sd1) bsum = bsum - wv;
                        dl = gb_d ? dk[k] : -dk[k];
                        row2[k*W +: W] = sat_add(wv, dl);
                    end
                    upd_we2 <= 1'b1; upd_waddr2 <= tag_d[AW2-1:0]; upd_wdata2 <= row2;
                    if (tag_d < N_HID) begin
                        back[tag_d[AW2-1:0]] <= bsum;
                        sum_abs <= sum_abs + {{(32-BK){1'b0}}, abs_bk(bsum)};
                    end
                end
                if (cnt > N_HID && !v_d) state <= S_NB;
            end
            // ------------------------------------------------------------ N_back EMA (shift only)
            S_NB: begin
                mabs = $signed(sum_abs >> SH);
                nb_fx <= nb_fx + (((mabs <<< 4) - nb_fx) >>> 7);
                state <= S_GATE;
            end
            // ------------------------------------------------------------ hidden gates
            S_GATE: begin
                msk = pow2_mask(nb_fx >>> 4);
                anyd = 1'b0;
                for (j = 0; j < N_HID; j = j + 1) begin
                    av = (abs_acc(acc_h[j]) <= r2_c[j]);
                    bv = (abs_bk(back[j]) > (r3_c[j] & msk));
                    dj[j]   <= av & bv;
                    dirp[j] <= (back[j] > 0);
                    anyd = anyd | (av & bv);
                end
                cnt <= 0; v_d <= 1'b0;
                if (anyd) begin
                    n_w1_updates <= n_w1_updates + 32'd1;
                    state <= S_W1U;
                end else state <= S_DONE;
            end
            // ------------------------------------------------------------ W1 update
            S_W1U: begin
                if (cnt <= N_IN) begin
                    v_d <= 1'b1; tag_d <= cnt; gb_d <= x_at_cnt; cnt <= cnt + 1'b1;
                end else v_d <= 1'b0;
                if (v_d) begin
                    for (j = 0; j < N_HID; j = j + 1) begin
                        wv = rdata1[j*W +: W];
                        if (dj[j]) dl = (dirp[j] == gb_d) ? 2'sd1 : -2'sd1;
                        else       dl = 2'sd0;
                        row1[j*W +: W] = sat_add(wv, dl);
                    end
                    upd_we1 <= 1'b1; upd_waddr1 <= tag_d[AW1-1:0]; upd_wdata1 <= row1;
                end
                if (cnt > N_IN && !v_d) state <= S_DONE;
            end
            // ------------------------------------------------------------ argmax (first max)
            S_ARGMAX: begin
                best = 4'd0; bestv = votes[0];
                for (k = 1; k < N_OUT; k = k + 1)
                    if (votes[k] > bestv) begin best = k; bestv = votes[k]; end
                pred <= best;
                state <= S_DONE;
            end
            S_DONE: begin
                busy <= 1'b0; done <= 1'b1;
                state <= S_IDLE;
            end
            default: state <= S_IDLE;
            endcase
        end
    end

`ifndef SYNTHESIS
    // ---- alignment check: every consumed row must be the row whose tag we hold
    always @(posedge clk) begin
        if (v_d && sweep1 && (tag_d[AW1-1:0] != raddr1_q)) begin
            $display("CORE ERROR: W1 tag %0d != RAM row %0d at t=%0t", tag_d, raddr1_q, $time);
            $fatal(1);
        end
        if (v_d && sweep2 && (tag_d[AW2-1:0] != raddr2_q)) begin
            $display("CORE ERROR: W2 tag %0d != RAM row %0d at t=%0t", tag_d, raddr2_q, $time);
            $fatal(1);
        end
    end
`endif
endmodule
