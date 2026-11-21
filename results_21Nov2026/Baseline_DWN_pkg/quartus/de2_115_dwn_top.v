// =============================================================================
// de2_115_dwn_top.v - DWN baseline on-board self-test (SENA V13 matched baseline)
// Same board, clock and indicator conventions as SENA's de2_115_calib_top.v:
//   CLOCK_50 -> clk_25 (CLOCK_50/2), KEY0 = reset, KEY1 = re-run, LEDG0 heartbeat,
//   LEDR0 busy, LEDG[8:1] = PASS, LEDR17 = FAIL, HEX3..HEX0 = 4 decimal digits.
// The DWN core (gen/dwn_core.v, generated) is fed all DWN_N_TEST test vectors
// from an on-chip ROM (gen/test_x.hex), one per clock. Every RTL prediction is
// compared with the Python integer model (gen/test_exp.hex) and with the label.
// PASS  = 0 mismatches against Python (bit-exact equivalence on the board).
// KEY2 cycles the HEX display:
//   mode 0 (LEDR[7:6]=00): number of correct predictions vs labels (e.g. of 2500)
//   mode 1 (01): number of RTL-vs-Python mismatches (must be 0000)
//   mode 2 (10): cycles from first input to last output (= N_TEST-1+LATENCY)
//   mode 3 (11): measured pipeline latency in cycles (first in -> first out)
// =============================================================================
`timescale 1ns/1ps
module de2_115_dwn_top #(
    parameter X_FILE = "gen/test_x.hex",
    parameter E_FILE = "gen/test_exp.hex"
) (
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
    `include "dwn_params.vh"
    localparam integer AW = (DWN_N_TEST <= 2) ? 1 : $clog2(DWN_N_TEST);

    reg clk_25 = 1'b0;
    always @(posedge CLOCK_50) clk_25 <= ~clk_25;
    wire clk = clk_25;

    reg [7:0] por = 8'd0;
    wire rst_n = por[7] & KEY[0];
    always @(posedge clk)
        if (!por[7]) por <= por + 8'd1;

    // ---------------- test-vector ROMs (M9K) ----------------
    reg [DWN_N_IN-1:0] xrom [0:DWN_N_TEST-1];
    reg [7:0]          erom [0:DWN_N_TEST-1];
    initial begin
        $readmemh(X_FILE, xrom);
        $readmemh(E_FILE, erom);
    end
    reg  [AW-1:0]       raddr;
    reg  [DWN_N_IN-1:0] xq;
    reg  [7:0]          eq;
    always @(posedge clk) begin
        xq <= xrom[raddr];
        eq <= erom[raddr];
    end

    // ---------------- control ----------------
    localparam ST_ARM = 2'd0, ST_RUN = 2'd1, ST_DRAIN = 2'd2, ST_DONE = 2'd3;
    reg [1:0]  st;
    reg [15:0] arm_wait;
    reg        issue, issue_d;          // issue: ROM read this cycle; issue_d: data valid
    reg [AW:0] n_out;
    reg [15:0] cyc, t_first_out, t_last_out;
    reg        seen_out;
    reg [13:0] n_correct, n_mism;

    wire core_ov;
    wire [DWN_IW-1:0] core_pred;
    wire [DWN_K*DWN_CW-1:0] core_counts;
    dwn_core u_core (
        .clk(clk), .rst_n(rst_n), .in_valid(issue_d), .x(xq),
        .out_valid(core_ov), .pred(core_pred), .counts(core_counts));

    // expected {label,pred} delayed to line up with the core output
    reg [7:0] edl [0:DWN_LATENCY-1];
    integer q;
    always @(posedge clk) begin
        edl[0] <= eq;
        for (q = 1; q < DWN_LATENCY; q = q + 1) edl[q] <= edl[q-1];
    end
    wire [7:0] e_now = edl[DWN_LATENCY-1];

    reg key1_d, key2_d;
    wire key1_rise = key1_d & ~KEY[1];
    wire key2_rise = key2_d & ~KEY[2];
    reg [1:0] mode;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st <= ST_ARM; arm_wait <= 0; issue <= 0; issue_d <= 0; raddr <= 0;
            n_out <= 0; cyc <= 0; t_first_out <= 0; t_last_out <= 0; seen_out <= 0;
            n_correct <= 0; n_mism <= 0; key1_d <= 1'b1; key2_d <= 1'b1; mode <= 2'd0;
        end else begin
            key1_d <= KEY[1]; key2_d <= KEY[2];
            if (key2_rise) mode <= mode + 2'd1;
            issue_d <= issue;
            if (key1_rise) begin
                st <= ST_ARM; arm_wait <= 0; issue <= 0; raddr <= 0; n_out <= 0;
                cyc <= 0; seen_out <= 0; t_first_out <= 0; t_last_out <= 0; n_correct <= 0; n_mism <= 0;
            end else begin
                if (issue_d || st == ST_DRAIN) cyc <= cyc + 16'd1;
                case (st)
                    ST_ARM: if (arm_wait == 16'd20000) begin
                                st <= ST_RUN; issue <= 1'b1; raddr <= 0;
                            end else arm_wait <= arm_wait + 16'd1;
                    ST_RUN: if (raddr == DWN_N_TEST - 1) begin
                                issue <= 1'b0; st <= ST_DRAIN;
                            end else raddr <= raddr + 1'b1;
                    ST_DRAIN: if (n_out == DWN_N_TEST) st <= ST_DONE;
                    default: ;
                endcase
                if (core_ov) begin
                    n_out <= n_out + 1'b1;
                    t_last_out <= cyc;
                    if (!seen_out) begin seen_out <= 1'b1; t_first_out <= cyc; end
                    if (core_pred == e_now[7:4]) n_correct <= n_correct + 14'd1;
                    if (core_pred != e_now[3:0]) n_mism <= n_mism + 14'd1;
                end
            end
        end
    end

    wire done = (st == ST_DONE);
    wire pass = done && (n_mism == 0) && (t_first_out == DWN_LATENCY);
    wire fail = done && !pass;
    // cycles from the clock edge that samples input 0 to the edge that presents
    // the last output; expected = DWN_N_TEST - 1 + DWN_LATENCY (1 inference/clock)
    wire [15:0] total_cycles = t_last_out;

    // ---------------- display ----------------
    reg [13:0] disp;
    always @* case (mode)
        2'd0: disp = n_correct;
        2'd1: disp = n_mism;
        2'd2: disp = (total_cycles > 16'd9999) ? 14'd9999 : total_cycles[13:0];
        default: disp = t_first_out[13:0];
    endcase
    // binary -> BCD (double dabble), 14 bits -> 4 digits
    reg [29:0] dd;
    integer b;
    always @* begin
        dd = {16'd0, disp};
        for (b = 0; b < 14; b = b + 1) begin
            if (dd[17:14] >= 5) dd[17:14] = dd[17:14] + 4'd3;
            if (dd[21:18] >= 5) dd[21:18] = dd[21:18] + 4'd3;
            if (dd[25:22] >= 5) dd[25:22] = dd[25:22] + 4'd3;
            if (dd[29:26] >= 5) dd[29:26] = dd[29:26] + 4'd3;
            dd = dd << 1;
        end
    end
    wire [3:0] d0 = done ? dd[17:14] : 4'hF;
    wire [3:0] d1 = done ? dd[21:18] : 4'hF;
    wire [3:0] d2 = done ? dd[25:22] : 4'hF;
    wire [3:0] d3 = done ? dd[29:26] : 4'hF;

    reg [24:0] hb_div; reg hb;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin hb_div <= 0; hb <= 0; end
        else if (hb_div == 25'd12_499_999) begin hb_div <= 0; hb <= ~hb; end
        else hb_div <= hb_div + 25'd1;
    end

    reg [DWN_IW-1:0] last_pred;
    always @(posedge clk) if (core_ov) last_pred <= core_pred;

    assign UART_TXD    = 1'b1;          // UART idle (not used by this self-test)
    assign LEDR[0]     = (st == ST_RUN) || (st == ST_DRAIN);
    assign LEDR[4:1]   = last_pred[3:0];
    assign LEDR[5]     = 1'b0;
    assign LEDR[7:6]   = mode;
    assign LEDR[16:8]  = 9'd0;
    assign LEDR[17]    = fail;
    assign LEDG[0]     = hb;
    assign LEDG[8:1]   = pass ? 8'hFF : 8'h00;

    seg7 s0 (.d(d0), .seg(HEX0));
    seg7 s1 (.d(d1), .seg(HEX1));
    seg7 s2 (.d(d2), .seg(HEX2));
    seg7 s3 (.d(d3), .seg(HEX3));
endmodule

module seg7 (input wire [3:0] d, output reg [6:0] seg);
    always @* case (d)
        4'd0: seg = 7'b1000000; 4'd1: seg = 7'b1111001;
        4'd2: seg = 7'b0100100; 4'd3: seg = 7'b0110000;
        4'd4: seg = 7'b0011001; 4'd5: seg = 7'b0010010;
        4'd6: seg = 7'b0000010; 4'd7: seg = 7'b1111000;
        4'd8: seg = 7'b0000000; 4'd9: seg = 7'b0010000;
        4'hF: seg = 7'b0111111;        // '-' while running
        default: seg = 7'b1111111;
    endcase
endmodule
