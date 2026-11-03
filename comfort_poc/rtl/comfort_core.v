// =============================================================================
// comfort_core.v — 16-bit features → 16 hid → 1 out, stochastic p_comfort
// x = {RH[7:0], T[7:0]} bit features; Y=1[acc>(K+R)]; p̂ = ones/T_passes
// Weights: int8 BRAM, member-selected (M=3 banks)
// =============================================================================
`timescale 1ns/1ps
module comfort_core #(
    parameter integer N_IN   = 16,
    parameter integer N_HID  = 16,
    parameter integer W_BITS = 8,
    parameter integer ACC_BITS = 20,
    parameter W_HID0 = "w_hid_m0.mem",
    parameter W_OUT0 = "w_out_m0.mem",
    parameter W_HID1 = "w_hid_m1.mem",
    parameter W_OUT1 = "w_out_m1.mem",
    parameter W_HID2 = "w_hid_m2.mem",
    parameter W_OUT2 = "w_out_m2.mem"
) (
    input  wire            clk,
    input  wire            rst_n,
    input  wire            start,
    input  wire [1:0]      member,       // 0..2
    input  wire [15:0]     t_passes,     // e.g. 100 or 1000
    input  wire [3:0]      n_log2,
    input  wire [7:0]      t_u8,
    input  wire [7:0]      rh_u8,
    output reg             busy,
    output reg             done,
    output reg  [15:0]     ones_count,
    output reg  [15:0]     passes_done,
    output reg  [31:0]     run_cycles
);
    (* ram_style = "block" *) reg [N_HID*W_BITS-1:0] ram_hid0 [0:N_IN];
    (* ram_style = "block" *) reg [N_HID*W_BITS-1:0] ram_hid1 [0:N_IN];
    (* ram_style = "block" *) reg [N_HID*W_BITS-1:0] ram_hid2 [0:N_IN];
    (* ram_style = "block" *) reg [W_BITS-1:0]       ram_out0 [0:N_HID];
    (* ram_style = "block" *) reg [W_BITS-1:0]       ram_out1 [0:N_HID];
    (* ram_style = "block" *) reg [W_BITS-1:0]       ram_out2 [0:N_HID];
    initial begin
        $readmemh(W_HID0, ram_hid0);
        $readmemh(W_HID1, ram_hid1);
        $readmemh(W_HID2, ram_hid2);
        $readmemh(W_OUT0, ram_out0);
        $readmemh(W_OUT1, ram_out1);
        $readmemh(W_OUT2, ram_out2);
    end

    reg [5:0]  idx;
    reg [4:0]  hid_addr;
    reg [4:0]  out_addr;
    reg [N_HID*W_BITS-1:0] hid_do;
    reg [W_BITS-1:0] out_do;
    always @(posedge clk) begin
        case (member)
            2'd0: hid_do <= ram_hid0[hid_addr];
            2'd1: hid_do <= ram_hid1[hid_addr];
            default: hid_do <= ram_hid2[hid_addr];
        endcase
        case (member)
            2'd0: out_do <= ram_out0[out_addr];
            2'd1: out_do <= ram_out1[out_addr];
            default: out_do <= ram_out2[out_addr];
        endcase
    end

    localparam [31:0] TAPS = 32'h8020_0003;
    reg [31:0] lfsr_h [0:N_HID-1];
    reg [31:0] lfsr_o;
    reg [31:0] cyc_free, cyc_start;

    function signed [ACC_BITS-1:0] sext8;
        input signed [7:0] w;
        begin sext8 = {{(ACC_BITS-8){w[7]}}, w}; end
    endfunction

    function signed [ACC_BITS-1:0] rnd_of;
        input [31:0] r;
        input [3:0] nl2;
        reg [16:0] half, mask, bits;
        begin
            half = 17'd1 << nl2;
            mask = (half << 1) - 17'd1;
            bits = r[16:0] & mask;
            rnd_of = $signed({{(ACC_BITS-18){1'b0}}, 1'b0, bits})
                   - $signed({{(ACC_BITS-18){1'b0}}, 1'b0, half});
        end
    endfunction

    wire [N_IN-1:0] x_bits = {rh_u8, t_u8};

    localparam S_IDLE=3'd0, S_HID_A=3'd1, S_HID_B=3'd2, S_HFIRE=3'd3,
               S_OUT_A=3'd4, S_OUT_B=3'd5, S_OFIRE=3'd6;
    reg [2:0] state;
    reg bit_d;
    reg [N_HID-1:0] h_bits;
    reg signed [ACC_BITS-1:0] acc_h [0:N_HID-1];
    reg signed [ACC_BITS-1:0] acc_o;
    reg y_bit;
    reg [15:0] pass_i;
    integer k;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            busy <= 0; done <= 0;
            ones_count <= 0; passes_done <= 0; run_cycles <= 0;
            cyc_free <= 0; cyc_start <= 0;
            hid_addr <= 0; out_addr <= 0;
            idx <= 0; bit_d <= 0; h_bits <= 0; acc_o <= 0; y_bit <= 0;
            pass_i <= 0; lfsr_o <= 32'hC0FF_EE01;
            for (k = 0; k < N_HID; k = k + 1) begin
                lfsr_h[k] <= 32'hA5A5_0001 + k * 32'h0101_0001;
                acc_h[k] <= 0;
            end
        end else begin
            done <= 0;
            cyc_free <= cyc_free + 1;
            if (busy) begin
                for (k = 0; k < N_HID; k = k + 1)
                    lfsr_h[k] <= {1'b0, lfsr_h[k][31:1]}
                                 ^ (lfsr_h[k][0] ? TAPS : 32'b0);
                lfsr_o <= {1'b0, lfsr_o[31:1]} ^ (lfsr_o[0] ? TAPS : 32'b0);
            end

            case (state)
                S_IDLE: if (start) begin
                    busy <= 1;
                    ones_count <= 0;
                    passes_done <= 0;
                    pass_i <= 0;
                    cyc_start <= cyc_free;
                    for (k = 0; k < N_HID; k = k + 1) acc_h[k] <= 0;
                    idx <= 0; hid_addr <= 0;
                    state <= S_HID_A;
                end

                S_HID_A: begin
                    hid_addr <= idx[4:0];
                    if (idx < N_IN)
                        bit_d <= x_bits[idx[3:0]];
                    else
                        bit_d <= 1'b1; // bias addr N_IN
                    state <= S_HID_B;
                end
                S_HID_B: begin
                    if (bit_d) begin
                        for (k = 0; k < N_HID; k = k + 1)
                            acc_h[k] <= acc_h[k] + sext8(hid_do[k*W_BITS +: W_BITS]);
                    end
                    if (idx == N_IN) begin
                        state <= S_HFIRE;
                    end else begin
                        idx <= idx + 1;
                        state <= S_HID_A;
                    end
                end

                S_HFIRE: begin
                    for (k = 0; k < N_HID; k = k + 1)
                        h_bits[k] <= (acc_h[k] > rnd_of(lfsr_h[k], n_log2));
                    acc_o <= 0;
                    idx <= 0;
                    state <= S_OUT_A;
                end

                S_OUT_A: begin
                    out_addr <= idx[4:0];
                    if (idx < N_HID)
                        bit_d <= h_bits[idx[3:0]];
                    else
                        bit_d <= 1'b1; // bias
                    state <= S_OUT_B;
                end
                S_OUT_B: begin
                    if (bit_d)
                        acc_o <= acc_o + sext8(out_do);
                    if (idx == N_HID) begin
                        state <= S_OFIRE;
                    end else begin
                        idx <= idx + 1;
                        state <= S_OUT_A;
                    end
                end

                S_OFIRE: begin
                    y_bit <= (acc_o > rnd_of(lfsr_o, n_log2));
                    if (acc_o > rnd_of(lfsr_o, n_log2))
                        ones_count <= ones_count + 16'd1;
                    passes_done <= passes_done + 16'd1;
                    pass_i <= pass_i + 16'd1;
                    if (pass_i + 16'd1 >= t_passes) begin
                        busy <= 0;
                        done <= 1;
                        run_cycles <= cyc_free - cyc_start;
                        state <= S_IDLE;
                    end else begin
                        for (k = 0; k < N_HID; k = k + 1) acc_h[k] <= 0;
                        idx <= 0;
                        state <= S_HID_A;
                    end
                end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
