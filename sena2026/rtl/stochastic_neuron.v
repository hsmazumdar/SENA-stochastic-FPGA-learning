// =============================================================================
// stochastic_neuron.v
// -----------------------------------------------------------------------------
// Mazumdar random-threshold neuron (Eq. 3):
//   acc = SUM (w_i if bit_i=1 else 0) + bias
//   Y   = 1 if (acc > (K + Rnd)) else 0
//
// Datapath ops: AND (bit-gate), ADD/SUB, COMPARE, RNG.  NO multiply.
// Weight width W_BITS (signed).  Input vector is 1-bit wide (binary fabric).
//
// PAR sets how many inputs are folded per clock:
//   PAR = 1   -> N_IN+2 cycles, 1 adder      (area-optimal)
//   PAR = 16  -> ceil(N_IN/16)+2 cycles, adder tree of 16 (throughput-optimal)
// =============================================================================
`timescale 1ns/1ps
module stochastic_neuron #(
    parameter integer N_IN   = 784,
    parameter integer W_BITS = 16,
    parameter integer ACC_BITS = W_BITS + 12,
    parameter integer PAR    = 1,
    // Random-threshold range is a POWER OF TWO: Rnd in [-2^n_log2, 2^n_log2-1].
    // A power-of-two range is a bare bit-slice of the LFSR (no divider, no
    // modulo), which is why the dynamic range N is held as log2 in a register.
    parameter integer N_LOG2_MAX = 15,
    parameter [31:0]  LFSR_SEED = 32'hACE10001
) (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  start,          // pulse: begin one presentation
    input  wire [N_IN-1:0]       bits,           // 1-bit inputs
    input  wire signed [W_BITS-1:0] weights [0:N_IN-1],
    input  wire signed [W_BITS-1:0] bias,
    input  wire signed [W_BITS-1:0] K,           // threshold offset
    input  wire [3:0]            n_log2,         // programmable range register
    output reg                   busy,
    output reg                   done,
    output reg                   y               // stochastic output bit
);
    localparam integer NSTEP    = (N_IN + PAR - 1) / PAR;
    localparam integer IDX_BITS = (NSTEP > 1) ? $clog2(NSTEP) : 1;
    localparam integer LOG2_PAR = (PAR > 1) ? $clog2(PAR) : 0;
    localparam integer IN_BITS  = (N_IN > 1) ? $clog2(N_IN) : 1;
    localparam integer FLAT_BITS = IDX_BITS + LOG2_PAR + 1;
    localparam [IDX_BITS-1:0]  LAST_STEP = IDX_BITS'(NSTEP - 1);
    localparam [FLAT_BITS-1:0] N_IN_C    = FLAT_BITS'(N_IN);
    localparam [FLAT_BITS-1:0] PAR_C     = FLAT_BITS'(PAR);

    reg  [IDX_BITS-1:0] step;
    reg  signed [ACC_BITS-1:0] acc;

    wire [31:0] rnd_raw;
    wire signed [ACC_BITS-1:0] rnd_scaled;
    wire signed [ACC_BITS-1:0] thr;

    lfsr32 #(.SEED(LFSR_SEED)) u_lfsr (
        .clk(clk), .rst_n(rst_n), .en(busy), .r(rnd_raw)
    );

    // Rnd = raw[n_log2:0] - 2^n_log2  ->  uniform in [-2^n_log2, 2^n_log2-1]
    wire [N_LOG2_MAX:0] rnd_half = {{N_LOG2_MAX{1'b0}}, 1'b1} << n_log2;
    wire [N_LOG2_MAX:0] rnd_bits = rnd_raw[N_LOG2_MAX:0] & ((rnd_half << 1) - 1'b1);
    assign rnd_scaled = $signed({{(ACC_BITS-N_LOG2_MAX-2){1'b0}}, 1'b0, rnd_bits})
                      - $signed({{(ACC_BITS-N_LOG2_MAX-2){1'b0}}, 1'b0, rnd_half});
    assign thr = $signed({{(ACC_BITS-W_BITS){K[W_BITS-1]}}, K}) + rnd_scaled;

    // ---- bit-gated partial sum over PAR lanes (synthesis infers adder tree) --
    // PAR must be a power of two so the lane offset is a shift, not a multiply.
    reg  [FLAT_BITS-1:0]       lane, flat;
    reg  signed [ACC_BITS-1:0] psum;
    wire [FLAT_BITS-1:0]       base = {{(FLAT_BITS-IDX_BITS){1'b0}}, step} << LOG2_PAR;
    always @* begin
        psum = {ACC_BITS{1'b0}};
        for (lane = 0; lane < PAR_C; lane = lane + 1'b1) begin
            flat = base + lane;
            if ((flat < N_IN_C) && bits[flat[IN_BITS-1:0]])
                psum = psum
                     + {{(ACC_BITS-W_BITS){weights[flat[IN_BITS-1:0]][W_BITS-1]}},
                        weights[flat[IN_BITS-1:0]]};
        end
    end

    wire signed [ACC_BITS-1:0] acc_next = acc + psum;
    wire last_step = (step == LAST_STEP);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy <= 1'b0;
            done <= 1'b0;
            y    <= 1'b0;
            step <= {IDX_BITS{1'b0}};
            acc  <= {ACC_BITS{1'b0}};
        end else begin
            done <= 1'b0;
            if (start && !busy) begin
                busy <= 1'b1;
                step <= {IDX_BITS{1'b0}};
                acc  <= {{(ACC_BITS-W_BITS){bias[W_BITS-1]}}, bias};
            end else if (busy) begin
                acc <= acc_next;
                if (last_step) begin
                    // compare the COMPLETE sum (acc_next) against K+Rnd
                    y    <= (acc_next > thr);
                    busy <= 1'b0;
                    done <= 1'b1;
                end else begin
                    step <= step + 1'b1;
                end
            end
        end
    end
endmodule
