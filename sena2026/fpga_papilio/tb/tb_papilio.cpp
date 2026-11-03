// =============================================================================
// tb_papilio.cpp — Verilator testbench for papilio_calib_top
// Drives real UART waveforms into rx and decodes tx, exactly like the FT2232
// will on the board. Build with CLKS_PER_BIT=16 to keep sim fast.
// =============================================================================
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "Vpapilio_calib_top.h"
#include "verilated.h"

static Vpapilio_calib_top* dut;
static uint64_t t = 0;
static const int CPB = 16;   // must match -GCLKS_PER_BIT

double sc_time_stamp() { return (double)t; }

static void tick() {
    dut->clk = 0; dut->eval();
    dut->clk = 1; dut->eval();
    t++;
}

static void uart_send_byte(uint8_t b) {
    dut->rx = 0;                       // start
    for (int i = 0; i < CPB; i++) tick();
    for (int bit = 0; bit < 8; bit++) {
        dut->rx = (b >> bit) & 1;
        for (int i = 0; i < CPB; i++) tick();
    }
    dut->rx = 1;                       // stop
    for (int i = 0; i < CPB; i++) tick();
    for (int i = 0; i < CPB / 2; i++) tick();   // inter-byte gap
}

static int uart_recv_byte(uint64_t timeout_cycles) {
    uint64_t start = t;
    while (dut->tx == 1) {
        tick();
        if (t - start > timeout_cycles) return -1;
    }
    // start bit found; sample mid-bit
    for (int i = 0; i < CPB / 2; i++) tick();
    uint8_t v = 0;
    for (int bit = 0; bit < 8; bit++) {
        for (int i = 0; i < CPB; i++) tick();
        v |= (dut->tx & 1) << bit;
    }
    for (int i = 0; i < CPB; i++) tick();       // stop bit
    return v;
}

static bool expect(int got, int want, const char* what) {
    if (got != want) {
        printf("FAIL %s: got %d (0x%02x) want %d\n", what, got, got, want);
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    dut = new Vpapilio_calib_top;
    dut->rx = 1;
    for (int i = 0; i < 400; i++) tick();       // power-on reset

    int fails = 0;

    // ---- V: version --------------------------------------------------------
    uart_send_byte('V');
    if (!expect(uart_recv_byte(100000), 0xA5, "V id")) fails++;
    if (!expect(uart_recv_byte(100000), 0x01, "V ver")) fails++;

    // ---- N: LFSR ranges ----------------------------------------------------
    uart_send_byte('N');
    uart_send_byte(9);
    uart_send_byte(8);
    if (!expect(uart_recv_byte(100000), 0x4E, "N ack")) fails++;

    // ---- X: load a sparse test image (every 5th pixel on) ------------------
    uart_send_byte('X');
    for (int b = 0; b < 98; b++) {
        uint8_t byte = 0;
        for (int bit = 0; bit < 8; bit++) {
            int px = b * 8 + bit;
            if (px < 784 && (px % 5) == 0) byte |= (1 << bit);
        }
        uart_send_byte(byte);
    }
    if (!expect(uart_recv_byte(100000), 0x58, "X ack")) fails++;

    // ---- R: fixed 4 passes, no early exit ----------------------------------
    uart_send_byte('R');
    uart_send_byte(4);
    uart_send_byte(0);
    int pred   = uart_recv_byte(4000000);
    int passes = uart_recv_byte(400000);
    uint32_t cyc = 0;
    for (int i = 0; i < 4; i++) {
        int b = uart_recv_byte(400000);
        if (b < 0) { fails++; printf("FAIL cycles byte %d timeout\n", i); b = 0; }
        cyc |= (uint32_t)b << (8 * i);
    }
    printf("run1: pred=%d passes=%d cycles=%u cycles/pass=%.2f\n",
           pred, passes, cyc, passes ? (double)cyc / passes : 0.0);
    if (!expect(passes, 4, "fixed passes")) fails++;
    // model: (784+2)+1+(16+2)+1+1 = 805 per pass, small FSM overhead allowed
    double cpp = (double)cyc / 4.0;
    if (cpp < 780 || cpp > 830) {
        printf("FAIL cycles/pass %.2f outside [780, 830]\n", cpp);
        fails++;
    }

    // ---- R: early exit at T_max=40 ------------------------------------------
    uart_send_byte('R');
    uart_send_byte(40);
    uart_send_byte(1);
    int pred2   = uart_recv_byte(40000000);
    int passes2 = uart_recv_byte(400000);
    uint32_t cyc2 = 0;
    for (int i = 0; i < 4; i++) {
        int b = uart_recv_byte(400000);
        if (b < 0) { fails++; b = 0; }
        cyc2 |= (uint32_t)b << (8 * i);
    }
    printf("run2: pred=%d passes=%d cycles=%u (early exit)\n",
           pred2, passes2, cyc2);
    if (passes2 < 1 || passes2 > 40) { printf("FAIL ee passes\n"); fails++; }

    printf(fails ? "TB FAILED (%d)\n" : "TB PASSED\n", fails);
    delete dut;
    return fails ? 1 : 0;
}
