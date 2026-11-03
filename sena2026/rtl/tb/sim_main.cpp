// =============================================================================
// sim_main.cpp -- Verilator testbench for sena_tile
// -----------------------------------------------------------------------------
// Reads a plain-text stimulus file (weights + input patterns), drives the tile
// pass-by-pass, and reports, for every sample:
//     pred  passes  exited  cycles_total  cycles_first_pass  votes...
// The cycle counts are what the calibrated speed model in sim_fpga_cycles.py is
// validated against.
// =============================================================================
#include <verilated.h>
#include "Vsena_tile.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static Vsena_tile *top = nullptr;
static uint64_t g_cycles = 0;

double sc_time_stamp() { return (double)g_cycles; }

static void tick() {
    top->clk = 0;
    top->eval();
    top->clk = 1;
    top->eval();
    ++g_cycles;
}

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    if (argc < 3) {
        std::cerr << "usage: sim <stimulus.txt> <out.txt>\n";
        return 2;
    }
    std::ifstream fin(argv[1]);
    if (!fin) { std::cerr << "cannot open " << argv[1] << "\n"; return 2; }

    int N_IN, N_HID, N_OUT, T_MAX, PAR, PAR_OUT;
    fin >> N_IN >> N_HID >> N_OUT >> T_MAX >> PAR >> PAR_OUT;
    int K_hid, K_out, nlog2_hid, nlog2_out;
    fin >> K_hid >> K_out >> nlog2_hid >> nlog2_out;

    std::vector<int> w_hid(N_HID * N_IN), b_hid(N_HID),
                     w_out(N_OUT * N_HID), b_out(N_OUT);
    for (auto &v : w_hid) fin >> v;
    for (auto &v : b_hid) fin >> v;
    for (auto &v : w_out) fin >> v;
    for (auto &v : b_out) fin >> v;

    int NS; fin >> NS;
    std::vector<uint64_t> xs(NS);
    for (int s = 0; s < NS; ++s) {
        uint64_t bits = 0;
        for (int i = 0; i < N_IN; ++i) { int b; fin >> b; if (b) bits |= (1ULL << i); }
        xs[s] = bits;
    }
    fin.close();

    top = new Vsena_tile;

    for (int i = 0; i < N_HID * N_IN; ++i) top->w_hid[i] = (uint16_t)(int16_t)w_hid[i];
    for (int i = 0; i < N_HID; ++i)        top->b_hid[i] = (uint16_t)(int16_t)b_hid[i];
    for (int i = 0; i < N_OUT * N_HID; ++i) top->w_out[i] = (uint16_t)(int16_t)w_out[i];
    for (int i = 0; i < N_OUT; ++i)        top->b_out[i] = (uint16_t)(int16_t)b_out[i];
    top->K_hid = (uint16_t)(int16_t)K_hid;
    top->K_out = (uint16_t)(int16_t)K_out;
    top->n_log2_hid = (uint8_t)nlog2_hid;
    top->n_log2_out = (uint8_t)nlog2_out;

    // ---- reset ----------------------------------------------------------
    top->rst_n = 0; top->sample_start = 0; top->pass_start = 0; top->x_bits = 0;
    for (int i = 0; i < 5; ++i) tick();
    top->rst_n = 1;
    tick();

    std::ofstream fout(argv[2]);
    fout << "# sample pred passes exited cycles_total cycles_pass1 votes\n";

    for (int s = 0; s < NS; ++s) {
        top->x_bits = xs[s];
        top->sample_start = 1; tick();
        top->sample_start = 0; tick();

        uint64_t c0 = g_cycles;
        uint64_t cyc_pass1 = 0;
        int passes_seen = 0;

        while (passes_seen < T_MAX) {
            uint64_t p0 = g_cycles;
            top->pass_start = 1; tick();
            top->pass_start = 0;
            // run until the tile returns to idle
            int guard = 0;
            while (top->busy && guard < 100000) { tick(); ++guard; }
            ++passes_seen;
            if (passes_seen == 1) cyc_pass1 = g_cycles - p0;
            if (top->exit_ready || top->done_force) break;
        }

        uint64_t cyc_total = g_cycles - c0;
        fout << s << " " << (int)top->pred << " " << (int)top->passes_used << " "
             << (int)(top->exit_ready ? 1 : 0) << " " << cyc_total << " " << cyc_pass1;
        for (int c = 0; c < N_OUT; ++c) fout << " " << (int)top->votes_dbg[c];
        fout << "\n";
    }

    fout.close();
    top->final();
    delete top;
    std::cout << "verilator sim done, total clock cycles = " << g_cycles << "\n";
    return 0;
}
