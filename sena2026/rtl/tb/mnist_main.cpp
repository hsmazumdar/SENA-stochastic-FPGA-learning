// Hardware-faithful MNIST runner for a full-width Verilated sena_tile.
// This testbench is compiled only for N_IN=784, hence x_bits is VlWide<25>.
#include <verilated.h>
#include "Vsena_tile.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

static Vsena_tile *top = nullptr;
static uint64_t cycles = 0;
double sc_time_stamp() { return static_cast<double>(cycles); }

static void tick() {
    top->clk = 0; top->eval();
    top->clk = 1; top->eval();
    ++cycles;
}

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    if (argc < 3) {
        std::cerr << "usage: mnist_sim stimulus.txt output.txt\n";
        return 2;
    }
    std::ifstream in(argv[1]);
    if (!in) return 2;

    int n_in, n_hid, n_out, t_max, par, par_out;
    int k_hid, k_out, nlog_hid, nlog_out;
    in >> n_in >> n_hid >> n_out >> t_max >> par >> par_out;
    in >> k_hid >> k_out >> nlog_hid >> nlog_out;
    if (n_in != 784) {
        std::cerr << "mnist_main requires N_IN=784\n";
        return 2;
    }

    std::vector<int> wh(n_hid * n_in), bh(n_hid);
    std::vector<int> wo(n_out * n_hid), bo(n_out);
    for (auto &v : wh) in >> v;
    for (auto &v : bh) in >> v;
    for (auto &v : wo) in >> v;
    for (auto &v : bo) in >> v;

    int ns; in >> ns;
    const int words = (n_in + 31) / 32;
    std::vector<std::vector<uint32_t>> x(ns, std::vector<uint32_t>(words, 0));
    for (int s = 0; s < ns; ++s) {
        for (int i = 0; i < n_in; ++i) {
            int bit; in >> bit;
            if (bit) x[s][i >> 5] |= uint32_t(1) << (i & 31);
        }
    }

    top = new Vsena_tile;
    for (int i = 0; i < n_hid * n_in; ++i)
        top->w_hid[i] = static_cast<uint16_t>(static_cast<int16_t>(wh[i]));
    for (int i = 0; i < n_hid; ++i)
        top->b_hid[i] = static_cast<uint16_t>(static_cast<int16_t>(bh[i]));
    for (int i = 0; i < n_out * n_hid; ++i)
        top->w_out[i] = static_cast<uint16_t>(static_cast<int16_t>(wo[i]));
    for (int i = 0; i < n_out; ++i)
        top->b_out[i] = static_cast<uint16_t>(static_cast<int16_t>(bo[i]));
    top->K_hid = static_cast<uint16_t>(static_cast<int16_t>(k_hid));
    top->K_out = static_cast<uint16_t>(static_cast<int16_t>(k_out));
    top->n_log2_hid = nlog_hid;
    top->n_log2_out = nlog_out;

    top->rst_n = 0; top->sample_start = 0; top->pass_start = 0;
    for (int w = 0; w < words; ++w) top->x_bits[w] = 0;
    for (int i = 0; i < 5; ++i) tick();
    top->rst_n = 1; tick();

    std::ofstream out(argv[2]);
    out << "# sample cycles votes[0..n_out-1]\n";
    for (int s = 0; s < ns; ++s) {
        for (int w = 0; w < words; ++w) top->x_bits[w] = x[s][w];
        top->sample_start = 1; tick();
        top->sample_start = 0; tick();
        const uint64_t c0 = cycles;
        for (int pass = 0; pass < t_max; ++pass) {
            top->pass_start = 1; tick();
            top->pass_start = 0;
            int guard = 0;
            while (top->busy && guard++ < 100000) tick();
            if (top->done_force) break;
        }
        out << s << " " << (cycles - c0);
        for (int c = 0; c < n_out; ++c)
            out << " " << static_cast<int>(top->votes_dbg[c]);
        out << "\n";
    }
    top->final();
    delete top;
    std::cout << "MNIST RTL done: " << ns << " samples, "
              << cycles << " clock cycles\n";
    return 0;
}
