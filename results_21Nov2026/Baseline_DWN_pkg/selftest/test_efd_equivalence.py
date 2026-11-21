"""SELF-TEST: pure-PyTorch EFD (dwn_ref.EFDTorch) vs a literal scalar Python
transcription of efd_cuda_kernel.cu (forward + backward loops). Small random
sizes; CPU only. Not a result of any kind - only checks the fallback code."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import dwn_ref

def kernel_ref(x, mapping, luts, alpha, beta, g):
    B, N, n = x.shape[0], mapping.shape[0], mapping.shape[1]
    out = torch.zeros(B, N); ig = torch.zeros_like(x); lg = torch.zeros_like(luts)
    for i in range(B):
        for j in range(N):
            addr = int(x[i, mapping[j][0]] > 0)
            for l in range(1, n):
                addr |= int(x[i, mapping[j][l]] > 0) << l
            out[i, j] = luts[j, addr]
            lg[j, addr] += g[i, j]
            for l in range(n):
                w = 0.0
                a_off = addr & ~(1 << l)
                for a2 in range(luts.shape[1]):
                    dist = bin(a_off ^ (a2 & ~(1 << l))).count("1")
                    fd = float(luts[j, a2]) * alpha * beta ** dist
                    w += fd if (a2 >> l) & 1 else -fd
                ig[i, mapping[j][l]] += w * g[i, j]
    return out, ig, lg

torch.manual_seed(1)
P = dwn_ref.torch_parts()
for n in (2, 4, 6):
    B, I, N = 5, 23, 7
    x = (torch.rand(B, I) > 0.5).float()
    mapping = torch.randint(0, I, (N, n), dtype=torch.int32)
    luts = torch.rand(N, 2 ** n) * 2 - 1
    alpha, beta = 0.5 * 0.75 ** (n - 1), 0.25 / 0.75
    g = torch.randn(B, N)
    o_ref, ig_ref, lg_ref = kernel_ref(x, mapping, luts, alpha, beta, g)
    xt = x.clone().requires_grad_(True); lt = luts.clone().requires_grad_(True)
    o = P["EFDTorch"].apply(xt, mapping, lt, torch.tensor(alpha), torch.tensor(beta))
    o.backward(g)
    e = [(o - o_ref).abs().max().item(), (xt.grad - ig_ref).abs().max().item(),
         (lt.grad - lg_ref).abs().max().item()]
    ok = e[0] == 0 and e[1] < 1e-5 and e[2] < 1e-5
    print(f"n={n}: fwd {e[0]:.2e} dX {e[1]:.2e} dLUT {e[2]:.2e} -> {'PASS' if ok else 'FAIL'}")
    assert ok
print("SELFTEST EFD fallback == CUDA-kernel transcription: PASS")
