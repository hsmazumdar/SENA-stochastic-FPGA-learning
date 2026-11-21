"""
dwn_ref.py - shared helpers for the SENA V13 DWN matched baseline.

Contents
  * EFDTorch / LUTLayerTorch : a pure-PyTorch re-implementation of the DWN
    LUT layer (forward + Extended Finite Difference backward), written as a
    line-by-line transcription of the official CUDA kernel
    (github.com/alanbacellar/DWN, src/torch_dwn/custom_operators/cuda/
    efd_cuda_kernel.cu, commit 2b456acf).  Runs on CPU or GPU without
    compiling anything.  Used (a) for the CPU self-test on the build box and
    (b) as a fallback on Windows if the official CUDA extension will not build.
    The official extension ("--efd ext") remains the reference choice.
  * export_model()    : turns a trained model (official torch_dwn layers or the
    torch fallback layers) into a plain JSON description: per LUT its n input
    indices and its 2^n-entry truth table (bit = lut_value > 0, exactly what the
    STE does at inference).
  * infer_int()       : integer/boolean reference inference from the JSON
    (numpy only).  This is the "Python model" against which the Verilog is
    checked bit-exactly.

DWN is by Bacellar et al., ICML 2024 (MIT licence).  The hardware path in this
package (JSON -> Verilog) is OUR re-implementation of the published DWN
inference structure; the DWN authors did not release an RTL generator.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

FORMAT = "sena-v13-dwn-lut-v1"


# --------------------------------------------------------------------------- #
#  Integer reference inference (numpy)                                        #
# --------------------------------------------------------------------------- #
def _tables(layer):
    n = layer["n"]
    tabs = np.zeros((layer["n_out"], 2 ** n), dtype=np.uint8)
    for j, h in enumerate(layer["luts_hex"]):
        v = int(h, 16)
        for a in range(2 ** n):
            tabs[j, a] = (v >> a) & 1
    return tabs


def group_slices(n_last, k):
    """GroupSum(k) in torch_dwn pads the last layer with zeros up to a
    multiple of k and then views it as (k, G): class c owns indices
    [c*G, (c+1)*G) (indices >= n_last are padding = 0)."""
    g = -(-n_last // k)
    return g, [(c * g, min((c + 1) * g, n_last)) for c in range(k)]


def infer_int(model, X, return_all=False):
    """X: (B, n_in) array of 0/1.  Returns (pred, counts[, layer_outputs])."""
    h = (np.asarray(X) > 0).astype(np.uint8)
    outs = []
    for layer in model["layers"]:
        m = np.asarray(layer["mapping"], dtype=np.int64)      # (N, n)
        tabs = _tables(layer)                                  # (N, 2^n)
        bits = h[:, m]                                         # (B, N, n)
        w = (1 << np.arange(layer["n"], dtype=np.int64))
        addr = (bits.astype(np.int64) * w).sum(-1)             # bit l = input l
        h = tabs[np.arange(len(m))[None, :], addr]             # (B, N)
        outs.append(h)
    k = model["n_classes"]
    _, sl = group_slices(h.shape[1], k)
    counts = np.stack([h[:, a:b].sum(1) for a, b in sl], 1).astype(np.int64)
    pred = counts.argmax(1)        # numpy/torch argmax: first maximum wins
    if return_all:
        return pred, counts, outs
    return pred, counts


def model_sha256(model):
    core = {"layers": model["layers"], "n_in": model["n_in"],
            "n_classes": model["n_classes"]}
    return hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()


def save_model(model, path):
    model["model_sha256"] = model_sha256(model)
    with open(path, "w") as f:
        json.dump(model, f)


def load_model(path):
    with open(path) as f:
        model = json.load(f)
    assert model.get("format") == FORMAT, "unexpected JSON format"
    return model


# --------------------------------------------------------------------------- #
#  PyTorch parts (imported lazily so the Verilog generator needs numpy only)  #
# --------------------------------------------------------------------------- #
def torch_parts():
    import torch
    from torch.nn.functional import softmax

    class EFDTorch(torch.autograd.Function):
        """Pure-PyTorch transcription of efd_cuda_forward/backward."""

        @staticmethod
        def forward(ctx, x, mapping, luts, alpha, beta):
            n = mapping.shape[1]
            bits = (x[:, mapping.long()] > 0)                       # (B,N,n)
            w = (1 << torch.arange(n, device=x.device))
            addr = (bits.long() * w).sum(-1)                         # (B,N)
            out = torch.gather(luts.unsqueeze(0).expand(x.shape[0], -1, -1),
                               2, addr.unsqueeze(-1)).squeeze(-1)
            ctx.save_for_backward(x, mapping, luts, alpha, beta, addr)
            return out

        @staticmethod
        def backward(ctx, g):
            x, mapping, luts, alpha, beta, addr = ctx.saved_tensors
            n = mapping.shape[1]
            A = 2 ** n
            dev = luts.device
            g = g.contiguous()
            # LUT grad: scatter-add output grad into the addressed entry
            luts_grad = torch.zeros_like(luts)
            luts_grad.scatter_add_(1, addr.t().contiguous(), g.t().contiguous())
            # Input grad: for every bit l, w(addr) = sum_a2 lut[a2]*alpha*
            #   beta^popc((addr&~(1<<l)) ^ (a2&~(1<<l))) * (+1 if a2 bit l else -1)
            a = torch.arange(A, device=dev)
            input_grad = torch.zeros_like(x)
            for l in range(n):
                msk = ~(1 << l)
                d = ((a[:, None] & msk) ^ (a[None, :] & msk))
                pop = torch.zeros_like(d)
                for b in range(n):
                    pop += (d >> b) & 1
                sign = torch.where(((a[None, :] >> l) & 1).bool(), 1.0, -1.0)
                D = alpha.to(dev).float() * torch.pow(beta.to(dev).float(),
                                                      pop.float()) * sign
                W = luts @ D.t()                      # (N, A): W[j, addr]
                wsel = torch.gather(W.unsqueeze(0).expand(x.shape[0], -1, -1),
                                    2, addr.unsqueeze(-1)).squeeze(-1)  # (B,N)
                input_grad.index_add_(1, mapping[:, l].long(),
                                      wsel * g)
            return input_grad, None, luts_grad, None, None

    class STEFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return (x > 0).float()

        @staticmethod
        def backward(ctx, g):
            return g

    class LearnableMappingFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weights, tau):
            mapping = weights.argmax(dim=0)
            ctx.save_for_backward(x, weights, tau)
            return x[:, mapping]

        @staticmethod
        def backward(ctx, g):
            x, weights, tau = ctx.saved_tensors
            weights_grad = ((2 * x - 1).T @ g)
            input_grad = g @ softmax(weights / tau, dim=0).T
            return input_grad, weights_grad, None

    class LearnableMapping(torch.nn.Module):
        def __init__(self, input_size, output_size, tau=0.001):
            super().__init__()
            self.weights = torch.nn.Parameter(
                torch.rand(input_size, output_size, dtype=torch.float32))
            self.tau = tau

        def forward(self, x):
            return LearnableMappingFunction.apply(
                x, self.weights, torch.tensor(self.tau, device=x.device))

    def layer_mapping(input_size, n, output_size, random=True):
        mapp = (lambda: torch.randperm(input_size)) if random else \
               (lambda: torch.arange(input_size))
        num_complete, offset = divmod(n * output_size, input_size)
        if num_complete == 0:
            return mapp()[:offset].reshape(output_size, n).int()
        parts = [mapp() for _ in range(num_complete)] + [mapp()[:offset]]
        return torch.cat(parts).reshape(output_size, n).int()

    class LUTLayerTorch(torch.nn.Module):
        """Same parameters/semantics as torch_dwn.LUTLayer (v1.1.1)."""

        def __init__(self, input_size, output_size, n, mapping="random",
                     alpha=None, beta=None, ste=True, clamp_luts=True,
                     lm_tau=0.001):
            super().__init__()
            self.input_size, self.output_size, self.n = input_size, output_size, n
            self.ste, self.clamp_luts = ste, clamp_luts
            self.alpha = torch.tensor(0.5 * 0.75 ** (n - 1) if alpha is None else alpha)
            self.beta = torch.tensor(0.25 / 0.75 if beta is None else beta)
            if mapping == "learnable":
                self.mapping = LearnableMapping(input_size, output_size * n, tau=lm_tau)
                self.register_buffer("dummy_mapping",
                                     torch.arange(output_size * n).reshape(output_size, n).int())
            else:
                self.mapping = torch.nn.Parameter(
                    layer_mapping(input_size, n, output_size, random=(mapping == "random")),
                    requires_grad=False)
            self.luts = torch.nn.Parameter(torch.rand(output_size, 2 ** n) * 2 - 1)

        def forward(self, x):
            if self.training and self.clamp_luts:
                with torch.no_grad():
                    self.luts.clamp_(-1, 1)
            if isinstance(self.mapping, LearnableMapping):
                x = self.mapping(x)
                mapping = self.dummy_mapping
            else:
                mapping = self.mapping
            x = EFDTorch.apply(x, mapping, self.luts, self.alpha, self.beta)
            if self.ste:
                x = STEFunction.apply(x)
            return x

    class GroupSum(torch.nn.Module):
        def __init__(self, k, tau=1.0):
            super().__init__()
            self.k, self.tau = k, tau

        def forward(self, x):
            pad = (-x.shape[-1]) % self.k
            if pad:
                x = torch.nn.functional.pad(x, (0, pad))
            x = x.view(*x.shape[:-1], self.k, x.shape[-1] // self.k)
            return x.sum(-1) / self.tau

    return dict(EFDTorch=EFDTorch, LUTLayerTorch=LUTLayerTorch,
                GroupSum=GroupSum, LearnableMapping=LearnableMapping)


def layer_effective_mapping(layer):
    """(N, n) int array of input indices for an official or fallback LUT layer."""
    import torch
    m = layer.mapping
    if isinstance(m, torch.nn.Parameter) or isinstance(m, torch.Tensor):
        return m.detach().cpu().numpy().astype(np.int64)
    # learnable: input of LUT j, bit l = x[argmax(W)[j*n + l]]
    idx = m.weights.detach().argmax(dim=0).cpu().numpy().astype(np.int64)
    return idx.reshape(layer.output_size, layer.n)


def export_model(model_seq, n_in, n_classes, tau, meta=None):
    """Export nn.Sequential(LUTLayer..., GroupSum) to the JSON dict."""
    layers = []
    for mod in model_seq:
        if hasattr(mod, "luts"):
            m = layer_effective_mapping(mod)
            t = (mod.luts.detach().cpu().numpy() > 0)
            hexes = []
            for row in t:
                v = 0
                for a, bit in enumerate(row):
                    if bit:
                        v |= (1 << a)
                hexes.append(format(v, "0%dx" % ((len(row) + 3) // 4)))
            layers.append({"n": int(mod.n), "n_in": int(mod.input_size),
                           "n_out": int(mod.output_size),
                           "mapping": m.tolist(), "luts_hex": hexes})
    return {"format": FORMAT, "n_in": int(n_in), "n_classes": int(n_classes),
            "groupsum_tau": float(tau), "layers": layers,
            "meta": meta or {}}
