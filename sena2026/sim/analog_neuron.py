"""
analog_neuron.py
================
Behavioral model of an analog mixed-signal stochastic neuron for the SENA demo.

Circuit abstraction (not transistor SPICE — suitable for interactive GUI /
pre-silicon exploration):

    Vin  ----+---->  + \\
             |         \\  Comparator  ---->  Y in {0,1}
    Vth_n ---+---->  - /
               ^
               |
         Thermal / flicker noise + mismatch
         Vth_n = V0 + n(t) ,  n ~ N(0, sigma(T)^2)

Averaged over samples, P(Y=1 | Vin) approximates a logistic sigmoid whose
slope is set by the noise temperature.  This is the analog dual of Mazumdar's
digital random-threshold neuron (Eq. 3).

Also models:
  * reverse characteristic R(x) via |Vin - V0| vs noise threshold
  * Monte-Carlo device mismatch (sigma_mis)
  * crude energy / area estimates for a comparator-based neuron
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class AnalogNeuronParams:
    v0: float = 0.0              # nominal threshold (V)
    sigma0: float = 0.35         # noise RMS at T0 (V)
    T0: float = 300.0            # reference temperature (K)
    T: float = 300.0             # operating temperature (K)
    mismatch: float = 0.05       # device-to-device threshold sigma (V)
    vdd: float = 1.0             # supply (V)
    c_load_fF: float = 5.0       # load capacitance (fF)
    area_um2: float = 12.0       # estimated neuron area (um^2)
    seed: int = 0


class AnalogStochasticNeuron:
    """Comparator + Gaussian threshold noise = stochastic analog neuron."""

    def __init__(self, p: AnalogNeuronParams | None = None):
        self.p = p or AnalogNeuronParams()
        self.rng = np.random.default_rng(self.p.seed)
        self.vth_static = self.p.v0 + self.rng.normal(0, self.p.mismatch)

    @property
    def sigma(self) -> float:
        """Thermal-noise RMS scales ~ sqrt(T/T0)."""
        return self.p.sigma0 * np.sqrt(max(self.p.T, 1.0) / self.p.T0)

    def sample_threshold(self, n: int = 1) -> np.ndarray:
        return self.vth_static + self.rng.normal(0, self.sigma, size=n)

    def fire(self, vin: float | np.ndarray, n_samples: int = 1) -> np.ndarray:
        """Binary firings for one or many Vin values.

        If vin is scalar: returns shape (n_samples,)
        If vin is array:  returns shape (len(vin), n_samples)
        """
        vin = np.asarray(vin, dtype=float)
        if vin.ndim == 0:
            thr = self.sample_threshold(n_samples)
            return (float(vin) > thr).astype(np.int64)
        thr = self.vth_static + self.rng.normal(
            0, self.sigma, size=(len(vin), n_samples))
        return (vin[:, None] > thr).astype(np.int64)

    def fire_prob(self, vin: np.ndarray, n_samples: int = 512) -> np.ndarray:
        return self.fire(vin, n_samples).mean(axis=1)

    def reverse_gate(self, vin: float | np.ndarray, n_samples: int = 1) -> np.ndarray:
        """Analog R(X): 1 if |Vin - V0| < |noise|, else 0 (learning sensitivity)."""
        vin = np.asarray(vin, dtype=float)
        if vin.ndim == 0:
            r = np.abs(self.rng.normal(0, self.sigma, size=n_samples))
            return (np.abs(float(vin) - self.p.v0) < r).astype(np.int64)
        r = np.abs(self.rng.normal(0, self.sigma, size=(len(vin), n_samples)))
        return (np.abs(vin[:, None] - self.p.v0) < r).astype(np.int64)

    def reverse_prob(self, vin: np.ndarray, n_samples: int = 512) -> np.ndarray:
        return self.reverse_gate(vin, n_samples).mean(axis=1)

    @staticmethod
    def ideal_sigmoid(vin: np.ndarray, scale: float = 0.35) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-vin / max(scale, 1e-6)))

    @staticmethod
    def ideal_sigmoid_deriv(vin: np.ndarray, scale: float = 0.35) -> np.ndarray:
        s = AnalogStochasticNeuron.ideal_sigmoid(vin, scale)
        return s * (1 - s) / max(scale, 1e-6)

    def energy_pj(self, activity: float = 0.5) -> float:
        """Rough dynamic energy per decision: a * C * Vdd^2  (pJ)."""
        c = self.p.c_load_fF * 1e-15
        e_j = activity * c * (self.p.vdd ** 2)
        return e_j * 1e12  # pJ

    def metrics(self) -> dict:
        return dict(
            sigma_V=float(self.sigma),
            T_K=float(self.p.T),
            vth_static=float(self.vth_static),
            mismatch_V=float(self.p.mismatch),
            area_um2=float(self.p.area_um2),
            energy_pJ=float(self.energy_pj(0.5)),
            vdd=float(self.p.vdd),
        )

    def monte_carlo_curves(self, vin: np.ndarray, n_devices: int = 40,
                           n_samples: int = 256) -> np.ndarray:
        """Return (n_devices, len(vin)) fire probabilities under mismatch."""
        curves = np.zeros((n_devices, len(vin)))
        base_seed = self.p.seed
        for i in range(n_devices):
            p = AnalogNeuronParams(**{**self.p.__dict__, "seed": base_seed + 17 * i + 3})
            n = AnalogStochasticNeuron(p)
            curves[i] = n.fire_prob(vin, n_samples=n_samples)
        return curves


def sweep_transfer(params: AnalogNeuronParams | None = None,
                   v_min=-2.0, v_max=2.0, n_pts=81, n_samples=800):
    """Convenience: Vin sweep → P(fire), ideal sigmoid, reverse P."""
    p = params or AnalogNeuronParams()
    neuron = AnalogStochasticNeuron(p)
    vin = np.linspace(v_min, v_max, n_pts)
    p_fire = neuron.fire_prob(vin, n_samples=n_samples)
    p_rev = neuron.reverse_prob(vin, n_samples=n_samples)
    ideal = neuron.ideal_sigmoid(vin, scale=neuron.sigma)
    ideal_d = neuron.ideal_sigmoid_deriv(vin, scale=neuron.sigma)
    # normalize ideal_d to [0,1] peak for overlay
    if ideal_d.max() > 0:
        ideal_d = ideal_d / ideal_d.max()
    return dict(vin=vin, p_fire=p_fire, p_rev=p_rev, ideal=ideal,
                ideal_d=ideal_d, neuron=neuron, metrics=neuron.metrics())
