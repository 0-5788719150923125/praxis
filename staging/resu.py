"""
Minimal demo of Rectified Spectral Units (ReSU).
Paper: https://arxiv.org/abs/2512.23146
Reference: https://github.com/ShawnQin/ReSU

ReSU learns temporal filters via past-future Canonical Correlation Analysis (CCA),
then applies ReLU rectification. No backprop. The "training" is a single SVD.

For a scalar time series y_t:
    past:    p_t = [y_t, y_{t-1}, ..., y_{t-m+1}]      (length m)
    future:  f_t = [y_{t+1}, ..., y_{t+h}]              (length h)

CCA finds U, V such that U^T p and V^T f are maximally correlated.
Solution: SVD of  K = C_pp^{-1/2} C_pf C_ff^{-1/2}  =  U S V^T

Filter (applied to whitened past):  w_i = u_i^T C_pp^{-1/2}
    ON-ReSU:  z+ = relu( w_i . p_t )
    OFF-ReSU: z- = relu(-w_i . p_t )
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import torch


def lag_matrix(y: torch.Tensor, m: int, h: int):
    """Build aligned past/future lag matrices from a 1D series y."""
    T = y.shape[0]
    N = T - m - h + 1
    past = torch.stack([y[i : i + m].flip(0) for i in range(N)])
    future = torch.stack([y[i + m : i + m + h] for i in range(N)])
    return past, future


def fit_resu(y: torch.Tensor, m: int, h: int, k: int, eps: float = 1e-4):
    """
    Fit k ReSU filters by CCA on a single time series.
    Returns the filter bank W (k, m) which is applied to raw past windows.
    """
    P, F = lag_matrix(y, m, h)
    P = P - P.mean(0, keepdim=True)
    F = F - F.mean(0, keepdim=True)

    N = P.shape[0]
    Cpp = (P.T @ P) / N + eps * torch.eye(m)
    Cff = (F.T @ F) / N + eps * torch.eye(h)
    Cpf = (P.T @ F) / N

    Wp = matrix_inv_sqrt(Cpp)
    Wf = matrix_inv_sqrt(Cff)
    K = Wp @ Cpf @ Wf

    U, S, _ = torch.linalg.svd(K, full_matrices=False)
    W = U[:, :k].T @ Wp
    return W, S[:k]


def matrix_inv_sqrt(C: torch.Tensor) -> torch.Tensor:
    s, V = torch.linalg.eigh(C)
    s = s.clamp(min=1e-8)
    return V @ torch.diag(s.rsqrt()) @ V.T


def apply_resu(y: torch.Tensor, W: torch.Tensor):
    """Apply ReSU filter bank as a causal 1D convolution, returning ON/OFF channels."""
    m = W.shape[1]
    kernel = W.flip(1).unsqueeze(1)  # (k, 1, m)
    x = y.view(1, 1, -1)
    proj = torch.nn.functional.conv1d(x, kernel).squeeze(0)
    return proj.clamp(min=0), (-proj).clamp(min=0)


def ou_process(T: int, theta: float = 0.05, sigma: float = 1.0, seed: int = 0):
    torch.manual_seed(seed)
    y = torch.zeros(T)
    for t in range(1, T):
        y[t] = (1 - theta) * y[t - 1] + sigma * math.sqrt(2 * theta) * torch.randn(1)
    return y


def ar2_oscillator(T: int, freq: float = 0.05, decay: float = 0.97, seed: int = 1):
    """AR(2) tuned to oscillate near `freq` cycles/sample - tests if ReSU picks it up."""
    torch.manual_seed(seed)
    phi1 = 2 * decay * math.cos(2 * math.pi * freq)
    phi2 = -(decay**2)
    y = torch.zeros(T)
    for t in range(2, T):
        y[t] = phi1 * y[t - 1] + phi2 * y[t - 2] + 0.3 * torch.randn(1)
    return y


def demo():
    m, h, k = 32, 16, 4
    T = 8000

    series = {"OU": ou_process(T), "AR2 oscillator": ar2_oscillator(T)}

    fig, axes = plt.subplots(len(series), 3, figsize=(14, 3.2 * len(series)))
    for row, (name, y) in enumerate(series.items()):
        W, S = fit_resu(y, m=m, h=h, k=k)
        on, off = apply_resu(y, W)
        print(f"{name}: top canonical correlations = {S.tolist()}")

        ax = axes[row, 0]
        ax.plot(y[:400].numpy(), lw=0.8)
        ax.set_title(f"{name}: input y_t")
        ax.set_xlabel("t")

        ax = axes[row, 1]
        for i in range(k):
            ax.plot(W[i].numpy(), label=f"k={i} (rho={S[i]:.2f})")
        ax.set_title("Learned ReSU filters (causal taps)")
        ax.set_xlabel("lag")
        ax.legend(fontsize=7)

        ax = axes[row, 2]
        t0 = 100
        seg = slice(t0, t0 + 300)
        ax.plot(on[0, seg].numpy(), label="ON unit 0", lw=0.9)
        ax.plot(-off[0, seg].numpy(), label="OFF unit 0", lw=0.9)
        ax.set_title("ReSU activations (top component)")
        ax.set_xlabel("t")
        ax.legend(fontsize=7)

    plt.tight_layout()
    out = "/tmp/resu_demo.png"
    plt.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    demo()
