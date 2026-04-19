"""
Prospective configuration: a predictive-coding-based alternative to backprop.

Paper:  Song et al., "Inferring neural activity before plasticity as a
        foundation for learning beyond backpropagation,"
        Nature Neuroscience (2024).
        https://www.nature.com/articles/s41593-023-01514-1

Reference implementation (Ray + Hydra, much larger):
        https://github.com/YuhangSong/Prospective-Configuration

Each layer owns its own activity x_l. Training alternates two local phases:

  1. Inference. Clamp input at x_0 and target at x_L. Relax the interior
     activities to minimise F = 1/2 sum_l ||eps_l||^2, with
         eps_l = x_l - (W_l phi(x_{l-1}) + b_l).
  2. Plasticity. With activities settled, update each weight from pre- and
     post-synaptic signals only:  dW_l propto eps_l phi(x_{l-1})^T.

Activities move first (toward a state that already predicts the target);
weights consolidate that change afterward. No global backward pass.

This script trains a 2-8-8-1 MLP on XOR using only these two phases.
"""

import math
import torch

torch.manual_seed(0)


def phi(z):
    return torch.tanh(z)


def dphi(z):
    return 1.0 - torch.tanh(z) ** 2


def predict(xs, Ws, bs):
    """Feedforward prediction mu_l for each layer given current xs."""
    mus = [None]
    for l, (W, b) in enumerate(zip(Ws, bs)):
        prev = xs[l] if l == 0 else phi(xs[l])
        mus.append(prev @ W.T + b)
    return mus


def forward_init(X, Ws, bs):
    """Initialise interior activities from a plain feedforward pass."""
    xs = [X]
    for l, (W, b) in enumerate(zip(Ws, bs)):
        prev = xs[-1] if l == 0 else phi(xs[-1])
        xs.append(prev @ W.T + b)
    return xs


def relax(xs, Ws, bs, alpha=0.1, steps=50):
    """Inference phase. xs[0] and xs[-1] stay clamped; interior xs relax."""
    for _ in range(steps):
        mus = predict(xs, Ws, bs)
        eps = [None] + [xs[l] - mus[l] for l in range(1, len(xs))]
        for l in range(1, len(xs) - 1):
            top_down = (eps[l + 1] @ Ws[l]) * dphi(xs[l])
            xs[l] = xs[l] + alpha * (-eps[l] + top_down)
    return xs


def train_step(X, Y, Ws, bs, eta=0.1):
    xs = forward_init(X, Ws, bs)
    xs[-1] = Y  # clamp target
    xs = relax(xs, Ws, bs)
    mus = predict(xs, Ws, bs)
    B = X.shape[0]
    for l in range(len(Ws)):
        eps_next = xs[l + 1] - mus[l + 1]
        prev = xs[l] if l == 0 else phi(xs[l])
        Ws[l] = Ws[l] + eta * (eps_next.T @ prev) / B
        bs[l] = bs[l] + eta * eps_next.mean(0)
    return Ws, bs


def feedforward(X, Ws, bs):
    h = X
    for l, (W, b) in enumerate(zip(Ws, bs)):
        h = (h if l == 0 else phi(h)) @ W.T + b
    return h


# XOR.
X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
Y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

dims = [2, 8, 8, 1]
Ws = [torch.randn(dims[i + 1], dims[i]) * math.sqrt(1.0 / dims[i])
      for i in range(len(dims) - 1)]
bs = [torch.zeros(d) for d in dims[1:]]

print("training 2-8-8-1 MLP on XOR via prospective configuration")
for epoch in range(3000):
    Ws, bs = train_step(X, Y, Ws, bs)
    if epoch % 300 == 0 or epoch == 2999:
        yhat = feedforward(X, Ws, bs)
        mse = 0.5 * ((yhat - Y) ** 2).mean().item()
        preds = [round(v, 3) for v in yhat.squeeze().tolist()]
        print(f"  epoch {epoch:4d}  mse={mse:.4f}  preds={preds}")
