# Predictive coding as a trainer backend

Reviewed 2026-09-02, prompted by [deepity](https://github.com/ra4ster/deepity).

**Declined. Speed is the decisive axis, and PC is slower by construction with
the gap widening as the model deepens.** Park unless the hardware changes.

Toy implementation already exists at `staging/prospective_config.py` - a working
2-8-8-1 XOR PCN in ~110 lines of pure PyTorch. That is the whole algorithm; if
this ever reopens, start there, not from a library.

## The numbers

**Wall-clock** (PCX, ICLR 2025, [arXiv:2407.01163](https://arxiv.org/abs/2407.01163),
JAX + JIT + vmap - a serious optimization effort). Seconds per epoch:

| | BP | PC |
| --- | --- | --- |
| MLP, FashionMNIST | 1.82 | 1.94 |
| AlexNet, CIFAR-10 | 1.04 | 3.86 |
| VGG-5, CIFAR-100 | 1.61 | 5.33 |
| VGG-7, Tiny ImageNet | 7.59 | 54.60 |

Never faster: 1.07x -> 7.2x as depth grows. Same paper, PC *accuracy* also
degrades with depth (VGG5 > VGG7 > VGG9 > ResNet18), inverted from BP.

**Language modeling has exactly one datapoint.** Pinchetti et al., NeurIPS 2022
([arXiv:2211.03481](https://arxiv.org/abs/2211.03481)): one transformer block,
one head, `d=128`, seq 32, 8k vocab.

| BP | PC (KL / categorical) | PC (Gaussian) |
| --- | --- | --- |
| 162.64 | 175.90 | 590.08 |

Plain Gaussian PC does not survive contact with softmax - 590 vs 163 is a broken
objective, not a tuning gap. Attention and the output head need per-layer
*categorical* energies, so a Praxis PC would need a custom energy per
non-Gaussian layer, and we have many.

The field agrees. *PC as a Neuromorphic Alternative to Backpropagation*
(Neural Computation 35:12) finds no wall-clock advantage on GPU or CPU.
Innocenti's [arXiv:2510.23323](https://arxiv.org/abs/2510.23323) makes 100+ layer
PCNs trainable and still closes with "the need for future research to focus on
**hardware co-design** if PC is to compete with BP at scale."

## The reference project does not claim what it looks like

deepity's "97.73% on MNIST in 60 seconds ... roughly 50x faster" is a
**self-comparison** - deepity-now vs deepity-a-few-months-ago, not vs backprop.
"Within 1% of standard PyTorch backprop" means slightly *worse*. Scale is
784-512-512-10, CPU only, CUDA on the roadmap. Its `DKPPCN` (one settling step
via Direct Kolen-Pollack) is the interesting part, but learned feedback matrices
converging to `W^T` is backprop with extra machinery to rediscover the transpose:
the closer PC gets to competitive, the more exactly it reduces to BP.

## Why Praxis specifically is a bad host

- **Recurrent depth cancels the one advantage.** PC's structural win is that
  layer `l` updates without waiting on `l+1`. Praxis reuses the same block
  across passes, so every pass writes the same `W` and the updates serialize
  again - paying the T-step relaxation for none of the parallelism.
- **The slot is taken.** `mono_forward` / `mono_forward_ray` already chase the
  no-end-to-end-backward-pass property, and get it from a `.detach()` plus
  ordinary autograd on a cut graph. Near-zero cost, no relaxation loop, no
  custom energies.
- **Memory goes up**: PC stores `mu` and `phi` separately, `M_PC < 2 * M_BP`.

## What would change the answer

Neuromorphic or analog hardware, where the relaxation is free and the backward
pass is the thing that cannot be built. That is the real case for PC and it is
not ours.

The one thread worth keeping warm is not about speed: **prospective
configuration** (Song et al., Nature Neuroscience 2024) argues that
activities-settle-first produces less interference than gradient descent, i.e.
better online/continual behavior. That is a *quality* claim and belongs with
[continual_learning.md](continual_learning.md), not a trainer rewrite.

## If the premise ever needs testing cheaply

Do not write a trainer. Wrap one Praxis block in the two-phase loop from
`staging/prospective_config.py` with a categorical energy on the softmax layers
and measure only **step-time ratio vs BP at `T = {1, 2, 5}`** on `small-b`. If
it is not under ~1.5x at a `T` that still trains, the thread closes on our own
numbers. Budget: a day. Expected: 3-7x, matching PCX.
