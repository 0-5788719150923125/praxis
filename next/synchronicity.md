# Synchronicity: related without a causal link

> Status: **parked** (2026-10-22). This note turns the idea into readings that
> could be tested. Reading 3 got built on 2026-10-23 as Kuramoto Attention:
> [kuramoto_attention.md](kuramoto_attention.md). Anything further lands
> under the parked oscillation thread
> ([oscillatory_axes.md](oscillatory_axes.md)), not as a new thread.

## The idea

Jung's synchronicity: two things meaningfully related with no causal
connection between them. The hope is a new mechanic for storing information
or compressing it better, of a kind nobody has tested.

As Jung meant it there is nothing to build. "Correlated, with no causal path
between them" does have three precise readings, though, and each is a
different mechanism.

## Three readings

### 1. Common cause

Reichenbach: if A and B correlate and neither causes the other, a hidden C
drives both. In a model, positions are related because they read the same
latent, not because one attends to the other.

This is latent-variable language modeling (document latents, VAE-style LMs),
which is well explored. Nothing new here.

### 2. Shared form, different content

`A B A` with different symbols. Two things are related if one short program
generates both; that is the compression reading, and it is the closest fit
to "store more information": store the form once, not each instance.

Built as the **relational bottleneck** (Webb et al. 2023) and the
**Abstractor** (Altabaa et al. 2023): pass along the pattern of similarities
between items and strip out the items themselves. Induction heads are a
learned, narrow case of the same thing.

### 3. Binding by synchrony

Neuroscience's answer to "related without wiring": features that belong to
one object fire in phase (von der Malsburg, Singer). Nothing connects them
except the timing. The harmonic latent is already phase-based, so this is
the reading closest to Praxis.

This corner filled up fast in 2026:

- **KoPE** (arXiv:2604.07904) - Kuramoto phase per token in a ViT, injected
  into attention as complex rotations; motivated by the binding problem.
- **Selective Synchronization Attention** (arXiv:2602.14445) - attention
  weight = synchronization strength between tokens, closed-form steady state.
- **Kuramoto Attention** (arXiv:2606.11585) - a language-modeling attention
  layer whose update is an explicit synchronization step.

## The gaps that are actually open

[magnetism.md](magnetism.md) searched this neighborhood and found two empty
spots, and synchronicity points at both:

1. **Phase synchronization as the cross-modal alignment operator.**
   Neuroscience has cross-modal phase reset; ML fusion has no phase in it.
2. **Learned, input-conditional higher-harmonic Kuramoto coupling.** Higher
   harmonics buy memory capacity with *fixed* coefficients
   (arXiv:2507.21984); nobody has made them learned. The amplitude field in
   `praxis/classifiers/harmonic.py` is unusually close already.

## A pattern worth keeping

The big concept rarely survives a literature search, but a narrow
combination often does. In magnetism.md the vocabulary died and two specific
couplings lived. If synchronicity yields anything, it will be one of the two
gaps above, not "synchronicity" as a module.
