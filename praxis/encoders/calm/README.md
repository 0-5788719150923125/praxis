# CALM encoder

Implementation of **Continuous Autoregressive Language Models** (arXiv
2510.27688) as an opt-in encoder for Praxis.

Flow inside a single optimizer step:

1. **VAE compression** (`vae.py`): K contiguous tokens are compressed
   into one continuous latent `z`. `decode()` returns per-token decoder
   *features* (not logits). The VAE's reconstruction + KL loss is
   returned from `encode()` and composes through the standard
   `LossContainer`.
2. **Latent autoregression** (outer praxis decoder): the global
   transformer runs over `z_0, z_1, ..., z_P` and predicts
   representations for `z_1, ..., z_{P+1}`.
3. **Energy-score matching** (`../../heads/energy.py` +
   `../../losses/energy_score.py`): a noise-conditioned head proposes
   latent candidates; pairwise distances against posterior samples of
   the next latent give an implicit generative loss. Stop-gradient on
   the posterior samples keeps the VAE's objective pure.
4. **Sampling** (`../../generation/lf_temperature.py` +
   `CALMEncoder.custom_generate`): locally-fair temperature is applied
   over candidates from the energy head, and the VAE's decoder
   materialises K tokens per selected latent.

## LM head

CALM does not own its token classifier. It declares its output layout
(`output_dim` = the VAE decoder feature width, `output_vocab_size` = the
tokenizer's true vocab), the model builds a head from `HEAD_REGISTRY`
sized to that, and injects it via `set_head()`. CALM applies the head to
the decoder features for both reconstruction and generation. So
`head_type: forward | crystal | harmonic` all work; `calm-a.yml` uses
`crystal`. (The energy head is separate - it is a latent sampler, not an
LM head, and stays out of the registry.)

## Vocabulary

CALM sizes its VAE to the tokenizer's *true* vocabulary via
`config.byte_vocab_size` (e.g. 264 for the byte tokenizer), falling back
to `config.vocab_size` when unset. This matters because byte-latent
overloads `config.vocab_size` as its hash-embedding bucket count, so it
is not a reliable token count; CALM, owning its own `tok_emb`, needs the
real one.

## Profiles

CALM ships as a set of registry profiles rather than per-hyperparam CLI
flags. Pick one with `--encoder-type`:

| Profile           | K  | latent_dim | ae_hidden | energy N / M | Use case                     |
| ----------------- | -- | ---------- | --------- | ------------ | ---------------------------- |
| `calm`            | 8  | 128        | 512       | 8 / 100      | Paper-scale, char tokenizer  |
| `calm_bpe`        | 4  | 128        | 512       | 8 / 100      | BPE / unigram tokenizer      |
| `calm_byte`       | 16 | 128        | 512       | 8 / 100      | Paper-scale, byte tokenizer  |
| `calm_small`      | 8  | 0.25       | 1.0       | 8 / 100      | Small, char tokenizer        |
| `calm_byte_small` | 16 | 0.25       | 1.0       | 4 / 16       | Small, byte tokenizer        |

`latent_dim`, `ae_hidden`, and `noise_dim` accept either an `int`
(absolute size) or a `float` (fraction of `config.hidden_size`), so the
small profiles scale with the model instead of pinning fixed widths. K
("one word of meaning per latent") tracks tokenizer granularity: BPE=4,
char=8, byte=16.

Profiles are defined in `praxis/encoders/__init__.py` using
`functools.partial`; add or tweak one there to avoid adding CLI knobs.
Pair with `--tokenizer-type char_level|byte_level|bpe` as appropriate.

## Smoke-test experiment

`experiments/calm-a.yml` trains a compact CALM model with the
`calm_byte_small` profile, the byte tokenizer, and the crystal head. Run
with `--calm-a`.
