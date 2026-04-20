# CALM encoder

Implementation of **Continuous Autoregressive Language Models** (arXiv
2510.27688) as an opt-in encoder for Praxis.

Flow inside a single optimizer step:

1. **VAE compression** (`vae.py`): K contiguous tokens are compressed
   into one continuous latent `z`. The VAE's reconstruction + KL loss
   is returned from `encode()` and composes through the standard
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
   `PraxisForCausalLM._calm_generate`): locally-fair temperature is
   applied over candidates from the energy head, and the VAE's decoder
   materialises K tokens per selected latent.

## Profiles

CALM ships as a set of registry profiles rather than per-hyperparam
CLI flags. Pick one with `--encoder-type`:

| Profile      | K  | latent_dim | ae_hidden | energy N / M | Use case                 |
| ------------ | -- | ---------- | --------- | ------------ | ------------------------ |
| `calm`       | 8  | 128        | 512       | 8 / 100      | Default, char tokenizer  |
| `calm_bpe`   | 4  | 128        | 512       | 8 / 100      | BPE / unigram tokenizer  |
| `calm_byte`  | 16 | 128        | 512       | 8 / 100      | Byte tokenizer           |
| `calm_small` | 8  | 64         | 256       | 4 / 16       | Smoke test / quick run   |

Profiles are defined in `praxis/encoders/__init__.py` using
`functools.partial`; add or tweak one there to avoid adding CLI knobs.
Pair with `--tokenizer-type char|byte` as appropriate.

## Smoke-test experiment

`experiments/april.yml` trains a compact CALM model with the
`calm_small` profile and the character tokenizer. Run with `--april`.
