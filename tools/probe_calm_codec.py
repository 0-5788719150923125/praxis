#!/usr/bin/env python
"""Isolate the CALM bottleneck: codec round-trip vs energy-head prediction.

Three decode paths over the same real text, all teacher-forced (no
compounding error), so a failure points at one component:

  A. Codec round-trip   - encode text -> latent -> decode. Upper bound on
     anything the head could achieve. If this is mush, the codec is lossy.
  B. Head best-guess    - the head's zero-noise (mean) next-latent prediction,
     conditioned on the TRUE prefix hidden state, decoded. If A is clean but
     this is mush, the energy head is the bottleneck.
  C. Head sample / vote - the actual inference sampler (single natural draw and
     the patch-vote), to see how much sampling noise costs on top of B.

Runs on CPU by default so it never touches a training GPU.

  python tools/probe_calm_codec.py [--run HASH] [--device cpu] [--text "..."]
"""

import argparse
import glob
import os

import torch


def load_model(run_hash, device):
    snaps = sorted(
        glob.glob(f"build/runs/{run_hash}/model/model-batch=*.ckpt"),
        key=os.path.getmtime,
    )
    if not snaps:
        snaps = [f"build/runs/{run_hash}/model/last.ckpt"]
    ckpt = snaps[-1]
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    step = ck.get("global_step", "?")

    # tie_word_embeddings collides with the config's tie_weights alias.
    hp = {
        k: v
        for k, v in ck["hyper_parameters"]["hparams"].items()
        if k != "tie_word_embeddings"
    }
    from praxis import PraxisConfig, PraxisForCausalLM
    from praxis.utils import initialize_lazy_modules

    config = PraxisConfig(**hp)
    model = PraxisForCausalLM(config)
    initialize_lazy_modules(model, device)
    sd = {
        k[len("model.") :]: v
        for k, v in ck["state_dict"].items()
        if k.startswith("model.")
    }
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[warn] missing={len(missing)} unexpected={len(unexpected)} params")
    model.eval().to(device)
    return model, config, ckpt, step


def make_tokenizer(config):
    import importlib.util

    sp = importlib.util.spec_from_file_location(
        "tm_int", "integrations/tokenmonster/main.py"
    )
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)  # registers 'tokenmonster' in the registry
    from praxis.tokenizers import create_tokenizer

    return create_tokenizer(
        vocab_size=config.vocab_size,
        encoder_type=config.encoder_type,
        tokenizer_type="tokenmonster",
        cache_dir="build",
    )


def token_acc(pred, target, pad_id):
    """Fraction of non-pad target tokens matched exactly."""
    mask = target != pad_id
    if mask.sum() == 0:
        return float("nan")
    return float((pred[mask] == target[mask]).float().mean())


def render(tokenizer, ids, pad_id):
    ids = [int(i) for i in ids.tolist() if int(i) != pad_id]
    try:
        return repr(tokenizer.decode(ids))
    except Exception as e:  # tokenizer round-trip is best-effort
        return f"<decode failed: {e}> ids={ids[:24]}"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=os.readlink("build/current").split("/")[-1])
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--text",
        default=(
            "The history of the world is the history of a few men who had "
            "faith in themselves. That faith calls out the divinity within."
        ),
    )
    ap.add_argument("--temperature", type=float, default=0.5)
    args = ap.parse_args()

    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    model, config, ckpt, step = load_model(args.run, args.device)
    tokenizer = make_tokenizer(config)
    enc = model.encoder
    pad_id = enc.pad_token_id
    dev = args.device

    print(f"run={args.run} step={step} ckpt={os.path.basename(ckpt)}")
    print(f"K={enc.K} frozen={enc._ae_is_frozen()} latent_dim={enc.vae.latent_dim}")
    print("=" * 72)

    ids = tokenizer.encode(args.text)
    input_ids = torch.tensor([ids], dtype=torch.long, device=dev)
    padded = enc._pad_to_chunk(input_ids)  # [1, N*K]
    print(f"INPUT ({padded.shape[1]} tokens, {padded.shape[1] // enc.K} patches):")
    print("  " + render(tokenizer, padded[0], pad_id))
    print("=" * 72)

    def decode_latents(z):
        logits = enc._classify(enc.vae.decode(z))
        return logits.argmax(-1)  # [B, N*K]

    # --- A: codec round-trip -------------------------------------------------
    mean, logvar = enc.vae.encode(padded)
    z_sample = enc.vae.reparameterize(mean, logvar)
    a_mean = decode_latents(mean)
    a_samp = decode_latents(z_sample)
    print("A. CODEC ROUND-TRIP (upper bound)")
    print(f"   posterior-mean  acc={token_acc(a_mean[0], padded[0], pad_id):.3f}")
    print("     " + render(tokenizer, a_mean[0], pad_id))
    print(f"   posterior-sample acc={token_acc(a_samp[0], padded[0], pad_id):.3f}")
    print("     " + render(tokenizer, a_samp[0], pad_id))
    print(
        f"   latent norm={mean.norm(dim=-1).mean():.2f} "
        f"post-std={(0.5 * logvar).exp().mean():.3f}"
    )
    print("-" * 72)

    # --- B/C: head prediction under teacher forcing --------------------------
    from praxis.modeling import PraxisModel

    out = PraxisModel.forward(model, input_ids=input_ids)
    h = out.last_hidden_state  # [1, N, hidden]
    N = h.shape[1]
    if N < 2:
        print("sequence too short for next-patch prediction")
        return
    h_cond = h[:, :-1]  # predicts patches 1..N-1
    t_cond = torch.arange(N - 1, device=dev)
    target = padded[:, enc.K :]  # true patches 1..N-1
    tgt_render = render(tokenizer, target[0], pad_id)

    # B: head zero-noise best guess
    zero_noise = h_cond.new_zeros(*h_cond.shape[:-1], enc.energy_head.noise_dim)
    z_hat = enc.energy_head(h_cond, zero_noise, t=t_cond)
    b_mean = decode_latents(z_hat)
    # C1: single natural sample
    z_one = enc.energy_head.sample(h_cond, num_samples=1, noise_scale=1.0, t=t_cond)[0]
    c_samp = decode_latents(z_one)
    # C2: patch-vote per position (the real inference sampler)
    voted = []
    for p in range(N - 1):
        voted.append(
            enc._patch_vote_sample(
                h_cond[0, p],
                temperature=args.temperature,
                num_samples=enc.vote_num_samples,
                t=t_cond[p : p + 1],
            )
        )
    c_vote = torch.stack(voted).reshape(1, -1)

    print("TARGET (true next patches 1..N-1):")
    print("     " + tgt_render)
    print()
    print("B. HEAD BEST-GUESS (zero-noise mean, true conditioning)")
    print(f"   acc={token_acc(b_mean[0], target[0], pad_id):.3f}")
    print("     " + render(tokenizer, b_mean[0], pad_id))
    print()
    print("C. HEAD SAMPLERS (true conditioning)")
    print(f"   single draw    acc={token_acc(c_samp[0], target[0], pad_id):.3f}")
    print("     " + render(tokenizer, c_samp[0], pad_id))
    print(
        f"   patch-vote T={args.temperature} "
        f"acc={token_acc(c_vote[0], target[0], pad_id):.3f}"
    )
    print("     " + render(tokenizer, c_vote[0], pad_id))
    print("=" * 72)
    print(
        "Read: if A is clean but B/C are mush -> the energy HEAD is the\n"
        "bottleneck (target brittle / objective too noisy). If A is mush ->\n"
        "the CODEC is lossy and no head swap helps."
    )


if __name__ == "__main__":
    main()
