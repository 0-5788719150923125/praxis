#!/usr/bin/env python3
"""Probe a CALM codec's reconstruction and latent robustness (sigma-sweep).

Splits the pipeline at the latent boundary: encode real text, decode at
increasing posterior-noise scales, and print what comes back. Separates
"codec froze undertrained" from "energy head can't hit the manifold":

    sigma=0   decode the posterior mean - the codec's ceiling
    sigma=1   a true posterior sample - what recon training sees
    sigma>1   off-manifold stress - what the LM head's imperfect
              predictions look like to the decoder

Usage (pass the same experiment flags as main.py, plus probe flags):
    python tools/probe_codec.py [main.py flags...] \
        [--probe-text "..."] [--probe-sigmas 0,0.5,1,2] [--probe-ckpt path]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _split_probe_args(argv):
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--probe-text", default=(
        "The quick brown fox jumps over the lazy dog. "
        "She sells sea shells by the sea shore."
    ))
    probe.add_argument("--probe-sigmas", default="0,0.5,1,2,4")
    probe.add_argument("--probe-ckpt", default=None)
    known, rest = probe.parse_known_args(argv)
    return known, rest


def _load_checkpoint(model, ckpt_path):
    import torch

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = state.get("state_dict", state)
    # Strip the Lightning wrapper's prefix ("model.", possibly with
    # torch.compile's "_orig_mod." inside).
    cleaned = {}
    for k, v in state.items():
        for prefix in ("model._orig_mod.", "model."):
            if k.startswith(prefix):
                k = k[len(prefix):]
                break
        cleaned[k] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[probe] {len(missing)} missing keys (e.g. {missing[:3]})")
    if unexpected:
        print(f"[probe] {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")


def main():
    probe_args, rest = _split_probe_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + rest

    import torch

    from praxis.cli import RunConfig, create_praxis_config, parse_cli
    from praxis.data.runs import setup_training_run
    from praxis.tokenizers import create_tokenizer
    from praxis.trainers import assemble_model
    from praxis.utils import resolve_resume_checkpoint

    args, processed_args = parse_cli()
    cfg = RunConfig.from_args(processed_args, args)
    run = setup_training_run(cfg)

    tokenizer = create_tokenizer(
        vocab_size=cfg.vocab_size,
        encoder_type=cfg.encoder_type,
        tokenizer_type=cfg.tokenizer_type,
        cache_dir=run.cache_dir,
    )
    config = create_praxis_config(args, tokenizer)
    bundle = assemble_model(cfg, config)
    model = bundle.model

    ckpt = probe_args.probe_ckpt or resolve_resume_checkpoint(run.cache_dir)
    if ckpt is None:
        print("[probe] no checkpoint found; probing at random init")
    else:
        print(f"[probe] loading {ckpt}")
        _load_checkpoint(model, ckpt)

    enc = model.encoder
    if not hasattr(enc, "vae"):
        sys.exit("[probe] encoder has no VAE codec; nothing to probe")

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    ids = tokenizer.encode(probe_args.probe_text)
    ids = torch.tensor([ids if isinstance(ids, list) else ids[0]], device=device)
    padded = enc._pad_to_chunk(ids)
    n_tokens = (padded != enc.pad_token_id).sum().item()

    sigmas = [float(s) for s in probe_args.probe_sigmas.split(",")]
    print(f"\n[probe] input ({n_tokens} tokens): {probe_args.probe_text!r}\n")

    with torch.no_grad():
        mean, logvar = enc.vae.encode(padded)
        std = (0.5 * logvar).exp()
        print(
            f"[probe] latent: mean_rms={mean.pow(2).mean().sqrt():.4f} "
            f"posterior_std_rms={std.pow(2).mean().sqrt():.4f}\n"
        )
        for sigma in sigmas:
            z = mean + sigma * std * torch.randn_like(std)
            logits = enc._classify(enc.vae.decode(z))
            tokens = logits.argmax(dim=-1)
            match = (tokens == padded).float()
            acc = match[padded != enc.pad_token_id].mean().item()
            text = tokenizer.decode(tokens[0].tolist(), skip_special_tokens=True)
            print(f"sigma={sigma:<4} acc={acc:6.1%}  {text!r}")

    print(
        "\n[probe] reading: acc@0 is the codec ceiling (must be ~100% for any"
        "\n        fluent generation); the falloff rate vs sigma is the latent"
        "\n        robustness the LM head depends on."
    )


if __name__ == "__main__":
    main()
