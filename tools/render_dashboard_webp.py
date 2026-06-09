#!/usr/bin/env python3
"""Render the terminal dashboard to a deterministic looping webp.

Drives ``TerminalDashboard`` headlessly with a scripted, periodic sequence of
the same signals the real training callback feeds it (see
praxis/callbacks/lightning/terminal.py), rasterizes each frame with a vendored
monospace font, and writes an animated webp. Because every input is seeded and
periodic, the output is byte-stable: it only churns when the dashboard's own
rendering changes.

  python tools/render_dashboard_webp.py            # -> static/terminal.webp

Seamless loop: the rolling CONTEXT generation scrolls a cyclic corpus exactly
one full cycle over the run, so the visible tail at the last frame wraps back
into the first. The loss chart uses the same period (a full rotation of its
ring buffer). Monotonic counters (step/batch/tokens/age) advance across the
loop and snap at the seam, as any looping progress display does.
"""

import argparse
import hashlib
import math
import os
import random
import sys
from collections import deque

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from praxis.interface import TerminalDashboard

FONT_PATH = os.path.join(REPO, "tools/assets/DejaVuSansMono.ttf")
OUT_PATH = os.path.join(REPO, "static/terminal.webp")

# Geometry / look. The dashboard scales to whatever grid it's given, so we just
# hand it a roomy COLS x LINES grid and rasterize at its native cell size (font
# 20 -> 12x24 px cells). That yields a ~960p canvas; the README/browser scales
# it down to fit, which is the "rendered big, shown smaller" crispness the old
# capture had - with no per-frame resample. A wider grid also stops the footer
# and info-panel keys from truncating.
COLS, LINES = 140, 38
FONT_SIZE = 20
FRAME_MS = 100  # 10fps - real-time, so the correlation animates at full speed
SEED = 42
# Real run hashes are the first 9 hex chars of a sha256 over the args (see
# praxis/cli/core/logger.py); match that format rather than a made-up string.
ARG_HASH = hashlib.sha256(b"praxis").hexdigest()[:9]
FG = (222, 224, 228)  # off-white, matching the training-loop terminal
BG = (13, 13, 16)

# Generation cadence: append one token every WORD_EVERY frames (~1.25 words/s at
# 10fps), close to the real loop's slow token stream. The buffer GROWS - tokens
# are appended at the end, exactly like generation - and the panel shows its
# tail, scrolling gently only when a new line is needed (never re-flowing the
# whole block).
#
# Render length and text length are decoupled: the webp is a short N_CLIP-frame
# sample, while the full corpus lives in the dashboard's buffer so the panel is
# always full of stable context. The clip sits deep in that buffer and only a
# couple dozen tokens stream in over the loop; the boundary is a plain cut
# between two full-text moments (generation can't wrap its tail back to its head).
WORD_EVERY = 8
N_CLIP = 220  # rendered frames (~22s loop) - keep this small; it sets the cost

# A believable small-model generation streamed one token at a time.
CONTEXT_CORPUS = """
the model settles into a harmonic basin where each attention head learns to
track a distinct band of the latent spectrum, and the slower oscillations carry
the long range structure of the argument while the faster ones resolve the
local detail of each clause. consensus among the heads grows roughly linearly
as the depth of the network increases, yet the interference between mismatched
phases grows only with its logarithm, which is the reason the deeper stacks
stay coherent for far longer spans of text before they begin to drift away from
the prompt. we deliberately keep the bias and the variance on two orthogonal
axes, so the optimizer can trade one against the other without ever collapsing
the representation into a single degenerate mode. each recurrent pass re-enters
the very same field it has only just left, a little more sharply tuned than
before, and the residual stream slowly sharpens toward the shape that was
already waiting inside the data. when the context window finally fills, the
rolling buffer quietly forgets its oldest tokens and the generation simply
continues, never quite repeating itself and yet always circling back near to
the themes where it first began. the watchmaker is blind, but the gradient is
patient, and over enough small steps the wave finds a standing pattern that
holds. we measure the error against the validation set, watch the loss descend
in uneven steps, and let the schedule cool the learning rate as the basin grows
narrow around the minimum. Nothing here is certain, only probable, and the
model goes on speaking its quiet harmonic language into the dark, one token at a
time, until the next reset. Between each reset the optimizer takes a thousand
small and careful steps, nudging the weights along the steepest reachable
direction while the momentum smooths the noise of the individual batches into a
single coherent drift. We watch the gradient norm rise and fall like a tide,
and when it spikes we know that some rare and difficult example has arrived to
reshape a corner of the manifold that the easy data never once touched. The
embeddings drift apart into clean and well separated clusters, the rare tokens
find their own quiet neighborhoods, and the attention maps grow sparse and
confident where they were once diffuse and uncertain. Nothing is memorized that
can instead be inferred, and nothing is inferred that the deeper structure of
the language does not already quietly imply. Slowly, and with great patience,
the model learns to predict not the surface of the text but the generating
process beneath it, the hidden grammar of cause and consequence, and that is
the whole of the task, repeated again and again until the loss can fall no
further and the basin closes gently over the weights like still water.
""".split()

# The clip streams REVEAL_WORDS over its length and ends on the full passage, so
# PREFILL is everything before that tail - a long, panel-filling context.
REVEAL_WORDS = N_CLIP // WORD_EVERY
PREFILL_WORDS = len(CONTEXT_CORPUS) - REVEAL_WORDS
DEFAULT_FRAMES = N_CLIP

# We feed the dashboard only from STATUS_START onward (the rest scrolled off long
# ago) so it isn't re-wrapping the whole passage each frame. The start is FIXED,
# not a sliding tail - that's what keeps appended tokens from re-flowing the
# panel; only the new word at the end moves.
STATUS_START = max(0, PREFILL_WORDS - 160)

# Realistic, timestamp-free log lines (timestamps would break determinism).
LOG_RING = [
    "praxis.trainer - INFO - resuming from checkpoint at step 12000",
    "praxis.data - INFO - dataset shard 3/8 streamed (cache warm)",
    "Training stage: preflight -> pretrain",
    "praxis.optim - INFO - lr warmup complete, entering cosine decay",
    "lightning - INFO - validation pass: val_loss improved to 1.84",
    "praxis.encoder - INFO - codec frozen, KL anneal locked to warmup",
    "praxis.memory - INFO - surprise within band, no reset triggered",
    "praxis.trainer - INFO - checkpoint written (step=12180)",
    "praxis.sampler - INFO - difficulty reweight applied to 4 tasks",
]


def _two_sig(x):
    return f"{x:.2f}"


def build_frames(n_frames):
    """Drive the dashboard and return a list of frames (each a list[str])."""
    dash = TerminalDashboard(seed=SEED, arg_hash=ARG_HASH, max_data_points=n_frames)
    # Headless: never start() (no thread / terminal takeover) and suppress the
    # atexit terminal restore since we never saved terminal state.
    dash.terminal_manager.terminal_restored = True
    dash._get_terminal_size = lambda: os.terminal_size((COLS, LINES))

    dash.update_params(int(82.4e6))
    dash.update_url("http://localhost:5000")
    dash.set_mode("train")
    dash.set_stage("pretrain")

    # Static-ish info panel: the real keys from terminal.py _update_dashboard.
    base_info = {
        "device": "cuda:0",
        "ram": "14.2GB/31.3GB",
        "vram": "6.10GB/8.00GB",
        "optimizer": "Muon",
        "strategy": "ddp",
        "policy": "reinforce",
        "vocab_size": 8192,
        "block_size": 512,
        "batch_size": 16,
        "target_batch": 256,
        "depth": 5,
        "local_layers": 5,
        "remote_layers": 0,
        "hidden_size": 512,
        "embed_size": 256,
        "dropout": 0.1,
        "debug": False,
        "meta": [],
    }
    dash.update_info(base_info)
    dash.update_layer_count(5, 0)

    # Pre-fill the loss ring with one full period so the chart starts mid-stream.
    losses = [_loss(f, n_frames) for f in range(n_frames)]
    dash.state.train_losses = deque(losses, maxlen=n_frames)

    # Pre-fill logs with the ring so the first frame already shows a full tail.
    for line in LOG_RING:
        dash.add_log(line)

    log_every = max(1, n_frames // len(LOG_RING))

    # Seed RNG immediately before the loop; the dashboard's correlation bar
    # (random) and forest-fire automata (np.random) are the only consumers, so
    # the stream stays reproducible run to run.
    random.seed(SEED)
    np.random.seed(SEED)

    frames = []
    for f in range(n_frames):
        # Monotonic counters (snap at the loop seam, like any looping recording).
        step = 12000 + f
        dash.update_step(step)
        dash.update_batch(step // 16)
        dash.update_tokens(0.102 + f * 2.0e-4)
        dash.hours_since = lambda f=f: 3.20 + f * (0.4 / n_frames)
        dash.update_rate(0.42 + 0.05 * math.sin(2 * math.pi * f / n_frames))

        # Header shows only ERROR and VALIDATION - the metrics actually present
        # in a typical run. Fitness/surprise/accuracy stay None (auto-hidden).
        loss = losses[f]
        dash.update_loss(loss)  # rotates the pre-filled ring by one each frame
        dash.update_val(loss * 1.03)

        # Streaming generation: the buffer grows by one token every WORD_EVERY
        # frames. We feed from a FIXED start so appended tokens only extend the
        # bottom line (the panel scrolls a line at a time, never re-flows); the
        # token chip still reports the full context length.
        revealed = f // WORD_EVERY
        context_len = PREFILL_WORDS + revealed + 1
        dash.update_status(" ".join(CONTEXT_CORPUS[STATUS_START:context_len]))
        dash.update_context_tokens(round(context_len * 1.3))

        if f > 0 and f % log_every == 0:
            dash.add_log(LOG_RING[(f // log_every) % len(LOG_RING)])

        frame = dash.frame_builder.correct_borders(dash._create_frame())
        frames.append(frame)

    return frames


def _loss(f, n):
    """Periodic, training-like EMA loss in ~[1.6, 2.2]."""
    t = 2 * math.pi * f / n
    return (
        1.88
        + 0.18 * math.sin(t)
        + 0.07 * math.sin(2 * t + 1.3)
        + 0.04 * math.sin(5 * t + 0.7)
    )


def rasterize(frames):
    """Glyph-atlas rasterizer: render each unique glyph once into an alpha tile,
    then blit tiles into a per-frame array. Far faster than per-line draw.text
    when there are many frames, and pixel-deterministic."""
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    cw = round(font.getlength("M"))
    ch = sum(font.getmetrics())

    atlas = {}
    for c in {c for fr in frames for line in fr for c in line if c != " "}:
        tile = Image.new("L", (cw, ch), 0)
        ImageDraw.Draw(tile).text((0, 0), c, font=font, fill=255)
        atlas[c] = np.asarray(tile, dtype=np.uint8)

    # alpha -> RGB lookup, so per-frame colorizing is a single fancy index.
    fg = np.array(FG, np.float32)
    bg = np.array(BG, np.float32)
    lut = (bg + (np.arange(256)[:, None] / 255.0) * (fg - bg)).astype(np.uint8)

    width, height = cw * COLS, ch * LINES
    images = []
    for frame in frames:
        alpha = np.zeros((height, width), np.uint8)
        for row, line in enumerate(frame):
            y = row * ch
            for col, c in enumerate(line[:COLS]):
                tile = atlas.get(c)
                if tile is not None:
                    alpha[y : y + ch, col * cw : col * cw + cw] = tile
        images.append(Image.fromarray(lut[alpha], "RGB"))
    return images


def render_webp(out_path, n_frames=DEFAULT_FRAMES):
    """Build, rasterize and write the looping webp. Idempotent: only writes
    when the encoded bytes differ. Returns True if the file changed."""
    import io

    if not os.path.exists(FONT_PATH):
        raise FileNotFoundError(f"vendored font missing: {FONT_PATH}")

    images = rasterize(build_frames(n_frames))
    buf = io.BytesIO()
    images[0].save(
        buf,
        save_all=True,
        append_images=images[1:],
        duration=FRAME_MS,
        loop=0,
        format="WEBP",
        lossless=True,
        method=6,
        exact=True,
    )
    data = buf.getvalue()

    if os.path.exists(out_path) and open(out_path, "rb").read() == data:
        # Source changed but bytes didn't - bump mtime so the launch-time
        # staleness check (sources newer than the webp?) doesn't re-fire forever.
        os.utime(out_path, None)
        return False
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = f"{out_path}.{os.getpid()}.tmp"  # temp + rename: never a half file
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, out_path)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=DEFAULT_FRAMES, help="loop length")
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    print(f"building {args.frames} frames...", file=sys.stderr)
    changed = render_webp(args.out, args.frames)
    size = os.path.getsize(args.out)
    verb = "wrote" if changed else "unchanged"
    print(
        f"{verb} {args.out} ({size / 1e6:.2f} MB, {args.frames} frames)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
