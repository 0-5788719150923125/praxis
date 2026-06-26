#!/usr/bin/env python3
"""Render the README's web-app stills from the real frontend.

Boots the Flask/API server with stub plumbing, drives the built frontend in
headless Chromium (Playwright), and captures one looping webp per scene:

  - static/chat.webp          Gymnasium tab, Evaluate mode, scripted chat
  - static/dashboard.webp     Terminal tab, live training dashboard
  - static/architecture.webp  Customs tab, focused on the Architecture card

  python tools/render_web.py                 # all scenes
  python tools/render_web.py --only chat      # one scene

Every scene is fed deterministic, canned data (a `--beta` run): the dashboard
from an injected LiveMetrics snapshot, the Customs tab from a stubbed
``/api/spec`` response, the chat from a scripted conversation. Each is a true
loop: the prism logo (prism.js, ``?seed=N&paused`` + ``window.prismStep``)
advances a fixed number of steps per frame and infinite CSS animations (caret,
tab glow) are paused and seeked per frame, so captures are byte-stable on a
given machine + browser. Every external request (fonts, chart.js, socket.io)
is intercepted and served from vendored bytes or stubs, so it runs offline.

Auto-regenerated on launch by praxis/docs.py when the web frontend changes.
"""

import argparse
import io
import json
import math
import os
import re
import sys
import time
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

FONT_PATH = os.path.join(REPO, "tools/assets/inter/inter-latin.woff2")

SEED = 42
ARG_HASH = "c73b55e61"  # the --beta run
FULL_HASH = "c73b55e61a7c50f0946f6ad508671b5ed8286b199dfe0abba3dc5edaf432b65e"
PORT = 2150  # clear of the real server's 2100-2120 range

# 10fps. 120 frames = 12s: a multiple of the caret blink (1s) and the active
# tab glow (3s alternate = 6s period), so both wrap seamlessly at the loop.
FRAME_MS = 100
N_FRAMES = 120
STEPS_PER_FRAME = 6  # prism runs fixed 60fps steps; 6 steps = one 100ms frame


# ---------------------------------------------------------------------------
# Canned --beta run data
# ---------------------------------------------------------------------------

CONVERSATION = [
    {"role": "user", "content": "Where do the words come from?"},
    {
        "role": "assistant",
        "content": (
            "From interference, mostly. Every token is a vote among "
            "oscillators - I keep the waves that agree, damp the ones that "
            "don't, and read what settles out of the standing pattern. What "
            "you are reading is the residue of a thousand small arguments."
        ),
    },
]

# --beta's instantiated module tree (captured from build/runs/<hash>/spec.json).
MODEL_ARCHITECTURE = """PraxisForCausalLM(
  (encoder): ByteLatentEncoder(architecture='conv', patching='space', n_encoders=2, n_decoders=2)
  (embeds): AdditiveEmbedding(ByteEmbedding(vocab=264, dim=64) + HashEmbedding(vocab=16384, dim=64, groups=[3, 4, 5], functions=1))
  (decoder): SequentialDecoder(
    (width): FullWidth()
    (controller): BaseController()
    (compressor): NoCompression()
    (order): NoSort()
    (halting): KLDivergenceHalting()
    (locals): ModuleList(
      (0): LocalLayer(
        (block): TransformerBlock(
          (attn_res): ResidualConnection()
          (attn_norm): SandwichNorm((96,), eps=1e-05, elementwise_affine=True)
          (attn): ArcAttention(
            (encoding): ArcHoPE(
              (depth_log_theta): Embedding(3, 1)
            )
            (qkv): Linear(in_features=96, out_features=288, bias=True)
            (output): Linear(in_features=96, out_features=96, bias=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (depth_qkv_bias): Embedding(3, 288)
            (depth_output_bias): Embedding(3, 96)
            (gate): Linear(in_features=96, out_features=96, bias=True)
          )
          (memory): MemoryBase()
          (ffn_res): ResidualConnection()
          (ffn_norm): SandwichNorm((96,), eps=1e-05, elementwise_affine=True)
          (ffn): ArcGLU(
            (up): Linear(in_features=96, out_features=256, bias=True)
            (act): ModuleList(
              (0-2): 3 x Serpent()
            )
            (dropout): Dropout(p=0.0, inplace=False)
            (down): Linear(in_features=128, out_features=96, bias=True)
          )
        )
      )
    )
  )
  (head): ForwardHead(hidden_size=64, vocab_size=264)
  (recall_policies): ModuleDict()
  (criterion): CrossEntropyLoss()
  (tasker): DifficultyTaskLossWeighter()
  (aux): ContrastiveIsotropyLoss()
  (strategy): NaiveSummation()
)
"""

SPEC_ARGS = {
    "experiment": "beta",
    "device": "cuda",
    "batch_size": 16,
    "seed": SEED,
    "encoder_type": "byte_latent",
    "block_type": "transformer",
    "attention_type": "arc",
    "encoding_type": "arc_hope",
    "ffn_type": "arc_glu",
    "activation": "serpent",
    "halting_type": "kl_divergence",
    "norm_type": "sandwich",
    "optimizer": "Lion",
    "sampler": "novelty",
    "depth": 3,
    "hidden_size": 96,
}

SPEC_PAYLOAD = {
    "truncated_hash": ARG_HASH,
    "full_hash": FULL_HASH,
    "args": SPEC_ARGS,
    "model_architecture": MODEL_ARCHITECTURE,
    "param_stats": {"model_parameters": 3376912, "optimizer_parameters": 0},
    "timestamp": "2026-06-26 09:14:02",
    "command": "python main.py --beta",
    "seed": SEED,
    "commit_timestamp": 1782400000,
    "git_url": "https://src.eco/praxis",
    "masked_git_url": "https://src.eco/praxis",
    "is_snapshot": False,
}


def _loss_history(n=50):
    """A smooth, training-like descent into ~1.9 with mild ripple. Pure
    function of the index, so the sparkline is byte-stable."""
    return [
        round(1.86 + 1.15 * math.exp(-i / 16.0) + 0.03 * math.sin(i / 2.3), 4)
        for i in range(n)
    ]


CONTEXTS = [
    {
        "name": "Ambient",
        "description": "always-on, low temperature",
        "temperature": 0.5,
        "chance": 1.0,
        "tokens": 512,
        "text": (
            "the model reads the corpus the way water finds a level, settling "
            "slowly into the shape of the text. each byte is a small vote, and "
            "the patches that agree are kept while the noise is damped away. "
            "what remains is a quiet consensus, a sentence assembled from the "
            "residue of a thousand interfering waves."
        ),
    },
    {
        "name": "Probe",
        "description": "mid temperature, chance-gated",
        "temperature": 0.7,
        "chance": 0.5,
        "tokens": 384,
        "text": (
            "a warmer draw lets the standing pattern wander further from the "
            "mean, surfacing rarer continuations before the schedule cools it "
            "back toward the basin. the difficulty weighter notices which tasks "
            "still resist and quietly leans the sampler their way."
        ),
    },
    {
        "name": "Dream",
        "description": "high temperature, rare",
        "temperature": 1.0,
        "chance": 0.25,
        "tokens": 256,
        "text": (
            "at the edge of the distribution the bytes loosen and recombine, "
            "half-words and invented punctuation drifting past as the halting "
            "head decides, again and again, that it has not yet seen enough to "
            "stop."
        ),
    },
]

INFO = {
    "optimizer": "Lion",
    "scheduler": "cosine",
    "sampler": "novelty",
    "halting": "kl_divergence",
    "encoder": "byte_latent",
    "attention": "arc",
    "device": "cuda",
    "precision": "bf16",
    "batch_size": 16,
    "target_batch": 256,
    "dropout": 0.0,
    "regularizers": "contrastive_isotropy",
}

LOG_LINES = [
    "praxis.trainer - INFO - resuming from checkpoint at step 14600",
    "praxis.data - INFO - interleaving 6 datasets (sampler=novelty)",
    "praxis.encoder - INFO - byte-latent codec frozen at step 12000",
    "lightning - INFO - validation pass: val_loss improved to 1.93",
    "praxis.sampler - INFO - difficulty reweight applied to 4 tasks",
    "praxis.trainer - INFO - checkpoint written (step=14820)",
]

SNAPSHOT = {
    "loss": 1.91,
    "loss_history": _loss_history(),
    "swarm_loss_history": [],
    "val_loss": 1.93,
    "accuracy": [0.318, 0.292],
    "fitness": None,
    "memory_churn": None,
    "batch": 256,
    "step": 14820,
    "rate": 0.41,
    "num_tokens": 0.834,
    "context_tokens": 512,
    "total_params": "3.38M",
    "local_layers": 3,
    "remote_layers": 0,
    "mode": "train",
    "stage": "pretrain",
    "events": [],
    "seed": SEED,
    "arg_hash": ARG_HASH,
    "url": "localhost:2100",
    "hours_elapsed": 6.27,
    "status_text": CONTEXTS[0]["text"],
    "contexts": CONTEXTS,
    "info": INFO,
    "update_count": 1,
    "log_lines": LOG_LINES,
}


# ---------------------------------------------------------------------------
# Server + request stubbing
# ---------------------------------------------------------------------------

FONT_CSS = """
@font-face {
    font-family: 'Inter';
    font-style: normal;
    font-weight: 100 900;
    font-display: block;
    src: url('/__assets/inter-latin.woff2') format('woff2');
}
"""

# socket.io stub: connects (green dot) then stays silent. No reconnect churn.
SOCKET_STUB_JS = """
(function () {
    function makeSocket() {
        const handlers = {};
        const sock = {
            connected: false,
            on(ev, cb) {
                (handlers[ev] = handlers[ev] || []).push(cb);
                if (ev === 'connect' && !sock._scheduled) {
                    sock._scheduled = true;
                    setTimeout(() => {
                        sock.connected = true;
                        (handlers['connect'] || []).forEach((cb) => cb());
                    }, 0);
                }
                return sock;
            },
            emit() { return sock; },
            off() { return sock; },
            disconnect() { sock.connected = false; return sock; },
        };
        return sock;
    }
    const io = () => makeSocket();
    io.connect = () => makeSocket();
    window.io = io;
})();
"""


class _StubGenerator:
    """Just enough generator surface for routes; never asked to generate."""

    model = None

    def request_generation(self, prompt, kwargs):
        return "capture_0"

    def get_result(self, request_id):
        return ""


def start_server():
    # praxis.cli parses sys.argv on import; hide this tool's flags from it.
    argv, sys.argv = sys.argv, sys.argv[:1]
    try:
        from praxis.web import APIServer
        from praxis.web.src.build import build_dev
    finally:
        sys.argv = argv

    build_dev()

    server = APIServer(
        _StubGenerator(),
        "127.0.0.1",
        PORT,
        tokenizer=None,
        integration_loader=None,
        param_stats={"total": 3_376_912, "trainable": 3_376_912},
        seed=SEED,
        truncated_hash=ARG_HASH,
        full_hash=FULL_HASH,
        dev_mode=False,
        launch_command="python main.py --beta",
    )
    server.start()

    url = f"http://127.0.0.1:{server.port}/"
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return server, url
        except OSError:
            time.sleep(0.2)
    raise RuntimeError("API server never became reachable")


def _route_external(context, origin):
    """Serve every non-local request from vendored bytes or stubs, and stub
    /api/spec with our canned --beta payload. Most-recently-registered wins."""
    font_bytes = open(FONT_PATH, "rb").read()

    def external(route):
        if route.request.url.startswith(origin):
            route.continue_()
        else:
            route.abort()

    context.route("**/*", external)
    context.route(
        "https://fonts.googleapis.com/**",
        lambda route: route.fulfill(content_type="text/css", body=FONT_CSS),
    )
    context.route(
        f"{origin}__assets/inter-latin.woff2",
        lambda route: route.fulfill(content_type="font/woff2", body=font_bytes),
    )
    context.route(
        "https://cdn.jsdelivr.net/**",
        lambda route: route.fulfill(
            content_type="application/javascript", body="/* chart.js stubbed */"
        ),
    )
    context.route(
        "https://cdnjs.cloudflare.com/**",
        lambda route: route.fulfill(
            content_type="application/javascript", body=SOCKET_STUB_JS
        ),
    )
    context.route(
        re.compile(r"/api/spec(\?.*)?$"),
        lambda route: route.fulfill(
            content_type="application/json", body=json.dumps(SPEC_PAYLOAD)
        ),
    )


# Installs window.__tick(f): advance the prism a fixed number of steps and
# seek every infinite CSS animation to the frame's timestamp, then sit still.
INSTALL_TICK_JS = """
([frameMs, steps]) => {
    window.__tick = (f) => {
        for (let i = 0; i < steps; i++) window.prismStep();
        for (const a of document.getAnimations()) {
            const t = a.effect && a.effect.getTiming();
            if (t && t.iterations === Infinity) {
                a.pause();
                a.currentTime = f * frameMs;
            }
        }
    };
    window.__tick(0);
}
"""


# ---------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------


def _setup_chat(page):
    page.wait_for_function("document.getElementById('message-input')")
    # Flip to Evaluate through the real action (button highlight, message
    # panel), then inject the conversation into the live module singletons.
    page.click('.tool-toggle[data-tool="evaluate"]')
    page.evaluate(
        """async (messages) => {
            const { state } = await import('/static/js/state.js');
            const { render, updateInputContainerStyling } =
                await import('/static/js/render.js');
            state.messages = messages;
            updateInputContainerStyling();
            render();
        }""",
        CONVERSATION,
    )
    page.wait_for_timeout(500)


def _clip_chat(page):
    # The app column, ending just below the input box.
    return page.evaluate("""() => {
        const app = document.querySelector('.app-container').getBoundingClientRect();
        const input = document.querySelector('.input-container').getBoundingClientRect();
        return {
            x: Math.round(app.x),
            y: Math.round(app.y),
            width: Math.round(app.width),
            height: Math.ceil(input.bottom - app.y) + 18,
        };
    }""")


def _setup_dashboard(page):
    page.evaluate(
        """async (snap) => {
            const { state } = await import('/static/js/state.js');
            state.liveMetrics.connected = true;
            state.liveMetrics.data = snap;
            state.terminal.connected = true;
        }""",
        SNAPSHOT,
    )
    page.click('.tab-button[data-tab="terminal"]')
    page.wait_for_selector(".live-dashboard", state="visible")
    page.wait_for_timeout(400)


def _clip_dashboard(page):
    return page.evaluate("""() => {
        const app = document.querySelector('.app-container').getBoundingClientRect();
        const dash = document.querySelector('.live-dashboard').getBoundingClientRect();
        return {
            x: Math.round(app.x),
            y: Math.round(app.y),
            width: Math.round(app.width),
            height: Math.ceil(dash.bottom - app.y) + 18,
        };
    }""")


def _setup_architecture(page):
    page.click('.tab-button[data-tab="spec"]')
    page.wait_for_selector("#spec-deck .chart-card", state="attached")
    # Jump the deck to the Architecture card and let the slide settle.
    page.evaluate("""async () => {
        const charts = await import('/static/js/charts.js');
        charts.requestDeckFocus('spec-deck', { title: 'Architecture' });
        charts.applyDeckFocus('spec-deck');
    }""")
    page.wait_for_timeout(900)


def _clip_architecture(page):
    # The full app column down to the bottom of the spec deck - a realistic
    # view of the Customs tab with the Architecture card in focus.
    return page.evaluate("""() => {
        const app = document.querySelector('.app-container').getBoundingClientRect();
        const deck = document.querySelector('#spec-deck').getBoundingClientRect();
        return {
            x: Math.round(app.x),
            y: Math.round(app.y),
            width: Math.round(app.width),
            height: Math.ceil(deck.bottom - app.y) + 24,
        };
    }""")


SCENES = {
    "chat": {
        "out": "static/chat.webp",
        "viewport": {"width": 944, "height": 720},
        "setup": _setup_chat,
        "clip": _clip_chat,
    },
    "dashboard": {
        "out": "static/dashboard.webp",
        "viewport": {"width": 944, "height": 1180},
        "setup": _setup_dashboard,
        "clip": _clip_dashboard,
    },
    "architecture": {
        "out": "static/architecture.webp",
        "viewport": {"width": 944, "height": 1320},
        "setup": _setup_architecture,
        "clip": _clip_architecture,
    },
}

BROWSER_ARGS = [
    "--force-color-profile=srgb",
    "--disable-lcd-text",
    "--font-render-hinting=none",
    "--disable-gpu",
    "--hide-scrollbars",
    f"--js-flags=--random-seed={SEED}",
]


def _capture_scene(browser, url, scene, n_frames):
    context = browser.new_context(viewport=scene["viewport"], device_scale_factor=1)
    _route_external(context, url)
    page = context.new_page()
    try:
        page.goto(f"{url}?seed={SEED}&paused=1", wait_until="load")
        # Boolean, not the function itself: wait_for_function can't serialize a
        # function return value to test its truthiness (it hangs).
        page.wait_for_function("typeof window.prismStep === 'function'")
        scene["setup"](page)
        page.evaluate(INSTALL_TICK_JS, [FRAME_MS, STEPS_PER_FRAME])
        clip = scene["clip"](page)
        if not clip:
            raise RuntimeError(f"scene '{scene['name']}' produced no clip")
        frames = [page.screenshot(clip=clip)]
        for f in range(1, n_frames):
            page.evaluate("f => window.__tick(f)", f)
            frames.append(page.screenshot(clip=clip))
        return frames
    finally:
        context.close()


def _encode_webp(frames):
    from PIL import Image

    images = [Image.open(io.BytesIO(png)).convert("RGB") for png in frames]
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
    return buf.getvalue()


def _write_if_changed(path, data):
    if os.path.exists(path) and open(path, "rb").read() == data:
        # Source changed but bytes didn't - bump mtime so the launch-time
        # staleness check doesn't re-fire.
        os.utime(path, None)
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"  # temp + rename: never a half file
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)
    return True


def render(names=None, n_frames=N_FRAMES):
    """Render the named scenes (default all) and write their webps. Returns
    the list of scenes whose bytes changed."""
    from playwright.sync_api import sync_playwright

    if not os.path.exists(FONT_PATH):
        raise FileNotFoundError(f"vendored font missing: {FONT_PATH}")

    names = names or list(SCENES)
    os.chdir(REPO)  # the server stamps repo_root from cwd
    server, url = start_server()
    captured = {}
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(args=BROWSER_ARGS)
            for name in names:
                scene = dict(SCENES[name], name=name)
                print(f"capturing {name}...", file=sys.stderr)
                captured[name] = _capture_scene(browser, url, scene, n_frames)
            browser.close()
    finally:
        server.stop()

    changed = []
    for name, frames in captured.items():
        out = os.path.join(REPO, SCENES[name]["out"])
        if _write_if_changed(out, _encode_webp(frames)):
            changed.append(name)
    return changed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--only",
        action="append",
        choices=list(SCENES),
        help="render only this scene (repeatable); default is all",
    )
    ap.add_argument("--frames", type=int, default=N_FRAMES, help="loop length")
    args = ap.parse_args()

    names = args.only or list(SCENES)
    print(f"rendering: {', '.join(names)}", file=sys.stderr)
    render(names, args.frames)
    for name in names:
        path = os.path.join(REPO, SCENES[name]["out"])
        size = os.path.getsize(path)
        print(f"wrote {path} ({size / 1e6:.2f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
