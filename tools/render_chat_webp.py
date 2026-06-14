#!/usr/bin/env python3
"""Render the web app's Gymnasium tab to a deterministic looping webp.

Boots the real Flask/API server with stub model plumbing, drives the real
frontend in headless Chromium (Playwright), and captures a scripted scene:
Evaluate mode active, a short fake conversation, the status dot connected.
Frames are clipped to the app container, so no browser chrome or margins.

  python tools/render_chat_webp.py            # -> static/chat.webp

Determinism: the prism logo animation runs under ?seed=N&paused (see
praxis/web/src/js/prism.js) and is advanced manually between screenshots;
infinite CSS animations (prompt caret, tab glow) are paused and seeked to
each frame's timestamp; every external request (fonts, chart.js, socket.io)
is intercepted and served from vendored bytes or stubs, so the capture is
offline and byte-stable on a given machine + browser build. The loop length
is a multiple of every CSS animation period, so only the prism cuts at the
seam - and it respawns tendrils constantly anyway.
"""

import argparse
import hashlib
import io
import os
import sys
import time
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

OUT_PATH = os.path.join(REPO, "static/chat.webp")
FONT_PATH = os.path.join(REPO, "tools/assets/inter/inter-latin.woff2")

SEED = 42
ARG_HASH = hashlib.sha256(b"praxis").hexdigest()[:9]
PORT = 2150  # clear of the real server's 2100-2120 range

# 10fps. 120 frames = 12s: a multiple of the caret blink (1s) and the active
# tab glow (3s alternate = 6s period), so both wrap seamlessly at the loop.
FRAME_MS = 100
N_CLIP = 120
STEPS_PER_FRAME = 6  # prism runs fixed 60fps steps; 6 steps = one 100ms frame

# Viewport sized so the 900px app column leaves a slim 22px gutter per side;
# the screenshot clips to the column itself.
VIEWPORT = {"width": 944, "height": 720}

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

# socket.io client stub: connects (so the status dot goes green), then stays
# silent forever - no reconnect churn, no server pushes, no network.
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

FONT_CSS = """
@font-face {
    font-family: 'Inter';
    font-style: normal;
    font-weight: 100 900;
    font-display: block;
    src: url('/__assets/inter-latin.woff2') format('woff2');
}
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
        param_stats={"total": 82_400_000, "trainable": 82_400_000},
        seed=SEED,
        truncated_hash=ARG_HASH,
        full_hash=hashlib.sha256(b"praxis").hexdigest(),
        dev_mode=False,
        launch_command="./launch",
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
    """Serve every non-local request from vendored bytes or stubs (or kill it).
    Playwright matches the most recently registered route first."""
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


def capture_frames(url, n_frames):
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(
            args=[
                "--force-color-profile=srgb",
                "--disable-lcd-text",
                "--font-render-hinting=none",
                "--disable-gpu",
                "--hide-scrollbars",
                f"--js-flags=--random-seed={SEED}",
            ]
        )
        context = browser.new_context(viewport=VIEWPORT, device_scale_factor=1)
        _route_external(context, url)
        page = context.new_page()

        page.goto(f"{url}?seed={SEED}&paused=1", wait_until="load")
        page.wait_for_function(
            "window.prismStep && document.getElementById('message-input')"
        )

        # Flip to Evaluate through the real action (button highlight, message
        # panel, the '< Shoot' placeholder), then inject the conversation into
        # the live module singletons.
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
        # Let the one-shot message fadeIn (0.3s) and scroll settle.
        page.wait_for_timeout(500)

        # Freeze the scene clock: prism advances only via prismStep, infinite
        # CSS animations only via explicit seeks.
        page.evaluate(
            """([frameMs, steps]) => {
                window.__chatTick = (f) => {
                    for (let i = 0; i < steps; i++) window.prismStep();
                    for (const a of document.getAnimations()) {
                        const t = a.effect && a.effect.getTiming();
                        if (t && t.iterations === Infinity) {
                            a.pause();
                            a.currentTime = f * frameMs;
                        }
                    }
                };
                window.__chatTick(0);
            }""",
            [FRAME_MS, STEPS_PER_FRAME],
        )

        # Crop to content: the app column, ending just below the input box
        # (the chat tab otherwise stretches to the full viewport height).
        clip = page.evaluate("""() => {
                const app = document.querySelector('.app-container')
                    .getBoundingClientRect();
                const input = document.querySelector('.input-container')
                    .getBoundingClientRect();
                return {
                    x: Math.round(app.x),
                    y: Math.round(app.y),
                    width: Math.round(app.width),
                    height: Math.ceil(input.bottom - app.y) + 18,
                };
            }""")

        frames = [page.screenshot(clip=clip)]
        for f in range(1, n_frames):
            page.evaluate("f => window.__chatTick(f)", f)
            frames.append(page.screenshot(clip=clip))

        browser.close()
    return frames


def render_webp(out_path, n_frames=N_CLIP):
    """Capture, encode and write the looping webp. Idempotent: only writes
    when the encoded bytes differ. Returns True if the file changed."""
    from PIL import Image

    if not os.path.exists(FONT_PATH):
        raise FileNotFoundError(f"vendored font missing: {FONT_PATH}")

    os.chdir(REPO)  # the server stamps repo_root from cwd
    server, url = start_server()
    try:
        shots = capture_frames(url, n_frames)
    finally:
        server.stop()

    images = [Image.open(io.BytesIO(png)).convert("RGB") for png in shots]
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
        # staleness check (sources newer than the webp?) doesn't re-fire.
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
    ap.add_argument("--frames", type=int, default=N_CLIP, help="loop length")
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    print(f"capturing {args.frames} frames...", file=sys.stderr)
    changed = render_webp(args.out, args.frames)
    size = os.path.getsize(args.out)
    verb = "wrote" if changed else "unchanged"
    print(
        f"{verb} {args.out} ({size / 1e6:.2f} MB, {args.frames} frames)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
