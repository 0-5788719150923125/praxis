"""The Gymnasium tab's half of streaming, driven in a real browser.

Three modules cooperate on the client: `api.js` mints a stream id and sends it
with the POST, `websocket.js` routes the frames that come back to whoever
registered that id, and `chatstream.js` turns the deltas into an assistant turn
that fills in. Every branch of that is something the user sees go wrong - a
turn that never appears, a turn per delta, stale text left standing after a
tool call, a reroll button offered on a reply still being written.

Uses `playwright.sync_api` directly rather than the pytest plugin, the way
`tools/render_web.py` does, so no new dependency is needed - the whole file
skips if the browser is unavailable.

WHAT IS SUBSTITUTED: the socket.io CLIENT is loaded from a CDN
(`templates/index.html`), so an offline browser cannot open a real one. `io` is
stubbed with a socket that can be handed frames from the test, which is what
lets the delta path be driven at all; the wire format those frames use is
pinned server-side in `tests/test_generation_stream_route.py`.
"""

import json

import pytest

sync_playwright = pytest.importorskip(
    "playwright.sync_api", reason="playwright is not installed"
).sync_playwright


# A socket.io stand-in that connects, records handlers, and lets the test push
# a frame in - which is the one thing the capture tool's stub cannot do.
SOCKET_STUB_JS = """
(function () {
    const sockets = [];
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
            _deliver(ev, payload) {
                (handlers[ev] || []).forEach((cb) => cb(payload));
            },
        };
        sockets.push(sock);
        return sock;
    }
    const io = () => makeSocket();
    io.connect = () => makeSocket();
    window.io = io;
    window.__deliver = (ev, payload) => sockets.forEach((s) => s._deliver(ev, payload));
})();
"""


class _StubGenerator:
    """Never asked to generate; the POST is fulfilled by a route."""

    model = None

    def request_generation(self, prompt, kwargs, deadline=None, **_):
        return "stub_0"

    def get_result(self, request_id):
        return ""


@pytest.fixture(scope="module")
def app_url():
    import sys
    import time
    import urllib.request

    from praxis.web.src.build import build_dev

    # praxis.cli parses sys.argv on import; hide pytest's flags from it.
    argv, sys.argv = sys.argv, sys.argv[:1]
    try:
        from praxis.web import APIServer
    finally:
        sys.argv = argv

    build_dev()
    server = APIServer(
        _StubGenerator(),
        "127.0.0.1",
        2199,
        tokenizer=None,
        integration_loader=None,
        dev_mode=False,
    )
    server.start()

    url = f"http://127.0.0.1:{server.port}/"
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    break
        except OSError:
            time.sleep(0.2)
    else:
        pytest.skip("API server never became reachable")
    yield url


@pytest.fixture(scope="module")
def page(app_url):
    with sync_playwright() as play:
        browser = play.chromium.launch()
        context = browser.new_context(viewport={"width": 1280, "height": 900})
        context.route(
            "**/*",
            lambda route: (
                route.continue_()
                if route.request.url.startswith(app_url)
                else route.abort()
            ),
        )
        context.route(
            "https://cdnjs.cloudflare.com/**",
            lambda route: route.fulfill(
                content_type="application/javascript", body=SOCKET_STUB_JS
            ),
        )
        context.route(
            "https://cdn.jsdelivr.net/**",
            lambda route: route.fulfill(
                content_type="application/javascript", body="/* chart.js stubbed */"
            ),
        )
        page = context.new_page()
        page.goto(app_url, wait_until="load")
        page.wait_for_timeout(400)
        yield page
        context.close()
        browser.close()


def _reset(page):
    page.evaluate("""async () => {
            const { state } = await import('/static/js/state.js');
            const { render } = await import('/static/js/render.js');
            // Messages only render outside Read mode - Read shows KB results
            // in the same pane. Evaluate is the texting-style chat view.
            state.conversationMode = 'evaluate';
            state.messages = [{ role: 'user', content: 'hi' }];
            state.isThinking = true;
            render();
        }""")


def _snapshot(page):
    return page.evaluate("""() => {
            // The thinking indicator is itself a `.message` (with no
            // `.message-content`), so it is excluded rather than counted.
            const nodes = document.querySelectorAll(
                '#chat-container .message:not(#thinking-message)');
            const last = nodes[nodes.length - 1];
            return {
                count: nodes.length,
                text: last ? last.querySelector('.message-content').textContent : null,
                streaming: last ? last.classList.contains('streaming') : false,
                reroll: !!document.getElementById('reroll-button'),
                thinking: !!document.querySelector('.thinking-status'),
            };
        }""")


# ---------------------------------------------------------------------------
# the whole client path: sendMessage -> socket frame -> turn -> DOM
# ---------------------------------------------------------------------------


def test_the_reply_fills_in_while_the_post_is_still_outstanding(page, app_url):
    """The point of the feature. The deltas arrive on the socket WHILE the POST
    is in flight, which is the ordering production has - and the id that routes
    them is the one `api.js` put in the request body."""
    _reset(page)
    seen = []

    def handle(route):
        body = json.loads(route.request.post_data)
        # The client mints it and sends it; the server only echoes it back.
        stream_id = body["stream_id"]
        for chunk in ("It ", "is ", "noo"):
            page.evaluate(
                "([id, text]) => window.__deliver('gen_delta', { id, text, seq: 1 })",
                [stream_id, chunk],
            )
            page.wait_for_timeout(40)
            seen.append(_snapshot(page))
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"response": "It is noon."}),
        )

    page.route(f"{app_url}messages/", handle)
    try:
        result = page.evaluate("""async () => {
                const { sendMessage } = await import('/static/js/api.js');
                const { streamingTurn } = await import('/static/js/chatstream.js');
                const { state } = await import('/static/js/state.js');
                const { render } = await import('/static/js/render.js');

                const turn = streamingTurn();
                const response = await sendMessage(state.messages, {
                    onDelta: turn.onDelta, onReset: turn.onReset
                });
                turn.settle(response.response);
                state.isThinking = false;
                render();
                return state.messages.map((m) => m.content);
            }""")
    finally:
        page.unroute(f"{app_url}messages/")

    # It grew, as ONE message beside the user's - never a bubble per delta.
    assert [s["text"] for s in seen] == ["It ", "It is ", "It is noo"]
    assert {s["count"] for s in seen} == {2}
    assert all(s["streaming"] for s in seen)
    # ...and settled on the POST's answer, which is what stays authoritative.
    assert result == ["hi", "It is noon."]
    assert _snapshot(page)["streaming"] is False


def test_deltas_for_another_request_are_ignored(page, app_url):
    """Ids keep concurrent requests from cross-talking, and stop a straggler
    from a finished turn landing in the next one."""
    _reset(page)

    def handle(route):
        page.evaluate(
            "() => window.__deliver('gen_delta', "
            "{ id: 'some-other-request', text: 'WRONG', seq: 1 })"
        )
        page.wait_for_timeout(40)
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"response": "mine"}),
        )

    page.route(f"{app_url}messages/", handle)
    try:
        result = page.evaluate("""async () => {
                const { sendMessage } = await import('/static/js/api.js');
                const { streamingTurn } = await import('/static/js/chatstream.js');
                const { state } = await import('/static/js/state.js');
                const { render } = await import('/static/js/render.js');
                const turn = streamingTurn();
                const response = await sendMessage(state.messages, {
                    onDelta: turn.onDelta, onReset: turn.onReset
                });
                turn.settle(response.response);
                state.isThinking = false;
                render();
                return state.messages.map((m) => m.content);
            }""")
    finally:
        page.unroute(f"{app_url}messages/")

    assert result == ["hi", "mine"]


def test_a_request_that_does_not_stream_sends_no_id(page, app_url):
    """The unchanged path: no `onDelta`, no `stream_id`, so the server builds no
    streamer at all and the behaviour is exactly what it was."""
    _reset(page)
    bodies = []

    def handle(route):
        bodies.append(json.loads(route.request.post_data))
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"response": "quiet reply"}),
        )

    page.route(f"{app_url}messages/", handle)
    try:
        result = page.evaluate("""async () => {
                const { sendMessage } = await import('/static/js/api.js');
                const { state } = await import('/static/js/state.js');
                const response = await sendMessage(state.messages);
                return response.response;
            }""")
    finally:
        page.unroute(f"{app_url}messages/")

    assert result == "quiet reply"
    assert "stream_id" not in bodies[0]


# ---------------------------------------------------------------------------
# the turn's own branches
# ---------------------------------------------------------------------------


def _drive(page, body):
    """Run a turn against the real modules, snapshotting after each step."""
    return page.evaluate(
        """async (body) => {
            const { streamingTurn } = await import('/static/js/chatstream.js');
            const { state } = await import('/static/js/state.js');
            const { render } = await import('/static/js/render.js');

            state.conversationMode = 'evaluate';
            state.messages = [{ role: 'user', content: 'hi' }];
            state.isThinking = true;
            render();

            const seen = [];
            const snap = () => {
                // The thinking indicator is itself a `.message` (with no
                // `.message-content`), so it is excluded rather than counted.
                const nodes = document.querySelectorAll(
                    '#chat-container .message:not(#thinking-message)');
                const last = nodes[nodes.length - 1];
                return {
                    count: nodes.length,
                    text: last ? last.querySelector('.message-content').textContent : null,
                    streaming: last ? last.classList.contains('streaming') : false,
                    reroll: !!document.getElementById('reroll-button'),
                    thinking: !!document.querySelector('.thinking-status'),
                };
            };

            const turn = streamingTurn();
            const step = async (fn) => {
                fn();
                // streamingTurn coalesces renders onto an animation frame.
                await new Promise((r) => requestAnimationFrame(() => r()));
                seen.push(snap());
            };

            await eval(body);
            return { seen, messages: state.messages.map((m) => m.content) };
        }""",
        body,
    )


def test_the_thinking_dots_give_way_to_text(page):
    out = _drive(
        page,
        """(async () => {
            await step(() => {});
            await step(() => turn.onDelta('answer'));
        })()""",
    )
    assert out["seen"][0]["thinking"] is True
    assert out["seen"][1]["thinking"] is False
    assert out["seen"][1]["text"] == "answer"


def test_a_turn_being_written_offers_no_reroll(page):
    """There is nothing to re-roll yet, and the request is still in flight."""
    out = _drive(
        page,
        """(async () => {
            await step(() => turn.onDelta('half a th'));
            await step(() => { turn.settle('half a thought'); render(); });
        })()""",
    )
    assert out["seen"][0]["streaming"] is True
    assert out["seen"][0]["reroll"] is False
    assert out["seen"][1]["streaming"] is False
    assert out["seen"][1]["reroll"] is True


def test_a_tool_call_clears_what_was_already_shown(page):
    """The runtime's turn anchor moves past each spliced tool result, so the
    reply is only what the model writes after it - a consumer that kept the
    pre-call chatter would show it beside an answer that excludes it."""
    out = _drive(
        page,
        """(async () => {
            await step(() => turn.onDelta('Checking now.'));
            await step(() => turn.onReset());
            await step(() => turn.onDelta('It is noon.'));
            await step(() => { turn.settle('It is noon.'); render(); });
        })()""",
    )
    seen = out["seen"]
    assert seen[0]["text"] == "Checking now."
    assert seen[1]["text"] == ""
    assert seen[-1]["text"] == "It is noon."
    assert "Checking now." not in out["messages"][-1]
    assert {s["count"] for s in seen} == {2}


def test_a_turn_that_streamed_nothing_is_still_appended(page):
    out = _drive(
        page,
        """(async () => {
            await step(() => { turn.settle('quiet reply'); render(); });
        })()""",
    )
    assert out["messages"] == ["hi", "quiet reply"]
    assert out["seen"][-1]["text"] == "quiet reply"
    assert out["seen"][-1]["streaming"] is False


def test_a_failed_request_leaves_no_half_written_turn(page):
    out = _drive(
        page,
        """(async () => {
            await step(() => turn.onDelta('partial...'));
            await step(() => {
                turn.discard();
                state.messages.push({ role: 'assistant', content: 'Error: boom' });
                render();
            });
        })()""",
    )
    assert out["messages"] == ["hi", "Error: boom"]
    assert out["seen"][-1]["text"] == "Error: boom"
    assert out["seen"][-1]["count"] == 2


# ---------------------------------------------------------------------------
# repaint discipline
#
# A byte-level model emits a delta per byte. Re-serializing the message list for
# each one destroyed and rebuilt every node, which is visible as flicker and
# takes the user's text selection and the caret's blink phase with it.
# ---------------------------------------------------------------------------


def test_streaming_does_not_rebuild_the_message_nodes(page):
    """The nodes on the page are the SAME objects across deltas - only the text
    inside one of them changes."""
    out = page.evaluate(
        """async () => {
            const { streamingTurn } = await import('/static/js/chatstream.js');
            const { state } = await import('/static/js/state.js');
            const { render } = await import('/static/js/render.js');

            state.conversationMode = 'evaluate';
            state.messages = [{ role: 'user', content: 'hi' }];
            state.isThinking = false;
            render();

            const frame = () => new Promise((r) => requestAnimationFrame(() => r()));

            // Captured AFTER the first delta: that one adds a message, which is
            // a structural change and legitimately rebuilds the list. What must
            // not happen is a rebuild for each of the deltas that follow.
            const turn = streamingTurn();
            turn.onDelta('one ');
            await frame();
            const nodes0 = document.querySelectorAll('#chat-container .message');
            const userNode = nodes0[0];
            const replyNode = nodes0[1];

            const identity = [];
            for (const chunk of ['two ', 'three ', 'four']) {
                turn.onDelta(chunk);
                await frame();
                const nodes = document.querySelectorAll('#chat-container .message');
                identity.push(nodes[0] === userNode && nodes[1] === replyNode);
            }
            return {
                identity,
                text: replyNode.querySelector('.message-content').textContent,
            };
        }"""
    )
    assert out["identity"] == [True, True, True], "nodes were rebuilt mid-stream"
    assert out["text"] == "one two three four"


def test_a_selection_in_an_earlier_message_survives_streaming(page):
    """Rebuilding the list dropped any text the user had selected. Patching one
    node leaves every other one - and its selection - untouched."""
    out = page.evaluate(
        """async () => {
            const { streamingTurn } = await import('/static/js/chatstream.js');
            const { state } = await import('/static/js/state.js');
            const { render } = await import('/static/js/render.js');

            state.conversationMode = 'evaluate';
            state.messages = [{ role: 'user', content: 'select me please' }];
            state.isThinking = false;
            render();

            const frame = () => new Promise((r) => requestAnimationFrame(() => r()));
            const turn = streamingTurn();

            // The first delta ADDS a message, which is a structural change and
            // legitimately rebuilds the list - one rebuild per turn, not per
            // byte. Select after that, which is also when a real user would:
            // while the reply is arriving.
            turn.onDelta('a');
            await frame();

            const target = document.querySelector('#chat-container .message-content');
            const range = document.createRange();
            range.selectNodeContents(target);
            const sel = window.getSelection();
            sel.removeAllRanges();
            sel.addRange(range);

            for (const chunk of ['b', 'c', 'd']) {
                turn.onDelta(chunk);
                await frame();
            }
            return window.getSelection().toString();
        }"""
    )
    assert out == "select me please"


# ---------------------------------------------------------------------------
# scroll discipline
# ---------------------------------------------------------------------------


def _stream_with_scroll(page, scroll_to_top):
    """Fill the pane past its height, optionally scroll up, then stream."""
    return page.evaluate(
        """async (scrollToTop) => {
            const { streamingTurn } = await import('/static/js/chatstream.js');
            const { state } = await import('/static/js/state.js');
            const { render } = await import('/static/js/render.js');

            state.conversationMode = 'evaluate';
            state.messages = Array.from({ length: 40 }, (_, i) => ({
                role: i % 2 ? 'assistant' : 'user',
                content: 'filler '.repeat(20) + i,
            }));
            state.isThinking = false;
            render();

            const box = document.getElementById('chat-container');
            const frame = () => new Promise((r) => requestAnimationFrame(() => r()));
            await frame();
            box.scrollTop = scrollToTop ? 0 : box.scrollHeight;
            const before = box.scrollTop;

            const turn = streamingTurn();
            for (const chunk of ['x'.repeat(40), 'y'.repeat(40), 'z'.repeat(40)]) {
                turn.onDelta(chunk);
                await frame();
            }
            await frame();
            return {
                scrollable: box.scrollHeight > box.clientHeight,
                before,
                after: box.scrollTop,
                atBottom:
                    box.scrollHeight - box.clientHeight - box.scrollTop <= 48,
            };
        }""",
        scroll_to_top,
    )


def test_a_reader_who_scrolled_up_is_not_yanked_back(page):
    """The one that actually hurts: re-reading an earlier turn while a reply
    streams used to drag the view to the bottom on every delta."""
    out = _stream_with_scroll(page, True)
    assert out["scrollable"], "the pane has to overflow for this to mean anything"
    assert out["before"] == 0
    assert out["after"] == 0, "streaming scrolled a user who had scrolled away"


def test_a_reader_at_the_tail_keeps_following(page):
    """...and the converse, which is what someone reading the newest reply
    wants: the view stays with the text as it arrives."""
    out = _stream_with_scroll(page, False)
    assert out["scrollable"]
    assert out["atBottom"], "the tail scrolled out of view while streaming"
    assert out["after"] > out["before"]


def test_the_users_own_new_turn_jumps_into_view(page):
    """What the user just did is worth jumping to whether or not they were
    following. A REPLY turn appearing is not - that one is the model's doing,
    and is covered by the sticky rule above."""
    out = page.evaluate(
        """async () => {
            const { state } = await import('/static/js/state.js');
            const { render } = await import('/static/js/render.js');

            state.conversationMode = 'evaluate';
            state.messages = Array.from({ length: 40 }, (_, i) => ({
                role: i % 2 ? 'assistant' : 'user',
                content: 'filler '.repeat(20) + i,
            }));
            state.isThinking = false;
            render();

            const box = document.getElementById('chat-container');
            const frame = () => new Promise((r) => requestAnimationFrame(() => r()));
            await frame();
            box.scrollTop = 0;

            const before = box.scrollTop;
            state.messages.push({ role: 'user', content: 'a brand new turn' });
            render();

            // The jump is SMOOTH here (it reads better for a turn the user just
            // sent), so wait for the animation to settle rather than guessing a
            // duration - it is proportional to the distance travelled.
            let last = -1;
            for (let i = 0; i < 60 && box.scrollTop !== last; i++) {
                last = box.scrollTop;
                await new Promise((r) => setTimeout(r, 50));
            }
            return {
                before,
                after: box.scrollTop,
                atBottom: box.scrollHeight - box.clientHeight - box.scrollTop <= 48,
            };
        }"""
    )
    assert out["before"] == 0
    assert out["atBottom"], f"stopped at {out['after']}"
