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
pinned server-side in `tests/web/test_routes.py`. The page is served by the
session's APIServer (tests/web/conftest.py); every POST is fulfilled here.
"""

import json
from types import SimpleNamespace

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


@pytest.fixture(scope="module")
def app_url(api_server):
    return f"http://127.0.0.1:{api_server.port}/"


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as play:
        browser = play.chromium.launch()
        yield browser
        browser.close()


def _open(browser, app_url, **context_args):
    """A page on the app with every off-origin request stubbed or refused."""
    context = browser.new_context(**context_args)
    context.route(
        "**/*",
        lambda route: (
            route.continue_() if route.request.url.startswith(app_url) else route.abort()
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
    return context, page


@pytest.fixture(scope="module")
def page(browser, app_url):
    """One desktop page shared by the module; tests reset the state they use."""
    context, page = _open(browser, app_url, viewport={"width": 1280, "height": 900})
    yield page
    context.close()


@pytest.fixture
def phone(browser, app_url):
    """A Pixel 6 sized touch viewport."""
    context, page = _open(
        browser,
        app_url,
        viewport={"width": 412, "height": 915},
        device_scale_factor=2.625,
        is_mobile=True,
        has_touch=True,
    )
    yield page
    context.close()


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
            // Name and count are separate elements, so the assertion reads
            // the way the row does rather than depending on whitespace.
            const chips = last
                ? [...last.querySelectorAll('.tool-chip')].map((c) => {
                      const count = c.querySelector('.tool-chip-count');
                      const name = c.querySelector('.tool-chip-name').textContent;
                      return count ? `${name} x${count.textContent}` : name;
                  })
                : [];
            return {
                count: nodes.length,
                text: last ? last.querySelector('.message-content').textContent : null,
                streaming: last ? last.classList.contains('streaming') : false,
                reroll: !!document.getElementById('reroll-button'),
                thinking: !!document.querySelector('.thinking-status'),
                tools: chips,
            };
        }""")


SEND_JS = """async (streaming) => {
    const { sendMessage } = await import('/static/js/api.js');
    const { streamingTurn } = await import('/static/js/chatstream.js');
    const { state } = await import('/static/js/state.js');
    const { render } = await import('/static/js/render.js');
    if (!streaming) {
        return [(await sendMessage(state.messages)).response];
    }
    const turn = streamingTurn();
    try {
        const response = await sendMessage(state.messages, {
            onDelta: turn.onDelta, onReset: turn.onReset, onTool: turn.onTool
        });
        turn.settle(response.response, response.tools);
    } catch (error) {
        turn.fail(`Error: ${error.message}`);
    }
    state.isThinking = false;
    render();
    return state.messages.map((m) => m.content);
}"""


def _send(page, app_url, frames=(), response=None, streaming=True):
    """Send one message through the real client modules.

    While the POST is in flight, each ``(event, payload)`` in ``frames`` is
    delivered on the socket carrying the request's own stream id (a payload
    ``id`` overrides it), with a DOM snapshot after each. The POST is then
    fulfilled with ``response``, or aborted when it is None.
    """
    _reset(page)
    seen, bodies = [], []

    def handle(route):
        body = json.loads(route.request.post_data)
        bodies.append(body)
        for event, payload in frames:
            page.evaluate(
                "([ev, data]) => window.__deliver(ev, data)",
                [event, {"id": body.get("stream_id"), "seq": 1, **payload}],
            )
            page.wait_for_timeout(40)
            seen.append(_snapshot(page))
        if response is None:
            route.abort()
        else:
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(response),
            )

    page.route(f"{app_url}messages/", handle)
    try:
        messages = page.evaluate(SEND_JS, streaming)
    finally:
        page.unroute(f"{app_url}messages/")
    return SimpleNamespace(seen=seen, bodies=bodies, messages=messages)


# ---------------------------------------------------------------------------
# the whole client path: sendMessage -> socket frame -> turn -> DOM
# ---------------------------------------------------------------------------


def test_the_reply_fills_in_while_the_post_is_still_outstanding(page, app_url):
    """The point of the feature. The deltas arrive on the socket WHILE the POST
    is in flight, which is the ordering production has - and the id that routes
    them is the one `api.js` put in the request body."""
    out = _send(
        page,
        app_url,
        frames=[("gen_delta", {"text": t}) for t in ("It ", "is ", "noo")],
        response={"response": "It is noon."},
    )

    # The client mints the id and sends it; the server only echoes it back.
    assert out.bodies[0]["stream_id"]
    # It grew, as ONE message beside the user's - never a bubble per delta.
    assert [s["text"] for s in out.seen] == ["It ", "It is ", "It is noo"]
    assert {s["count"] for s in out.seen} == {2}
    assert all(s["streaming"] for s in out.seen)
    # ...and settled on the POST's answer, which is what stays authoritative.
    assert out.messages == ["hi", "It is noon."]
    assert _snapshot(page)["streaming"] is False


def test_deltas_for_another_request_are_ignored(page, app_url):
    """Ids keep concurrent requests from cross-talking, and stop a straggler
    from a finished turn landing in the next one."""
    out = _send(
        page,
        app_url,
        frames=[("gen_delta", {"id": "some-other-request", "text": "WRONG"})],
        response={"response": "mine"},
    )
    assert out.messages == ["hi", "mine"]


def test_a_request_that_does_not_stream_sends_no_id(page, app_url):
    """The unchanged path: no `onDelta`, no `stream_id`, so the server builds no
    streamer at all and the behaviour is exactly what it was."""
    out = _send(page, app_url, response={"response": "quiet reply"}, streaming=False)
    assert out.messages == ["quiet reply"]
    assert "stream_id" not in out.bodies[0]


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


@pytest.mark.parametrize(
    "streamed,ending,shown",
    [
        (None, "settle('quiet reply')", "quiet reply"),
        ("partial", "settle('final answer')", "final answer"),
        # A long reply streams in, the client's 60s patience runs out, and the
        # POST comes back "". The stream is closed by then, so what is on
        # screen IS everything the model produced: keep it and caption it.
        ("partial", "settle('')", "partial"),
        # With nothing on screen there is nothing to protect.
        (None, "settle('')", "Error: No response"),
        # An outright failure is a footnote on streamed text, not a replacement.
        ("partial", "fail('Error: boom')", "partial"),
        (None, "fail('Error: boom')", "Error: boom"),
    ],
)
def test_how_a_turn_ends(page, streamed, ending, shown):
    """(streamed or not) x (answer, empty answer, failure): the turn is always
    appended exactly once, and streamed text is never thrown away."""
    delta = f"await step(() => turn.onDelta({streamed!r}));" if streamed else ""
    out = _drive(
        page,
        f"""(async () => {{
            {delta}
            await step(() => {{ turn.{ending}; render(); }});
        }})()""",
    )
    assert out["messages"] == ["hi", shown]
    assert out["seen"][-1]["count"] == 2
    assert out["seen"][-1]["streaming"] is False
    assert shown in out["seen"][-1]["text"]


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
    out = page.evaluate("""async () => {
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
        }""")
    assert out["identity"] == [True, True, True], "nodes were rebuilt mid-stream"
    assert out["text"] == "one two three four"


def test_a_selection_in_an_earlier_message_survives_streaming(page):
    """Rebuilding the list dropped any text the user had selected. Patching one
    node leaves every other one - and its selection - untouched."""
    out = page.evaluate("""async () => {
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
        }""")
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
    out = page.evaluate("""async () => {
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
        }""")
    assert out["before"] == 0
    assert out["atBottom"], f"stopped at {out['after']}"


# ---------------------------------------------------------------------------
# tool badges: the row of chips under a turn that used tools
# ---------------------------------------------------------------------------


def test_a_tool_chip_appears_while_the_reply_is_still_arriving(page, app_url):
    """A tool that ran leaves no trace in the reply text - the server strips
    the whole call/result exchange - so the chip is the only thing that shows
    it, and it shows up as it happens rather than at the end."""
    out = _send(
        page,
        app_url,
        frames=[
            ("gen_delta", {"text": "let me look"}),
            ("gen_tool", {"name": "read_file"}),
            ("gen_delta", {"text": " - found it"}),
        ],
        response={
            "response": "let me look - found it",
            "tools": [{"name": "read_file", "count": 1}],
        },
    )
    assert [s["tools"] for s in out.seen] == [[], ["read_file"], ["read_file"]]
    # One turn throughout - the chip is part of the message, not a bubble.
    assert {s["count"] for s in out.seen} == {2}
    assert _snapshot(page)["tools"] == ["read_file"]


def test_a_reset_clears_the_text_but_not_the_chips(page, app_url):
    """`gen_reset` fires BECAUSE a tool ran: the runtime spliced the result and
    moved the turn anchor past it, so the model's pre-call chatter stopped
    being part of the answer. The call itself still happened."""
    out = _send(
        page,
        app_url,
        frames=[
            ("gen_delta", {"text": "let me look"}),
            ("gen_tool", {"name": "read_file"}),
            ("gen_reset", {}),
            ("gen_delta", {"text": "it says 42"}),
        ],
        response={"response": "it says 42", "tools": [{"name": "read_file", "count": 1}]},
    )
    # Measured MID-STREAM, the only place the rule is visible: the response's
    # own tally would restore the chips at the end either way.
    after_reset = out.seen[2]
    assert after_reset["text"] == ""  # the chatter was dropped
    assert after_reset["tools"] == ["read_file"]  # the call was not

    final = _snapshot(page)
    assert final["text"] == "it says 42"
    assert final["tools"] == ["read_file"]


def test_repeated_use_of_one_tool_is_counted_not_repeated(page, app_url):
    """Two chips reading `read_file` would be noise. One with a count is the
    Discord-reaction shape the row is modelled on."""
    _send(
        page,
        app_url,
        frames=[("gen_tool", {"name": n}) for n in ("read_file", "search", "read_file")],
        response={
            "response": "done",
            "tools": [{"name": "read_file", "count": 2}, {"name": "search", "count": 1}],
        },
    )
    # First-use order, and the count only shows past one.
    assert _snapshot(page)["tools"] == ["read_file x2", "search"]


def test_the_response_tally_is_what_the_turn_settles_on(page, app_url):
    """The server tallies tools in the branch that runs each one and sends them
    whether or not the socket was up, so a client whose socket dropped every
    frame still ends up with the right row."""
    _send(
        page,
        app_url,
        response={"response": "done", "tools": [{"name": "calc", "count": 3}]},
    )
    assert _snapshot(page)["tools"] == ["calc x3"]


def test_a_failed_request_keeps_the_chips_it_earned(page, app_url):
    """A tool that ran still ran. The error is a footnote on the turn, not a
    replacement for the record of what it did."""
    _send(page, app_url, frames=[("gen_tool", {"name": "search"})], response=None)

    assert _snapshot(page)["tools"] == ["search"]
    assert page.evaluate("""async () => {
            const { state } = await import('/static/js/state.js');
            return state.messages[state.messages.length - 1].caption;
        }""").startswith("Error:")


# ---------------------------------------------------------------------------
# phone layout (css/responsive.css, mobile.js)
# ---------------------------------------------------------------------------


TABS = ["chat", "terminal", "agents", "research", "dynamics", "spec"]


def test_no_tab_overflows_a_phone_screen(phone):
    width = phone.viewport_size["width"]
    for tab in TABS:
        # The tab bar is rendered once per layout; click the visible one.
        phone.locator(f'button[data-tab="{tab}"]:visible').first.click()
        phone.wait_for_timeout(100)
        body = phone.evaluate("() => document.body.scrollWidth")
        assert body <= width + 5, f"{tab}: body scrollWidth {body} > {width}"
        panel = phone.locator(f"#{tab}-content").bounding_box()
        assert panel, f"{tab}: panel not shown"
        assert panel["x"] + panel["width"] <= width, f"{tab}: panel overflows"


def test_the_message_input_fits_a_phone_screen(phone):
    field = phone.locator("#message-input")
    assert field.is_visible()
    box = field.bounding_box()
    assert box["x"] + box["width"] <= phone.viewport_size["width"]
    assert box["width"] >= 200, f"input too narrow: {box['width']}px"
