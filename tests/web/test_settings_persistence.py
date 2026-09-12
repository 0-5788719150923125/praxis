"""The Settings form survives a refresh, driven in a real browser.

Source guards cannot see this one. Every read and write went through
``storage``, which looks the key up in ``STORAGE_KEYS`` and, on a miss,
console.warns and returns without touching localStorage - so the whole form
round-tripped through state, rendered correctly, and persisted nothing. The
only way to catch that is to actually reload the page.

Shares the session's APIServer and the page-opening helper with
``test_browser.py``; skips wholesale when playwright is unavailable.
"""

import pytest

sync_playwright = pytest.importorskip(
    "playwright.sync_api", reason="playwright is not installed"
).sync_playwright

from tests.web.test_browser import _open  # noqa: E402  (after the skip guard)


@pytest.fixture(scope="module")
def app_url(api_server):
    return f"http://127.0.0.1:{api_server.port}/"


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as play:
        browser = play.chromium.launch()
        yield browser
        browser.close()


@pytest.fixture
def page(browser, app_url):
    """A page with localStorage cleared, so each test starts from the run's
    own defaults rather than whatever a previous one saved."""
    context, page = _open(browser, app_url, viewport={"width": 1280, "height": 900})
    page.evaluate("localStorage.clear()")
    page.reload(wait_until="load")
    page.wait_for_timeout(400)
    yield page
    context.close()


def _save_generation_kwargs(page, text):
    """Type into the Settings form and press its Save button, as a user would."""
    page.click("#settings-button")
    page.wait_for_selector("#generation-kwargs", timeout=5000)
    page.fill("#generation-kwargs", text)
    page.click("#save-settings")
    page.wait_for_timeout(300)


def _reload(page):
    page.reload(wait_until="load")
    page.wait_for_timeout(400)


def test_generation_kwargs_survive_a_refresh(page):
    _save_generation_kwargs(page, "max_new_tokens=42\ntemperature=0.9")
    _reload(page)

    page.click("#settings-button")
    page.wait_for_selector("#generation-kwargs", timeout=5000)
    assert "max_new_tokens=42" in page.input_value("#generation-kwargs")


def test_the_saved_value_is_what_the_next_request_sends(page):
    """Persisting into a box nothing reads would look identical from the UI."""
    _save_generation_kwargs(page, "max_new_tokens=42")
    _reload(page)

    sent = page.evaluate("""async () => {
            const api = await import('./static/js/api.js');
            let body = null;
            const real = window.fetch;
            window.fetch = (url, opts) => {
                if (String(url).includes('/messages')) body = JSON.parse(opts.body);
                return real(url, opts);
            };
            try { await api.sendMessage([{role: 'user', content: 'hi'}]); }
            catch (e) { /* the reply does not matter, the payload does */ }
            window.fetch = real;
            return body;
        }""")
    assert sent is not None, "the chat never issued a request"
    assert "max_new_tokens=42" in sent["generation_kwargs"]


def test_an_edited_value_is_not_overwritten_by_the_runs_default(page):
    """A value the user changed outranks the run's; only an untouched one
    yields to a new `--generation-kwargs`."""
    _save_generation_kwargs(page, "max_new_tokens=7")
    _reload(page)

    stored = page.evaluate("localStorage.getItem('praxis_generation_kwargs')")
    assert "max_new_tokens=7" in stored


def test_clearing_the_box_falls_back_to_the_runs_defaults(page):
    _save_generation_kwargs(page, "max_new_tokens=7")
    _reload(page)
    _save_generation_kwargs(page, "")
    _reload(page)

    sent = page.evaluate("""async () => {
            const api = await import('./static/js/api.js');
            let body = null;
            const real = window.fetch;
            window.fetch = (url, opts) => {
                if (String(url).includes('/messages')) body = JSON.parse(opts.body);
                return real(url, opts);
            };
            try { await api.sendMessage([{role: 'user', content: 'hi'}]); }
            catch (e) { /* payload only */ }
            window.fetch = real;
            return body;
        }""")
    assert sent["generation_kwargs"] == [], "an empty box must send nothing"


def test_the_developer_prompt_survives_a_refresh(page):
    """It saves on blur rather than through the modal, so it is a separate
    path through the same broken helper."""
    page.wait_for_selector("#developer-prompt", timeout=5000)
    page.evaluate("""() => {
            const el = document.getElementById('developer-prompt');
            el.focus();
            el.textContent = 'speak only in haiku';
            el.blur();
        }""")
    page.wait_for_timeout(200)
    _reload(page)

    assert page.text_content("#developer-prompt").strip() == "speak only in haiku"


def test_a_list_valued_kwarg_round_trips_through_the_form(page):
    """Template interpolation renders [64, 1.03] as `64,1.03` - brackets gone -
    and the server reads that back as a STRING, which raises inside the logits
    processor and surfaces as "(model produced an empty turn)". The form has to
    write a value the server can parse back to a list."""
    rendered = page.evaluate("""async () => {
            const cfg = await import('./static/js/config.js');
            return cfg.generationKwargText({
                temperature: 0.7,
                exponential_decay_length_penalty: [64, 1.03],
            });
        }""")
    assert "exponential_decay_length_penalty=[64,1.03]" in rendered
    # Scalars stay bare - the common line must not become a quoted value.
    assert "temperature=0.7" in rendered

    lines = page.evaluate(f"""async () => {{
            const cfg = await import('./static/js/config.js');
            return cfg.generationKwargLines({rendered!r});
        }}""")
    from praxis.inference import parse_generation_kwargs

    parsed = parse_generation_kwargs(lines)
    assert list(parsed["exponential_decay_length_penalty"]) == [64, 1.03]
