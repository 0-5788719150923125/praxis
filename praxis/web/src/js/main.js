/**
 * Praxis Web - Main Entry Point
 * Wire up events, initialize app
 * Architecture: Events → Update State → Render
 */

import { state, CONSTANTS, DEFAULT_SYSTEM_PROMPT } from './state.js';
import { render, renderAppStructure, updateInputContainerStyling, renderPrintButton, renderKbResults } from './render.js';
import { sendMessage, kbSearch, testApiConnection, printAsk, printRespond, printEnergy, loopGenerate } from './api.js';
import { connectMetricsLive, setupLiveReload, renderCurrentMetrics } from './websocket.js';
import { kbSlideWindow, setupKbPrefetch } from './kbcache.js';
import { loadSpec, loadAgents, loadResearchMetrics } from './tabs.js';
import { setupTabCarousel, setupTabSwipe } from './mobile.js';
import { storage, FORM_FIELDS, readFormValues, updateRangeDisplay } from './config.js';
import { CLICK_HANDLERS, delegateClick } from './events.js';
import { executeAction } from './actions.js';
import { beginPrewarm, endPrewarm } from './prefetch.js';
import { setupAccentRetint } from './charts.js';
import './prism.js';

// Mode-aware input cues, all basketball: Read invites a query ("> Look"),
// Evaluate invites a shot ("< Shoot"), Print invites an answer to the model's
// question - a "< Pass" back. Prefixes are 2 chars, so cursor/length logic in
// the input handlers is unaffected.
const prefixForMode = (mode) => (mode === 'read' ? '> ' : '< ');
export const inputPrefix = () => prefixForMode(state.conversationMode);
const inputPlaceholderText = () => {
    if (state.conversationMode === 'read') return 'Look';
    if (state.conversationMode === 'print') return 'Pass';
    return 'Shoot';
};

/**
 * Keep the input box consistent when the conversation mode flips. Either rebuild
 * the placeholder for the new mode, or swap just the prefix while preserving any
 * typed text - so the prefix never doubles up (e.g. "< > Perplexity").
 */
export function syncInputToMode(prevMode) {
    const input = document.getElementById('message-input');
    if (!input) return;
    if (state.isShowingPlaceholder) {
        showPlaceholder();
        return;
    }
    const oldPrefix = prefixForMode(prevMode);
    const newPrefix = inputPrefix();
    let body = input.value;
    if (body.startsWith(oldPrefix)) body = body.slice(oldPrefix.length);
    else if (body.startsWith(newPrefix)) body = body.slice(newPrefix.length);
    input.value = newPrefix + body;
}

/**
 * Lifecycle function registry - maps string names to actual functions
 * This allows tabs to specify activation/deactivation hooks as strings in state.js
 */
const lifecycleFunctions = {
    renderCurrentMetrics,
    loadSpec,
    loadAgents,
    loadResearchMetrics
};

/**
 * Resolve lifecycle function by name
 * @param {string} name - Function name
 * @returns {Function|null} The function or null if not found
 */
function getLifecycleFunction(name) {
    return lifecycleFunctions[name] || null;
}

/**
 * Initialize the application
 */
function init() {
    // Disable Chart.js tooltips on touch devices: the tap-triggered labels
    // cover the chart and can't be dismissed. Taps synthesize mouse/click
    // events too, so we strip all events on hover-less devices.
    if (window.Chart) {
        const isTouch = window.matchMedia('(hover: none)').matches;
        Chart.defaults.events = isTouch ? [] : ['mousemove', 'mouseout', 'click'];
    }

    // Charts auto-recolor when the accent (logs blue mode) flips - no Refresh needed.
    setupAccentRetint();

    // Set API URL
    const pathPrefix = window.location.pathname.endsWith('/')
        ? window.location.pathname.slice(0, -1)
        : window.location.pathname;
    state.settings.apiUrl = window.location.origin + (pathPrefix === '/' ? '' : pathPrefix);

    // Load saved settings from localStorage
    loadSettings();

    // Build entire app structure from JavaScript (UI = render(state))
    renderAppStructure();

    // Initial render of dynamic content
    render();

    // Set up event listeners
    setupEventListeners();

    // Connect to metrics-live WebSocket
    connectMetricsLive();

    // Setup live reload
    setupLiveReload();

    // Setup mobile tab carousel
    setupTabCarousel();

    // Setup mobile swipe-to-switch-tabs
    setupTabSwipe();

    // Setup KB sliding-window prefetch (scroll-driven precache/postcache)
    setupKbPrefetch();

    // Setup window resize handler for dashboard scaling
    setupWindowResizeHandler();

    // Initialize input placeholder
    showPlaceholder();

    // Poll the Print hook: periodically query the model for a self-led question.
    // When one arrives the conditional Print button appears (see render).
    setupPrintHook();

    // Warm the other tabs in the background so their charts/spec are ready before
    // the user navigates (and so KB card deep-links land on a built deck). Decks
    // built while hidden are re-measured on tab activation. Refresh still forces.
    prefetchTabs();

    console.log('[Praxis] Initialized');
}

/**
 * Background-build the data-heavy tabs after the initial paint, sequentially so
 * we don't fire a burst of fetches that competes with the live metrics stream.
 * Each is laid out off-screen at the active region's size so its charts measure
 * and render fully - making the first navigation instant, like a revisit.
 */
// Re-warm hidden tabs this often so their data is fresh BEFORE navigation.
// Endpoints are cheap (ETag 304s for metrics; small JSON elsewhere) and the
// refresh only ever touches hidden tabs, so it can never flash the screen.
const TAB_REFRESH_MS = 60000;

async function prefetchTabs() {
    const { loadResearchMetrics, loadDynamics, loadSpec, loadAgents } = await import('./tabs.js');
    // [tabId, contentId, loader, needsLayout] - deck tabs must be laid out
    // off-screen so their charts measure; the Stage fleet is a plain list, so
    // a background load is enough.
    const jobs = [
        ['spec', 'spec-content', loadSpec, true],
        ['research', 'research-content', loadResearchMetrics, true],
        ['dynamics', 'dynamics-content', loadDynamics, true],
        ['agents', 'agents-content', loadAgents, false],
    ];
    // Building a hidden deck is a long synchronous chunk (dozens of charts);
    // run each only in an idle slice so typing and hover never block.
    const idle = () => new Promise(resolve =>
        'requestIdleCallback' in window
            ? requestIdleCallback(() => resolve(), { timeout: 4000 })
            : setTimeout(resolve, 200)
    );
    const warm = async (force) => {
        for (const [tabId, contentId, load, needsLayout] of jobs) {
            // Never touch the visible tab from the background path: its loader
            // rebuilds the DOM, which would flash mid-read. It refreshes via
            // its own controls; hidden tabs swap invisibly and arrive fresh.
            if (force && state.currentTab === tabId) continue;
            await idle();
            try {
                const loader = () => load(force);
                await (needsLayout ? prewarmTab(tabId, contentId, loader) : loader());
            } catch (error) {
                console.warn('[Praxis] Tab prewarm failed:', contentId, error);
            }
        }
    };
    setTimeout(async () => {
        await warm(false);  // initial fill: first navigation lands warm
        warmKbIndex();      // build the KB index now so the first "> Look" is instant
        // Steady-state: keep hidden tabs fresh. Sequential + paused while the
        // page itself is hidden, so it never competes with the live stream.
        setInterval(() => {
            if (document.hidden) return;
            warm(true);
        }, TAB_REFRESH_MS);
    }, 600);
}

/**
 * Lay out a hidden tab off-screen at the active region's exact size, run its
 * loader so the deck measures + renders for real, then return it to the hidden
 * state render() manages. Two frames let the deck settle before we hide it.
 */
async function prewarmTab(tabId, contentId, load) {
    const el = document.getElementById(contentId);
    const active = document.querySelector('.tab-content.active');
    const rect = active ? active.getBoundingClientRect() : null;
    // Already on screen (or unsizeable): plain load, never touch its styles -
    // off-screen styling of a visible tab is exactly the blank-flash bug.
    if (!el || state.currentTab === tabId || !rect || rect.height < 50) {
        await load();
        return;
    }

    Object.assign(el.style, {
        display: 'flex',
        flexDirection: 'column',
        position: 'absolute',
        left: `${rect.left}px`,
        top: `${rect.top}px`,
        width: `${rect.width}px`,
        height: `${rect.height}px`,
        visibility: 'hidden',
        pointerEvents: 'none',
        zIndex: '-1',
    });
    // Register so SWITCH_TAB can strip these styles instantly if the user
    // navigates here mid-warm (revealPrewarmed) instead of staring at a
    // vanished container until the load settles.
    beginPrewarm(tabId, el);
    try {
        await load();
        await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
    } finally {
        endPrewarm(tabId);
        el.removeAttribute('style');  // back to .tab-content (display:none) until activated
    }
}

/**
 * Setup window resize handler
 */
function setupWindowResizeHandler() {
    let resizeTimeout = null;

    // Handle window resize
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => {
            // Re-render to update any size-dependent elements
            render();
            // Re-render live dashboard if on terminal tab
            if (state.currentTab === 'terminal') {
                renderCurrentMetrics();
            }
        }, 250);
    });

    // Handle orientation changes (mobile)
    window.addEventListener('orientationchange', () => {
        setTimeout(() => {
            if (state.currentTab === 'terminal') {
                renderCurrentMetrics();
            }
        }, 100);
    });
}

/**
 * Load settings from localStorage using centralized storage utilities
 */
function loadSettings() {
    // Load theme
    const savedTheme = storage.get('theme');
    if (savedTheme) {
        state.theme = savedTheme;
    }

    // Load system prompt
    const savedPrompt = storage.get('developerPrompt');
    if (savedPrompt) {
        state.settings.systemPrompt = savedPrompt;
    }

    // Load API URL
    const savedApiUrl = storage.get('apiUrl');
    if (savedApiUrl) {
        // Skip if we're on ngrok (use dynamic URL)
        if (!window.location.hostname.includes('ngrok')) {
            state.settings.apiUrl = savedApiUrl;
        }
    }

    // Load generation params
    const savedParams = storage.get('genParams');
    if (savedParams) {
        Object.assign(state.settings, savedParams);
    }

    // Load debug flag
    const debugLogging = storage.get('debugLogging');
    if (debugLogging !== null) {
        state.settings.debugLogging = debugLogging === 'true' || debugLogging === true;
    }
}

/**
 * Save settings to localStorage using centralized storage utilities
 */
function saveSettings() {
    storage.set('theme', state.theme);
    storage.set('developerPrompt', state.settings.systemPrompt);
    storage.set('apiUrl', state.settings.apiUrl);
    storage.set('genParams', {
        maxTokens: state.settings.maxTokens,
        temperature: state.settings.temperature,
        repetitionPenalty: state.settings.repetitionPenalty,
        doSample: state.settings.doSample,
        useCache: state.settings.useCache
    });
    storage.set('debugLogging', state.settings.debugLogging);
}

/**
 * Setup all event listeners using delegation
 */
function setupEventListeners() {
    // Global click handler (event delegation)
    document.addEventListener('click', handleClick);

    // Input handlers
    const messageInput = document.getElementById('message-input');
    if (messageInput) {
        messageInput.addEventListener('keydown', handleInputKeydown);
        messageInput.addEventListener('input', handleInputChange);
        messageInput.addEventListener('focus', handleInputFocus);
        messageInput.addEventListener('blur', handleInputBlur);
        messageInput.addEventListener('click', handleInputClick);
        messageInput.addEventListener('select', handleInputSelect);
    }

    // System prompt editing
    const systemPrompt = document.getElementById('developer-prompt');
    if (systemPrompt) {
        systemPrompt.addEventListener('focus', () => {
            systemPrompt.classList.remove('default-prompt');
        });
        systemPrompt.addEventListener('blur', () => {
            state.settings.systemPrompt = systemPrompt.textContent;
            saveSettings();
            if (state.settings.systemPrompt === DEFAULT_SYSTEM_PROMPT) {
                systemPrompt.classList.add('default-prompt');
            }
        });
    }

    // Settings modal range inputs (live readout) + joke score slider.
    document.addEventListener('input', handleRangeInput);
    // Joke score slider commits on release (change), not on every drag step.
    document.addEventListener('change', handleSliderChange);
}

/** Submit a joke score when the want->need slider is released. */
async function handleSliderChange(e) {
    if (e.target.classList && e.target.classList.contains('joke-slider')) {
        await executeAction('SCORE_JOKE', Number(e.target.value));
    }
}

/**
 * Handle all click events using declarative event delegation
 * Pure delegation pattern - configuration in event-handlers.js, logic in actions.js
 */
async function handleClick(e) {
    const action = delegateClick(e, CLICK_HANDLERS);
    if (action) {
        await executeAction(action.type, action.payload, action.meta);
    }
}

/**
 * Handle input keydown (send on Enter)
 */
async function handleInputKeydown(e) {
    const input = e.target;

    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (state.loop.enabled) {
            rerollLoopNow();  // re-roll the current task now
        } else if (state.conversationMode === 'read') {
            openTopKbResult();
        } else {
            await sendUserMessage();
        }
        return;
    }

    // Prevent deleting into the prefix - but only for a bare cursor; a real
    // selection should still be deletable (maintain-prefix re-adds it after).
    const collapsed = input.selectionStart === input.selectionEnd;
    if ((e.key === 'Backspace' || e.key === 'Delete')
        && collapsed && input.selectionStart <= inputPrefix().length) {
        e.preventDefault();
        return;
    }

    // Prevent a bare cursor from moving before the prefix.
    if (e.key === 'ArrowLeft' && collapsed && input.selectionStart <= inputPrefix().length) {
        e.preventDefault();
        return;
    }

    if (e.key === 'Home') {
        e.preventDefault();
        setCursorAfterPrefix();
        return;
    }
}

/**
 * Handle input changes (maintain prefix)
 */
function handleInputChange(e) {
    const input = e.target;
    const currentValue = input.value;

    // Handle placeholder state
    if (state.isShowingPlaceholder) {
        const newChars = currentValue.replace(inputPrefix() + inputPlaceholderText(), '').replace(inputPrefix(), '');
        hidePlaceholder();
        if (newChars) {
            input.value = inputPrefix() + newChars;
            input.setSelectionRange(input.value.length, input.value.length);
        }
        return;
    }

    // Maintain prefix
    if (!currentValue.startsWith(inputPrefix())) {
        input.value = inputPrefix() + currentValue;
        setCursorAfterPrefix();
    }

    // Auto-resize
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 200) + 'px';

    // Read mode: live KB search as you type.
    if (state.conversationMode === 'read') {
        scheduleKbSearch(currentQuery(input));
    }
}

// --- KB search (Read mode) ---------------------------------------------------

let kbSearchTimer = null;
let kbSearchSeq = 0;

// The recent feed cached from a background warm-up. Populated by warmKbIndex()
// at startup so the first focus paints instantly instead of waiting on the
// one-time FTS index build; kept fresh by every later empty-query search.
let kbWarmFeed = null;
let kbWarmPromise = null;

/**
 * Build the KB index and cache the recent feed in the background. Does NOT touch
 * the UI - results only surface when the user focuses the "> Look" box. The
 * expensive first-build happens here, off the click path.
 */
export function warmKbIndex() {
    kbWarmPromise ||= kbSearch('')
        .then(hits => (kbWarmFeed = hits))
        .catch(() => { kbWarmPromise = null; return null; });
    return kbWarmPromise;
}

/** Strip the input prefix to get the raw query text. */
function currentQuery(input) {
    const v = input.value;
    return (v.startsWith(inputPrefix()) ? v.slice(inputPrefix().length) : v).trim();
}

/**
 * Append a KB label to the search query as a comma-delimited term (skipping
 * duplicates) and re-run the search. The comma form mirrors prompt syntax -
 * each click narrows the topic the way you'd stack terms in a prompt. Wired to
 * the green KB labels (see events.js ADD_KB_LABEL_FILTER).
 */
export function addKbSearchTerm(label) {
    const input = document.getElementById('message-input');
    const term = (label || '').trim();
    if (!input || !term) return;
    let raw = currentQuery(input);
    if (state.isShowingPlaceholder) raw = '';  // ignore the ghost placeholder
    // Leave placeholder state cleanly, or the input handler will treat our
    // appended term as ghost text and mangle the next keystroke.
    hidePlaceholder();
    const terms = raw.split(',').map(t => t.trim()).filter(Boolean);
    if (!terms.includes(term)) terms.push(term);
    const query = terms.join(', ');
    input.value = inputPrefix() + query;
    input.focus();
    input.setSelectionRange(input.value.length, input.value.length);
    scheduleKbSearch(query);
}

/**
 * Debounced live search; stale responses are dropped via a sequence guard. An
 * empty query fetches the recent feed (the backend returns newest-first), so the
 * box is never blank - it's the default view when you click in with no text.
 */
function scheduleKbSearch(query) {
    clearTimeout(kbSearchTimer);
    state.kbOpenItem = null;  // typing a new query returns to the results list
    const seq = ++kbSearchSeq;
    // Empty query (the recent feed) paints instantly from the warm cache - the
    // first focus never shows a spinner. A revalidating fetch still runs below.
    if (!query && kbWarmFeed) {
        state.kbResults = kbWarmFeed;
        state.kbSearching = false;
        renderKbResults();
        kbSlideWindow(0);
    } else {
        state.kbSearching = true;
    }
    kbSearchTimer = setTimeout(async () => {
        try {
            const hits = await kbSearch(query);  // '' -> recent feed
            if (seq !== kbSearchSeq) return; // a newer keystroke superseded us
            state.kbResults = hits;
            if (!query) kbWarmFeed = hits;  // keep the warm cache fresh
        } catch (error) {
            if (seq !== kbSearchSeq) return;
            console.error('[KB] Search error:', error);
            state.kbResults = [];
        } finally {
            if (seq === kbSearchSeq) {
                state.kbSearching = false;
                // Re-render only the KB panel - a search changes nothing else, and
                // a full render() (chat, mobile tab carousel, modal, ...) on every
                // keystroke-debounced query is what made typing feel blocked.
                renderKbResults();
                // Prime the sliding-window prefetch on the fresh result set so the
                // first screenful of items opens instantly.
                kbSlideWindow(0);
            }
        }
    }, query ? 120 : 0);
}

/** Enter in Read mode opens the top-ranked hit (inline, or external for links). */
function openTopKbResult() {
    const top = state.kbResults[0];
    if (top) executeAction('OPEN_KB_ITEM', { id: top.id, type: top.type, uri: top.uri, title: top.title });
}

/**
 * Handle input focus. In Read mode, focusing the box loads results for the
 * current query - empty included, which yields the recent feed.
 */
function handleInputFocus() {
    hidePlaceholder();
    if (state.conversationMode === 'read' && !state.kbOpenItem) {
        const input = document.getElementById('message-input');
        if (input) scheduleKbSearch(currentQuery(input));
    }
}

/**
 * Handle input blur
 */
function handleInputBlur(e) {
    const input = e.target;
    if (input.value === inputPrefix() || input.value === '') {
        showPlaceholder();
    }
}

/**
 * Handle input click (clear placeholder and prevent cursor before prefix)
 */
function handleInputClick(e) {
    const input = e.target;

    // Clear placeholder on click
    if (state.isShowingPlaceholder) {
        hidePlaceholder();
    }

    // Nudge a bare cursor out of the prefix, but never disturb an active
    // selection - collapsing it here is what made highlighting feel like it was
    // fighting back.
    const collapsed = input.selectionStart === input.selectionEnd;
    if (collapsed && input.selectionStart < inputPrefix().length) {
        setCursorAfterPrefix();
    }
}

/**
 * Keep selections out of the prefix. Fires on any selection change (drag,
 * Ctrl+A, shift-arrows) and snaps the start past the "< " so the prefix can
 * never be highlighted or deleted as part of a selection.
 */
function handleInputSelect(e) {
    const input = e.target;
    if (state.isShowingPlaceholder) return;
    const p = inputPrefix().length;
    if (input.selectionStart < p && input.selectionEnd > input.selectionStart) {
        input.setSelectionRange(p, Math.max(input.selectionEnd, p), input.selectionDirection);
    }
}

/**
 * Set cursor position after prefix
 */
function setCursorAfterPrefix() {
    const input = document.getElementById('message-input');
    if (input) {
        input.setSelectionRange(inputPrefix().length, inputPrefix().length);
    }
}

/**
 * Show placeholder in input
 */
export function showPlaceholder() {
    const input = document.getElementById('message-input');
    if (!input) return;

    input.value = inputPrefix() + inputPlaceholderText();
    input.style.color = 'var(--light-text)';
    input.style.fontStyle = 'italic';
    input.style.height = 'auto'; // Reset height to default (collapse to single line)
    state.isShowingPlaceholder = true;
    setCursorAfterPrefix();
}

/**
 * Hide placeholder
 */
function hidePlaceholder() {
    const input = document.getElementById('message-input');
    if (!input) return;

    if (state.isShowingPlaceholder) {
        input.value = inputPrefix();
        input.style.color = '';
        input.style.fontStyle = '';
        state.isShowingPlaceholder = false;
        setCursorAfterPrefix();
    }
}

/**
 * Handle range input changes (update display) using form config
 */
function handleRangeInput(e) {
    // Joke score slider: live-update its numeric readout while dragging.
    if (e.target.classList && e.target.classList.contains('joke-slider')) {
        const val = e.target.parentElement?.querySelector('.joke-score-val');
        if (val) val.textContent = Number(e.target.value).toFixed(2);
        return;
    }
    updateRangeDisplay(FORM_FIELDS.settings, e.target.id, e.target.value);
}

/**
 * Send user message to API
 */
async function sendUserMessage() {
    const input = document.getElementById('message-input');
    if (!input) return;

    // Don't send if just showing placeholder
    if (state.isShowingPlaceholder) return;

    // Extract message (remove prefix)
    const fullValue = input.value;
    const content = fullValue.startsWith(inputPrefix())
        ? fullValue.slice(inputPrefix().length).trim()
        : fullValue.trim();

    if (!content) return;

    // Print flow: in Print mode with a model-led question awaiting an answer,
    // this message IS the answer - score it instead of generating. The pending
    // question survives Read/Evaluate switches, so we also require Print mode
    // here (a message typed in Evaluate must still chat normally).
    if (state.print.awaitingResponse && state.conversationMode === 'print') {
        input.value = inputPrefix();
        input.style.height = 'auto';
        state.isShowingPlaceholder = false;
        setCursorAfterPrefix();
        await handlePrintResponse(content);
        return;
    }

    // Add user message to state
    state.messages.push({ role: 'user', content });

    // Clear input without showing placeholder (let blur handler do that if needed)
    input.value = inputPrefix();
    input.style.height = 'auto'; // Reset height to single line
    input.style.color = '';
    input.style.fontStyle = '';
    state.isShowingPlaceholder = false;
    setCursorAfterPrefix();

    // Show thinking indicator
    state.isThinking = true;
    updateInputContainerStyling();
    render();

    try {
        // Send to API
        const response = await sendMessage(state.messages);

        // Add assistant response
        state.messages.push({
            role: 'assistant',
            content: response.response || response.content || 'Error: No response'
        });

        // Trim history if too long
        if (state.messages.length > CONSTANTS.MAX_HISTORY_LENGTH) {
            state.messages = state.messages.slice(-CONSTANTS.MAX_HISTORY_LENGTH);
        }

    } catch (error) {
        console.error('[Chat] Error:', error);
        state.messages.push({
            role: 'assistant',
            content: `Error: ${error.message}`
        });
    } finally {
        state.isThinking = false;
        render();
    }
}

// --- Print (model-leads question + live reward) ------------------------------

let printHookTimer = null;

/**
 * Keep the live engagement-energy badge fresh. The model is only queried for a
 * question when the user clicks Print (see PRESENT_PRINT_QUESTION) - this poll
 * never asks on its own, it just reads the energy snapshot.
 */
function setupPrintHook() {
    const POLL_MS = 15000;
    const tick = async () => {
        try {
            const snap = await printEnergy();
            if (snap && typeof snap.energy === 'number') {
                state.print.energy = snap;
                renderPrintButton();  // update only the badge, not a full render
            }
        } catch (e) { /* backend not ready */ }
    };
    clearInterval(printHookTimer);
    printHookTimer = setInterval(tick, POLL_MS);
    tick();  // also try once on load
}

/**
 * Query the model for a self-led question and present it as an assistant turn.
 * No-op (just refocus) if a question is already awaiting an answer, so the model
 * is prompted once and only again after the user has responded.
 */
export async function fetchAndPresentQuestion(reroll = false) {
    const input = document.getElementById('message-input');
    if (state.print.awaitingResponse) {
        if (input) input.focus();
        return;
    }

    state.isThinking = true;
    render();
    try {
        const res = await printAsk(reroll);
        if (res && res.available && res.question) {
            state.print.available = true;
            state.print.question = res.question;
            state.print.id = res.id;
        }
    } catch (e) {
        // Backend / generator not ready - leave the chat quiet.
    }
    state.isThinking = false;

    if (!state.print.available || !state.print.question) {
        render();
        return;
    }

    // The model leads with its question; the next user message answers it.
    state.messages.push({ role: 'assistant', content: state.print.question });
    state.print.awaitingResponse = true;
    state.print.available = false;
    render();
    if (input) input.focus();
}

/**
 * Submit the user's answer, attach the reward as a muted caption on that answer
 * (not a chat bubble), then auto-ask the next question inline.
 */
async function handlePrintResponse(content) {
    const userMsg = { role: 'user', content };
    state.messages.push(userMsg);
    const id = state.print.id;
    state.print.awaitingResponse = false;
    state.print.question = null;
    state.print.id = null;
    state.isThinking = true;
    render();

    try {
        const r = await printRespond(id, content);
        if (r && r.status === 'ok') {
            state.print.lastReward = r;
            const pct = Math.round((r.recall || 0) * 100);
            userMsg.caption = `recall ${pct}% · energy ${Number(r.energy).toFixed(3)}`;
        }
    } catch (error) {
        // Scoring failed - leave the answer un-captioned rather than noisy.
    }
    state.isThinking = false;
    render();

    // Now that the user has responded, prompt the next question.
    await fetchAndPresentQuestion();
}

// --- Loop (run one task; re-roll a section on demand, never autonomously) ----

// Generation runs server-side via /api/loop/generate (short, time-capped): the
// active loop mode owns the prompt and parses off any self-predicted score.
// Reroll is always manual (a section's reroll button) - no interval.

/** The task text in the input (prefix stripped), or '' while the placeholder shows. */
function currentInputTask() {
    if (state.isShowingPlaceholder) return '';
    const input = document.getElementById('message-input');
    if (!input) return '';
    const v = input.value;
    return (v.startsWith(inputPrefix()) ? v.slice(inputPrefix().length) : v).trim();
}

/** Drop literal text into the input (replacing any placeholder). */
function setInputText(text) {
    const input = document.getElementById('message-input');
    if (!input) return;
    input.value = inputPrefix() + text;
    input.style.color = '';
    input.style.fontStyle = '';
    input.style.height = 'auto';
    state.isShowingPlaceholder = false;
}

export function stopLoop() {
    state.loop.enabled = false;
    state.loop.generating = false;
}

export function startLoop() {
    state.loop.enabled = true;
    // Seed the canonical "< joke" task unless the user already typed one.
    if (!currentInputTask()) setInputText('joke');
    state.messages = [];  // fresh loop - independent of any prior chat
    render();
    runLoopCycle();  // one section; further rolls are manual
}

/** Run the task once and present the section (task + output) with score + reroll.
 *  Never schedules another run - re-rolling is the user's call. */
async function runLoopCycle() {
    if (!state.loop.enabled || state.isThinking) return;
    const task = currentInputTask() || 'joke';

    state.loop.generating = true;
    state.isThinking = true;
    state.messages = [{ role: 'user', content: task }];
    render();

    let answer, loopId = null;
    try {
        const res = await loopGenerate(task);
        answer = (res.text || '').trim() || '(silence - reroll to try again)';
        if (res.available) loopId = res.id;
    } catch (error) {
        answer = '(the model stumbled - reroll to try again)';
    }

    state.isThinking = false;
    state.loop.generating = false;
    if (!state.loop.enabled) return;  // disabled mid-generation - drop the result

    // The section: this challenge + its output, with a want->need score slider
    // (the human signal). The slider always starts at 0; any self-predicted
    // score lives backend-side (loopId ties the score to it on submit). Even
    // gibberish is scorable; reroll regenerates it.
    state.messages = [
        { role: 'user', content: task },
        { role: 'assistant', content: answer, jokeScore: true, loopId }
    ];
    render();
}

/** Re-roll this section on demand (the section's reroll button / Enter). */
export function rerollLoopNow() {
    if (!state.loop.enabled) return;
    runLoopCycle();
}

/**
 * Handle reroll - regenerate last assistant response
 */
async function handleReroll() {
    // Find last user message
    let lastUserMessage = null;
    for (let i = state.messages.length - 1; i >= 0; i--) {
        if (state.messages[i].role === 'user') {
            lastUserMessage = state.messages[i].content;
            break;
        }
    }

    if (!lastUserMessage) return;

    // Remove last assistant message if present
    if (state.messages.length > 0 && state.messages[state.messages.length - 1].role === 'assistant') {
        state.messages.pop();
    }

    // Show thinking
    state.isThinking = true;
    render();

    try {
        // Re-send to API
        const response = await sendMessage(state.messages);

        // Add new response
        state.messages.push({
            role: 'assistant',
            content: response.response || response.content || 'Error: No response'
        });

    } catch (error) {
        console.error('[Chat] Reroll error:', error);
        state.messages.push({
            role: 'assistant',
            content: `Error: ${error.message}`
        });
    } finally {
        state.isThinking = false;
        render();
    }
}

/**
 * Read settings from modal inputs using form config
 */
function readSettingsFromModal() {
    const updates = readFormValues(FORM_FIELDS.settings);
    Object.assign(state, updates);
}

/**
 * Handle save settings with API test
 */
async function handleSaveSettings() {
    const apiUrlInput = document.getElementById('api-url');
    const newApiUrl = apiUrlInput ? apiUrlInput.value : state.settings.apiUrl;

    // Test API connection if URL changed
    if (newApiUrl !== state.settings.apiUrl) {
        const testResult = await testApiConnection(newApiUrl);

        if (!testResult.success) {
            if (!confirm(`API connection test failed: ${testResult.message}\n\nSave anyway?`)) {
                return;
            }
        }
    }

    // Read and save settings
    readSettingsFromModal();
    saveSettings();

    // Show confirmation
    const confirmation = document.getElementById('save-confirmation');
    if (confirmation) {
        confirmation.classList.add('show');
        setTimeout(() => {
            confirmation.classList.remove('show');
            state.modals.settingsOpen = false;
            render();
        }, 1500);
    }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
