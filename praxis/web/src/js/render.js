/**
 * Praxis Web - Render Function
 * UI = render(state)
 * Pure functional rendering - state goes in, DOM comes out
 */

import { state, CONSTANTS, DEFAULT_SYSTEM_PROMPT } from './state.js';
import {
    createMessage,
    createThinkingIndicator,
    createKbResult,
    createKbCard,
    createKbExplorer,
    createTab,
    createSettingsModal,
    createTerminalStatus,
    createAppStructure
} from './components.js';
import { centerLoopedTabs, TAB_LOOP_COPIES } from './mobile.js';
import { attachWheelMomentum } from './momentum.js';
import { kbSearch } from './api.js';

/**
 * Initial render - builds entire app structure from scratch
 * Called once on initialization
 */
export function renderAppStructure() {
    const appContainer = document.querySelector('.app-container');
    if (!appContainer) {
        console.error('[Render] No .app-container found');
        return;
    }

    // Generate entire UI structure from state
    appContainer.innerHTML = createAppStructure(state);
}

/**
 * Main render function - updates entire UI based on state
 * This is the heart of the functional approach
 */
export function render() {
    renderConversation();
    renderTabs();
    renderTheme();
    renderTerminalStatus();
    renderModal();
    renderSystemPrompt();
    renderNotifications();
}

/**
 * Render the header notification bell: unread badge + dropdown panel.
 * Pure projection of state.notifications onto the DOM.
 */
export function renderNotifications() {
    const badge = document.getElementById('notification-badge');
    if (badge) {
        const unread = state.notifications.unread;
        badge.textContent = unread > 9 ? '9+' : String(unread);
        badge.hidden = unread === 0;
    }

    const panel = document.getElementById('notification-panel');
    if (!panel) return;

    // The pop-out shows on hover (desktop) or tap (touch, via .open) - so the content is
    // ALWAYS rendered and ready; visibility is owned by CSS, not the `hidden` attribute.
    const items = state.notifications.items;
    // Re-serialize the panel body only when the event set actually changes -
    // render() runs on nearly every action, and rebuilding this innerHTML each
    // time is pure waste when the list is identical. Keyed by event ids. The
    // open/position/hover tail below still runs every call (panelOpen can change
    // without the items changing).
    const sig = items.map((e) => e.id).join(',');
    if (panel._notifSig !== sig) {
        panel._notifSig = sig;
        renderNotificationItems(panel, items);
    }

    // Touch devices toggle the pop-out open with .open; hover-capable devices ignore it.
    panel.classList.toggle('open', state.notifications.panelOpen);

    // Mobile: the panel is viewport-fixed (see responsive.css). Anchor its top to
    // the bell's live position, since the header height varies (the system prompt
    // can wrap). Clear it otherwise so the CSS rule owns positioning.
    if (state.notifications.panelOpen && window.innerWidth <= 768) {
        const btn = document.getElementById('notifications-btn');
        if (btn) panel.style.top = `${Math.round(btn.getBoundingClientRect().bottom + 6)}px`;
    } else {
        panel.style.top = '';
    }

    // Desktop: hovering the bell counts as reading - clear the unread badge. Bind once
    // per (re)rendered wrapper.
    const wrapper = panel.closest('.notification-wrapper');
    if (wrapper && !wrapper._notifHoverBound) {
        wrapper._notifHoverBound = true;
        wrapper.addEventListener('mouseenter', () => {
            if (state.notifications.unread !== 0) {
                state.notifications.unread = 0;
                const b = document.getElementById('notification-badge');
                if (b) b.hidden = true;
            }
        });
    }
}

/** Serialize the notification event list into the panel body (newest first). */
function renderNotificationItems(panel, items) {
    if (!items.length) {
        panel.innerHTML = '<div class="notification-empty">No events yet.</div>';
        return;
    }
    panel.innerHTML = items
        .slice()
        .reverse()
        .map((ev) => {
            const age = typeof ev.hours_elapsed === 'number'
                ? `${ev.hours_elapsed.toFixed(2)}h`
                : '';
            const level = ev.level || 'info';
            return `
                    <div class="notification-item notification-${level}">
                        <span class="notification-message">${escapeNotification(ev.message)}</span>
                        <span class="notification-age">${age}</span>
                    </div>
                `;
        })
        .join('');
}

/** Minimal HTML escaping for event text. */
function escapeNotification(str) {
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
}

/**
 * Render the Gymnasium conversation area: KB results in Read mode, chat
 * messages in Evaluate mode. Only the active panel is shown.
 */
function renderConversation() {
    const mode = state.conversationMode;
    const readMode = mode === 'read';

    // Read / Evaluate / Print are discrete, mutually-exclusive modes: exactly one
    // is highlighted, mirroring state.conversationMode onto the toolbar.
    document.querySelectorAll(
        '.tool-toggle[data-tool="read"], .tool-toggle[data-tool="evaluate"], .tool-toggle[data-tool="print"]'
    ).forEach(btn => btn.classList.toggle('active', btn.dataset.tool === mode));
    document.documentElement.toggleAttribute('data-eval', mode === 'evaluate');

    // Loop is coupled to Print (inert unless Print mode is active) but is never
    // greyed - it keeps the same color as every other button, just lighting up
    // via .active while a loop is running.
    const loopBtn = document.querySelector('.tool-toggle[data-tool="loop"]');
    if (loopBtn) {
        loopBtn.classList.toggle('active', state.loop.enabled);
    }

    const results = document.getElementById('kb-results');
    const chat = document.getElementById('chat-container');
    if (results) results.hidden = !readMode;
    if (chat) chat.hidden = readMode;

    renderPrintButton();

    // Read floats the input up with results/content below; Evaluate is texting
    // style (messages above, input pinned at bottom). Driven by a pane class.
    const pane = document.getElementById('chat-content');
    if (pane) pane.classList.toggle('mode-read', readMode);

    if (readMode) {
        renderKbResults();
    } else {
        renderMessages();
    }
}

/**
 * The Print button's highlight is driven by mode (renderConversation), like Read
 * and Evaluate - it lights up only while Print mode is active. This handles just
 * the live-energy badge, which appears once a real-user Print reward exists.
 */
// Harmonic glyph: two half-moon "planes", each a circular segment anchored to 2
// of 3 shared points (both share the top point A; one fans to B, one to C). They
// breathe out of phase and the overlap is punched to the background (evenodd), so
// it reads as two warping planes in (and out of) harmony. currentColor -> accent.
function buildHarmonicIcon() {
    const rnd = (a, b) => a + Math.random() * (b - a);
    // Three anchors, lightly randomized each load (so no two are identical).
    const A = { x: 12, y: rnd(4.5, 5.5) };
    const B = { x: rnd(5.5, 6.5), y: rnd(17.5, 18.5) };
    const C = { x: rnd(17.5, 18.5), y: rnd(17.5, 18.5) };
    const n = (v) => v.toFixed(1);
    // A half-moon (minor circular segment) from p to q: arc out, chord back.
    const seg = (p, q, r, sweep) =>
        `M${n(p.x)} ${n(p.y)}A${n(r)} ${n(r)} 0 0 ${sweep} ${n(q.x)} ${n(q.y)}Z`;
    // One warp state: the two planes' radii swing sinusoidally, phase-shifted, so
    // they flex in harmony. Same path structure each frame (only radii change).
    const state = (k) => {
        const r1 = 13 + 3.5 * Math.sin(k * 2 * Math.PI);
        const r2 = 13 + 3.5 * Math.sin(k * 2 * Math.PI + Math.PI * 0.66);
        return seg(A, B, r1, 1) + seg(A, C, r2, 0);
    };
    const frames = [0, 0.33, 0.66].map(state);
    const values = [...frames, frames[0]].join(';');
    // The shapes stay white; the icon's BACKGROUND is the gauge - green filling
    // from the bottom to the energy fraction, gray above (CSS gradient driven by
    // the --energy custom property, set in renderPrintButton).
    return `
<svg class="harmonic-icon" viewBox="0 0 24 24" aria-hidden="true">
  <path fill="#fff" fill-rule="evenodd" d="${frames[0]}">
    <animate attributeName="d" dur="7s" repeatCount="indefinite" calcMode="spline"
      keyTimes="0;0.33;0.66;1"
      keySplines="0.4 0 0.6 1;0.4 0 0.6 1;0.4 0 0.6 1" values="${values}"/>
  </path>
</svg>`;
}

const HARMONIC_ICON = buildHarmonicIcon();

export function renderPrintButton() {
    const badge = document.getElementById('print-energy-badge');
    if (!badge) return;
    const snap = state.print.energy;
    const live = snap && snap.count > 0;
    badge.hidden = !live;
    if (!live) return;
    // Set the (animating) icon once, then only move the gauge split - so
    // re-renders don't restart the SMIL warp. No numeric readout: the icon's
    // background IS the gauge (green fills from the bottom; gray = depleted).
    if (!badge.querySelector('.harmonic-icon')) {
        badge.innerHTML = `<span class="badge-chip">${HARMONIC_ICON}</span>`;
    }
    const e = Math.max(0, Math.min(1, Number(snap.energy) || 0));
    badge.style.setProperty('--energy', e.toFixed(3));
    badge.title = `Live engagement energy: ${Math.round(e * 100)}%`;
}

/**
 * Render the Read panel: a full-height content card if one is open, else the
 * ranked search hits.
 */
export function renderKbResults() {
    const container = document.getElementById('kb-results');
    if (!container) return;

    let html;
    const results = state.kbResults;
    if (state.kbOpenItem) {
        html = createKbCard(state.kbOpenItem, state.kbOpenItem.html || '');
    } else if (!results.length) {
        html = state.kbSearching ? '<div class="kb-empty">Searching...</div>' : '';
    } else {
        html = results.map(createKbResult).join('');
    }

    // Same flicker guard as renderMessages: don't re-write identical results on
    // every periodic render.
    if (container._kbSig === html) return;
    container._kbSig = html;
    container.innerHTML = html;
    // Wheel input coasts like a touch flick (idempotent; the list container
    // persists, an opened card's body is recreated each open).
    attachWheelMomentum(container);
    if (state.kbOpenItem) {
        // The full document is rendered; land on the section we matched (a note's
        // inner heading) so you can scroll above/below it. No anchor -> top.
        const body = container.querySelector('.kb-card-body');
        if (body) {
            attachWheelMomentum(body);
            scrollKbBodyToAnchor(body, state.kbOpenItem.anchor);
        }
        // Code cards carry their directory context: the ancestor fan mounts
        // above the source so traversal is one tap away.
        if (state.kbOpenItem.type === 'code' && body) {
            mountCodeExplorer(body, state.kbOpenItem.title);
        }
    }
}

// One fetch of the code listing serves every explorer mount; paths only churn
// on reindex, and a stale tree merely omits brand-new files until reopen.
let codeTreePromise = null;

async function mountCodeExplorer(body, path) {
    codeTreePromise ||= kbSearch('', ['code']).catch(() => (codeTreePromise = null, []));
    const tree = await codeTreePromise;
    if (!tree.length || state.kbOpenItem?.title !== path) return;
    body.insertAdjacentHTML('afterbegin', createKbExplorer(tree, path));
}

/** Normalize heading text for fuzzy matching: drop markdown markers/arrows and
 *  collapse whitespace, so a section title matches its rendered element. */
function normalizeAnchor(s) {
    return (s || '').replace(/[#*_`>\-]+/g, ' ').replace(/\s+/g, ' ').trim().toLowerCase();
}

/** Scroll the card body to the heading (or bold list item) matching `anchor`.
 *  Falls back to the top when there's no anchor or no match. */
function scrollKbBodyToAnchor(body, anchor) {
    const want = normalizeAnchor(anchor);
    if (!want) { body.scrollTop = 0; return; }
    let target = null;
    // Headings first (the common case), then list items / paragraphs for the
    // bold-bullet sections notes also split on.
    for (const el of body.querySelectorAll('h1,h2,h3,h4,h5,h6,li,p,strong')) {
        const text = normalizeAnchor(el.textContent);
        if (text && (text === want || text.startsWith(want) || want.startsWith(text))) {
            target = el;
            break;
        }
    }
    if (!target) { body.scrollTop = 0; return; }
    const top = target.getBoundingClientRect().top - body.getBoundingClientRect().top + body.scrollTop;
    body.scrollTop = Math.max(0, top - 8);   // small breathing gap above the heading
}

/**
 * Everything about a message that shapes its DOM, MINUS its text.
 *
 * Splitting this out is what lets a growing reply update in place. A streaming
 * turn changes one thing - the characters inside one `.message-content` - and
 * re-serializing the list for each of them destroyed and rebuilt every message
 * node, taking the user's text selection, the caret's blink phase and a frame
 * of layout with it. That is the flicker.
 */
function messageStructure(msg, isLast) {
    return [
        msg.role,
        isLast ? 'last' : '',
        msg.streaming ? 'streaming' : '',
        // Caption and score are rendered as their own elements, so a change to
        // either really is structural and has to fall through to a rebuild.
        msg.caption ?? '',
        msg.jokeScore ? `score:${msg.score ?? 0}` : '',
    ].join('');
}

/**
 * Write each message's text into the nodes already on the page.
 *
 * Returns whether anything actually changed, or `null` when the DOM does not
 * line up with `state.messages` (a caller raced the rebuild) so the caller can
 * fall back rather than paint nonsense.
 */
function patchMessageText(container) {
    const nodes = container.querySelectorAll('.message-content');
    if (nodes.length !== state.messages.length) return null;

    let changed = false;
    state.messages.forEach((msg, index) => {
        const node = nodes[index];
        const next = msg.content ?? '';
        const current = node.textContent;
        if (current === next) return;
        changed = true;
        // The streaming case is an APPEND, and appending a text node leaves the
        // existing ones - and any selection inside them - untouched. Assigning
        // textContent would replace the lot on every byte. Text nodes never
        // parse markup, so this is as safe as the escapeHtml path it mirrors.
        if (next.startsWith(current)) {
            node.appendChild(document.createTextNode(next.slice(current.length)));
        } else {
            node.textContent = next;
        }
    });
    return changed;
}

/**
 * Render chat messages
 */
function renderMessages() {
    const container = document.getElementById('chat-container');
    if (!container) return;

    const isDarkMode = state.theme === 'dark';
    const lastIndex = state.messages.length - 1;

    const structure =
        state.messages.map((m, i) => messageStructure(m, i === lastIndex)).join('') +
        `${state.isThinking ? 1 : 0}${isDarkMode ? 1 : 0}`;

    // Measured BEFORE the DOM changes: whether the user is reading the tail (so
    // following along is what they want) or has scrolled up (so yanking them
    // back is not).
    const pinned = isPinnedToBottom(container);

    // Same nodes, different text - the streaming case. Patch in place and leave
    // every other node, and the caret's animation, alone.
    if (container._msgStructure === structure) {
        const changed = patchMessageText(container);
        if (changed !== null) {
            // The html signature describes a build that no longer matches the
            // DOM, so retire it rather than let a later render trust it. The
            // count is kept CURRENT rather than retired: it is what tells the
            // next render whether a turn appeared, and a stale one would make
            // that answer arbitrary.
            container._msgSig = null;
            container._msgCount = state.messages.length;
            // Instant, not smooth: a smooth scroll retriggered on every delta
            // spends its whole animation being restarted, which reads as
            // stutter. Only on a real change, so the 2Hz metrics render does
            // not scroll a pinned reader for nothing.
            if (changed && pinned) scrollToBottom(container, 'auto');
            return;
        }
    }

    const messagesHTML = state.messages
        .map((msg, index) => createMessage(msg, isDarkMode, index === lastIndex))
        .join('');
    const thinkingHTML = state.isThinking ? createThinkingIndicator(isDarkMode) : '';
    const html = messagesHTML + thinkingHTML;

    // Skip the rebuild when nothing changed. Periodic renders (the energy poll,
    // websocket events) would otherwise re-write innerHTML every cycle, which
    // restarts the thinking-dots animation (flicker) and clobbers a mid-drag
    // slider. The signature lives on the element, so a fresh container (tab
    // rebuild) has none and always renders.
    if (container._msgSig === html) return;

    // The USER adding a turn is worth jumping to whether or not they were
    // pinned - they just did it, and it is what they want to see. Nothing else
    // is: a reply turn appears because the MODEL started writing, and dragging
    // the view down for that is precisely the yank that makes re-reading an
    // earlier turn impossible while a reply streams.
    const last = state.messages[lastIndex];
    const userActed =
        state.messages.length !== container._msgCount && last && last.role === 'user';
    // Smooth reads well for a turn the user just sent. It does NOT for a reply
    // arriving: the animation would still be running when the next delta
    // scrolls again, and the two fight.
    const behavior = userActed ? 'smooth' : 'auto';

    container._msgSig = html;
    container._msgStructure = structure;
    container._msgCount = state.messages.length;
    container.innerHTML = html;

    if (userActed || pinned) ensureLastMessageVisible(behavior);
}

/**
 * Repaint the conversation alone, for a reply arriving a few characters at a
 * time. `render()` walks the whole app - tabs, theme, modal, notifications -
 * which is a lot of DOM to touch per animation frame for a change confined to
 * one text node.
 *
 * Read mode shows KB results in this pane instead, and owns it; a stream
 * landing while the user is over there is left to the next full render.
 */
export function renderStreamingMessages() {
    if (state.conversationMode === 'read') return;
    renderMessages();
}

/**
 * Render tabs
 */
// Signature of the rendered strip (copies + tab ids/labels). render() runs on
// every metrics tick / notification, so we rebuild the strip's innerHTML only
// when the set actually changes - a blind rebuild every render thrashes the DOM
// and resets the carousel scroll mid-swipe.
let lastTabSig = null;

function renderTabs() {
    const container = document.querySelector('.tab-buttons');
    if (!container) return;

    // Natural fixed order. On mobile the strip is an infinite carousel: lay the
    // same set out several times over so native scroll (smooth, no pop-in) always
    // has buffer copies in both directions, and mobile.js re-centers the scroll
    // toward the middle copy so it never reaches an end. Content visibility is
    // keyed by tab id, so the duplicate buttons are inert copies; clicking any of
    // them switches by data-tab just the same.
    const tabs = state.tabs;
    const loop = window.innerWidth <= 768 && tabs.length > 1;
    const copies = loop ? TAB_LOOP_COPIES : 1;

    const sig = copies + '|' + tabs.map(t => {
        const label = typeof t.label === 'function' ? t.label(state.theme) : t.label;
        return `${t.id}:${label}`;
    }).join(',');

    const existing = container.querySelectorAll('.tab-button');
    const intact = existing.length === tabs.length * copies;
    let rebuilt = false;
    if (sig !== lastTabSig || !intact) {
        const buttons = tabs.map(tab => createTab(tab)).join('');
        container.innerHTML = loop ? buttons.repeat(TAB_LOOP_COPIES) : buttons;
        lastTabSig = sig;
        rebuilt = true;
    } else {
        // Same set: just move the .active class on the existing buttons.
        existing.forEach(btn =>
            btn.classList.toggle('active', btn.dataset.tab === state.currentTab));
    }

    // Show/hide tab content
    let activeChanged = false;
    state.tabs.forEach(tab => {
        const content = document.getElementById(`${tab.id}-content`);
        if (content && content.classList.contains('active') !== tab.active) {
            content.classList.toggle('active', tab.active);
            if (tab.active) activeChanged = true;
        }
    });

    // Seat the active tab in the middle copy so a full set of strip sits on each
    // side, ready to scroll either way. Only on rebuild or an actual tab change -
    // re-seating on every incidental render would jerk the strip mid-scroll. A
    // pure active change (a switch) animates; a rebuild seats instantly (the
    // fresh buttons have no prior scroll position to glide from).
    if (loop && (rebuilt || activeChanged)) centerLoopedTabs(activeChanged && !rebuilt);
}

/**
 * Render theme
 */
function renderTheme() {
    const isDark = state.theme === 'dark';

    if (isDark) {
        document.documentElement.setAttribute('data-theme', 'dark');
    } else {
        document.documentElement.removeAttribute('data-theme');
    }

    // Update theme icon
    const themeIcon = document.getElementById('theme-icon');
    if (themeIcon) {
        themeIcon.innerHTML = isDark ? CONSTANTS.THEME_ICONS.sun : CONSTANTS.THEME_ICONS.moon;
    }
}

/**
 * Render terminal connection status
 */
export function renderTerminalStatus() {
    const indicator = document.getElementById('status-indicator');
    if (!indicator) return;

    if (state.terminal.connected) {
        indicator.classList.add('connected');
    } else {
        indicator.classList.remove('connected');
    }
}

/**
 * Render settings modal
 */
function renderModal() {
    const modal = document.getElementById('settings-modal');
    if (!modal) return;

    // Toggle open class
    modal.classList.toggle('open', state.modals.settingsOpen);

    // Render content if open
    if (state.modals.settingsOpen) {
        const modalContent = modal.querySelector('.modal-content');
        if (modalContent) {
            modalContent.innerHTML = createSettingsModal(state.settings);
        }
    }
}

/**
 * Render system prompt header
 */
function renderSystemPrompt() {
    const element = document.getElementById('developer-prompt');
    if (element && element.textContent !== state.settings.systemPrompt) {
        element.textContent = state.settings.systemPrompt;
    }
    if (element) {
        element.classList.toggle('default-prompt', state.settings.systemPrompt === DEFAULT_SYSTEM_PROMPT);
    }
}

// How far from the bottom still counts as "reading the tail". One line's worth
// of slack, so a scroll that lands a pixel short does not read as "the user
// deliberately scrolled up".
const STICK_TO_BOTTOM_SLACK = 48;

/**
 * Whether the user is at the tail of the conversation.
 *
 * The distinction this draws is the whole point: following along is what
 * someone reading the newest reply wants, and is exactly what someone who
 * scrolled up to re-read an earlier turn does NOT want. A container too short
 * to scroll counts as pinned - there is nowhere else to be.
 */
function isPinnedToBottom(container) {
    const slack = container.scrollHeight - container.clientHeight - container.scrollTop;
    return slack <= STICK_TO_BOTTOM_SLACK;
}

/** Put the tail in view without touching focus or selection. */
function scrollToBottom(container, behavior = 'smooth') {
    container.scrollTo({ top: container.scrollHeight, behavior });
}

/**
 * Ensure last message is visible (scroll to bottom)
 *
 * Deferred a frame so the freshly written nodes have been laid out and
 * `scrollHeight` is the real one. Scrolls the CONTAINER rather than calling
 * `scrollIntoView` on the message: that walks up to the nearest scrollable
 * ancestor, which on a short conversation is the page, and moving the page
 * shifts the input box out from under the user.
 */
function ensureLastMessageVisible(behavior = 'smooth') {
    const container = document.getElementById('chat-container');
    if (!container) return;
    if (!container.lastElementChild) return;
    requestAnimationFrame(() => scrollToBottom(container, behavior));
}

/**
 * Update input container styling based on message presence
 */
export function updateInputContainerStyling() {
    const inputContainer = document.querySelector('.input-container');
    if (inputContainer) {
        inputContainer.classList.toggle('with-messages', state.messages.length > 0);
    }
}
