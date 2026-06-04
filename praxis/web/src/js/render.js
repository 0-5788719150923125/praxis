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
    createTab,
    createSettingsModal,
    createTerminalStatus,
    createAppStructure
} from './components.js';
import { centerLoopedTabs, TAB_LOOP_COPIES } from './mobile.js';

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
    renderAgentsTitle();
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
    if (!items.length) {
        panel.innerHTML = '<div class="notification-empty">No events yet.</div>';
    } else {
        // Newest first.
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
    return `
<svg class="harmonic-icon" viewBox="0 0 24 24" aria-hidden="true">
  <path fill="currentColor" fill-rule="evenodd" d="${frames[0]}">
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
    // Set the (animating) icon once, then only update the value text - so re-renders
    // don't restart the SMIL warp.
    let value = badge.querySelector('.badge-value');
    if (!value) {
        // The chip (icon + value) is the visible, content-width element; the
        // badge itself is just a positioning wrapper (full-width line on mobile).
        badge.innerHTML = `<span class="badge-chip">${HARMONIC_ICON}<span class="badge-value"></span></span>`;
        value = badge.querySelector('.badge-value');
    }
    value.textContent = Number(snap.energy).toFixed(2);
}

/**
 * Render the Read panel: a full-height content card if one is open, else the
 * ranked search hits.
 */
function renderKbResults() {
    const container = document.getElementById('kb-results');
    if (!container) return;

    let html;
    if (state.kbOpenItem) {
        html = createKbCard(state.kbOpenItem, state.kbOpenItem.html || '');
    } else if (!state.kbResults.length) {
        html = state.kbSearching ? '<div class="kb-empty">Searching...</div>' : '';
    } else {
        html = state.kbResults.map(createKbResult).join('');
    }

    // Same flicker guard as renderMessages: don't re-write identical results on
    // every periodic render.
    if (container._kbSig === html) return;
    container._kbSig = html;
    container.innerHTML = html;
    if (state.kbOpenItem) container.scrollTop = 0;
}

/**
 * Render chat messages
 */
function renderMessages() {
    const container = document.getElementById('chat-container');
    if (!container) return;

    const isDarkMode = state.theme === 'dark';

    // Render all messages
    const messagesHTML = state.messages
        .map((msg, index) => {
            const isLast = index === state.messages.length - 1;
            return createMessage(msg, isDarkMode, isLast);
        })
        .join('');

    // Add thinking indicator if needed
    const thinkingHTML = state.isThinking ? createThinkingIndicator(isDarkMode) : '';
    const html = messagesHTML + thinkingHTML;

    // Skip the rebuild when nothing changed. Periodic renders (the energy poll,
    // websocket events) would otherwise re-write innerHTML every cycle, which
    // restarts the thinking-dots animation (flicker) and clobbers a mid-drag
    // slider. The signature lives on the element, so a fresh container (tab
    // rebuild) has none and always renders.
    if (container._msgSig === html) return;
    container._msgSig = html;
    container.innerHTML = html;

    // Scroll to bottom
    ensureLastMessageVisible();
}

/**
 * Render tabs
 */
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
    const buttons = tabs.map(tab => createTab(tab)).join('');

    container.innerHTML = loop ? buttons.repeat(TAB_LOOP_COPIES) : buttons;

    // Show/hide tab content
    state.tabs.forEach(tab => {
        const content = document.getElementById(`${tab.id}-content`);
        if (content) {
            content.classList.toggle('active', tab.active);
        }
    });

    // Seat the active tab in the middle copy so a full set of strip sits on
    // each side, ready to scroll either way (no-op on desktop).
    if (loop) centerLoopedTabs();
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
function renderTerminalStatus() {
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

/**
 * Render agents tab title (theme-aware: Hangar/Wire)
 */
function renderAgentsTitle() {
    // Only update if agents tab is loaded
    if (!state.agents.loaded) return;

    const container = document.getElementById('agents-container');
    if (!container) return;

    // Find the title element in the tab header
    const titleElement = container.querySelector('.tab-header h2');
    if (!titleElement) return;

    // Update title based on current theme
    const title = state.theme === 'dark' ? 'Hangar' : 'Wire';
    if (titleElement.textContent !== title) {
        titleElement.textContent = title;
    }
}

/**
 * Ensure last message is visible (scroll to bottom)
 */
function ensureLastMessageVisible() {
    const container = document.getElementById('chat-container');
    if (!container) return;

    const lastMessage = container.lastElementChild;
    if (lastMessage) {
        requestAnimationFrame(() => {
            lastMessage.scrollIntoView({ behavior: 'smooth', block: 'end' });
        });
    }
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
