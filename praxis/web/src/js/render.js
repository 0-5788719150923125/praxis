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
    const readMode = state.conversationMode === 'read';
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
 * The Print button is always present and always looks the same - it never greys
 * out and never auto-highlights when a question arrives. It's simply inert
 * (clicking does nothing) until the model has posed a question to answer; the
 * PRESENT_PRINT_QUESTION action guards on state.print.available.
 */
function renderPrintButton() {
    const btn = document.querySelector('.tool-toggle[data-tool="print"]');
    if (btn) {
        // Defensive: ensure no stale conditional styling lingers across renders.
        btn.classList.remove('inactive', 'available', 'active');
    }

    // Live-energy badge: appears once at least one real-user Print reward exists.
    const badge = document.getElementById('print-energy-badge');
    if (badge) {
        const snap = state.print.energy;
        const live = snap && snap.count > 0;
        badge.hidden = !live;
        if (live) badge.textContent = `⚡ ${Number(snap.energy).toFixed(2)}`;
    }
}

/**
 * Render the Read panel: a full-height content card if one is open, else the
 * ranked search hits.
 */
function renderKbResults() {
    const container = document.getElementById('kb-results');
    if (!container) return;

    if (state.kbOpenItem) {
        container.innerHTML = createKbCard(state.kbOpenItem, state.kbOpenItem.html || '');
        container.scrollTop = 0;
        return;
    }

    if (!state.kbResults.length) {
        container.innerHTML = state.kbSearching ? '<div class="kb-empty">Searching...</div>' : '';
        return;
    }
    container.innerHTML = state.kbResults.map(createKbResult).join('');
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

    container.innerHTML = messagesHTML + thinkingHTML;

    // Scroll to bottom
    ensureLastMessageVisible();
}

/**
 * Render tabs
 */
function renderTabs() {
    const container = document.querySelector('.tab-buttons');
    if (!container) return;

    // On mobile the strip is a continuous loop: rotate the list so the active
    // tab leads and the rest trail in wrap order (the tab after the last is the
    // first). Paired with the wrap-swipe, the ribbon circles endlessly and the
    // active tab is always anchored at the left edge - no deep overflow dead-end.
    // Desktop keeps the natural fixed order. (Content visibility is keyed by
    // tab id, not DOM order, so reordering the buttons is safe.)
    let tabs = state.tabs;
    if (window.innerWidth <= 768 && tabs.length > 1) {
        const a = tabs.findIndex(t => t.active);
        if (a > 0) tabs = [...tabs.slice(a), ...tabs.slice(0, a)];
    }

    container.innerHTML = tabs
        .map(tab => createTab(tab))
        .join('');

    // Show/hide tab content
    state.tabs.forEach(tab => {
        const content = document.getElementById(`${tab.id}-content`);
        if (content) {
            content.classList.toggle('active', tab.active);
        }
    });
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
