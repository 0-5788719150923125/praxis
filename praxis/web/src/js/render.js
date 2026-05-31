/**
 * Praxis Web - Render Function
 * UI = render(state)
 * Pure functional rendering - state goes in, DOM comes out
 */

import { state, CONSTANTS, DEFAULT_SYSTEM_PROMPT } from './state.js';
import {
    createMessage,
    createThinkingIndicator,
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
    renderMessages();
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

    panel.hidden = !state.notifications.panelOpen;
    if (!state.notifications.panelOpen) return;

    const items = state.notifications.items;
    if (!items.length) {
        panel.innerHTML = '<div class="notification-empty">No events yet.</div>';
        return;
    }

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

/** Minimal HTML escaping for event text. */
function escapeNotification(str) {
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
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

    container.innerHTML = state.tabs
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
