/**
 * Praxis Web - Render Function
 * UI = render(state)
 * Pure functional rendering - state goes in, DOM comes out
 */

import { state, CONSTANTS } from './state.js';
import { createMessage, createThinkingIndicator, createTab, createSettingsModal, createTerminalStatus } from './components.js';

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
    const thinkingHTML = state.isThinking ? createThinkingIndicator() : '';

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
    const container = document.querySelector('.terminal-status');
    if (!container) return;

    container.innerHTML = createTerminalStatus(state.terminal.connected);
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
