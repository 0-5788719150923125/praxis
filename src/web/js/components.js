/**
 * Praxis Web - Pure Functional Components
 * All components are pure functions: data → DOM
 */

import { CONSTANTS } from './state.js';

/**
 * Create a chat message element
 * @param {Object} msg - Message data { role: 'user'|'assistant', content: string }
 * @param {boolean} isDarkMode - Current theme
 * @param {boolean} isLast - Whether this is the last message
 * @returns {string} HTML string
 */
export function createMessage({ role, content }, isDarkMode, isLast = false) {
    const headerText = isDarkMode
        ? (role === 'user' ? 'Me' : 'You')
        : (role === 'user' ? 'You' : 'Me');

    // Add reroll button for last assistant message
    const rerollButton = (role === 'assistant' && isLast)
        ? '<button class="reroll-button" id="reroll-button">🔄 Reroll</button>'
        : '';

    return `
        <div class="message ${role}">
            ${rerollButton}
            <div class="message-header">${headerText}</div>
            <div class="message-content">${escapeHtml(content)}</div>
        </div>
    `;
}

/**
 * Create thinking indicator
 * @returns {string} HTML string
 */
export function createThinkingIndicator() {
    return `
        <div class="message assistant" id="thinking-message">
            <div class="message-header">
                <span class="header-name">Me</span>
                <span class="thinking-status">
                    thinking<span class="dots">
                        <span class="dot"></span>
                        <span class="dot"></span>
                        <span class="dot"></span>
                    </span>
                </span>
            </div>
        </div>
    `;
}

/**
 * Create tab button
 * @param {Object} tab - Tab data { id, label, active }
 * @returns {string} HTML string
 */
export function createTab({ id, label, active }) {
    return `
        <button
            class="tab-button ${active ? 'active' : ''}"
            data-tab="${id}"
        >
            ${label}
        </button>
    `;
}

/**
 * Create settings modal content
 * @param {Object} settings - Settings data
 * @returns {string} HTML string
 */
export function createSettingsModal(settings) {
    return `
        <div class="modal-header">
            <h3 class="modal-title">Settings</h3>
            <button class="close-button" id="close-modal">&times;</button>
        </div>

        <div class="form-group">
            <label for="api-url">API Server URL</label>
            <input type="text" id="api-url" value="${settings.apiUrl}">
        </div>

        <h4 class="modal-subheading">Generation Parameters</h4>

        <div class="form-group">
            <label for="max-tokens">Max New Tokens (50-1000)</label>
            <input type="number" id="max-tokens" min="50" max="1000" value="${settings.maxTokens}">
        </div>

        <div class="form-group">
            <label for="temperature">Temperature (0.1-2.0)</label>
            <input type="range" id="temperature" min="0.1" max="2.0" step="0.05" value="${settings.temperature}">
            <span id="temperature-value">${settings.temperature}</span>
        </div>

        <div class="form-group">
            <label for="repetition-penalty">Repetition Penalty (1.0-2.0)</label>
            <input type="range" id="repetition-penalty" min="1.0" max="2.0" step="0.05" value="${settings.repetitionPenalty}">
            <span id="repetition-penalty-value">${settings.repetitionPenalty}</span>
        </div>

        <div class="form-group">
            <label>
                <input type="checkbox" id="do-sample" ${settings.doSample ? 'checked' : ''}>
                Enable Sampling
            </label>
        </div>

        <h4 class="modal-subheading">Developer Options</h4>

        <div class="form-group">
            <label>
                <input type="checkbox" id="debug-logging" ${settings.debugLogging ? 'checked' : ''}>
                Debug Logging
            </label>
        </div>

        <div class="button-group">
            <button class="save-button" id="save-settings">Save Settings</button>
            <button class="reset-button" id="reset-settings">Reset All</button>
        </div>
        <div class="save-confirmation" id="save-confirmation">Settings saved! Refreshing...</div>
    `;
}

/**
 * Create terminal status indicator
 * @param {boolean} connected - Connection status
 * @returns {string} HTML string
 */
export function createTerminalStatus(connected) {
    return `
        <span class="status-indicator ${connected ? 'connected' : ''}" id="status-indicator"></span>
        <span id="terminal-status">${connected ? 'Connected' : 'Disconnected'}</span>
    `;
}

/**
 * Create terminal line
 * @param {string} text - Terminal line text
 * @returns {string} HTML string
 */
export function createTerminalLine(text) {
    return `<div class="terminal-line">${escapeHtml(text)}</div>`;
}

/**
 * Escape HTML to prevent XSS
 * @param {string} str - String to escape
 * @returns {string} Escaped string
 */
function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}
