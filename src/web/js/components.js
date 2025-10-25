/**
 * Praxis Web - Pure Functional Components
 * All components are pure functions: data → DOM
 * Following the principle: UI = render(data)
 */

import { CONSTANTS } from './state.js';

/**
 * Create entire app structure - main container for everything
 * @param {Object} state - Application state
 * @returns {string} HTML string
 */
export function createAppStructure(state) {
    return `
        ${createHeader(state)}
        ${createTabNav()}
        ${createTabContents()}
        ${createSettingsModalContainer()}
    `;
}

/**
 * Create header with logo, system prompt, and controls
 * @param {Object} state - Application state
 * @returns {string} HTML string
 */
export function createHeader(state) {
    return `
        <header class="header">
            <div class="logo">
                <div class="prism-logo">
                    <canvas id="prism-canvas"></canvas>
                </div>
                <span class="logo-separator">|</span>
                <span class="system-prompt-header" id="developer-prompt" contenteditable="true" spellcheck="false">${escapeHtml(state.settings.systemPrompt)}</span>
            </div>
            <div class="header-actions">
                <button class="theme-toggle-button" id="theme-toggle">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16" id="theme-icon">
                        ${state.theme === 'dark' ? CONSTANTS.THEME_ICONS.sun : CONSTANTS.THEME_ICONS.moon}
                    </svg>
                </button>
                <button class="settings-button" id="settings-button">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
                        <path d="M8 4.754a3.246 3.246 0 1 0 0 6.492 3.246 3.246 0 0 0 0-6.492zM5.754 8a2.246 2.246 0 1 1 4.492 0 2.246 2.246 0 0 1-4.492 0z"/>
                        <path d="M9.796 1.343c-.527-1.79-3.065-1.79-3.592 0l-.094.319a.873.873 0 0 1-1.255.52l-.292-.16c-1.64-.892-3.433.902-2.54 2.541l.159.292a.873.873 0 0 1-.52 1.255l-.319.094c-1.79.527-1.79 3.065 0 3.592l.319.094a.873.873 0 0 1 .52 1.255l-.16.292c-.892 1.64.901 3.434 2.541 2.54l.292-.159a.873.873 0 0 1 1.255.52l.094.319c.527 1.79 3.065 1.79 3.592 0l.094-.319a.873.873 0 0 1 1.255-.52l.292.16c1.64.893 3.434-.902 2.54-2.541l-.159-.292a.873.873 0 0 1 .52-1.255l.319-.094c1.79-.527 1.79-3.065 0-3.592l-.319-.094a.873.873 0 0 1-.52-1.255l.16-.292c.893-1.64-.902-3.433-2.541-2.54l-.292.159a.873.873 0 0 1-1.255-.52l-.094-.319zm-2.633.283c.246-.835 1.428-.835 1.674 0l.094.319a1.873 1.873 0 0 0 2.693 1.115l.291-.16c.764-.415 1.6.42 1.184 1.185l-.159.292a1.873 1.873 0 0 0 1.116 2.692l.318.094c.835.246.835 1.428 0 1.674l-.319.094a1.873 1.873 0 0 0-1.115 2.693l.16.291c.415.764-.42 1.6-1.185 1.184l-.291-.159a1.873 1.873 0 0 0-2.693 1.116l-.094.318c-.246.835-1.428.835-1.674 0l-.094-.319a1.873 1.873 0 0 0-2.692-1.115l-.292.16c-.764.415-1.6-.42-1.184-1.185l.159-.291A1.873 1.873 0 0 0 1.945 8.93l-.319-.094c-.835-.246-.835-1.428 0-1.674l.319-.094A1.873 1.873 0 0 0 3.06 4.377l-.16-.292c-.415-.764.42-1.6 1.185-1.184l.292.159a1.873 1.873 0 0 0 2.692-1.115l.094-.319z"/>
                    </svg>
                    Settings
                </button>
            </div>
        </header>
    `;
}

/**
 * Create tab navigation structure
 * @returns {string} HTML string
 */
export function createTabNav() {
    return `
        <div class="tab-nav">
            <div class="tab-buttons" id="tab-buttons">
                <!-- Rendered dynamically by renderTabs() -->
            </div>
            <div class="terminal-status">
                <!-- Rendered dynamically by renderTerminalStatus() -->
            </div>
        </div>
    `;
}

/**
 * Create all tab content containers
 * @returns {string} HTML string
 */
export function createTabContents() {
    return `
        <!-- Chat Tab -->
        <div class="tab-content active" id="chat-content">
            <div class="chat-container" id="chat-container">
                <!-- Messages rendered dynamically -->
            </div>
            <div class="input-container">
                <textarea class="message-input" id="message-input" rows="1"></textarea>
            </div>
        </div>

        <!-- Terminal Tab -->
        <div class="tab-content" id="terminal-content">
            <div class="terminal-container" id="terminal-display">
                <div class="terminal-line">Terminal ready. Dashboard will connect automatically when available.</div>
            </div>
        </div>

        <!-- Spec Tab -->
        <div class="tab-content" id="spec-content">
            <div class="spec-container" id="spec-container">
                <div class="loading-placeholder">Loading specification...</div>
            </div>
        </div>

        <!-- Agents Tab -->
        <div class="tab-content" id="agents-content">
            <div class="agents-container" id="agents-container">
                <div class="loading-placeholder">Loading agents...</div>
            </div>
        </div>

        <!-- Research Tab -->
        <div class="tab-content" id="research-content">
            <div class="research-container" id="research-container">
                <div class="loading-placeholder">Loading metrics...</div>
            </div>
        </div>
    `;
}

/**
 * Create settings modal container
 * @returns {string} HTML string
 */
export function createSettingsModalContainer() {
    return `
        <div class="settings-modal" id="settings-modal">
            <div class="modal-content">
                <!-- Rendered dynamically when opened -->
            </div>
        </div>
    `;
}

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
