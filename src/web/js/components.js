/**
 * Praxis Web - Pure Functional Components
 * All components are pure functions: data → DOM
 * Following the principle: UI = render(data)
 */

import { CONSTANTS } from './state.js';

// ============================================================================
// GENERIC UI PRIMITIVES
// ============================================================================

/**
 * Escape HTML to prevent XSS
 * @param {string} str - String to escape
 * @returns {string} Escaped string
 */
export const escapeHtml = (str) => {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
};

/**
 * Create a section container with title
 * @param {string} title - Section title
 * @param {string} content - Section content HTML
 * @param {string} id - Optional section ID
 * @returns {string} HTML string
 */
export const createSection = (title, content, id = '') => `
    <div class="spec-section"${id ? ` id="${id}"` : ''}>
        ${title ? `<div class="spec-title">${title}</div>` : ''}
        ${content}
    </div>
`;

/**
 * Create a code block
 * @param {string} code - Code to display
 * @param {string} className - Additional CSS class
 * @returns {string} HTML string
 */
export const createCodeBlock = (code, className = '') => `
    <div class="spec-code ${className}">${escapeHtml(code)}</div>
`;

/**
 * Create a pre-formatted code block
 * @param {string} code - Code to display
 * @returns {string} HTML string
 */
export const createPreBlock = (code) => `
    <pre class="spec-code">${escapeHtml(code)}</pre>
`;

/**
 * Create a metadata label
 * @param {string} text - Label text (can include HTML)
 * @returns {string} HTML string
 */
export const createMetadata = (text) => `
    <div class="spec-metadata">${text}</div>
`;

/**
 * Create a key-value pair
 * @param {string} key - Key label
 * @param {string|number} value - Value to display
 * @param {boolean} highlight - Whether to highlight the value
 * @returns {string} HTML string
 */
export const createKeyValue = (key, value, highlight = true) => {
    const valueClass = highlight ? 'spec-value-highlight' : '';
    return `
        <div class="spec-metadata">
            ${key}: <span class="${valueClass}">${value}</span>
        </div>
    `;
};

/**
 * Create a numbered steps list
 * @param {Array} steps - Array of {instruction, code} objects
 * @returns {string} HTML string
 */
export const createStepsList = (steps) => {
    return steps.map((step, index) => `
        <div class="spec-metadata">${index + 1}. ${step.instruction}</div>
        <div class="spec-code spec-code-step">${escapeHtml(step.code)}</div>
    `).join('');
};

/**
 * Create a button
 * @param {string} text - Button text
 * @param {string} className - CSS class
 * @param {Object} dataset - Data attributes as key-value pairs
 * @returns {string} HTML string
 */
export const createButton = (text, className, dataset = {}) => {
    const dataAttrs = Object.entries(dataset)
        .map(([key, value]) => `data-${key}="${escapeHtml(value)}"`)
        .join(' ');

    return `
        <button class="${className}" ${dataAttrs}>
            ${text}
        </button>
    `;
};

/**
 * Create a link
 * @param {string} text - Link text
 * @param {string} href - Link URL
 * @param {string} className - CSS class
 * @param {Object} attrs - Additional attributes
 * @returns {string} HTML string
 */
export const createLink = (text, href, className = 'spec-link', attrs = {}) => {
    const attrString = Object.entries(attrs)
        .map(([key, value]) => `${key}="${escapeHtml(value)}"`)
        .join(' ');

    return `<a href="${escapeHtml(href)}" class="${className}" ${attrString}>${text}</a>`;
};

/**
 * Create a wrapper div
 * @param {string} content - Content HTML
 * @param {string} className - CSS class
 * @param {string} style - Inline styles (use sparingly)
 * @returns {string} HTML string
 */
export const createWrapper = (content, className = '', style = '') => {
    const styleAttr = style ? ` style="${style}"` : '';
    const classAttr = className ? ` class="${className}"` : '';
    return `<div${classAttr}${styleAttr}>${content}</div>`;
};

/**
 * Create a list of items from array
 * @param {Array} items - Array of items
 * @param {Function} renderFn - Function to render each item
 * @returns {string} HTML string
 */
export const createList = (items, renderFn) => {
    return items.map(renderFn).join('');
};

/**
 * Conditionally render content
 * @param {boolean} condition - Whether to render
 * @param {Function|string} content - Content to render (function or string)
 * @returns {string} HTML string or empty string
 */
export const renderIf = (condition, content) => {
    if (!condition) return '';
    return typeof content === 'function' ? content() : content;
};

/**
 * Create a hash display with highlighted truncated part
 * @param {string} fullHash - Full hash string
 * @param {number} truncLength - Length of truncated part
 * @param {string} linkHref - Optional link for truncated part
 * @returns {string} HTML string
 */
export const createHashDisplay = (fullHash, truncLength, linkHref = '') => {
    const truncPart = fullHash.substring(0, truncLength);
    const restPart = fullHash.substring(truncLength);

    const truncHtml = linkHref
        ? createLink(truncPart, linkHref, 'spec-link-primary')
        : `<span class="spec-hash-trunc">${truncPart}</span>`;

    return `
        <div class="spec-hash">
            ${truncHtml}<span class="spec-hash-rest">${restPart}</span>
        </div>
    `;
};

/**
 * Format a number with locale-specific separators
 * @param {number} num - Number to format
 * @returns {string} Formatted number string
 */
export const formatNumber = (num) => num.toLocaleString();

/**
 * Format JSON for display
 * @param {Object} obj - Object to format
 * @param {number} indent - Indentation spaces
 * @returns {string} Formatted JSON string
 */
export const formatJSON = (obj, indent = 2) => JSON.stringify(obj, null, indent);

// ============================================================================
// TAB TEMPLATE GENERATORS
// ============================================================================

/**
 * Create a simple container with loading placeholder
 * Generic template for lazy-loaded tabs
 * @param {string} containerClass - CSS class for container
 * @param {string} containerId - DOM ID for container
 * @param {string} placeholderText - Loading placeholder text
 * @returns {string} HTML string
 */
export const createSimpleTabContent = (containerClass, containerId, placeholderText) => `
    <div class="${containerClass}" id="${containerId}">
        <div class="loading-placeholder">${placeholderText}</div>
    </div>
`;

/**
 * Create container with initial content
 * @param {string} containerClass - CSS class for container
 * @param {string} containerId - DOM ID for container
 * @param {string} initialContent - Initial HTML content
 * @returns {string} HTML string
 */
export const createContainerWithContent = (containerClass, containerId, initialContent) => `
    <div class="${containerClass}" id="${containerId}">
        ${initialContent}
    </div>
`;

/**
 * Create chat tab with message container and input
 * @param {Object} config - Configuration {chatContainerId, inputId, inputRows}
 * @returns {string} HTML string
 */
export const createChatTabContent = (config) => `
    <div class="chat-container" id="${config.chatContainerId}">
        <!-- Messages rendered dynamically -->
    </div>
    <div class="input-container">
        <textarea class="message-input" id="${config.inputId}" rows="${config.inputRows}"></textarea>
    </div>
`;

// ============================================================================
// APP-SPECIFIC COMPONENTS
// ============================================================================


/**
 * Create entire app structure - main container for everything
 * @param {Object} state - Application state
 * @returns {string} HTML string
 */
export function createAppStructure(state) {
    return `
        ${createHeader(state)}
        ${createTabNav()}
        ${createTabContents(state.tabs)}
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
 * Create all tab content containers - data-driven from tabs configuration
 * @param {Array} tabs - Array of tab configurations
 * @returns {string} HTML string
 */
export function createTabContents(tabs) {
    return tabs.map(tab => {
        // Build CSS classes dynamically
        const classes = ['tab-content', ...tab.customClasses];
        if (tab.active) classes.push('active');

        return `
            <div class="${classes.join(' ')}" id="${tab.id}-content">
                ${tab.template()}
            </div>
        `;
    }).join('');
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

