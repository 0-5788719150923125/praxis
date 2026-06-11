/**
 * Praxis Web - Pure Functional Components
 * All components are pure functions: data → DOM
 * Following the principle: UI = render(data)
 */

import { CONSTANTS, DEFAULT_SYSTEM_PROMPT, state } from './state.js';

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
    <div class="tab-container ${containerClass}" id="${containerId}">
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
    <div class="tab-container ${containerClass}" id="${containerId}">
        ${initialContent}
    </div>
`;

/**
 * Create chat tab with message container and input
 * @param {Object} config - Configuration {chatContainerId, inputId, inputRows}
 * @returns {string} HTML string
 */
export const createChatTabContent = (config) => `
    <div class="conversation-toolbar">
        <button class="tool-toggle active" data-tool="read" data-toggleable>Read</button>
        <button class="tool-toggle" data-tool="evaluate" data-toggleable>Evaluate</button>
        <button class="tool-toggle" data-tool="print">Print</button>
        <button class="tool-toggle" data-tool="loop">Loop</button>
        <span class="print-energy-badge" id="print-energy-badge" hidden title="Live engagement energy (real-user Print rewards)"></span>
    </div>
    <div class="kb-results" id="kb-results">
        <!-- KB search hits rendered dynamically in Read mode -->
    </div>
    <div class="chat-container" id="${config.chatContainerId}">
        <!-- Messages rendered dynamically in Evaluate mode -->
    </div>
    <div class="input-container">
        <textarea class="message-input" id="${config.inputId}" rows="${config.inputRows}"></textarea>
    </div>
`;

/**
 * Create iframe tab content
 * Generic, reusable template for rendering iframes scaled to container
 * @param {Object} config - Configuration {url, containerClass, containerId}
 * @returns {string} HTML string
 */
export const createIframeTabContent = (config) => `
    <div class="tab-container ${config.containerClass}" id="${config.containerId}">
        <iframe src="${escapeHtml(config.url)}" class="iframe-content" title="${escapeHtml(config.title || 'Content')}"></iframe>
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
    // Donations URL injected by the backend (--donations) via a meta tag; empty hides it.
    const donations = (document.querySelector('meta[name="praxis-donations"]')?.content || '').trim();
    const donationButton = donations ? `
                <a class="donation-button" href="${escapeHtml(donations)}" target="_blank" rel="noopener noreferrer" title="Support Praxis" aria-label="Support Praxis">
                    <!-- The rallying flag: a call to action, not a coin.
                         The whole glyph leans right off its pole base. -->
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
                        <g transform="rotate(14 8 8) translate(1.2 0.2) scale(1.04)">
                            <path d="M14.778.085A.5.5 0 0 1 15 .5V8a.5.5 0 0 1-.314.464L14.5 8l.186.464-.003.001-.006.003-.023.009a12 12 0 0 1-.397.15c-.264.095-.631.223-1.047.35-.816.252-1.879.523-2.71.523-.847 0-1.548-.28-2.158-.525l-.028-.01C7.68 8.71 7.14 8.5 6.5 8.5c-.7 0-1.638.23-2.437.477A20 20 0 0 0 3 9.342V15.5a.5.5 0 0 1-1 0V.5a.5.5 0 0 1 1 0v.282c.226-.079.496-.17.79-.26C4.606.272 5.67 0 6.5 0c.84 0 1.524.277 2.121.519l.043.018C9.286.788 9.828 1 10.5 1c.7 0 1.638-.23 2.437-.477a20 20 0 0 0 1.349-.476l.019-.007.004-.002h.001"/>
                        </g>
                    </svg>
                </a>` : '';
    return `
        <header class="header">
            <div class="logo">
                <div class="prism-logo">
                    <canvas id="prism-canvas"></canvas>
                </div>
                <span class="logo-separator">|</span>
                <span class="system-prompt-header${state.settings.systemPrompt === DEFAULT_SYSTEM_PROMPT ? ' default-prompt' : ''}" id="developer-prompt" contenteditable="true" spellcheck="false">${escapeHtml(state.settings.systemPrompt)}</span>
            </div>
            <div class="header-actions">
                <div class="notification-wrapper">
                    <button class="notification-button" id="notifications-btn" title="Notifications" aria-label="Notifications">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
                            <path d="M8 16a2 2 0 0 0 2-2H6a2 2 0 0 0 2 2zm.995-14.901a1 1 0 1 0-1.99 0A5.002 5.002 0 0 0 3 6c0 1.098-.5 6-2 7h14c-1.5-1-2-5.902-2-7 0-2.42-1.72-4.44-4.005-4.901z"/>
                        </svg>
                        <span class="notification-badge" id="notification-badge" hidden></span>
                    </button>
                    <div class="notification-panel" id="notification-panel"></div>
                </div>
                ${donationButton}
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
                <span class="status-indicator" id="status-indicator"></span>
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
export function createMessage({ role, content, caption, jokeScore, score = 0 }, isDarkMode, isLast = false) {
    const headerText = isDarkMode
        ? (role === 'user' ? 'Me' : 'You')
        : (role === 'user' ? 'You' : 'Me');

    // Reroll button on the last assistant turn - re-rolls just this section.
    const rerollButton = (role === 'assistant' && isLast)
        ? '<button class="reroll-button" id="reroll-button">🔄 Reroll</button>'
        : '';

    // Optional muted footnote (e.g. the Print reward shown alongside an answer).
    const captionHtml = caption
        ? `<div class="message-caption">${escapeHtml(caption)}</div>`
        : '';

    // Loop mode: a want->need score slider (the human signal). A continuous -1..1
    // judgement of how much we need what the model produced, not a binary vote.
    // The want/need framing persists across all loop modes; any self-predicted
    // score stays backend-side (calibration compares against it on submit).
    const scoreHtml = jokeScore
        ? `<div class="joke-score">
               <span class="joke-score-end">want</span>
               <input type="range" class="joke-slider" min="-1" max="1" step="0.05" value="${score}"
                      aria-label="Score from want to need">
               <span class="joke-score-end">need</span>
               <span class="joke-score-val">${Number(score).toFixed(2)}</span>
           </div>`
        : '';

    return `
        <div class="message ${role}">
            ${rerollButton}
            <div class="message-header">${headerText}</div>
            <div class="message-content">${escapeHtml(content)}</div>
            ${captionHtml}
            ${scoreHtml}
        </div>
    `;
}

/**
 * Create a single KB search result. Snippet highlight markers (\x01/\x02,
 * emitted by FTS5 snippet()) are converted to <mark> after escaping.
 * @param {Object} hit - {type, title, uri, origin, summary, snippet}
 * @returns {string} HTML string
 */
export function createKbResult({ id, type, label, title, uri, origin, summary, snippet, meta }) {
    const highlighted = escapeHtml(snippet || summary || '')
        .replaceAll('\x01', '<mark>')
        .replaceAll('\x02', '</mark>');
    const from = origin ? `<span class="kb-result-origin">${escapeHtml(origin)}</span>` : '';
    const video = (uri || '').match(/youtube\.com\/watch\?v=([\w-]{11})/);
    // Thumbnail: a stored preview image (og:image, crawled by the spider), or
    // one derived from the video id for watch URLs that predate image capture.
    const image = (meta && meta.image) || (video ? `https://i.ytimg.com/vi/${video[1]}/mqdefault.jpg` : '');
    // Pages label themselves with their domain, which already shows by the
    // title (origin) - so the chip uses the generic type instead of repeating it.
    const chipLabel = type === 'page' ? type : (label || type);
    const chip = `<span class="kb-result-type kb-label-filter" data-kb-type="${escapeHtml(type)}"
              data-kb-label="${escapeHtml(chipLabel)}" role="button"
              title="Add to search">${escapeHtml(chipLabel)}</span>`;
    const open = `<div class="kb-result${image ? ' has-thumb' : ''}" role="button" tabindex="0"
             data-kb-id="${escapeHtml(id)}" data-kb-type="${escapeHtml(type)}"
             data-kb-uri="${escapeHtml(uri)}" data-kb-title="${escapeHtml(title)}">`;
    if (image) {
        // Enriched rows: two panels. Media (chip + thumb) left; text right
        // with a clean title and a shortened link on its own line.
        const cleanTitle = title.replace(/\s*-\s*YouTube\s*$/, '');
        let host = '';
        try { host = new URL(uri).hostname.replace(/^www\./, ''); } catch { /* tab routes etc. */ }
        const link = host
            ? `<a class="kb-result-url" href="${escapeHtml(uri)}" target="_blank"
                   rel="noopener noreferrer">[${escapeHtml(host)}]</a>`
            : '';
        return `
        ${open}
            <div class="kb-result-media">
                ${chip}
                <img class="kb-result-thumb" loading="lazy" decoding="async" alt=""
                     src="${escapeHtml(image)}">
            </div>
            <div class="kb-result-text">
                <span class="kb-result-title">${escapeHtml(cleanTitle)}</span>
                ${link}
                <span class="kb-result-snippet">${highlighted}</span>
            </div>
        </div>
        `;
    }
    return `
        ${open}
            ${chip}
            <span class="kb-result-title">${escapeHtml(title)}${from}</span>
            <span class="kb-result-snippet">${highlighted}</span>
        </div>
    `;
}

/**
 * File explorer: the source tree as a vertical fan of directory cards.
 * Directories sort top-to-bottom in path order; collapsed cards overlap so
 * only their header lips show (a fanned deck of folders). Tapping a header
 * pulls that card out of the fan and reveals its file list; tapping a file
 * opens the code card. Toggling is a CSS class flip - no re-render.
 * Given a focus path, the fan shows only that file's lineage: one card per
 * ancestor directory, each listing its files, with the open file marked.
 * @param {Array} results - KB hits, all type "code" (title = repo path)
 * @param {string} [focusPath] - Open file's repo path; scopes the fan
 * @returns {string} HTML string
 */
export function createKbExplorer(results, focusPath) {
    const dirs = new Map();
    for (const r of results) {
        const path = r.title || '';
        const dir = path.includes('/') ? path.slice(0, path.lastIndexOf('/')) : '.';
        if (!dirs.has(dir)) dirs.set(dir, []);
        dirs.get(dir).push(r);
    }
    let keys = [...dirs.keys()].sort();
    if (focusPath) {
        const parts = focusPath.split('/').slice(0, -1);
        const lineage = parts.map((_, i) => parts.slice(0, i + 1).join('/'));
        keys = keys.filter(d => lineage.includes(d));
    }
    const leaf = focusPath ? focusPath.slice(0, focusPath.lastIndexOf('/')) : null;
    const cards = keys.map(dir => {
        const files = dirs.get(dir)
            .sort((a, b) => a.title.localeCompare(b.title))
            .map(f => `
                <div class="kb-file${f.title === focusPath ? ' current' : ''}"
                     role="button" tabindex="0"
                     data-kb-id="${escapeHtml(f.id)}" data-kb-type="code"
                     data-kb-uri="${escapeHtml(f.uri)}" data-kb-title="${escapeHtml(f.title)}">
                    ${escapeHtml(f.title.split('/').pop())}
                </div>`).join('');
        // Rank counts back from the front (deepest) card: 0 = front, parents
        // recede behind it like the chart-deck fan.
        const rank = keys.length - 1 - keys.indexOf(dir);
        const open = dir === leaf ? ' open' : '';
        return `
        <div class="kb-dir-card${open}" style="--dir-rank: ${rank}">
            <div class="kb-dir-header" role="button" tabindex="0">
                <span class="kb-dir-name">${escapeHtml(dir)}/</span>
                <span class="kb-dir-count">${dirs.get(dir).length}</span>
            </div>
            <div class="kb-dir-files">${files}</div>
        </div>`;
    }).join('');
    return `<div class="kb-explorer">${cards}</div>`;
}

/**
 * Full-height card showing one KB entry's content, rendered from data.
 * @param {Object} item - {type, title, uri, body, meta}
 * @param {string} bodyHtml - Pre-rendered body HTML (markdown/json)
 * @returns {string} HTML string
 */
export function createKbCard(item, bodyHtml) {
    const source = item.origin ? ` · ${escapeHtml(item.origin)}` : '';
    // Crawled pages read inline but keep a path back to the live original.
    const external = /^https?:\/\//.test(item.uri || '')
        ? `<a class="kb-card-external" href="${escapeHtml(item.uri)}" target="_blank"
              rel="noopener" aria-label="Open original">↗</a>`
        : '';
    // Crawled video pages get an inline player above the extracted text.
    const video = (item.uri || '').match(/youtube\.com\/watch\?v=([\w-]{11})/);
    const embed = video
        ? `<iframe class="kb-card-embed" src="https://www.youtube-nocookie.com/embed/${video[1]}"
              loading="lazy" allowfullscreen
              allow="encrypted-media; picture-in-picture"
              referrerpolicy="strict-origin-when-cross-origin"></iframe>`
        : '';
    return `
        <div class="kb-card">
            <div class="kb-card-header">
                <button class="kb-card-back" aria-label="Back to results">← Back</button>
                <span class="kb-result-type" data-kb-type="${escapeHtml(item.type)}">${escapeHtml(item.label || item.type)}</span>
                <span class="kb-card-title">${escapeHtml(item.title)}${source}</span>
                ${external}
            </div>
            ${embed}
            <div class="kb-card-body" data-kb-type="${escapeHtml(item.type)}">${bodyHtml}</div>
        </div>
    `;
}

/**
 * Create thinking indicator
 * @param {boolean} isDarkMode - Current theme
 * @returns {string} HTML string
 */
export function createThinkingIndicator(isDarkMode) {
    // In dark mode: assistant is "You", so "You are thinking"
    // In light mode: assistant is "Me", so "Me—andering" (wordplay with em dash)
    const thinkingText = isDarkMode ? 'You are thinking' : 'Me—andering';

    return `
        <div class="message assistant" id="thinking-message">
            <div class="message-header">
                <span class="thinking-status">
                    ${thinkingText}<span class="dots">
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
    // Resolve label - can be string or function(theme)
    const resolvedLabel = typeof label === 'function' ? label(state.theme) : label;

    return `
        <button
            class="tab-button ${active ? 'active' : ''}"
            data-tab="${id}"
        >
            ${resolvedLabel}
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
 * Create tab header - pure functional component
 * Reusable header with title, optional buttons, and optional metadata
 * @param {Object} config - Header configuration
 * @param {string} config.title - Main title text
 * @param {Array} config.buttons - Array of button configs [{id, label, icon, action, dataAttrs, className}]
 * @param {string} config.metadata - Optional metadata/subtitle below title
 * @param {Object} config.additionalContent - Optional HTML content to inject into controls
 * @returns {string} HTML string
 */
export function createTabHeader(config) {
    if (!config || !config.title) return '';

    // Build buttons HTML
    const buttonsHTML = (config.buttons || []).map(btn => {
        const iconHTML = btn.icon || '';
        const className = btn.className || 'tab-header-button';
        const actionAttr = btn.action ? `data-action="${btn.action}"` : '';
        const customAttrs = btn.dataAttrs || '';

        return `
            <button class="${className}" id="${btn.id || ''}" ${actionAttr} ${customAttrs}>
                ${iconHTML}
                ${btn.label || ''}
            </button>
        `;
    }).join('');

    // Combine additional content with buttons
    const controlsContent = (config.additionalContent || '') + buttonsHTML;
    const controlsHTML = controlsContent ? `<div class="tab-header-controls">${controlsContent}</div>` : '';

    // Build metadata HTML (subtitle below title)
    const metadataHTML = config.metadata ? `<div class="tab-header-metadata">${config.metadata}</div>` : '';

    return `
        <div class="tab-header">
            <div class="tab-header-title-row">
                <h2>${escapeHtml(config.title)}</h2>
                ${controlsHTML}
            </div>
            ${metadataHTML}
        </div>
    `;
}

// Document glyph (folded corner) for the "Download PDF" button. ``currentColor``
// so it inherits the active theme like every other tab-header button.
export const PDF_ICON = `
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
        <path d="M9.5 0H4a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2V4.5L9.5 0zM9 1.5 12.5 5H9.5a.5.5 0 0 1-.5-.5V1.5z"/>
    </svg>
`;

/** Tab-header button config that downloads the living research paper PDF. */
export function pdfButton(id) {
    return {
        id,
        label: 'PDF',
        icon: PDF_ICON,
        className: 'tab-header-button download-pdf-btn',
    };
}

