/**
 * Praxis Web - Main Entry Point
 * Wire up events, initialize app
 * Architecture: Events → Update State → Render
 */

import { state, CONSTANTS, DEFAULT_SYSTEM_PROMPT } from './state.js';
import { render, renderAppStructure, updateInputContainerStyling } from './render.js';
import { sendMessage, kbSearch, testApiConnection } from './api.js';
import { connectMetricsLive, setupLiveReload, renderCurrentMetrics } from './websocket.js';
import { loadSpec, loadAgents, loadResearchMetrics } from './tabs.js';
import { setupTabCarousel, setupTabSwipe } from './mobile.js';
import { storage, FORM_FIELDS, readFormValues, updateRangeDisplay } from './config.js';
import { CLICK_HANDLERS, delegateClick } from './events.js';
import { executeAction } from './actions.js';
import { setupAccentRetint } from './charts.js';
import './prism.js';

// Mode-aware input cues: Read invites a query ("> Look"), Evaluate invites a
// chat turn ("< Shoot"). Both prefixes are 2 chars, so cursor/length logic in
// the input handlers is unaffected.
const prefixForMode = (mode) => (mode === 'read' ? '> ' : '< ');
export const inputPrefix = () => prefixForMode(state.conversationMode);
const inputPlaceholderText = () => (state.conversationMode === 'read' ? 'Look' : 'Shoot');

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

    // Setup window resize handler for dashboard scaling
    setupWindowResizeHandler();

    // Initialize input placeholder
    showPlaceholder();

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
async function prefetchTabs() {
    const { loadResearchMetrics, loadDynamics, loadSpec, loadAgents } = await import('./tabs.js');
    // [contentId, loader, needsLayout] - deck tabs must be laid out off-screen so
    // their charts measure; the Stage fleet is a plain list, so a background load
    // is enough.
    const jobs = [
        ['spec-content', loadSpec, true],
        ['research-content', loadResearchMetrics, true],
        ['dynamics-content', loadDynamics, true],
        ['agents-content', loadAgents, false],
    ];
    setTimeout(async () => {
        for (const [contentId, load, needsLayout] of jobs) {
            try {
                await (needsLayout ? prewarmTab(contentId, load) : load(false));
            } catch (error) {
                console.warn('[Praxis] Tab prewarm failed:', contentId, error);
            }
        }
    }, 600);
}

/**
 * Lay out a hidden tab off-screen at the active region's exact size, run its
 * loader so the deck measures + renders for real, then return it to the hidden
 * state render() manages. Two frames let the deck settle before we hide it.
 */
async function prewarmTab(contentId, load) {
    const el = document.getElementById(contentId);
    const active = document.querySelector('.tab-content.active');
    const rect = active ? active.getBoundingClientRect() : null;
    if (!el || !rect || rect.height < 50) {
        await load(false);  // can't size it; fall back to a plain background load
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
    try {
        await load(false);
        await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
    } finally {
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

    // Settings modal range inputs
    document.addEventListener('input', handleRangeInput);
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
        if (state.conversationMode === 'read') {
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

/** Strip the input prefix to get the raw query text. */
function currentQuery(input) {
    const v = input.value;
    return (v.startsWith(inputPrefix()) ? v.slice(inputPrefix().length) : v).trim();
}

/** Debounced live search; stale responses are dropped via a sequence guard. */
function scheduleKbSearch(query) {
    clearTimeout(kbSearchTimer);
    state.kbOpenItem = null;  // typing a new query returns to the results list
    if (!query) {
        state.kbResults = [];
        state.kbSearching = false;
        render();
        return;
    }
    state.kbSearching = true;
    const seq = ++kbSearchSeq;
    kbSearchTimer = setTimeout(async () => {
        try {
            const hits = await kbSearch(query);
            if (seq !== kbSearchSeq) return; // a newer keystroke superseded us
            state.kbResults = hits;
        } catch (error) {
            if (seq !== kbSearchSeq) return;
            console.error('[KB] Search error:', error);
            state.kbResults = [];
        } finally {
            if (seq === kbSearchSeq) {
                state.kbSearching = false;
                render();
            }
        }
    }, 120);
}

/** Enter in Read mode opens the top-ranked hit (inline, or external for links). */
function openTopKbResult() {
    const top = state.kbResults[0];
    if (top) executeAction('OPEN_KB_ITEM', { id: top.id, type: top.type, uri: top.uri });
}

/**
 * Handle input focus
 */
function handleInputFocus() {
    hidePlaceholder();
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
