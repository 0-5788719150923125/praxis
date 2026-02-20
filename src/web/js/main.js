/**
 * Praxis Web - Main Entry Point
 * Wire up events, initialize app
 * Architecture: Events → Update State → Render
 */

import { state, CONSTANTS } from './state.js';
import { render, renderAppStructure, updateInputContainerStyling } from './render.js';
import { sendMessage, testApiConnection } from './api.js';
import { connectMetricsLive, setupLiveReload, renderCurrentMetrics } from './websocket.js';
import { loadSpec, loadAgents, loadResearchMetrics } from './tabs.js';
import { setupTabCarousel } from './mobile.js';
import { storage, FORM_FIELDS, readFormValues, updateRangeDisplay } from './config.js';
import { CLICK_HANDLERS, delegateClick } from './events.js';
import { executeAction } from './actions.js';
import './prism.js';

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

    // Setup window resize handler for dashboard scaling
    setupWindowResizeHandler();

    // Initialize input placeholder
    showPlaceholder();

    console.log('[Praxis] Initialized');
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
        systemPrompt.addEventListener('blur', () => {
            state.settings.systemPrompt = systemPrompt.textContent;
            saveSettings();
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
        await sendUserMessage();
        return;
    }

    // Prevent deleting prefix
    if ((e.key === 'Backspace' || e.key === 'Delete') && input.selectionStart <= CONSTANTS.PREFIX.length) {
        e.preventDefault();
        return;
    }

    // Prevent cursor from moving before prefix
    if (e.key === 'ArrowLeft' && input.selectionStart <= CONSTANTS.PREFIX.length) {
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
        const newChars = currentValue.replace(CONSTANTS.PREFIX + CONSTANTS.PLACEHOLDER_TEXT, '').replace(CONSTANTS.PREFIX, '');
        hidePlaceholder();
        if (newChars) {
            input.value = CONSTANTS.PREFIX + newChars;
            input.setSelectionRange(input.value.length, input.value.length);
        }
        return;
    }

    // Maintain prefix
    if (!currentValue.startsWith(CONSTANTS.PREFIX)) {
        input.value = CONSTANTS.PREFIX + currentValue;
        setCursorAfterPrefix();
    }

    // Auto-resize
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 200) + 'px';
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
    if (input.value === CONSTANTS.PREFIX || input.value === '') {
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

    // Prevent cursor before prefix
    if (input.selectionStart < CONSTANTS.PREFIX.length) {
        setCursorAfterPrefix();
    }
}

/**
 * Set cursor position after prefix
 */
function setCursorAfterPrefix() {
    const input = document.getElementById('message-input');
    if (input) {
        input.setSelectionRange(CONSTANTS.PREFIX.length, CONSTANTS.PREFIX.length);
    }
}

/**
 * Show placeholder in input
 */
function showPlaceholder() {
    const input = document.getElementById('message-input');
    if (!input) return;

    input.value = CONSTANTS.PREFIX + CONSTANTS.PLACEHOLDER_TEXT;
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
        input.value = CONSTANTS.PREFIX;
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
    const content = fullValue.startsWith(CONSTANTS.PREFIX)
        ? fullValue.slice(CONSTANTS.PREFIX.length).trim()
        : fullValue.trim();

    if (!content) return;

    // Add user message to state
    state.messages.push({ role: 'user', content });

    // Clear input without showing placeholder (let blur handler do that if needed)
    input.value = CONSTANTS.PREFIX;
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
