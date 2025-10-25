/**
 * Praxis Web - Main Entry Point
 * Wire up events, initialize app
 * Architecture: Events → Update State → Render
 */

import { state, CONSTANTS } from './state.js';
import { render, renderAppStructure, updateInputContainerStyling } from './render.js';
import { sendMessage, testApiConnection } from './api.js';
import { connectTerminal, setupLiveReload, recalculateDashboardScale } from './websocket.js';
import { loadSpec, loadAgents, loadResearchMetrics } from './tabs.js';
import { setupTabCarousel } from './mobile.js';
import { toggleAgentSelector, toggleAgentSelection, loadResearchMetricsWithCharts } from './charts.js';

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

    // Connect to terminal WebSocket
    connectTerminal();

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
            // Recalculate dashboard scaling if on terminal tab
            if (state.currentTab === 'terminal') {
                recalculateDashboardScale();
            }
            // Re-render to update any size-dependent elements
            render();
        }, 250);
    });

    // Handle orientation changes (mobile)
    window.addEventListener('orientationchange', () => {
        setTimeout(() => {
            if (state.currentTab === 'terminal') {
                recalculateDashboardScale();
            }
        }, 100);
    });
}

/**
 * Load settings from localStorage
 */
function loadSettings() {
    // Load theme
    const savedTheme = localStorage.getItem('praxis_theme');
    if (savedTheme) {
        state.theme = savedTheme;
    }

    // Load system prompt
    const savedPrompt = localStorage.getItem('praxis_developer_prompt');
    if (savedPrompt) {
        state.settings.systemPrompt = savedPrompt;
    }

    // Load API URL
    const savedApiUrl = localStorage.getItem('praxis_api_url');
    if (savedApiUrl) {
        // Skip if we're on ngrok (use dynamic URL)
        if (!window.location.hostname.includes('ngrok')) {
            state.settings.apiUrl = savedApiUrl;
        }
    }

    // Load generation params
    const savedParams = localStorage.getItem('praxis_gen_params');
    if (savedParams) {
        try {
            const params = JSON.parse(savedParams);
            Object.assign(state.settings, params);
        } catch (e) {
            console.error('[Settings] Failed to parse saved params:', e);
        }
    }

    // Load debug flag
    const debugLogging = localStorage.getItem('praxis_debug_logging');
    if (debugLogging) {
        state.settings.debugLogging = debugLogging === 'true';
    }
}

/**
 * Save settings to localStorage
 */
function saveSettings() {
    localStorage.setItem('praxis_theme', state.theme);
    localStorage.setItem('praxis_developer_prompt', state.settings.systemPrompt);
    localStorage.setItem('praxis_api_url', state.settings.apiUrl);
    localStorage.setItem('praxis_gen_params', JSON.stringify({
        maxTokens: state.settings.maxTokens,
        temperature: state.settings.temperature,
        repetitionPenalty: state.settings.repetitionPenalty,
        doSample: state.settings.doSample,
        useCache: state.settings.useCache
    }));
    localStorage.setItem('praxis_debug_logging', state.settings.debugLogging.toString());
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
 * Handle all click events (delegation)
 */
async function handleClick(e) {
    // Reroll button
    if (e.target.matches('#reroll-button')) {
        handleReroll();
        return;
    }

    // Theme toggle
    if (e.target.closest('#theme-toggle')) {
        state.theme = state.theme === 'light' ? 'dark' : 'light';
        saveSettings();
        render();
        return;
    }

    // Settings button
    if (e.target.closest('#settings-button')) {
        state.modals.settingsOpen = true;
        render();
        return;
    }

    // Close modal
    if (e.target.closest('#close-modal') || e.target.matches('.settings-modal')) {
        if (e.target.matches('.settings-modal') && !e.target.closest('.modal-content')) {
            state.modals.settingsOpen = false;
            render();
        } else if (e.target.closest('#close-modal')) {
            state.modals.settingsOpen = false;
            render();
        }
        return;
    }

    // Save settings
    if (e.target.matches('#save-settings')) {
        await handleSaveSettings();
        return;
    }

    // Reset settings
    if (e.target.matches('#reset-settings')) {
        if (confirm('Reset all settings to defaults?')) {
            localStorage.clear();
            location.reload();
        }
        return;
    }

    // Tab switching
    if (e.target.matches('.tab-button')) {
        const tabId = e.target.dataset.tab;

        // Update current tab
        state.currentTab = tabId;

        state.tabs = state.tabs.map(t => ({
            ...t,
            active: t.id === tabId
        }));
        render();

        // Lazy-load tab content and handle special cases
        switch (tabId) {
            case 'terminal':
                // Recalculate dashboard scale after tab is visible
                setTimeout(() => {
                    recalculateDashboardScale();
                }, 200);
                break;
            case 'spec':
                loadSpec();
                break;
            case 'agents':
                loadAgents();
                break;
            case 'research':
                loadResearchMetrics();
                break;
        }

        return;
    }

    // Agent selector toggle
    if (e.target.closest('#agent-selector-btn')) {
        toggleAgentSelector();
        return;
    }

    // Refresh metrics button
    if (e.target.closest('#refresh-metrics-btn')) {
        loadResearchMetricsWithCharts(true);
        return;
    }

    // Agent selection checkbox
    if (e.target.matches('.run-selector-item input[type="checkbox"]')) {
        const agentName = e.target.dataset.agentName;
        if (agentName) {
            toggleAgentSelection(agentName);
        }
        return;
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
function handleInputFocus(e) {
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
 * Handle input click (prevent cursor before prefix)
 */
function handleInputClick(e) {
    const input = e.target;
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
 * Handle range input changes (update display)
 */
function handleRangeInput(e) {
    if (e.target.id === 'temperature') {
        const display = document.getElementById('temperature-value');
        if (display) display.textContent = e.target.value;
    }
    if (e.target.id === 'repetition-penalty') {
        const display = document.getElementById('repetition-penalty-value');
        if (display) display.textContent = e.target.value;
    }
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

    // Reset input to placeholder
    showPlaceholder();

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
 * Read settings from modal inputs
 */
function readSettingsFromModal() {
    const apiUrl = document.getElementById('api-url');
    const maxTokens = document.getElementById('max-tokens');
    const temperature = document.getElementById('temperature');
    const repPenalty = document.getElementById('repetition-penalty');
    const doSample = document.getElementById('do-sample');
    const debugLogging = document.getElementById('debug-logging');

    if (apiUrl) state.settings.apiUrl = apiUrl.value;
    if (maxTokens) state.settings.maxTokens = parseInt(maxTokens.value);
    if (temperature) state.settings.temperature = parseFloat(temperature.value);
    if (repPenalty) state.settings.repetitionPenalty = parseFloat(repPenalty.value);
    if (doSample) state.settings.doSample = doSample.checked;
    if (debugLogging) state.settings.debugLogging = debugLogging.checked;
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
