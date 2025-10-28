/**
 * Praxis Web - Action Handlers
 * Pure action handler functions - business logic separated from event handling
 */

import { state } from './state.js';
import { render } from './render.js';
import { storage, readFormValues, FORM_FIELDS } from './config.js';
import { toggleAgentSelector, toggleAgentSelection } from './charts.js';
import { loadResearchMetrics } from './tabs.js';
import { sendMessage, testApiConnection } from './api.js';

/**
 * Get lifecycle function by name
 * @param {string} name - Function name
 * @returns {Promise<Function|null>}
 */
const getLifecycleFunction = async (name) => {
    // Import dynamically to avoid circular dependencies
    const { recalculateDashboardScale } = await import('./websocket.js');
    const { loadSpec, loadAgents } = await import('./tabs.js');

    const lifecycleFunctions = {
        recalculateDashboardScale,
        loadSpec,
        loadAgents,
        loadResearchMetrics
    };

    return lifecycleFunctions[name] || null;
};

/**
 * Call lifecycle hook with configuration
 * @param {string|Function} hook - Hook function or name
 * @param {Object} config - Tab configuration with delay and params
 */
const callLifecycleHook = async (hook, config = {}) => {
    if (!hook) return;

    const fn = typeof hook === 'string'
        ? await getLifecycleFunction(hook)
        : hook;

    if (!fn) return;

    const params = config.activateParams || [];
    const delay = config.activateDelay || 0;

    if (delay > 0) {
        setTimeout(() => fn(...params), delay);
    } else {
        fn(...params);
    }
};

/**
 * Action handler registry
 * Maps action types to handler functions
 */
export const ACTION_HANDLERS = {
    /**
     * Handle reroll button click - resend last user message
     */
    REROLL: async () => {
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
    },

    /**
     * Toggle theme between light and dark
     */
    TOGGLE_THEME: () => {
        state.theme = state.theme === 'light' ? 'dark' : 'light';
        storage.set('theme', state.theme);
        render();
    },

    /**
     * Switch active tab
     * @param {string} tabId - ID of tab to switch to
     */
    SWITCH_TAB: async (tabId) => {
        const oldTab = state.tabs.find(t => t.active);
        const newTab = state.tabs.find(t => t.id === tabId);

        if (!newTab || newTab.id === oldTab?.id) return;

        // Call deactivation hook
        if (oldTab?.onDeactivate) {
            await callLifecycleHook(oldTab.onDeactivate, oldTab);
        }

        // Update state immutably
        state.currentTab = tabId;
        state.tabs = state.tabs.map(t => ({
            ...t,
            active: t.id === tabId
        }));

        render();

        // Call activation hook
        if (newTab.onActivate) {
            await callLifecycleHook(newTab.onActivate, newTab);
        }
    },

    /**
     * Open settings modal
     */
    OPEN_SETTINGS_MODAL: () => {
        state.modals.settingsOpen = true;
        render();
    },

    /**
     * Close modal
     */
    CLOSE_MODAL: () => {
        state.modals.settingsOpen = false;
        render();
    },

    /**
     * Save settings from modal with API validation
     */
    SAVE_SETTINGS: async () => {
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

        // Read form values
        const updates = readFormValues(FORM_FIELDS.settings);
        Object.assign(state, updates);

        // Save to localStorage
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
    },

    /**
     * Reset all settings
     */
    RESET_SETTINGS: () => {
        storage.clear();

        // Show confirmation
        const confirmationDiv = document.getElementById('save-confirmation');
        if (confirmationDiv) {
            confirmationDiv.textContent = 'All settings cleared! Refreshing...';
            confirmationDiv.style.display = 'block';
        }

        setTimeout(() => window.location.reload(), 500);
    },

    /**
     * Toggle agent selector dropdown
     */
    TOGGLE_AGENT_SELECTOR: () => {
        toggleAgentSelector();
    },

    /**
     * Toggle agent selection checkbox
     * @param {string} agentName - Name of agent to toggle
     */
    TOGGLE_AGENT: (agentName) => {
        toggleAgentSelection(agentName);
    },

    /**
     * Copy text to clipboard and show notification
     * @param {string} text - Text to copy
     * @param {Object} meta - Metadata including button element
     */
    COPY_TO_CLIPBOARD: async (text, meta) => {
        try {
            await navigator.clipboard.writeText(text);

            // Create notification positioned absolutely
            const notification = document.createElement('div');
            notification.textContent = 'Copied git remote to clipboard.';
            notification.style.cssText = `
                position: absolute;
                top: 0;
                right: 100%;
                margin-right: 1rem;
                padding: 0.5rem 1rem;
                background: var(--success-bg, #0B9A6D);
                color: white;
                border-radius: 4px;
                font-size: 14px;
                white-space: nowrap;
                animation: fadeIn 0.2s ease-in;
            `;

            // Insert notification into parent container
            meta.button.parentNode.appendChild(notification);

            // Remove after 2 seconds
            setTimeout(() => {
                notification.style.animation = 'fadeOut 0.2s ease-out';
                setTimeout(() => notification.remove(), 200);
            }, 2000);
        } catch (err) {
            console.error('[Clipboard] Failed to copy:', err);
            alert('Failed to copy to clipboard. Please copy manually.');
        }
    },

    /**
     * Refresh current tab data
     */
    REFRESH_TAB_DATA: async () => {
        const currentTab = state.tabs.find(t => t.active);
        if (currentTab?.onActivate) {
            await callLifecycleHook(currentTab.onActivate, {
                ...currentTab,
                activateParams: [true]  // Force refresh
            });
        }
    }
};

/**
 * Execute action by type
 * @param {string} type - Action type
 * @param {any} payload - Action payload
 * @param {any} meta - Action metadata
 */
export const executeAction = async (type, payload, meta) => {
    const handler = ACTION_HANDLERS[type];
    if (!handler) {
        console.warn(`[Actions] Unknown action type: ${type}`);
        return;
    }

    try {
        await handler(payload, meta);
    } catch (err) {
        console.error(`[Actions] Error executing ${type}:`, err);
    }
};
