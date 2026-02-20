/**
 * Praxis Web - Action Handlers
 * Pure action handler functions - business logic separated from event handling
 */

import { state } from './state.js';
import { render } from './render.js';
import { storage, readFormValues, FORM_FIELDS } from './config.js';
import { toggleAgentSelector, toggleAgentSelection } from './charts.js';
import { loadResearchMetrics, loadDynamics } from './tabs.js';
import { sendMessage, testApiConnection } from './api.js';

/**
 * Get lifecycle function by name
 * @param {string} name - Function name
 * @returns {Promise<Function|null>}
 */
const getLifecycleFunction = async (name) => {
    // Import dynamically to avoid circular dependencies
    const { renderCurrentMetrics } = await import('./websocket.js');
    const { loadSpec, loadAgents } = await import('./tabs.js');

    const lifecycleFunctions = {
        renderCurrentMetrics,
        loadSpec,
        loadAgents,
        loadResearchMetrics,
        loadDynamics
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
     * 10% chance to trigger hybrid "split-reality" mode
     */
    TOGGLE_THEME: () => {
        const oldTheme = state.theme;
        const oldHybrid = state.isHybridMode;

        // 10% chance to enter hybrid mode (only when switching to dark)
        const willEnterHybrid = Math.random() < 0.1;

        if (state.theme === 'light' && willEnterHybrid) {
            // Entering hybrid mode - dark base with light split
            state.theme = 'dark';
            state.isHybridMode = true;
            console.log('[Theme] ⚡ HYBRID MODE ACTIVATED ⚡');
        } else if (state.isHybridMode) {
            // Exit hybrid mode
            state.isHybridMode = false;
            state.theme = state.theme === 'light' ? 'dark' : 'light';
        } else {
            // Normal theme toggle
            state.theme = state.theme === 'light' ? 'dark' : 'light';
        }

        storage.set('theme', state.theme);
        render();

        // Start/stop hybrid rendering if needed
        if (state.isHybridMode) {
            import('./hybrid.js')
                .then(({ startHybridMode }) => {
                    startHybridMode();
                })
                .catch((err) => {
                    console.error('[Theme] Failed to load hybrid.js:', err);
                });
        } else {
            import('./hybrid.js')
                .then(({ stopHybridMode }) => {
                    stopHybridMode();
                })
                .catch(() => {}); // Ignore if module not loaded yet
        }

        // Data-driven: check if current tab has charts via feature flag
        const currentTab = state.tabs.find(t => t.active);
        if (currentTab?.hasCharts) {
            // Dynamically update chart colors without redrawing
            import('./charts.js').then(({ updateChartColors }) => {
                updateChartColors();
            });
            // Also update dynamics charts if on Dynamics tab
            if (currentTab.id === 'dynamics') {
                import('./dynamics.js').then(({ updateDynamicsChartColors }) => {
                    updateDynamicsChartColors();
                });
            }
        }
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
        let copySuccess = false;

        // Try modern clipboard API first
        try {
            await navigator.clipboard.writeText(text);
            copySuccess = true;
        } catch (err) {
            console.warn('[Clipboard] Modern API failed, trying fallback:', err);

            // Fallback: use legacy execCommand method
            try {
                const textArea = document.createElement('textarea');
                textArea.value = text;
                textArea.style.cssText = 'position:fixed;top:-999px;left:-999px;opacity:0;';
                document.body.appendChild(textArea);
                textArea.focus();
                textArea.select();

                copySuccess = document.execCommand('copy');
                document.body.removeChild(textArea);
            } catch (execErr) {
                console.error('[Clipboard] Fallback also failed:', execErr);
            }
        }

        if (copySuccess) {
            // Create notification with viewport-aware positioning
            const notification = document.createElement('div');
            notification.textContent = 'Copied git remote to clipboard.';
            notification.className = 'copy-notification';

            // Get button position
            const buttonRect = meta.button.getBoundingClientRect();
            const viewportWidth = window.innerWidth;
            const viewportHeight = window.innerHeight;

            // Notification dimensions (estimate before rendering)
            const notificationWidth = 240; // approximate width
            const notificationHeight = 40; // approximate height

            // Calculate position: try to place near button
            let top = buttonRect.top + (buttonRect.height / 2) - (notificationHeight / 2);
            let left = buttonRect.left - notificationWidth - 16; // 16px gap

            // Adjust if would overflow left
            if (left < 16) {
                left = buttonRect.right + 16; // Place on right instead
            }

            // Adjust if would overflow right
            if (left + notificationWidth > viewportWidth - 16) {
                left = viewportWidth - notificationWidth - 16;
            }

            // Adjust if would overflow top
            if (top < 16) {
                top = 16;
            }

            // Adjust if would overflow bottom
            if (top + notificationHeight > viewportHeight - 16) {
                top = viewportHeight - notificationHeight - 16;
            }

            // Apply positioning
            notification.style.cssText = `
                position: fixed;
                top: ${top}px;
                left: ${left}px;
                padding: 0.5rem 1rem;
                background: #0B9A6D;
                color: white;
                border-radius: 4px;
                font-size: 14px;
                white-space: nowrap;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
                z-index: 1000;
                animation: fadeIn 0.2s ease-in;
            `;

            // Append to body (ignores all container boundaries)
            document.body.appendChild(notification);

            // Remove after 2 seconds
            setTimeout(() => {
                notification.style.animation = 'fadeOut 0.2s ease-out';
                setTimeout(() => notification.remove(), 200);
            }, 2000);
        } else {
            // Both methods failed - show modal with selectable text
            const modal = document.createElement('div');
            modal.className = 'copy-fallback-modal';
            modal.innerHTML = `
                <div class="copy-fallback-content">
                    <h3>Copy Manually</h3>
                    <p style="margin: 0.5rem 0; font-size: 0.9em; opacity: 0.8;">
                        Automatic clipboard access is not available. Please copy the text below:
                    </p>
                    <input type="text" readonly value="${text.replace(/"/g, '&quot;')}"
                           class="copy-fallback-input" id="copy-fallback-input">
                    <button class="copy-fallback-close" id="copy-fallback-close">Close</button>
                </div>
            `;

            modal.style.cssText = `
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0, 0, 0, 0.5);
                display: flex;
                align-items: center;
                justify-content: center;
                z-index: 10000;
                padding: 1rem;
            `;

            document.body.appendChild(modal);

            // Auto-select the text
            setTimeout(() => {
                const input = document.getElementById('copy-fallback-input');
                if (input) {
                    input.focus();
                    input.select();
                }
            }, 100);

            // Close handler
            const closeModal = () => modal.remove();
            document.getElementById('copy-fallback-close').addEventListener('click', closeModal);
            modal.addEventListener('click', (e) => {
                if (e.target === modal) closeModal();
            });
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
