/**
 * Praxis Web - Event Handler Registry
 * Declarative event handler configuration - maps selectors to actions
 */

/**
 * Click handler registry
 * Each entry maps a selector to an action type
 */
export const CLICK_HANDLERS = [
    {
        selector: '#reroll-button',
        match: 'exact',
        action: () => ({ type: 'REROLL' })
    },
    {
        selector: '#theme-toggle',
        match: 'closest',
        action: () => ({ type: 'TOGGLE_THEME' })
    },
    {
        selector: '.tab-button',
        match: 'exact',
        action: (e) => ({
            type: 'SWITCH_TAB',
            payload: e.target.dataset.tab
        })
    },
    {
        selector: '#settings-button',
        match: 'closest',
        action: () => ({ type: 'OPEN_SETTINGS_MODAL' })
    },
    {
        selector: '#close-modal',
        match: 'closest',
        action: () => ({ type: 'CLOSE_MODAL' })
    },
    {
        selector: '.settings-modal',
        match: 'exact',
        action: (e) => {
            // Only close if clicking backdrop (not modal content)
            if (e.target.closest('.modal-content')) return null;
            return { type: 'CLOSE_MODAL' };
        }
    },
    {
        selector: '#save-settings',
        match: 'exact',
        action: () => ({ type: 'SAVE_SETTINGS' })
    },
    {
        selector: '#reset-settings',
        match: 'exact',
        action: () => ({ type: 'RESET_SETTINGS' })
    },
    {
        selector: '#agent-selector-btn',
        match: 'closest',
        action: () => ({ type: 'TOGGLE_AGENT_SELECTOR' })
    },
    {
        selector: '.run-selector-item input[type="checkbox"]',
        match: 'closest',
        action: (e) => ({
            type: 'TOGGLE_AGENT',
            payload: e.target.dataset.agentName
        })
    },
    {
        selector: '.copy-git-remote-btn',
        match: 'exact',
        action: (e) => ({
            type: 'COPY_TO_CLIPBOARD',
            payload: e.target.dataset.command,
            meta: { button: e.target }
        })
    },
    {
        selector: '#refresh-metrics-btn',
        match: 'closest',
        action: () => ({ type: 'REFRESH_TAB_DATA' })
    },
    {
        selector: '#refresh-dynamics-btn',
        match: 'closest',
        action: () => ({ type: 'REFRESH_TAB_DATA' })
    },
    {
        selector: '.refresh-button',
        match: 'exact',
        action: (e) => {
            // Don't handle if it's a copy button (already handled above)
            if (e.target.classList.contains('copy-git-remote-btn')) return null;
            return { type: 'REFRESH_TAB_DATA' };
        }
    }
];

/**
 * Pure event delegation function
 * @param {Event} e - Click event
 * @param {Array} handlers - Array of handler configurations
 * @returns {Object|null} Action object or null if no match
 */
export const delegateClick = (e, handlers) => {
    for (const handler of handlers) {
        let element = null;

        if (handler.match === 'exact') {
            element = e.target.matches(handler.selector) ? e.target : null;
        } else if (handler.match === 'closest') {
            element = e.target.closest(handler.selector);
        }

        if (element) {
            const action = handler.action(e);
            if (action) return action;
        }
    }
    return null;
};
