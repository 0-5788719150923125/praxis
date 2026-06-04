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
        selector: '#notifications-btn',
        match: 'closest',
        action: () => ({ type: 'TOGGLE_NOTIFICATIONS' })
    },
    {
        selector: '.tool-toggle[data-tool="print"]',
        match: 'closest',
        action: () => ({ type: 'PRESENT_PRINT_QUESTION' })
    },
    {
        selector: '.tool-toggle[data-tool="loop"]',
        match: 'closest',
        action: () => ({ type: 'TOGGLE_LOOP' })
    },
    {
        selector: '.tool-toggle[data-toggleable]',
        match: 'closest',
        action: (e) => ({
            type: 'TOGGLE_TOOL',
            meta: { button: e.target.closest('.tool-toggle') }
        })
    },
    {
        selector: '.kb-card-back',
        match: 'closest',
        action: () => ({ type: 'CLOSE_KB_ITEM' })
    },
    {
        selector: '.kb-result',
        match: 'closest',
        action: (e) => {
            const el = e.target.closest('.kb-result');
            return {
                type: 'OPEN_KB_ITEM',
                payload: {
                    id: el.dataset.kbId,
                    type: el.dataset.kbType,
                    uri: el.dataset.kbUri,
                    title: el.dataset.kbTitle
                }
            };
        }
    },
    {
        selector: '#contracts-toggle',
        match: 'closest',
        action: () => ({ type: 'TOGGLE_CONTRACTS_VIEW' })
    },
    {
        selector: '.contract-agree-btn',
        match: 'closest',
        action: (e) => ({
            type: 'AGREE_CONTRACT',
            payload: e.target.closest('.contract-agree-btn').dataset.contractId
        })
    },
    {
        selector: '.agent-sever',
        match: 'closest',
        action: (e) => ({
            type: 'SEVER_AGENT',
            payload: e.target.closest('.agent-sever').dataset.agentId,
            meta: { button: e.target.closest('.agent-sever') }
        })
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
        selector: '.run-selector-item input[data-run-hash]',
        match: 'closest',
        action: (e) => ({
            type: 'TOGGLE_RUN',
            payload: e.target.dataset.runHash
        })
    },
    {
        selector: '.copy-git-remote-btn',
        match: 'exact',
        action: (e) => ({
            type: 'COPY_TO_CLIPBOARD',
            payload: e.target.dataset.command,
            meta: { button: e.target, label: 'Copied git remote to clipboard.' }
        })
    },
    {
        selector: '.spec-copy-btn',
        match: 'closest',
        action: (e) => {
            const pre = e.target.closest('.chart-card')?.querySelector('pre.spec-code');
            return {
                type: 'COPY_TO_CLIPBOARD',
                payload: pre ? pre.textContent : '',
                meta: { button: e.target.closest('.spec-copy-btn'), label: 'Copied to clipboard.' }
            };
        }
    },
    {
        selector: '#run-selector-btn',
        match: 'closest',
        action: () => ({ type: 'TOGGLE_RUN_SELECTOR' })
    },
    {
        selector: '.run-selector-item input[data-dynamics-run-hash]',
        match: 'closest',
        action: (e) => ({
            type: 'SELECT_DYNAMICS_RUN',
            payload: e.target.dataset.dynamicsRunHash
        })
    },
    {
        selector: '#dynamics-run-selector-btn',
        match: 'closest',
        action: () => ({ type: 'TOGGLE_DYNAMICS_RUN_SELECTOR' })
    },
    {
        selector: '.run-selector-item input[data-spec-run-hash]',
        match: 'closest',
        action: (e) => ({
            type: 'SELECT_SPEC_RUN',
            payload: e.target.dataset.specRunHash
        })
    },
    {
        selector: '#spec-run-selector-btn',
        match: 'closest',
        action: () => ({ type: 'TOGGLE_SPEC_RUN_SELECTOR' })
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
        selector: '.download-pdf-btn',
        match: 'closest',
        action: () => ({ type: 'DOWNLOAD_PAPER_PDF' })
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
