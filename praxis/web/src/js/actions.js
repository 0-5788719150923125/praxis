/**
 * Praxis Web - Action Handlers
 * Pure action handler functions - business logic separated from event handling
 */

import { state } from './state.js';
import { render, renderNotifications } from './render.js';
import { storage, readFormValues, FORM_FIELDS } from './config.js';
import {
    toggleRunSelector, toggleRunSelection, repaintDeckCards,
    requestDeckFocus, applyDeckFocus, isDeckFocusPending, clearDeckFocus,
} from './charts.js';
import { toggleDynamicsRunSelector, selectDynamicsRun } from './dynamics.js';
import { loadResearchMetrics, loadDynamics, toggleSpecRunSelector, selectSpecRun, toggleContractsView, agreeContract, severSwarmAgent } from './tabs.js';
import { sendMessage, kbFetchItem, testApiConnection, loopApprove } from './api.js';
import { kbCacheFetch } from './kbcache.js';
import { renderMarkdown, renderJson } from './markdown.js';
import { revealPrewarmed } from './prefetch.js';
import { syncInputToMode, fetchAndPresentQuestion, startLoop, stopLoop, rerollLoopNow, addKbSearchTerm } from './main.js';

/**
 * Floating confirmation popup anchored to a button - the "copied to
 * clipboard" style used by the git copy buttons. Viewport-aware placement.
 * @param {Element} button - Anchor element
 * @param {string} label - Popup text
 */
function showActionNotification(button, label) {
    const notification = document.createElement('div');
    notification.textContent = label;
    notification.className = 'copy-notification';

    const buttonRect = button.getBoundingClientRect();
    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;

    // Notification dimensions (estimate before rendering)
    const notificationWidth = 240;
    const notificationHeight = 40;

    // Try to place near the button, flipping/clamping at viewport edges.
    let top = buttonRect.top + (buttonRect.height / 2) - (notificationHeight / 2);
    let left = buttonRect.left - notificationWidth - 16; // 16px gap

    if (left < 16) {
        left = buttonRect.right + 16; // Place on right instead
    }
    if (left + notificationWidth > viewportWidth - 16) {
        left = viewportWidth - notificationWidth - 16;
    }
    top = Math.min(Math.max(top, 16), viewportHeight - notificationHeight - 16);

    notification.style.cssText = `
        position: fixed;
        top: ${top}px;
        left: ${left}px;
        padding: 0.5rem 1rem;
        background: var(--accent);
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

    setTimeout(() => {
        notification.style.animation = 'fadeOut 0.2s ease-out';
        setTimeout(() => notification.remove(), 200);
    }, 2000);
}

/**
 * Crop a captured tab frame down to the app-container, bottom-clipped to
 * the last visible card (the app has no footer). The frame spans the
 * viewport, so CSS px are mapped to frame px by the capture scale.
 * @param {HTMLCanvasElement} frame - Full captured frame
 * @returns {HTMLCanvasElement} Cropped canvas (or the frame if cropping fails)
 */
function cropToAppContainer(frame) {
    const app = document.querySelector('.app-container');
    if (!app) return frame;
    const rect = app.getBoundingClientRect();

    const scaleX = frame.width / window.innerWidth;
    const scaleY = frame.height / window.innerHeight;

    let bottom = rect.top;
    for (const el of app.querySelectorAll('.chart-card, .kb-card, .kb-dir-card')) {
        if (!el.offsetParent) continue; // hidden tab
        bottom = Math.max(bottom, el.getBoundingClientRect().bottom);
    }
    if (bottom <= rect.top) bottom = rect.bottom; // no cards: full container
    bottom = Math.min(bottom, rect.bottom, window.innerHeight);

    const x = Math.max(rect.left, 0) * scaleX;
    const y = Math.max(rect.top, 0) * scaleY;
    const w = Math.min(rect.right, window.innerWidth) * scaleX - x;
    const h = bottom * scaleY - y;
    if (w <= 0 || h <= 0) return frame;

    const out = document.createElement('canvas');
    out.width = Math.round(w);
    out.height = Math.round(h);
    out.getContext('2d').drawImage(frame, x, y, w, h, 0, 0, w, h);
    return out;
}

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
    // Click swallowed on purpose (e.g. native <a> navigation inside a row).
    NOOP: () => {},

    /**
     * Handle reroll button click - resend last user message
     */
    REROLL: async () => {
        // In Loop mode, reroll regenerates just this section's output.
        if (state.loop.enabled) {
            rerollLoopNow();
            return;
        }
        // In Print mode, reroll discards the pending question and asks a fresh
        // one (there's no user turn to resend - the model leads).
        if (state.conversationMode === 'print' && state.print.awaitingResponse) {
            const last = state.messages[state.messages.length - 1];
            if (last && last.role === 'assistant') state.messages.pop();
            state.print.awaitingResponse = false;
            state.print.question = null;
            state.print.id = null;
            await fetchAndPresentQuestion(true);   // discard server-side pending too
            return;
        }
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

        // Recolor every live chart, not just the active tab's: cards cached on an
        // inactive chart tab don't rebuild on revisit, so they'd keep stale colors.
        import('./charts.js').then(({ updateChartColors }) => {
            updateChartColors();
        });
        import('./dynamics.js').then(({ updateDynamicsChartColors }) => {
            updateDynamicsChartColors();
        });

        // Force the composited fan cards to re-rasterize so their font colors follow the
        // new theme (a CSS-var change alone leaves cards 2-4 stale). Next frame, after the
        // DOM has settled.
        requestAnimationFrame(repaintDeckCards);
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

        // If a background warm has this tab laid out off-screen, strip those
        // styles NOW: the old content stays visible while the refresh swaps in
        // underneath. Without this the inline styles override .active and the
        // whole container vanishes until the warm settles.
        revealPrewarmed(tabId);

        // Call activation hook
        if (newTab.onActivate) {
            await callLifecycleHook(newTab.onActivate, newTab);
        }

        // Lay out the deck only if it never got a visible measure (prefetch retry
        // lapsed); an already-laid-out deck is left untouched so switching tabs
        // doesn't re-render it (which flashed the cards). Then consume any pending
        // deep-link focus.
        const { relayoutDeckOnActivate } = await import('./charts.js');
        requestAnimationFrame(() => {
            relayoutDeckOnActivate(DECK_BY_TAB[tabId]);
            applyDeckFocus(DECK_BY_TAB[tabId]);
        });

        // First time on a deck tab, play the staggered "unroll" reveal.
        playDeckReveal(tabId);
    },

    /**
     * Toggle the Stage tab between discovered actors and the CONTRACTS view.
     */
    TOGGLE_CONTRACTS_VIEW: () => {
        toggleContractsView();
    },

    /**
     * Agree to a contract: spawn an in-page swarm agent (IDLE for now).
     */
    AGREE_CONTRACT: (contractId) => {
        agreeContract(contractId);
    },

    /**
     * Sever a spawned swarm agent's connection. On touch devices (no hover)
     * this is two-tap: the first tap arms the button (revealing SEVER), the
     * second confirms. On hover-capable devices SEVER is already shown on
     * hover, so a single click confirms.
     */
    SEVER_AGENT: (agentId, meta) => {
        const btn = meta && meta.button;
        const canHover = window.matchMedia && window.matchMedia('(hover: hover)').matches;
        if (btn && !canHover && !btn.classList.contains('armed')) {
            // First tap: arm this button, disarm any others.
            document.querySelectorAll('.agent-sever.armed').forEach(b => b.classList.remove('armed'));
            btn.classList.add('armed');
            return;
        }
        severSwarmAgent(agentId);
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
     * Toggle run selector dropdown
     */
    /**
     * Toggle a conversation tool button on/off (blue when active).
     * Local UI state only - does not touch the theme/accent.
     */
    TOGGLE_TOOL: (_payload, meta) => {
        if (!meta || !meta.button) return;
        const tool = meta.button.dataset.tool;

        // Read/Evaluate/Print are mutually exclusive modes (radio-like): one is
        // always active. render() mirrors state.conversationMode onto the toolbar.
        if (tool === 'read' || tool === 'evaluate') {
            const prevMode = state.conversationMode;
            if (prevMode === tool) return; // already active, keep it
            state.conversationMode = tool;
            state.kbOpenItem = null;  // leaving Read closes any open content card
            stopLoop();  // Loop is coupled to Print; leaving Print ends the loop.
            // NB: a pending Print question is intentionally preserved across mode
            // switches - the model is prompted once and not re-asked until the
            // user answers, so toggling Read<->Print never re-queries the model.
            // Swap the input prefix ("> " <-> "< ") in place so it never doubles up.
            syncInputToMode(prevMode);
            render();
            return;
        }

        meta.button.classList.toggle('active');
    },

    /**
     * Clicking Print enters Print mode (a discrete mode like Read/Evaluate) and
     * queries the model for a self-led question on demand; the model may take a
     * moment to respond. The question is presented as an assistant turn the user
     * answers. A no-op only while a request is in flight or a question is already
     * awaiting an answer.
     */
    PRESENT_PRINT_QUESTION: async () => {
        if (state.isThinking) return;  // a request is already in flight

        // Enter Print mode (discrete): highlights Print, drops Read/Evaluate.
        if (state.conversationMode !== 'print') {
            const prevMode = state.conversationMode;
            state.conversationMode = 'print';
            state.kbOpenItem = null;
            syncInputToMode(prevMode);
            render();
        }

        // Prompt the model only if no question is already waiting (avoids
        // re-querying when toggling Read<->Print); otherwise just refocus.
        await fetchAndPresentQuestion();
    },

    /**
     * Loop is coupled to Print: only enable-able from Print mode. Toggling it on
     * repeats one task (default "joke") on a timer, replacing the response each
     * cycle; toggling off stops the loop.
     */
    TOGGLE_LOOP: () => {
        if (state.conversationMode !== 'print') return;  // press Print first
        if (state.loop.enabled) {
            stopLoop();
            render();
        } else {
            state.print.awaitingResponse = false;  // suspend the Q&A flow
            startLoop();
        }
    },

    /**
     * Score the current looped output on the want->need slider (the live human
     * signal). Records the reward and captions the section with it. Does NOT
     * reroll - re-rolling is a separate, deliberate action.
     */
    SCORE_JOKE: async (score) => {
        const msg = state.messages[state.messages.length - 1];
        if (!msg || msg.role !== 'assistant') return;
        msg.score = score;  // persist so the slider re-renders where it was left
        const s = Number(score).toFixed(2);
        try {
            const r = await loopApprove(score, msg.loopId);
            // Calibration mode appends the correction magnitude (the learning
            // signal); the want<->need framing is constant across modes.
            const corr = (r.correction != null)
                ? ` · off by ${Number(r.correction).toFixed(2)}`
                : '';
            msg.caption = `scored ${s} (want↔need)${corr} · energy ${Number(r.energy).toFixed(3)}`;
        } catch (e) {
            msg.caption = `scored ${s} (want↔need)`;
        }
        render();
    },

    /**
     * Open a KB result inline. Links open externally; doc/note/run content is
     * fetched and rendered as a full-height card from data.
     */
    /**
     * Follow a wiki edge by stem: a relative .md reference inside a card body
     * names another KB node. Notes are indexed per-section (#0 = the top),
     * docs by stem - try both, then fall back to a search on the stem so a
     * dangling edge still lands somewhere useful.
     */
    OPEN_KB_WIKI: async ({ stem }) => {
        if (!stem) return;
        const ids = stem.endsWith('.py')
            ? [`code:${stem}`]
            : [`note:${stem}#0`, `doc:${stem}`];
        for (const id of ids) {
            try {
                const item = await kbCacheFetch(id);
                if (item) {
                    state.kbOpenItem = item;
                    render();
                    return;
                }
            } catch {
                // not this id form; try the next
            }
        }
        await executeAction('ADD_KB_LABEL_FILTER', { label: stem.replace(/_/g, ' ') });
    },

    OPEN_KB_ITEM: async ({ id, type, uri, title }) => {
        if (type === 'link') {
            if (uri) window.open(uri, '_blank', 'noopener');
            return;
        }
        if (type === 'card') {
            // id is "card:<tab>:<key>"; deep-link to that tab's deck and card.
            const tab = (uri || '').split(':')[1];
            const key = (id || '').split(':').slice(2).join(':');
            await navigateToCard(tab, key, title);
            return;
        }
        if (type === 'agent') {
            // Switch to the Stage tab and flag the matching fleet row.
            await navigateToAgent(title);
            return;
        }
        try {
            // Served instantly when the sliding-window prefetch already cached it;
            // otherwise this fetches + hydrates (and caches) on demand.
            const item = await kbCacheFetch(id);
            if (!item) return;
            state.kbOpenItem = item;
            render();
        } catch (error) {
            console.error('[KB] Open error:', error);
        }
    },

    /**
     * Close the open KB card and return to the results list.
     */
    CLOSE_KB_ITEM: () => {
        state.kbOpenItem = null;
        render();
    },

    /**
     * Notification bell. On desktop the pop-out is hover-driven (CSS), so a click does
     * nothing. On touch devices there's no hover, so a tap toggles it open/closed (and
     * opening clears the unread badge).
     */
    TOGGLE_NOTIFICATIONS: () => {
        const isTouch = window.matchMedia('(hover: none)').matches;
        if (!isTouch) return;
        state.notifications.panelOpen = !state.notifications.panelOpen;
        if (state.notifications.panelOpen) state.notifications.unread = 0;
        renderNotifications();
    },

    TOGGLE_RUN_SELECTOR: () => {
        toggleRunSelector();
    },

    /**
     * Toggle historical run selection checkbox
     * @param {string} runHash - Hash of run to toggle
     */
    TOGGLE_RUN: (runHash) => {
        toggleRunSelection(runHash);
    },

    /**
     * Toggle the Dynamics tab run selector dropdown
     */
    TOGGLE_DYNAMICS_RUN_SELECTOR: () => {
        toggleDynamicsRunSelector();
    },

    /**
     * Select a single run for the Dynamics tab (single-select)
     */
    SELECT_DYNAMICS_RUN: (runHash) => {
        selectDynamicsRun(runHash);
    },

    /**
     * Toggle the Identity tab run selector dropdown
     */
    TOGGLE_SPEC_RUN_SELECTOR: () => {
        toggleSpecRunSelector();
    },

    /**
     * Select a single run for the Identity tab (single-select)
     */
    SELECT_SPEC_RUN: (runHash) => {
        selectSpecRun(runHash);
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
            showActionNotification(meta.button, (meta && meta.label) || 'Copied to clipboard.');
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
     * Screenshot the app via the native screen-capture API (current tab),
     * cropped to the app-container, and copy the PNG to the clipboard.
     */
    TAKE_SCREENSHOT: async (_payload, meta) => {
        // Surface unsupported environments instead of failing silently:
        // insecure contexts (plain HTTP off localhost) have no mediaDevices
        // or clipboard at all; mobile browsers lack getDisplayMedia entirely.
        if (!navigator.mediaDevices?.getDisplayMedia) {
            showActionNotification(meta.button, window.isSecureContext
                ? 'Screen capture is not supported on this device.'
                : 'Screenshots need HTTPS (or localhost).');
            return;
        }
        if (!navigator.clipboard?.write || typeof ClipboardItem === 'undefined') {
            showActionNotification(meta.button, window.isSecureContext
                ? 'Image clipboard is not supported in this browser.'
                : 'Screenshots need HTTPS (or localhost).');
            return;
        }

        let stream;
        try {
            stream = await navigator.mediaDevices.getDisplayMedia({
                // Tab capture only: window/screen surfaces include browser
                // chrome (URL bar etc), which we never want in a shot.
                video: { displaySurface: 'browser' },
                preferCurrentTab: true,
                selfBrowserSurface: 'include',
                monitorTypeSurfaces: 'exclude',
                audio: false
            });

            // The picker can't be forced; if the user shared a window or
            // screen anyway, bail rather than capture the URL bar.
            const surface = stream.getVideoTracks()[0].getSettings().displaySurface;
            if (surface && surface !== 'browser') {
                stream.getTracks().forEach(track => track.stop());
                stream = null;
                showActionNotification(meta.button, 'Share this tab (not a window) to screenshot.');
                return;
            }

            const video = document.createElement('video');
            video.srcObject = stream;
            await video.play();
            // The first frames can be blank; wait for a decoded one.
            if (video.requestVideoFrameCallback) {
                await new Promise(r => video.requestVideoFrameCallback(r));
            }

            const frame = document.createElement('canvas');
            frame.width = video.videoWidth;
            frame.height = video.videoHeight;
            frame.getContext('2d').drawImage(video, 0, 0);

            // Stop tracks promptly so the recording indicator turns off.
            stream.getTracks().forEach(track => track.stop());
            stream = null;

            const crop = cropToAppContainer(frame);
            const blob = await new Promise(r => crop.toBlob(r, 'image/png'));

            // Clipboard only - never auto-download on top of a copy.
            try {
                await navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })]);
                showActionNotification(meta.button, 'Copied screenshot to clipboard.');
            } catch (err) {
                console.warn('[Screenshot] Clipboard write failed:', err);
                showActionNotification(meta.button, 'Screenshot copy failed - check clipboard permissions.');
            }
        } catch (error) {
            if (error && error.name === 'NotAllowedError') {
                // User dismissed the share prompt - no popup needed.
                console.info('[Screenshot] Capture canceled.');
            } else {
                console.error('[Screenshot] Capture failed:', error);
                showActionNotification(meta.button, 'Screenshot failed - see console.');
            }
        } finally {
            if (stream) stream.getTracks().forEach(track => track.stop());
        }
    },

    /**
     * Append a clicked KB label to the search query (comma-delimited) and search.
     */
    ADD_KB_LABEL_FILTER: (payload) => {
        addKbSearchTerm(payload?.label || '');
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
    },

    /**
     * Download the living research paper PDF. Fetched as a blob so a 404
     * (not built yet) surfaces as a message instead of downloading an error.
     */
    DOWNLOAD_PAPER_PDF: async () => {
        try {
            const res = await fetch('/api/paper.pdf');
            if (!res.ok) throw new Error(`status ${res.status}`);
            const blob = await res.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'praxis-research.pdf';
            document.body.appendChild(a);
            a.click();
            a.remove();
            URL.revokeObjectURL(url);
        } catch (e) {
            console.warn('[PDF] download failed:', e);
            alert('The research PDF is not available yet. It is rebuilt during '
                + 'training once latexmk is present in the environment.');
        }
    }
};

// Deck element id per tab, for deep-linking a KB card to its dashboard card.
const DECK_BY_TAB = { research: 'chart-deck', dynamics: 'dynamics-deck', spec: 'spec-deck' };

// Tabs whose first-activation "unroll" has already played (once per session).
const revealedDecks = new Set();

/** Play the staggered card reveal the first time a deck tab is opened. */
function playDeckReveal(tabId) {
    if (revealedDecks.has(tabId) || !DECK_BY_TAB[tabId]) return;
    revealedDecks.add(tabId);
    const content = document.getElementById(`${tabId}-content`);
    if (!content) return;
    content.classList.add('deck-reveal');
    setTimeout(() => content.classList.remove('deck-reveal'), 800);
}

/**
 * Deep-link to a dashboard card. Registers an event-driven focus request the
 * deck consumes when it becomes visible or rebuilds (see charts.applyDeckFocus),
 * unifying this with user swiping - both end in the same slideTo. If the deck
 * lacks the card (prefetched before the metric logged), a refresh rebuilds it
 * and the request is reapplied. Still missing => no data yet; notify and clear.
 */
async function navigateToCard(tab, key, title) {
    const deckId = DECK_BY_TAB[tab];
    if (!deckId) return;
    requestDeckFocus(deckId, { key, title });
    await executeAction('SWITCH_TAB', tab);  // activation applies the focus once visible

    if (await waitFocusConsumed(deckId)) return;

    await forceRefreshTab(tab);  // rebuild deck with latest data; init reapplies focus
    if (await waitFocusConsumed(deckId)) return;

    clearDeckFocus();
    notifyMissingCard(title);
}

async function waitFocusConsumed(deckId, tries = 20) {
    for (let i = 0; i < tries; i++) {
        applyDeckFocus(deckId);  // idempotent; consumes once the card is present
        if (!isDeckFocusPending(deckId)) return true;
        await new Promise(r => setTimeout(r, 100));
    }
    return false;
}

async function forceRefreshTab(tab) {
    const t = await import('./tabs.js');
    if (tab === 'research') await t.loadResearchMetrics(true);
    else if (tab === 'dynamics') await t.loadDynamics(true);
    else if (tab === 'spec') await t.loadSpec(true);
}

function notifyMissingCard(title) {
    state.notifications.items.push({
        id: `kb-missing-${Date.now()}`,
        message: `"${title}" has no data yet - its card appears once the run logs it.`,
        level: 'info',
    });
    state.notifications.items = state.notifications.items.slice(-100);
    state.notifications.unread += 1;
    renderNotifications();
}

/** Switch to the Stage tab and scroll/flash the fleet row for `name`. */
async function navigateToAgent(name) {
    await executeAction('SWITCH_TAB', 'agents');
    for (let i = 0; i < 40; i++) {
        const row = document.querySelector(`.agent-row[data-agent-name="${CSS.escape(name)}"]`);
        if (row) {
            row.scrollIntoView({ block: 'center', behavior: 'smooth' });
            row.classList.add('kb-flash');
            setTimeout(() => row.classList.remove('kb-flash'), 1200);
            return;
        }
        await new Promise(r => setTimeout(r, 100));
    }
}

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
