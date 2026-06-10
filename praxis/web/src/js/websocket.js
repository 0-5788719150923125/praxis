/**
 * Praxis Web - WebSocket Service
 * Handle metrics-live and live-reload WebSocket connections
 */

import { state } from './state.js';
import { render, renderNotifications } from './render.js';
import { renderLiveDashboard } from './dashboard.js';

/**
 * Merge backend events from a metrics snapshot into the notification feed.
 * Deduped by monotonic id; bumps unread while the panel is closed.
 */
function mergeNotifications(events) {
    if (!Array.isArray(events) || events.length === 0) return;
    const known = new Set(state.notifications.items.map((e) => e.id));
    let added = 0;
    for (const ev of events) {
        if (!known.has(ev.id)) {
            state.notifications.items.push(ev);
            added++;
        }
    }
    if (added === 0) return;
    state.notifications.items = state.notifications.items.slice(-100);
    if (!state.notifications.panelOpen) {
        state.notifications.unread += added;
    }
    renderNotifications();
}

// Live reload socket
let liveReloadSocket = null;

// Metrics live socket
let metricsSocket = null;
let metricsReconnectTimeout = null;
let metricsReconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 5;

/**
 * Get the socket.io path based on current URL
 * @returns {string}
 */
function getSocketPath() {
    const pathname = window.location.pathname;
    let socketPath = '/socket.io';
    if (pathname && pathname !== '/') {
        const cleanPath = pathname.replace(/\/$/, '');
        socketPath = cleanPath + '/socket.io';
    }
    return socketPath;
}

/**
 * Connect to metrics-live WebSocket for real-time dashboard data
 */
export function connectMetricsLive() {
    if (metricsSocket && metricsSocket.connected) {
        return;
    }

    if (metricsReconnectTimeout) {
        clearTimeout(metricsReconnectTimeout);
        metricsReconnectTimeout = null;
    }

    console.log('[WS] Connecting to metrics-live (attempt', metricsReconnectAttempts + 1, ')');

    metricsSocket = io.connect('/metrics-live', {
        path: getSocketPath(),
        reconnection: true,
        reconnectionDelay: 1000,
        reconnectionDelayMax: 5000,
        reconnectionAttempts: MAX_RECONNECT_ATTEMPTS,
        transports: ['websocket', 'polling']
    });

    metricsSocket.on('connect', () => {
        console.log('[WS] Metrics-live connected');
        metricsReconnectAttempts = 0;
        state.liveMetrics.connected = true;
        state.terminal.connected = true;
        render();
    });

    metricsSocket.on('disconnect', () => {
        console.log('[WS] Metrics-live disconnected');
        state.liveMetrics.connected = false;
        state.terminal.connected = false;
        render();

        if (metricsReconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
            metricsReconnectTimeout = setTimeout(() => {
                metricsReconnectAttempts++;
                connectMetricsLive();
            }, 2000);
        }
    });

    metricsSocket.on('metrics_snapshot', (data) => {
        state.liveMetrics.data = data;

        // Surface any backend events in the notification bell.
        mergeNotifications(data.events);

        // Only render if the Terminal tab is active
        if (state.currentTab === 'terminal') {
            renderLiveDashboard(data);
        }
    });

    // Server-pushed invalidations ("metrics" on each flushed training step,
    // "snapshots" when a model-probe snapshot actually changes). Re-broadcast
    // as a DOM event so the refresh scheduler (main.js) stays decoupled from
    // the socket lifecycle.
    metricsSocket.on('invalidate', (data) => {
        window.dispatchEvent(new CustomEvent('praxis:data-invalidate', { detail: data }));
    });

    metricsSocket.on('connect_error', (error) => {
        console.error('[WS] Metrics-live connection error:', error);
        state.liveMetrics.connected = false;
        state.terminal.connected = false;
        render();
    });
}

/**
 * Disconnect metrics-live WebSocket
 */
export function disconnectMetricsLive() {
    if (metricsSocket) {
        metricsSocket.disconnect();
        metricsSocket = null;
    }
    if (metricsReconnectTimeout) {
        clearTimeout(metricsReconnectTimeout);
        metricsReconnectTimeout = null;
    }
}

/**
 * Render the latest metrics snapshot when switching to Terminal tab
 */
export function renderCurrentMetrics() {
    if (state.liveMetrics.data) {
        renderLiveDashboard(state.liveMetrics.data);
    }
}

/**
 * Setup live reload WebSocket for template hot-reloading
 */
export function setupLiveReload() {
    if (state.settings.debugLogging) {
        console.log('[LiveReload] Connecting...');
    }

    if (state.settings.debugLogging) {
        console.log('[LiveReload] Socket path:', getSocketPath());
    }

    liveReloadSocket = io.connect('/live-reload', {
        path: getSocketPath()
    });

    liveReloadSocket.on('connect', () => {
        console.log('[LiveReload] Connected');
    });

    liveReloadSocket.on('reload', () => {
        console.log('[LiveReload] Template change detected, reloading...');
        window.location.reload();
    });

    liveReloadSocket.on('disconnect', () => {
        console.log('[LiveReload] Disconnected');
    });
}
