/**
 * Praxis Web - WebSocket Service
 * Handle metrics-live and live-reload WebSocket connections
 */

import { state } from './state.js';
import { renderTerminalStatus, renderNotifications } from './render.js';
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
// Guards the one-time foreground/online revival listeners (see below).
let revivalWired = false;

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
    // Reuse the existing socket: if it's mid-reconnect just nudge it awake,
    // never build a second one (orphaned sockets keep firing on shared state).
    if (metricsSocket) {
        if (!metricsSocket.connected) metricsSocket.connect();
        return;
    }

    console.log('[WS] Connecting to metrics-live');

    // Let socket.io own reconnection entirely - no hand-rolled retry loop, and
    // crucially no attempt cap (default ['polling','websocket'] handshake is the
    // robust path through mobile carriers/proxies that block a bare WS upgrade).
    metricsSocket = io.connect('/metrics-live', {
        path: getSocketPath(),
        reconnection: true,
        reconnectionDelay: 1000,
        reconnectionDelayMax: 10000,
        reconnectionAttempts: Infinity
    });

    metricsSocket.on('connect', () => {
        console.log('[WS] Metrics-live connected');
        state.liveMetrics.connected = true;
        state.terminal.connected = true;
        // Only the connection dot reflects this; a full render() here re-touched
        // the tab carousel mid-swipe on every connect/disconnect flap.
        renderTerminalStatus();
    });

    metricsSocket.on('disconnect', () => {
        console.log('[WS] Metrics-live disconnected');
        state.liveMetrics.connected = false;
        state.terminal.connected = false;
        renderTerminalStatus();
        // No manual reconnect here: socket.io's built-in reconnection handles it.
    });

    // Mobile browsers suspend backgrounded tabs, killing the socket and (while
    // throttled) stalling socket.io's own backoff. Force a reconnect the moment
    // the page returns to the foreground or the network comes back.
    wireConnectionRevival();

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
        renderTerminalStatus();
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
}

/**
 * Revive the metrics socket whenever the page returns to the foreground or the
 * network is restored. Wired once. socket.io reconnects on its own, but mobile
 * timer throttling can leave that backoff stalled while the tab is hidden - this
 * gives it an immediate kick on resume so the dot/terminal recover without a
 * full page reload.
 */
function wireConnectionRevival() {
    if (revivalWired) return;
    revivalWired = true;

    const revive = () => {
        if (metricsSocket && !metricsSocket.connected) {
            metricsSocket.connect();
        }
    };

    document.addEventListener('visibilitychange', () => {
        if (!document.hidden) revive();
    });
    window.addEventListener('online', revive);
    window.addEventListener('focus', revive);
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
