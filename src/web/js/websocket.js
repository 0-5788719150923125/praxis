/**
 * Praxis Web - WebSocket Service
 * Handle metrics-live and live-reload WebSocket connections
 */

import { state } from './state.js';
import { render } from './render.js';
import { renderLiveDashboard } from './dashboard.js';

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

        // Only render if the Terminal tab is active
        if (state.currentTab === 'terminal') {
            renderLiveDashboard(data);
        }
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
