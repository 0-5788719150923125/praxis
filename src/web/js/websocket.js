/**
 * Praxis Web - WebSocket Service
 * Handle terminal/dashboard and live-reload WebSocket connections
 */

import { state } from './state.js';
import { render } from './render.js';

// Live reload socket
let liveReloadSocket = null;

// Terminal socket
let socket = null;
let reconnectTimeout = null;
let reconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 5;

// Dashboard rendering state
let lastFrameHash = null;
let frameStuckCounter = 0;
let lastFrameUpdateTime = Date.now();
let frameValidationInterval = null;
let dashboardScale = null;
let currentFrameContainer = null;
let currentWrapperDiv = null;

/**
 * Connect to terminal WebSocket
 */
export function connectTerminal() {
    if (socket && socket.connected) {
        return;
    }

    // Clear any pending reconnect
    if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
        reconnectTimeout = null;
    }

    console.log('[WS] Connecting to terminal (attempt', reconnectAttempts + 1, ')');

    // Determine socket path
    const pathname = window.location.pathname;
    let socketPath = '/socket.io';

    if (pathname && pathname !== '/') {
        const cleanPath = pathname.replace(/\/$/, '');
        socketPath = cleanPath + '/socket.io';
    }

    // Connect to terminal namespace
    socket = io.connect('/terminal', {
        path: socketPath,
        reconnection: true,
        reconnectionDelay: 1000,
        reconnectionDelayMax: 5000,
        reconnectionAttempts: MAX_RECONNECT_ATTEMPTS,
        transports: ['websocket', 'polling']
    });

    socket.on('connect', () => {
        console.log('[WS] Terminal connected');
        reconnectAttempts = 0;
        frameStuckCounter = 0;
        lastFrameUpdateTime = Date.now();
        state.terminal.connected = true;
        render();

        // Request initial frame and start capture
        setTimeout(() => {
            if (socket && socket.connected) {
                startTerminalCapture();
                startFrameValidation();
            }
        }, 100);
    });

    socket.on('disconnect', () => {
        console.log('[WS] Terminal disconnected');
        state.terminal.connected = false;
        render();

        // Attempt reconnect
        if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
            reconnectTimeout = setTimeout(() => {
                reconnectAttempts++;
                connectTerminal();
            }, 2000);
        }
    });

    socket.on('dashboard_update', (data) => {
        renderDashboardUpdate(data);
    });

    socket.on('dashboard_frame', (data) => {
        renderDashboardFrame(data);
    });

    socket.on('output', (data) => {
        appendTerminalLine(data.data || data);
    });

    socket.on('connect_error', (error) => {
        console.error('[WS] Connection error:', error);
        state.terminal.connected = false;
        render();
    });
}

/**
 * Start terminal capture - request frames from server
 */
function startTerminalCapture() {
    if (socket && socket.connected) {
        socket.emit('start_capture', {
            command: 'connect_existing'
        });
        console.log('[WS] Requested terminal capture start');
    }
}

/**
 * Disconnect terminal WebSocket
 */
export function disconnectTerminal() {
    if (socket) {
        socket.disconnect();
        socket = null;
    }
    if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
        reconnectTimeout = null;
    }
}

/**
 * Render dashboard update (handles both full and diff updates)
 * @param {Object} data - Update data from server
 */
function renderDashboardUpdate(data) {
    if (data.type === 'full' && data.frame) {
        // Full frame update: data.frame is an array of lines
        renderDashboardFrame(data.frame);
    } else if (data.type === 'diff' && data.changes) {
        // Differential update: apply character-level changes
        applyDashboardDiff(data.changes);
    }
}

/**
 * Render dashboard frame (ANSI art)
 * @param {Array} frame - Array of frame lines from server
 */
function renderDashboardFrame(frame) {
    const container = document.getElementById('terminal-display');
    if (!container) return;

    // Update frame timestamp
    lastFrameUpdateTime = Date.now();

    // Calculate frame hash for change detection
    const frameString = frame.join('\n');
    const currentHash = simpleHash(frameString);

    if (currentHash === lastFrameHash) {
        return;
    }
    lastFrameHash = currentHash;

    // Clear container
    container.innerHTML = '';

    // Create wrapper div with styling
    const wrapperDiv = document.createElement('div');
    wrapperDiv.style.position = 'relative';
    wrapperDiv.style.overflow = 'hidden';
    wrapperDiv.style.backgroundColor = '#0d0d0d';
    wrapperDiv.style.display = 'block';
    wrapperDiv.style.margin = '0 auto';
    wrapperDiv.style.userSelect = 'text';
    wrapperDiv.style.cursor = 'text';

    // Create frame container
    const frameContainer = document.createElement('div');
    frameContainer.className = 'dashboard-frame';
    frameContainer.style.userSelect = 'text';

    // Detect mobile for ASCII fallback
    const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);

    // Render each line
    frame.forEach((line, index) => {
        const lineDiv = document.createElement('div');
        lineDiv.className = 'dashboard-line';
        lineDiv.setAttribute('data-line-index', index);
        lineDiv.style.userSelect = 'text';

        // Replace Unicode box chars with ASCII on mobile
        if (isMobile) {
            line = line.replace(/[█▓▒░]/g, '#')
                      .replace(/[▀▄]/g, '=')
                      .replace(/[▌▐]/g, '|')
                      .replace(/[═]/g, '=')
                      .replace(/[─━]/g, '-')
                      .replace(/[║│]/g, '|')
                      .replace(/[┏┌┓┐┗└┛┘]/g, '+')
                      .replace(/[┣├┫┤┳┬┻┴╋┼]/g, '+');
        }

        lineDiv.textContent = line;
        frameContainer.appendChild(lineDiv);
    });

    wrapperDiv.appendChild(frameContainer);
    container.appendChild(wrapperDiv);

    // Store references for scaling
    currentFrameContainer = frameContainer;
    currentWrapperDiv = wrapperDiv;

    // Calculate and apply scaling
    if (dashboardScale === null) {
        // First frame - calculate scale in next animation frame
        requestAnimationFrame(() => {
            calculateDashboardScale();
        });
    } else if (dashboardScale) {
        // Subsequent frames - apply existing scale
        applyDashboardScale();
    }
}

/**
 * Append a line to terminal output
 * @param {string} text - Line text
 */
function appendTerminalLine(text) {
    const container = document.getElementById('terminal-display');
    if (!container) return;

    const line = document.createElement('div');
    line.className = 'terminal-line';
    line.textContent = text;
    container.appendChild(line);

    // Auto-scroll
    container.scrollTop = container.scrollHeight;
}

/**
 * Setup live reload WebSocket for template hot-reloading
 */
export function setupLiveReload() {
    if (state.settings.debugLogging) {
        console.log('[LiveReload] Connecting...');
    }

    const pathname = window.location.pathname;
    let socketPath = '/socket.io';

    if (pathname && pathname !== '/') {
        const cleanPath = pathname.replace(/\/$/, '');
        socketPath = cleanPath + '/socket.io';
    }

    if (state.settings.debugLogging) {
        console.log('[LiveReload] Socket path:', socketPath);
    }

    liveReloadSocket = io.connect('/live-reload', {
        path: socketPath
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

/**
 * Calculate dashboard scaling based on container dimensions
 */
function calculateDashboardScale() {
    if (!currentFrameContainer || !currentWrapperDiv) {
        return;
    }

    const terminalDisplay = document.getElementById('terminal-display');
    if (!terminalDisplay) {
        return;
    }

    const naturalWidth = currentFrameContainer.scrollWidth;
    const naturalHeight = currentFrameContainer.scrollHeight;

    currentFrameContainer.style.width = naturalWidth + 'px';
    currentFrameContainer.style.height = naturalHeight + 'px';
    currentFrameContainer.style.maxWidth = naturalWidth + 'px';
    currentFrameContainer.style.overflow = 'hidden';

    const containerWidth = terminalDisplay.clientWidth;
    const containerHeight = terminalDisplay.clientHeight || window.innerHeight * 0.6;
    const padding = 20;

    const widthScale = (containerWidth - padding) / naturalWidth;
    const heightScale = (containerHeight - padding) / naturalHeight;

    dashboardScale = Math.min(widthScale, heightScale, 1.5);

    if (window.innerWidth <= 768 && dashboardScale > 1) {
        dashboardScale = 1;
    }

    applyDashboardScale();
}

/**
 * Apply scaling to dashboard
 */
function applyDashboardScale() {
    if (!currentFrameContainer || !currentWrapperDiv || !dashboardScale) {
        return;
    }

    const terminalDisplay = document.getElementById('terminal-display');
    if (!terminalDisplay) {
        return;
    }

    const naturalWidth = currentFrameContainer.scrollWidth || parseInt(currentFrameContainer.style.width);
    const naturalHeight = currentFrameContainer.scrollHeight || parseInt(currentFrameContainer.style.height);

    currentFrameContainer.style.transform = `scale(${dashboardScale})`;
    currentFrameContainer.style.transformOrigin = 'top left';

    const scaledHeight = naturalHeight * dashboardScale;
    const scaledWidth = naturalWidth * dashboardScale;

    currentWrapperDiv.style.width = scaledWidth + 'px';
    currentWrapperDiv.style.height = scaledHeight + 'px';
    currentWrapperDiv.style.overflow = 'hidden';
    currentWrapperDiv.style.margin = '0 auto';
    currentWrapperDiv.style.display = 'block';

    terminalDisplay.style.minHeight = (scaledHeight + 20) + 'px';
    terminalDisplay.scrollTop = 0;
}

/**
 * Recalculate and apply dashboard scaling
 */
export function recalculateDashboardScale() {
    if (!currentFrameContainer || !currentWrapperDiv) {
        return;
    }

    // Check if we're on the terminal tab
    if (state.currentTab !== 'terminal') {
        return;
    }

    // Reset scale and transform before recalculating
    dashboardScale = null;
    currentFrameContainer.style.transform = 'none';
    calculateDashboardScale();
}

/**
 * Simple hash function for frame comparison
 */
function simpleHash(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
        const char = str.charCodeAt(i);
        hash = ((hash << 5) - hash) + char;
        hash = hash & hash;
    }
    return hash;
}

/**
 * Start frame validation to detect stuck frames
 */
function startFrameValidation() {
    if (frameValidationInterval) {
        clearInterval(frameValidationInterval);
    }

    frameValidationInterval = setInterval(() => {
        validateTerminalRendering();
    }, 5000);
}

/**
 * Validate terminal rendering and force refresh if stuck
 */
function validateTerminalRendering() {
    const now = Date.now();
    const timeSinceLastFrame = now - lastFrameUpdateTime;

    // If no frame update in 10 seconds and terminal is connected
    if (timeSinceLastFrame > 10000 && state.terminal.connected) {
        frameStuckCounter++;

        if (frameStuckCounter >= 2) {
            console.log('[Terminal] Frame appears stuck, forcing refresh...');
            forceTerminalRefresh();
            frameStuckCounter = 0;
        }
    } else {
        frameStuckCounter = 0;
    }
}

/**
 * Force terminal refresh by reconnecting
 */
function forceTerminalRefresh() {
    if (socket && socket.connected) {
        socket.emit('request_frame');
    }
}

/**
 * Apply dashboard diff (for incremental updates)
 * @param {Array} changes - Array of change objects: {row, col, text, length}
 */
function applyDashboardDiff(changes) {
    if (!currentFrameContainer) {
        console.warn('[Dashboard] No frame container to apply diff');
        return;
    }

    // Update frame timestamp
    lastFrameUpdateTime = Date.now();

    // Apply each change to the DOM
    changes.forEach(change => {
        const lineDiv = currentFrameContainer.querySelector(`[data-line-index="${change.row}"]`);
        if (!lineDiv) {
            console.warn(`[Dashboard] Line ${change.row} not found`);
            return;
        }

        // Get current line text
        let lineText = lineDiv.textContent;

        // Apply the change: replace 'length' characters at 'col' with 'text'
        const before = lineText.substring(0, change.col);
        const after = lineText.substring(change.col + change.length);
        lineText = before + change.text + after;

        // Update the line
        lineDiv.textContent = lineText;
    });

    // Note: We don't update the hash for diff updates to allow continuous changes
    // The hash is only used for detecting duplicate full frames
}
