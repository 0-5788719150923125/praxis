/**
 * Praxis Web - Live Dashboard Renderer
 * Renders a terminal-styled dashboard from streamed metrics data
 */

// Persistent UI state across re-renders
let logPanelOpen = false;

/**
 * Toggle the log panel open/closed
 */
export function toggleLogPanel() {
    logPanelOpen = !logPanelOpen;
    const panel = document.getElementById('ld-log-panel');
    const btn = document.getElementById('ld-log-toggle');
    if (panel) panel.classList.toggle('open', logPanelOpen);
    if (btn) btn.textContent = logPanelOpen ? 'LOGS [-]' : 'LOGS [+]';
    if (logPanelOpen) {
        requestAnimationFrame(() => {
            const logContent = panel && panel.querySelector('.ld-log-content');
            if (logContent) logContent.scrollTop = logContent.scrollHeight;
        });
    }
}

// Expose globally so the onclick works from innerHTML
if (typeof window !== 'undefined') {
    window.toggleLogPanel = toggleLogPanel;
}

/**
 * Escape HTML to prevent XSS
 * @param {string} text
 * @returns {string}
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Format a number with fixed decimals, or return a fallback
 * @param {number|null} value
 * @param {number} decimals
 * @param {string} fallback
 * @returns {string}
 */
function fmt(value, decimals = 4, fallback = '---') {
    if (value === null || value === undefined) return fallback;
    return Number(value).toFixed(decimals);
}

/**
 * Render the info panel as key-value rows
 * @param {Object} info
 * @returns {string}
 */
function renderInfoPanel(info) {
    if (!info || Object.keys(info).length === 0) {
        return '<div class="ld-info-row"><span class="ld-info-key">status</span><span class="ld-info-val">waiting for data...</span></div>';
    }

    const displayKeys = [
        'device', 'rank', 'node', 'ram', 'vram', 'vram_gb',
        'optimizer', 'strategy', 'policy',
        'vocab_size', 'block_size', 'batch_size', 'target_batch',
        'depth', 'num_layers', 'hidden_size', 'embed_size', 'dropout'
    ];

    return displayKeys
        .filter(key => info[key] !== undefined && info[key] !== null)
        .map(key => {
            const label = key.replace(/_/g, ' ');
            return `<div class="ld-info-row"><span class="ld-info-key">${escapeHtml(label)}</span><span class="ld-info-val">${escapeHtml(String(info[key]))}</span></div>`;
        })
        .join('');
}

/**
 * Draw a sparkline on a canvas element
 * @param {number[]} data - Array of loss values
 * @param {string} canvasId - Canvas element ID
 */
function drawSparkline(data, canvasId) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !data || data.length < 2) return;

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();

    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;
    const padding = 4;

    // Clear
    ctx.clearRect(0, 0, width, height);

    // Calculate bounds
    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;

    // Draw line
    ctx.beginPath();
    ctx.strokeStyle = '#0B9A6D';
    ctx.lineWidth = 1.5;
    ctx.lineJoin = 'round';

    data.forEach((val, i) => {
        const x = padding + (i / (data.length - 1)) * (width - 2 * padding);
        const y = height - padding - ((val - min) / range) * (height - 2 * padding);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });

    ctx.stroke();

    // Draw current value dot
    if (data.length > 0) {
        const lastVal = data[data.length - 1];
        const x = width - padding;
        const y = height - padding - ((lastVal - min) / range) * (height - 2 * padding);
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, 2 * Math.PI);
        ctx.fillStyle = '#0B9A6D';
        ctx.fill();
    }

    // Draw min/max labels
    ctx.font = '9px monospace';
    ctx.fillStyle = '#666';
    ctx.textAlign = 'left';
    ctx.fillText(fmt(max, 4), padding, 10);
    ctx.fillText(fmt(min, 4), padding, height - 2);
}

/**
 * Render the live dashboard from a metrics snapshot
 * @param {Object} m - Metrics snapshot from server
 */
/**
 * Check if a scrollable element is at (or near) the bottom
 * @param {Element} el
 * @returns {boolean}
 */
function isScrolledToBottom(el) {
    if (!el) return true;
    return el.scrollHeight - el.scrollTop - el.clientHeight < 8;
}

export function renderLiveDashboard(m) {
    const container = document.getElementById('terminal-display');
    if (!container) return;

    const expertStr = (m.local_experts || m.remote_experts)
        ? `${m.local_experts || 0} local, ${m.remote_experts || 0} remote`
        : 'none';

    // Check if dashboard already exists so we can update in-place
    const existing = container.querySelector('.live-dashboard');
    if (existing) {
        // Update in-place to preserve scroll positions

        // Header
        const headerLeft = existing.querySelector('.ld-header-left');
        if (headerLeft) headerLeft.innerHTML = `<span class="ld-label">HASH:</span> <span class="ld-hash">${escapeHtml(m.arg_hash || '------')}</span> <span class="ld-sep">||</span> <span class="ld-label">CONTEXT:</span> ${m.context_tokens || 0} tokens`;

        const headerMetrics = existing.querySelector('.ld-header-metrics');
        if (headerMetrics) headerMetrics.innerHTML = `
            <span class="ld-metric">ERROR <span class="ld-val">${fmt(m.loss)}</span></span>
            ${m.val_loss !== null ? `<span class="ld-metric">VALIDATION <span class="ld-val">${fmt(m.val_loss)}</span></span>` : ''}
            ${m.fitness !== null && m.fitness !== undefined ? `<span class="ld-metric">FITNESS <span class="ld-val">${fmt(m.fitness, 2)}%</span></span>` : ''}
            ${m.accuracy ? `<span class="ld-metric">ACCURACY <span class="ld-val">${fmt(m.accuracy[0], 3)}</span></span>` : ''}
        `;

        // Status text - auto-scroll only if already at bottom
        const statusText = existing.querySelector('.ld-status-text');
        if (statusText) {
            const wasAtBottom = isScrolledToBottom(statusText);
            statusText.textContent = m.status_text || '_initializing';
            if (wasAtBottom) statusText.scrollTop = statusText.scrollHeight;
        }

        // Info panel
        const infoGrid = existing.querySelector('.ld-info-grid');
        if (infoGrid) infoGrid.innerHTML = renderInfoPanel(m.info);

        // Log count
        const logCount = existing.querySelector('.ld-log-count');
        if (logCount) logCount.textContent = `${(m.log_lines || []).length} lines`;

        // Log content - auto-scroll only if already at bottom
        const logContent = existing.querySelector('.ld-log-content');
        if (logContent && logPanelOpen) {
            const wasAtBottom = isScrolledToBottom(logContent);
            logContent.innerHTML = (m.log_lines || []).map(l => `<div class="ld-log-line">${escapeHtml(l)}</div>`).join('');
            if (wasAtBottom) logContent.scrollTop = logContent.scrollHeight;
        }

        // Footer
        const footer = existing.querySelector('.ld-footer');
        if (footer) footer.innerHTML = `
            <span class="ld-footer-item">PRAXIS:${m.seed || '?'}</span>
            <span class="ld-footer-sep">|</span>
            <span class="ld-footer-item">${m.total_params || '0M'}</span>
            <span class="ld-footer-sep">|</span>
            <span class="ld-footer-item">${(m.mode || 'train').toUpperCase()}</span>
            <span class="ld-footer-sep">|</span>
            <span class="ld-footer-item">AGE ${fmt(m.hours_elapsed, 2, '0.00')}h</span>
            <span class="ld-footer-sep">|</span>
            <span class="ld-footer-item">TOKENS ${fmt(m.num_tokens, 3, '0.000')}B</span>
            <span class="ld-footer-sep">|</span>
            <span class="ld-footer-item">BATCH ${Math.round(m.batch) || 0}</span>
            <span class="ld-footer-sep">|</span>
            <span class="ld-footer-item">STEP ${m.step || 0}</span>
            <span class="ld-footer-sep">|</span>
            <span class="ld-footer-item">RATE ${fmt(m.rate, 2, '0.00')}s</span>
            <span class="ld-footer-sep">|</span>
            <span class="ld-footer-item">EXPERTS ${expertStr}</span>
            <span class="ld-footer-sep">|</span>
            <span class="ld-footer-item ld-url">${escapeHtml(m.url || 'N/A')}</span>
        `;

        // Redraw sparkline
        requestAnimationFrame(() => {
            drawSparkline(m.loss_history, 'ld-sparkline-canvas');
        });
        return;
    }

    // First render - build full DOM
    container.innerHTML = `
        <div class="live-dashboard">
            <div class="ld-header">
                <span class="ld-header-left"><span class="ld-label">HASH:</span> <span class="ld-hash">${escapeHtml(m.arg_hash || '------')}</span> <span class="ld-sep">||</span> <span class="ld-label">CONTEXT:</span> ${m.context_tokens || 0} tokens</span>
                <span class="ld-header-metrics">
                    <span class="ld-metric">ERROR <span class="ld-val">${fmt(m.loss)}</span></span>
                    ${m.val_loss !== null ? `<span class="ld-metric">VALIDATION <span class="ld-val">${fmt(m.val_loss)}</span></span>` : ''}
                    ${m.fitness !== null && m.fitness !== undefined ? `<span class="ld-metric">FITNESS <span class="ld-val">${fmt(m.fitness, 2)}%</span></span>` : ''}
                    ${m.accuracy ? `<span class="ld-metric">ACCURACY <span class="ld-val">${fmt(m.accuracy[0], 3)}</span></span>` : ''}
                </span>
            </div>
            <div class="ld-body">
                <div class="ld-panel ld-panel-status">
                    <div class="ld-panel-title">Status</div>
                    <div class="ld-status-text">${escapeHtml(m.status_text || '_initializing')}</div>
                </div>
                <div class="ld-panel ld-panel-chart">
                    <div class="ld-panel-title">Training Loss</div>
                    <canvas id="ld-sparkline-canvas" class="ld-sparkline-canvas"></canvas>
                </div>
            </div>
            <div class="ld-info-panel">
                <div class="ld-panel-title">System</div>
                <div class="ld-info-grid">
                    ${renderInfoPanel(m.info)}
                </div>
            </div>
            <div class="ld-log-bar">
                <button id="ld-log-toggle" class="ld-log-toggle" onclick="toggleLogPanel()">${logPanelOpen ? 'LOGS [-]' : 'LOGS [+]'}</button>
                <span class="ld-log-count">${(m.log_lines || []).length} lines</span>
            </div>
            <div id="ld-log-panel" class="ld-log-panel ${logPanelOpen ? 'open' : ''}">
                <div class="ld-log-content">${(m.log_lines || []).map(l => `<div class="ld-log-line">${escapeHtml(l)}</div>`).join('')}</div>
            </div>
            <div class="ld-footer">
                <span class="ld-footer-item">PRAXIS:${m.seed || '?'}</span>
                <span class="ld-footer-sep">|</span>
                <span class="ld-footer-item">${m.total_params || '0M'}</span>
                <span class="ld-footer-sep">|</span>
                <span class="ld-footer-item">${(m.mode || 'train').toUpperCase()}</span>
                <span class="ld-footer-sep">|</span>
                <span class="ld-footer-item">AGE ${fmt(m.hours_elapsed, 2, '0.00')}h</span>
                <span class="ld-footer-sep">|</span>
                <span class="ld-footer-item">TOKENS ${fmt(m.num_tokens, 3, '0.000')}B</span>
                <span class="ld-footer-sep">|</span>
                <span class="ld-footer-item">BATCH ${Math.round(m.batch) || 0}</span>
                <span class="ld-footer-sep">|</span>
                <span class="ld-footer-item">STEP ${m.step || 0}</span>
                <span class="ld-footer-sep">|</span>
                <span class="ld-footer-item">RATE ${fmt(m.rate, 2, '0.00')}s</span>
                <span class="ld-footer-sep">|</span>
                <span class="ld-footer-item">EXPERTS ${expertStr}</span>
                <span class="ld-footer-sep">|</span>
                <span class="ld-footer-item ld-url">${escapeHtml(m.url || 'N/A')}</span>
            </div>
        </div>
    `;

    // Draw sparkline and scroll to bottom on first render
    requestAnimationFrame(() => {
        drawSparkline(m.loss_history, 'ld-sparkline-canvas');
        const statusText = container.querySelector('.ld-status-text');
        if (statusText) statusText.scrollTop = statusText.scrollHeight;
        if (logPanelOpen) {
            const logContent = container.querySelector('.ld-log-content');
            if (logContent) logContent.scrollTop = logContent.scrollHeight;
        }
    });
}
