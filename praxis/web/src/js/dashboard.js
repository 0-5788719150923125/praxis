/**
 * Praxis Web - Live Dashboard Renderer
 * Renders a terminal-styled dashboard from streamed metrics data
 */

import { statusLabel, state } from './state.js';
import { getOfflineAccent } from './theme.js';

// Persistent UI state across re-renders (survives tab switches; resets on reload).
let logPanelOpen = false;

// Opening the logs flips the whole app to the blue accent (the "hidden mode"). The
// flag lives on <html> so every tab + the Praxis animation re-tint, and it persists
// across tab navigation because we never clear it on re-render. Closing it restores
// the registered offline accent (if any) rather than clearing it outright - an
// offline export must stay red once the logs panel closes again.
function applyLogAccent() {
    if (typeof document === 'undefined') return;
    if (logPanelOpen) {
        document.documentElement.setAttribute('data-accent', 'blue');
    } else if (getOfflineAccent()) {
        document.documentElement.setAttribute('data-accent', getOfflineAccent());
    } else {
        document.documentElement.removeAttribute('data-accent');
    }
}

/**
 * Switch the System/Logs card view. The two tabs swap which inner card shows;
 * landing on Logs also flips the app to the blue accent (the hidden mode).
 */
export function switchDashCard(view) {
    logPanelOpen = view === 'logs';
    applyLogAccent();
    const root = document.querySelector('.live-dashboard');
    const panel = root && root.querySelector('.ld-info-panel');
    if (!panel) return;
    const grid = panel.querySelector('.ld-info-grid');
    const logs = panel.querySelector('.ld-log-view');
    if (grid) grid.hidden = logPanelOpen;
    if (logs) logs.hidden = !logPanelOpen;
    // Logs view tints the card black (keeps its chrome + tab header).
    panel.classList.toggle('ld-info-panel--logs', logPanelOpen);
    panel.querySelectorAll('.ld-card-tab').forEach(t =>
        t.classList.toggle('active', t.dataset.view === view));
    if (logPanelOpen && logs) {
        requestAnimationFrame(() => {
            const c = logs.querySelector('.ld-log-content');
            if (c) c.scrollTop = c.scrollHeight;
        });
    }
}

// Expose globally so the onclick works from innerHTML
if (typeof window !== 'undefined') {
    window.switchDashCard = switchDashCard;
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

// Internal flags streamed for the TUI dashboard but not surfaced in the web
// System panel. Everything else the backend sends is rendered, in the order
// it arrives - so a new info field shows up without any frontend edit.
const INTERNAL_INFO_KEYS = new Set(['debug', 'meta']);

/**
 * Render the info panel as key-value rows, driven entirely by what the
 * backend puts in ``m.info`` (the training process owns the field list).
 * @param {Object} info
 * @returns {string}
 */
function renderInfoPanel(info) {
    const entries = Object.entries(info || {}).filter(([key, value]) =>
        !INTERNAL_INFO_KEYS.has(key) && value !== undefined && value !== null
    );

    if (entries.length === 0) {
        return '<div class="ld-info-row"><span class="ld-info-key">status</span><span class="ld-info-val">waiting for data...</span></div>';
    }

    // A "heavy" row is one whose value would wrap (a long value, e.g. several
    // comma-joined policies) and inflate the grid row it shares with tidy
    // single-line metrics. Threshold ~ a cell's worth of characters.
    const HEAVY_LEN = 22;
    const isHeavy = ([key, value]) =>
        (key.length + String(value).length) > HEAVY_LEN || String(value).includes(',');

    const shorts = entries.filter(e => !isHeavy(e));
    const heavies = entries.filter(isHeavy);

    // Float heavy rows to the bottom, but seat them at the LEFT of the final row
    // with the leftover single-line metrics filling in to their right - tidier
    // than a tall wrapping cell stranded on the right. The grid is ~4 columns on
    // desktop (mobile is one column, so this just stacks them near the end); each
    // heavy is also forced to grid-column 1 (the --heavy class) so it cleanly
    // begins a row regardless of the exact column count.
    const COLS = 4;
    let ordered = shorts;
    if (heavies.length) {
        const rightFill = Math.min(shorts.length, COLS - (heavies.length % COLS || COLS));
        ordered = [
            ...shorts.slice(0, shorts.length - rightFill),
            ...heavies,
            ...shorts.slice(shorts.length - rightFill),
        ];
    }

    return ordered
        .map(([key, value]) => {
            const label = key.replace(/_/g, ' ');
            const cls = isHeavy([key, value]) ? 'ld-info-row ld-info-row--heavy' : 'ld-info-row';
            return `<div class="${cls}"><span class="ld-info-key">${escapeHtml(label)}</span><span class="ld-info-val">${escapeHtml(String(value))}</span></div>`;
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

    // Brand accent (green, or blue in logs mode) - follows the central --accent.
    const accent = getComputedStyle(canvas).getPropertyValue('--accent').trim() || '#0B9A6D';

    // Clear
    ctx.clearRect(0, 0, width, height);

    // Calculate bounds
    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;

    // Draw line
    ctx.beginPath();
    ctx.strokeStyle = accent;
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
        ctx.fillStyle = accent;
        ctx.fill();
    }

    // Draw min/max labels
    ctx.font = '9px monospace';
    ctx.fillStyle = getComputedStyle(canvas).getPropertyValue('--term-dim2').trim() || '#666';
    ctx.textAlign = 'left';
    ctx.fillText(fmt(max, 4), padding, 10);
    ctx.fillText(fmt(min, 4), padding, height - 2);
}

// The sparklines are hand-drawn canvases, not Chart.js instances (compare
// setupAccentRetint in charts.js), so an accent/theme flip only ever reaches
// them via a redraw. In live mode the ~2Hz metrics_snapshot tick redraws them
// anyway, so this went unnoticed; on a frozen/offline export that tick fires
// once and never again, so without this observer a theme change (the offline
// orange accent, or a light/dark toggle) never appears on these charts at all.
let _sparklineThemeRedrawReady = false;
export function setupSparklineThemeRedraw() {
    if (_sparklineThemeRedrawReady || typeof document === 'undefined') return;
    _sparklineThemeRedrawReady = true;
    new MutationObserver(() => {
        const m = state.liveMetrics.data;
        if (!m) return;
        drawSparkline(m.loss_history, 'ld-sparkline-canvas');
        if (hasSwarm(m)) drawSparkline(m.swarm_loss_history, 'ld-swarm-sparkline-canvas');
    }).observe(document.documentElement, {
        attributes: true, attributeFilter: ['data-accent', 'data-theme'],
    });
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

// Rolling contexts: the server sends `contexts` (one block per temperature
// experiment). Fall back to the single status_text for the legacy single-context
// path so older producers still render one block.
function contextsOf(m) {
    if (Array.isArray(m.contexts) && m.contexts.length) return m.contexts;
    return [{ name: 'Context', description: '', temperature: null, chance: null, text: m.status_text }];
}

// Block meta line: "T0.5  10% | 42 tok". Chance always shows as a positive percent
// (100% for the always-on block); every block carries its own token count (falling
// back to the legacy global on block 0), set off from temperature/chance by a pipe.
function ctxMetaHtml(c, m, i) {
    const parts = [];
    if (c.temperature != null) parts.push(`T${(+c.temperature).toFixed(1)}`);
    if (c.chance != null) {
        const p = c.chance * 100;
        parts.push(`${p < 1 ? p.toFixed(1) : Math.round(p)}%`);
    }
    // Prefer each block's own count; fall back to the global primary count for any
    // block the producer didn't tokenize, so a seeded card never reads "0 tok".
    const toks = c.tokens != null ? c.tokens : (m.context_tokens || 0);
    const left = parts.length ? `${escapeHtml(parts.join('  '))}<span class="ld-meta-sep">|</span>` : '';
    return `${left}<span class="ld-meta-tok">${toks} tok</span>`;
}

function renderContextBlock(c, i, m) {
    return `
        <div class="ld-panel ld-context" data-ctx-index="${i}">
            <div class="ld-panel-title">${escapeHtml(c.name || 'Context')}<span class="ld-context-meta">${ctxMetaHtml(c, m, i)}</span></div>
            ${c.description ? `<div class="ld-context-desc">${escapeHtml(c.description)}</div>` : ''}
            <div class="ld-status-text">${escapeHtml(c.text || '_initializing')}</div>
        </div>`;
}

// The swarm-loss sparkline only appears once an expert pool has logged at least
// two sampled points (drawSparkline needs a segment to draw).
function hasSwarm(m) {
    return Array.isArray(m.swarm_loss_history) && m.swarm_loss_history.length > 1;
}

export function renderLiveDashboard(m) {
    const container = document.getElementById('terminal-display');
    if (!container) return;

    const ctxs = contextsOf(m);

    // Update in-place to preserve scroll positions - but only when the block count
    // matches; if it changed (e.g. contexts arrived after a status_text-only frame)
    // fall through to a full rebuild.
    const existing = container.querySelector('.live-dashboard');
    const swarmPanelPresent = !!(existing && existing.querySelector('#ld-swarm-sparkline-canvas'));
    if (existing
        && existing.querySelectorAll('.ld-context').length === ctxs.length
        && swarmPanelPresent === hasSwarm(m)) {

        // Per-context rolling text - auto-scroll only if already at bottom.
        const blocks = existing.querySelectorAll('.ld-context');
        ctxs.forEach((c, i) => {
            const blk = blocks[i];
            const st = blk && blk.querySelector('.ld-status-text');
            if (st) {
                const wasAtBottom = isScrolledToBottom(st);
                st.textContent = c.text || '_initializing';
                if (wasAtBottom) st.scrollTop = st.scrollHeight;
            }
            const meta = blk && blk.querySelector('.ld-context-meta');
            if (meta) meta.innerHTML = ctxMetaHtml(c, m, i);
        });

        // Header
        const headerLeft = existing.querySelector('.ld-header-left');
        if (headerLeft) headerLeft.innerHTML = `<span class="ld-label">HASH:</span> <span class="ld-hash">${escapeHtml(m.arg_hash || '------')}</span>`;

        const headerMetrics = existing.querySelector('.ld-header-metrics');
        if (headerMetrics) headerMetrics.innerHTML = `
            <span class="ld-metric">ERROR <span class="ld-val">${fmt(m.loss)}</span></span>
            ${m.val_loss !== null ? `<span class="ld-metric">VALIDATION <span class="ld-val">${fmt(m.val_loss)}</span></span>` : ''}
            ${m.fitness !== null && m.fitness !== undefined ? `<span class="ld-metric">FITNESS <span class="ld-val">${fmt(m.fitness, 2)}%</span></span>` : ''}
            ${m.accuracy ? `<span class="ld-metric">ACCURACY <span class="ld-val">${fmt(m.accuracy[0], 3)}</span></span>` : ''}
        `;

        // (rolling-context text is updated per-block above)

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
            <span class="ld-footer-item">${m.total_params || '0.00M'}</span>
            <span class="ld-footer-sep">|</span>
            <span class="ld-footer-item">${statusLabel(m)}</span>
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
            <span class="ld-footer-item ld-url">${escapeHtml(m.url || 'N/A')}</span>
        `;

        // Redraw sparklines
        requestAnimationFrame(() => {
            drawSparkline(m.loss_history, 'ld-sparkline-canvas');
            if (hasSwarm(m)) drawSparkline(m.swarm_loss_history, 'ld-swarm-sparkline-canvas');
        });
        return;
    }

    // First render - build full DOM
    container.innerHTML = `
        <div class="live-dashboard">
            <div class="ld-header">
                <span class="ld-header-left"><span class="ld-label">HASH:</span> <span class="ld-hash">${escapeHtml(m.arg_hash || '------')}</span></span>
                <span class="ld-header-metrics">
                    <span class="ld-metric">ERROR <span class="ld-val">${fmt(m.loss)}</span></span>
                    ${m.val_loss !== null ? `<span class="ld-metric">VALIDATION <span class="ld-val">${fmt(m.val_loss)}</span></span>` : ''}
                    ${m.fitness !== null && m.fitness !== undefined ? `<span class="ld-metric">FITNESS <span class="ld-val">${fmt(m.fitness, 2)}%</span></span>` : ''}
                    ${m.accuracy ? `<span class="ld-metric">ACCURACY <span class="ld-val">${fmt(m.accuracy[0], 3)}</span></span>` : ''}
                </span>
            </div>
            <div class="ld-info-panel ${logPanelOpen ? 'ld-info-panel--logs' : ''}">
                <div class="ld-card-tabs">
                    <button class="ld-card-tab ${logPanelOpen ? '' : 'active'}" data-view="system" onclick="switchDashCard('system')">SYSTEM</button>
                    <button class="ld-card-tab ${logPanelOpen ? 'active' : ''}" data-view="logs" onclick="switchDashCard('logs')">LOGS</button>
                </div>
                <div class="ld-info-grid" ${logPanelOpen ? 'hidden' : ''}>
                    ${renderInfoPanel(m.info)}
                </div>
                <div class="ld-log-view" ${logPanelOpen ? '' : 'hidden'}>
                    <div class="ld-log-panel-title">Logs <span class="ld-log-count">${(m.log_lines || []).length} lines</span></div>
                    <div class="ld-log-content">${(m.log_lines || []).map(l => `<div class="ld-log-line">${escapeHtml(l)}</div>`).join('')}</div>
                </div>
            </div>
            <div class="ld-body">
                ${ctxs.map((c, i) => renderContextBlock(c, i, m)).join('')}
                <div class="ld-panel ld-panel-chart">
                    <div class="ld-panel-title">Training Loss</div>
                    <canvas id="ld-sparkline-canvas" class="ld-sparkline-canvas"></canvas>
                </div>
                ${hasSwarm(m) ? `
                <div class="ld-panel ld-panel-chart">
                    <div class="ld-panel-title">Swarm Loss</div>
                    <canvas id="ld-swarm-sparkline-canvas" class="ld-sparkline-canvas"></canvas>
                </div>` : ''}
            </div>
            <div class="ld-footer">
                <span class="ld-footer-item">PRAXIS:${m.seed || '?'}</span>
                <span class="ld-footer-sep">|</span>
                <span class="ld-footer-item">${m.total_params || '0.00M'}</span>
                <span class="ld-footer-sep">|</span>
                <span class="ld-footer-item">${statusLabel(m)}</span>
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
                <span class="ld-footer-item ld-url">${escapeHtml(m.url || 'N/A')}</span>
            </div>
        </div>
    `;

    // Draw sparkline and scroll to bottom on first render
    requestAnimationFrame(() => {
        drawSparkline(m.loss_history, 'ld-sparkline-canvas');
        if (hasSwarm(m)) drawSparkline(m.swarm_loss_history, 'ld-swarm-sparkline-canvas');
        container.querySelectorAll('.ld-status-text').forEach(st => { st.scrollTop = st.scrollHeight; });
        if (logPanelOpen) {
            const logContent = container.querySelector('.ld-log-content');
            if (logContent) logContent.scrollTop = logContent.scrollHeight;
        }
    });
}
