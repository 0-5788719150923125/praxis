/**
 * Praxis Web - Chart.js Integration
 * Full Chart.js implementation for Research tab
 */

import { state, CONSTANTS, chartLineColor, currentAccentHue, rotateHexHue } from './state.js';
import { fetchAPI } from './api.js';
import { createTabHeader, pdfButton } from './components.js';
import { dedupe, hasRealContent } from './prefetch.js';
import { SCROLL_TAU, SCROLL_MIN_VEL } from './momentum.js';

// Chart instances storage (exported for hybrid mode)
export const charts = {};

// ── Accent auto-retint ──────────────────────────────────────────────────────
// Chart.js caches dataset colors at creation, so flipping the accent (the logs
// "blue mode") wouldn't recolor live charts. A plugin stamps each chart with the
// hue it was built at; a MutationObserver on <html data-accent> then rotates every
// chart's line colors by the hue delta - no rebuild/Refresh needed.
const _accentCharts = new Set();
let _accentRetintReady = false;

function retintAccentCharts() {
    const hue = currentAccentHue();
    _accentCharts.forEach(chart => {
        const from = chart.$accentHue == null ? hue : chart.$accentHue;
        const delta = hue - from;
        if (!delta) return;
        let changed = false;
        (chart.data?.datasets || []).forEach(ds => {
            const b = rotateHexHue(ds.borderColor, delta);
            if (b !== ds.borderColor) { ds.borderColor = b; changed = true; }
            const g = rotateHexHue(ds.backgroundColor, delta);
            if (g !== ds.backgroundColor) { ds.backgroundColor = g; changed = true; }
        });
        chart.$accentHue = hue;
        if (changed) chart.update('none');
    });
}

export function setupAccentRetint() {
    if (_accentRetintReady || typeof document === 'undefined') return;
    _accentRetintReady = true;
    if (window.Chart) {
        window.Chart.register({
            id: 'accentRetint',
            afterInit(chart) { chart.$accentHue = currentAccentHue(); _accentCharts.add(chart); },
            afterDestroy(chart) { _accentCharts.delete(chart); },
        });
    }
    new MutationObserver(retintAccentCharts).observe(document.documentElement, {
        attributes: true, attributeFilter: ['data-accent'],
    });
}

// Layer selection state for per-layer metrics
const layerSelectionState = {};

// ETag caches for the metrics endpoints so a 304 Not Modified on a manual
// refresh can short-circuit re-render.
let lastMetricsEtag = null;
let lastDataMetricsEtag = null;

/**
 * Detect if an element is within a hybrid overlay (light theme context)
 * @param {HTMLElement} element - The element to check
 * @returns {boolean} True if in hybrid overlay
 */
function isInHybridOverlay(element) {
    if (!element) return false;
    return element.closest('.hybrid-overlay') !== null;
}

/**
 * Get the appropriate theme for rendering based on context
 * @param {HTMLElement} element - The element being rendered
 * @returns {string} 'light' or 'dark'
 */
function getContextTheme(element) {
    return isInHybridOverlay(element) ? 'light' : state.theme;
}

/**
 * Pure function: Get theme-appropriate colors
 * Always recalculates from current theme state - functional approach
 * @param {string} [forceTheme] - Optional theme override ('light' or 'dark')
 * @returns {Object} Color palette for current theme
 */
function getThemeColors(forceTheme) {
    const theme = forceTheme || state.theme;
    const isDark = theme === 'dark';
    return {
        textColor: isDark ? '#e0e0e0' : '#1f1f1f',
        gridColor: isDark ? 'rgba(255, 255, 255, 0.12)' : 'rgba(0, 0, 0, 0.12)',
        tooltipBg: isDark ? '#1e1e1e' : '#ffffff'
    };
}

/**
 * Recolor a single live chart for the current theme, without redrawing data.
 * Colors are derived from the chart's OWN canvas context (getContextTheme), so a
 * chart living in the light hybrid overlay stays light even when the base is dark.
 * @param {Chart} chart - a Chart.js instance
 */
export function applyChartTheme(chart) {
    if (!chart) return;
    const { textColor, gridColor, tooltipBg } = getThemeColors(getContextTheme(chart.canvas));

    // Scale colors (axes)
    if (chart.options.scales) {
        Object.values(chart.options.scales).forEach(scale => {
            if (scale.title) scale.title.color = textColor;
            if (scale.ticks) scale.ticks.color = textColor;
            if (scale.grid) scale.grid.color = gridColor;
        });
    }

    // Legend / title text
    if (chart.options.plugins?.legend?.labels) {
        chart.options.plugins.legend.labels.color = textColor;
    }
    if (chart.options.plugins?.title) {
        chart.options.plugins.title.color = textColor;
    }

    // Tooltip
    if (chart.options.plugins?.tooltip) {
        chart.options.plugins.tooltip.backgroundColor = tooltipBg;
        chart.options.plugins.tooltip.titleColor = textColor;
        chart.options.plugins.tooltip.bodyColor = textColor;
        chart.options.plugins.tooltip.borderColor = gridColor;
    }

    // Dataset-level colors that were seeded from textColor at creation.
    chart.data?.datasets?.forEach(ds => {
        if ('pointHoverBorderColor' in ds) ds.pointHoverBorderColor = textColor;
    });

    chart.update('none');   // apply instantly, no animation
}

/**
 * Recolor every research chart for the current theme. Iterates the full registry
 * so cards cached on an inactive tab are recolored too (they don't rebuild on
 * revisit), and each chart picks up its own context theme.
 */
export function updateChartColors() {
    Object.values(charts).forEach(applyChartTheme);
}

// A theme toggle only changes CSS-variable colors, but the deck cards are GPU-composited
// (will-change: transform, opacity), and a composited layer does NOT re-rasterize its
// text on a pure variable change - so the fan cards behind the head keep stale font
// colors (card 1 updates, 2-4 don't). A layout flush isn't enough; we have to dirty an
// actual PAINT property. Nudge each card's background this frame (forces a repaint, which
// re-rasterizes the text with the now-current --text), then clear it next frame. The
// nudge value is invisible (0.001 alpha) so nothing flickers.
export function repaintDeckCards() {
    if (typeof document === 'undefined') return;
    const cards = document.querySelectorAll('.chart-deck .chart-card');
    if (!cards.length) return;
    cards.forEach((c) => { c.style.backgroundColor = 'rgba(0, 0, 0, 0.001)'; });
    requestAnimationFrame(() => {
        cards.forEach((c) => { c.style.backgroundColor = ''; });
    });
}

/**
 * Load available runs (local on-disk + remote agents) into a unified list
 */
export async function loadAvailableRuns() {
    try {
        // Fetch local runs and agents in parallel
        const [runsResponse, agentsResponse] = await Promise.all([
            fetch('/api/runs').catch(() => null),
            fetchAPI('agents').catch(() => null)
        ]);

        const localRuns = [];
        if (runsResponse && runsResponse.ok) {
            const data = await runsResponse.json();
            if (data.runs) {
                data.runs.forEach(run => {
                    localRuns.push({
                        ...run,
                        source: 'local'
                    });
                });
            }
        }

        // Build a set of local run hashes for dedup
        const localHashes = new Set(localRuns.map(r => r.hash));

        // Add online remote agents (skip self-* since they duplicate local runs)
        const remoteAgents = [];
        if (agentsResponse && agentsResponse.agents) {
            for (const agent of agentsResponse.agents) {
                if (agent.status !== 'online') continue;
                if (agent.name.startsWith('self-')) continue;

                // Fetch the agent's truncated_hash from its spec endpoint
                let argsHash = null;
                try {
                    const baseUrl = agent.url.replace(/\/praxis(\.git)?$/, '');
                    const specResp = await fetch(`${baseUrl}/api/spec`);
                    if (specResp.ok) {
                        const spec = await specResp.json();
                        argsHash = spec.truncated_hash;
                    }
                } catch { /* skip if unreachable */ }

                // Skip if this agent's args hash already exists as a local run
                if (argsHash && localHashes.has(argsHash)) {
                    // Annotate the local run with the agent name instead
                    const localRun = localRuns.find(r => r.hash === argsHash);
                    if (localRun) localRun.agentName = agent.name;
                    continue;
                }

                remoteAgents.push({
                    hash: argsHash || agent.short_hash || agent.name,
                    agentName: agent.name,
                    agentUrl: agent.url.replace(/\/praxis(\.git)?$/, ''),
                    is_current: false,
                    num_steps: 0,
                    metrics_updated: agent.commit_timestamp || 0,
                    source: 'remote'
                });
            }
        }

        state.research.historicalRuns = [...localRuns, ...remoteAgents];

        // Auto-select the current (active) run on first load
        if (state.research.selectedHistoricalRuns.length === 0) {
            const currentRun = localRuns.find(r => r.is_current);
            if (currentRun) {
                state.research.selectedHistoricalRuns = [currentRun.hash];
            }
        }

        return state.research.historicalRuns;
    } catch (error) {
        console.error('[Charts] Error loading runs:', error);
        return [];
    }
}

/**
 * Fetch metrics for all selected runs (local and remote)
 */
async function fetchSelectedRunMetrics() {
    const selected = state.research.selectedHistoricalRuns;
    if (selected.length === 0) return [];

    // Split selected into local and remote
    const localHashes = [];
    const remoteEntries = [];

    for (const hash of selected) {
        const entry = state.research.historicalRuns.find(r => r.hash === hash);
        if (!entry) continue;
        if (entry.source === 'remote' && entry.agentUrl) {
            remoteEntries.push(entry);
        } else {
            localHashes.push(hash);
        }
    }

    const results = [];

    // Fetch local runs in a single batch. Send If-None-Match so the
    // server can short-circuit with 304 when nothing has changed; on 304
    // we reuse the last render's local-run data.
    let localUnchanged = false;
    if (localHashes.length > 0) {
        try {
            const runsParam = localHashes.join(',');
            const headers = lastMetricsEtag ? { 'If-None-Match': lastMetricsEtag } : {};
            const response = await fetch(
                `/api/metrics?since=0&limit=1000&downsample=lttb&runs=${runsParam}`,
                { headers, cache: 'no-cache' }
            );
            if (response.status === 304) {
                localUnchanged = true;
                const previousLocal = (state.research.lastRuns || []).filter(
                    r => localHashes.includes(r.hash)
                );
                results.push(...previousLocal);
            } else if (response.ok) {
                const newEtag = response.headers.get('ETag');
                if (newEtag) lastMetricsEtag = newEtag;
                const data = await response.json();
                if (data.runs) {
                    data.runs.forEach(run => {
                        results.push({
                            name: run.hash,
                            hash: run.hash,
                            is_current: run.is_current,
                            metrics: run.metrics,
                            metadata: run.metadata
                        });
                    });
                }
                // Stash the registries so callers can build chart configs
                // from them. Riding alongside as array properties keeps
                // the function signature unchanged.
                if (data.registry) results.registry = data.registry;
                if (data.composite_registry) results.compositeRegistry = data.composite_registry;
            }
        } catch (error) {
            console.error('[Charts] Error loading local run metrics:', error);
        }
    }

    // Fetch remote agent metrics individually
    const remotePromises = remoteEntries.map(async (entry) => {
        try {
            const response = await fetch(`${entry.agentUrl}/api/metrics?since=0&limit=1000&downsample=lttb`);
            if (!response.ok) return null;
            const data = await response.json();
            if (data.status === 'no_data' || !data.runs || data.runs.length === 0) return null;
            return {
                name: entry.hash,
                hash: entry.hash,
                is_current: false,
                metrics: data.runs[0].metrics,
                metadata: data.runs[0].metadata
            };
        } catch {
            return null;
        }
    });

    const remoteResults = await Promise.all(remotePromises);
    remoteResults.forEach(r => { if (r) results.push(r); });

    // Signal "nothing changed" only when local was unchanged AND there are
    // no remote runs to re-fetch (remotes aren't etag-tracked yet).
    if (localUnchanged && remoteEntries.length === 0) {
        results.unchanged = true;
    }

    return results;
}

/**
 * Fetch data metrics (sampling weights, etc.) from the local server
 */
async function fetchDataMetrics() {
    try {
        const headers = lastDataMetricsEtag ? { 'If-None-Match': lastDataMetricsEtag } : {};
        const response = await fetch(
            '/api/data-metrics?since=0&limit=1000&downsample=lttb',
            { headers, cache: 'no-cache' }
        );
        if (response.status === 304) return { unchanged: true };
        if (!response.ok) return [];

        const newEtag = response.headers.get('ETag');
        if (newEtag) lastDataMetricsEtag = newEtag;

        const data = await response.json();
        if (data.status === 'no_data' || !data.runs || data.runs.length === 0) return [];

        return data.runs.map(run => ({
            name: run.hash || 'local',
            data_metrics: run.data_metrics || run.metrics,
            metadata: run.metadata
        }));
    } catch {
        return [];
    }
}

/**
 * Toggle run selector dropdown
 */
export function toggleRunSelector() {
    state.research.runSelectorOpen = !state.research.runSelectorOpen;
    const dropdown = document.getElementById('run-selector-dropdown');
    if (dropdown) {
        dropdown.style.display = state.research.runSelectorOpen ? 'block' : 'none';
    }
}

/**
 * Toggle historical run selection
 */
export function toggleRunSelection(hash) {
    const index = state.research.selectedHistoricalRuns.indexOf(hash);

    if (index > -1) {
        state.research.selectedHistoricalRuns.splice(index, 1);
    } else {
        state.research.selectedHistoricalRuns.push(hash);
    }

    // Reload charts only (header stays intact)
    loadResearchMetricsWithCharts(true);
}

/**
 * Format a relative time string from a timestamp
 */
export function formatRelativeTime(timestamp) {
    if (!timestamp) return 'unknown';
    const now = Date.now() / 1000;
    const diff = now - timestamp;

    if (diff < 60) return 'just now';
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    if (diff < 604800) return `${Math.floor(diff / 86400)}d ago`;
    return new Date(timestamp * 1000).toLocaleDateString();
}

/**
 * Load and render research metrics with full Chart.js integration.
 *
 * Fetches run on tab activation, on run-selector changes, and when a
 * server-pushed invalidation marks the data dirty (see main.js). There is
 * no blind polling; charts only rebuild when something actually changed.
 *
 * @param {boolean} force - If true, re-fetch even if already loaded.
 */
export async function loadResearchMetricsWithCharts(force = false) {
    await dedupe('tab:research', () => loadResearchInner(force));
}

async function loadResearchInner(force) {
    if (state.research.loaded && !force) return;

    const container = document.getElementById('research-container');
    if (!container) return;

    // Stale-while-revalidate: keep the current charts painted during a
    // refresh; only first loads show the placeholder.
    const chartsArea = document.getElementById('metrics-charts-area');
    const target = chartsArea || container;
    if (!hasRealContent(target)) {
        target.innerHTML = '<div class="loading-placeholder">Loading metrics...</div>';
    }

    try {
        await loadAvailableRuns();

        const [runs, dataMetrics] = await Promise.all([
            fetchSelectedRunMetrics(),
            fetchDataMetrics()
        ]);

        // Normalize 304-unchanged sentinels back to the previous render's data.
        const runsUnchanged = runs && runs.unchanged === true;
        const dataUnchanged = dataMetrics && dataMetrics.unchanged === true;
        const runsToRender = runsUnchanged ? (state.research.lastRuns || []) : runs;
        const dataToRender = dataUnchanged ? (state.research.lastDataMetrics || []) : dataMetrics;

        // Registry only refreshes when local runs do (the 304 path reuses
        // whatever was cached). Remote-only refreshes also fall back to
        // the cached registry since remotes aren't etag-tracked.
        if (runs && runs.registry) {
            state.research.metricRegistry = runs.registry;
        }
        if (runs && runs.compositeRegistry) {
            state.research.compositeRegistry = runs.compositeRegistry;
        }

        state.research.lastRuns = runsToRender;
        state.research.lastDataMetrics = dataToRender;

        await renderMetricsCharts({
            runs: runsToRender,
            dataMetrics: dataToRender,
            registry: state.research.metricRegistry,
            compositeRegistry: state.research.compositeRegistry,
        }, container);
        state.research.loaded = true;

    } catch (error) {
        console.error('[Charts] Error loading metrics:', error);
        const errorHTML = `
            <div class="error-message">
                <h3>Error Loading Metrics</h3>
                <p>${error.message}</p>
            </div>
        `;
        const chartsAreaEl = document.getElementById('metrics-charts-area');
        if (chartsAreaEl) {
            chartsAreaEl.innerHTML = errorHTML;
        } else {
            container.innerHTML = errorHTML;
        }
    }
}

/**
 * Render full metrics charts
 */
/**
 * Translate the backend training-metric registry into chart configs.
 *
 * Each registry entry with a ``chart`` hint becomes a scalar chart on
 * the Research tab. Chartless entries (e.g. ``batch``, ``local_layers``)
 * are persisted as columns but don't render. Sorted by ``chart.order``
 * so backend declares display ordering.
 */
function buildScalarConfigsFromRegistry(registry) {
    const entries = Object.entries(registry || {})
        .filter(([, v]) => v && v.chart)
        .sort(([, a], [, b]) => (a.chart.order ?? 0) - (b.chart.order ?? 0));

    return entries.map(([key, entry]) => ({
        key,
        canvasId: `chart-metric-${key.replace(/_/g, '-')}`,
        title: entry.chart.title || key,
        label: entry.chart.y_label || entry.chart.title || key,
        type: entry.chart.type || 'line',
        description: entry.description || '',
    }));
}

/**
 * Translate the backend composite-metric registry into chart configs.
 *
 * These are the multi-expert / sampling / heatmap charts that used to be
 * hardcoded in state.js. The backend declares each one's type, title,
 * key pattern, and source; here we just compile the pattern to a RegExp
 * and derive a canvas id. Adding a composite chart is now a one-entry
 * change in praxis/metrics/training_metrics.py - no JS edits.
 */
function buildCompositeConfigsFromRegistry(registry) {
    return (registry || [])
        .slice()
        .sort((a, b) => (a.order ?? 0) - (b.order ?? 0))
        .map(entry => ({
            key: entry.key,
            canvasId: `chart-metric-${entry.key.replace(/[/_]/g, '-')}`,
            title: entry.title || entry.key,
            label: entry.y_label || entry.title || entry.key,
            type: entry.type || 'line',
            description: entry.description || '',
            source: entry.source || 'metrics',
            stepped: entry.stepped || false,
            keyPattern: entry.key_pattern ? new RegExp(entry.key_pattern) : null,
        }));
}

// Maps a config's `type` to a renderer. Each takes (config, { runs,
// dataMetrics }); the underlying chart builders are declared below and
// hoisted. Add a renderer here to support a new backend chart `type`.
const METRIC_RENDERERS = {
    bar: (config, { runs }) =>
        createTokensBarChart(config.canvasId, config.label, runs, config.key),
    sampling: (config, { dataMetrics }) =>
        createSamplingWeightsChart(config.canvasId, dataMetrics),
    multi_expert_line: (config, { runs }) =>
        createMultiExpertChart(config.canvasId, config.title, config.label, runs, config.keyPattern, config),
    expert_routing_heatmap: (config, { runs }) =>
        createExpertRoutingChart(config.canvasId, runs),
    line: (config, { runs }) =>
        createRunComparisonChart(config.canvasId, config.label, runs, config.key),
    evolution: (config) => createEvolutionChart(config.canvasId),
    spider_citations: (config) => createSpiderCitationsChart(config.canvasId),
};

// Spider link graph: top cited URLs as horizontal bars, ranked by the same
// citation counts that order the crawl frontier. Fetches /api/spider; the
// payload also carries top referrers, surfaced in each bar's tooltip footer.
async function createSpiderCitationsChart(canvasId) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    const fail = (msg) => {
        const c = ctx.getContext('2d');
        const w = ctx.parentElement.clientWidth || 600, h = ctx.parentElement.clientHeight || 280;
        ctx.width = w; ctx.height = h;
        c.fillStyle = '#888'; c.font = '12px monospace'; c.textAlign = 'center';
        c.fillText(msg, w / 2, h / 2);
    };

    let data;
    try {
        const r = await fetch('/api/spider');
        const j = await r.json();
        if (j.status !== 'ok' || !j.data) return fail('No crawl data yet (enable with --spider)');
        data = j.data;
    } catch (e) {
        return fail('Spider data unavailable');
    }

    if (charts[canvasId]) {
        charts[canvasId].destroy();
    }
    const theme = getContextTheme(ctx);
    const { textColor, gridColor, tooltipBg } = getThemeColors(theme);

    // Label by path (host only when it's a site root) so bars stay legible;
    // the tooltip shows the full URL.
    const shortLabel = (url) => {
        const m = url.match(/^https?:\/\/([^/]+)(\/.*)?$/);
        if (!m) return url;
        const path = m[2] && m[2] !== '/' ? m[2] : '';
        const label = path ? `${m[1]}${path}` : m[1];
        return label.length > 42 ? `…${label.slice(-41)}` : label;
    };
    const referrers = (data.referrers || [])
        .map(r => `${shortLabel(r.url)} (${r.count})`);

    charts[canvasId] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.cited.map(d => shortLabel(d.url)),
            datasets: [{
                label: 'Citations',
                data: data.cited.map(d => d.count),
                backgroundColor: data.cited.map((d, i) => chartLineColor(i) + '80'),
                borderColor: data.cited.map((d, i) => chartLineColor(i)),
                borderWidth: 2,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: tooltipBg,
                    titleColor: textColor,
                    bodyColor: textColor,
                    borderColor: gridColor,
                    borderWidth: 1,
                    padding: 12,
                    callbacks: {
                        title: (items) => data.cited[items[0].dataIndex].url,
                        label: (item) => `${item.parsed.x} citation(s)`,
                        footer: () => referrers.length
                            ? ['', 'Top referrers:', ...referrers.slice(0, 5)]
                            : [],
                    },
                },
            },
            scales: {
                x: {
                    beginAtZero: true,
                    ticks: { color: textColor, precision: 0 },
                    grid: { color: gridColor },
                },
                y: {
                    ticks: { color: textColor, font: { size: 10 } },
                    grid: { display: false },
                },
            },
        },
    });
}

// Isometric recency-weighted terrain. MUST match praxis/pillars/evolution.py
// (_iso_boxes) so the web card and the LaTeX figure are the same picture: height
// is recency-weighted churn (recent towers, history settles to the prior valley),
// peaks taper, color fades from the subsystem hue toward a neutral prior into the
// past, banded in phased strata over a non-linear timescale.
const ISO = {
    TX: 1.0, TY: 0.5, TZ: 1.0, HMAX: 4.4, GAP: 0.14,
    DECAY: 2.4, FLOOR: 0.0, TAPER: 0.6, PHASE_AMP: 0.24, PHASE_CYCLES: 3.0,
    MAX_STACK: 6, BASE_UNIT: 0.12, STACK_GAP: 0.28, CAP_HMAX: 0.32,
};
const PRIOR_RGB = [87, 97, 115];  // neutral the past fades into (0.34,0.38,0.45)
const isoProj = (x, y, z) => [(x - y) * ISO.TX, z * ISO.HMAX * ISO.TZ - (x + y) * ISO.TY];
const isoHexRgb = (hex) => {
    const n = parseInt((hex || '#586072').slice(1), 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
};
const isoShadeRgb = (rgb, f) =>
    `rgb(${Math.min(255, rgb[0] * f) | 0},${Math.min(255, rgb[1] * f) | 0},${Math.min(255, rgb[2] * f) | 0})`;

// Repo-level git-churn evolution card. Fetches /api/evolution - the SAME
// computation the LaTeX figure renders (praxis.pillars.evolution.evolution_data)
// - and draws it as the same isometric block field. The data is unified; this is
// just the web output format.
async function createEvolutionChart(canvasId) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const fail = (msg) => {
        const ctx = canvas.getContext('2d');
        const w = canvas.parentElement.clientWidth || 600, h = canvas.parentElement.clientHeight || 280;
        canvas.width = w; canvas.height = h;
        ctx.fillStyle = '#0f1117'; ctx.fillRect(0, 0, w, h);
        ctx.fillStyle = '#888'; ctx.font = '12px monospace'; ctx.textAlign = 'center';
        ctx.fillText(msg, w / 2, h / 2);
    };

    let data;
    try {
        const r = await fetch('/api/evolution');
        const j = await r.json();
        if (j.status !== 'ok' || !j.data) return fail('No git history');
        data = j.data;
    } catch (e) {
        return fail('Evolution data unavailable');
    }

    const { subsystems: subs, series, totals, colors, bins: B } = data;
    const S = subs.length;
    let cmax = 1e-6, tmax = 1e-6;
    for (const s of subs) for (let i = 0; i < B; i++) {
        cmax = Math.max(cmax, series[s][i]);
        tmax = Math.max(tmax, (totals[s] || [])[i] || 0);
    }

    const mix = (a, b, t) => [0, 1, 2].map(k => a[k] + t * (b[k] - a[k]));
    // Three visible iso faces of a box from base footprint (z0) to top (z1).
    const isoBox = (x0, x1, y0, y1, tx0, tx1, ty0, ty1, z0, z1, col, depth) => {
        const top = [isoProj(tx0, ty0, z1), isoProj(tx1, ty0, z1), isoProj(tx1, ty1, z1), isoProj(tx0, ty1, z1)];
        const east = [isoProj(x1, y0, z0), isoProj(x1, y1, z0), isoProj(tx1, ty1, z1), isoProj(tx1, ty0, z1)];
        const south = [isoProj(x0, y1, z0), isoProj(x1, y1, z0), isoProj(tx1, ty1, z1), isoProj(tx0, ty1, z1)];
        return [
            { pts: south, color: isoShadeRgb(col, 0.6), depth },
            { pts: east, color: isoShadeRgb(col, 0.8), depth },
            { pts: top, color: isoShadeRgb(col, 1.0), depth },
        ];
    };

    // Build the block faces once (geometry is size-independent in unit coords).
    // Each cell: a stack of base blocks (total lines, recency-weighted) topped
    // by a tapered churn cap. Painter-ordered by i+j.
    const faces = [];
    for (let j = 0; j < S; j++) {
        const base = isoHexRgb(colors[subs[j]]);
        for (let i = 0; i < B; i++) {
            const u = i / Math.max(B - 1, 1);              // 0 oldest .. 1 now
            const w = Math.exp(-ISO.DECAY * (1 - u));       // recency weight
            const g = ISO.GAP / 2;
            const x0 = i + g, x1 = i + 1 - g, y0 = j + g, y1 = j + 1 - g;
            const phase = 1 + ISO.PHASE_AMP * Math.cos(2 * Math.PI * ISO.PHASE_CYCLES * Math.pow(1 - u, 1.3));
            const depth = i + j;
            // Base stack: total lines, recency-weighted -> N rectangular blocks.
            const tot = ((totals[subs[j]] || [])[i] || 0) / tmax;
            const nStack = Math.round(Math.sqrt(tot) * w * ISO.MAX_STACK);
            const baseCol = mix(PRIOR_RGB, base, Math.min(1, w * 0.7)).map(c => Math.min(255, c * phase));
            for (let k = 0; k < nStack; k++) {
                const z0 = ISO.FLOOR + k * ISO.BASE_UNIT;
                const z1 = z0 + ISO.BASE_UNIT * (1 - ISO.STACK_GAP);
                faces.push(...isoBox(x0, x1, y0, y1, x0, x1, y0, y1, z0, z1, baseCol, depth));
            }
            const baseTop = ISO.FLOOR + nStack * ISO.BASE_UNIT;
            // Churn cap: vivid + tapered, on top of the stack.
            const c = series[subs[j]][i] / cmax;
            if (c > 0) {
                const capH = c * w * ISO.CAP_HMAX;
                const tp = ISO.TAPER * (capH / ISO.CAP_HMAX);
                const ix = tp * (x1 - x0) / 2, iy = tp * (y1 - y0) / 2;
                const capCol = mix(PRIOR_RGB, base, w).map(c2 => Math.min(255, c2 * phase));
                faces.push(...isoBox(x0, x1, y0, y1, x0 + ix, x1 - ix, y0 + iy, y1 - iy,
                                     baseTop, baseTop + capH, capCol, depth));
            }
        }
    }
    faces.sort((a, b) => a.depth - b.depth);  // back to front
    // Unit bounding box for the fit.
    let uxmin = Infinity, uxmax = -Infinity, uymin = Infinity, uymax = -Infinity;
    for (const f of faces) for (const [ux, uy] of f.pts) {
        if (ux < uxmin) uxmin = ux; if (ux > uxmax) uxmax = ux;
        if (uy < uymin) uymin = uy; if (uy > uymax) uymax = uy;
    }

    const draw = () => {
        const wrap = canvas.parentElement;
        const w = wrap.clientWidth || 600, h = wrap.clientHeight || 280;
        if (w < 2 || h < 2 || !faces.length) return;
        const dpr = window.devicePixelRatio || 1;
        canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
        canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
        const ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.fillStyle = '#0f1117'; ctx.fillRect(0, 0, w, h);

        const padL = 8, padR = 8, padT = 26, padB = 20;
        const plotW = w - padL - padR, plotH = h - padT - padB;
        const scale = Math.min(plotW / Math.max(uxmax - uxmin, 1e-6),
                               plotH / Math.max(uymax - uymin, 1e-6));
        const offX = padL + (plotW - (uxmax - uxmin) * scale) / 2;
        const sx = (ux) => offX + (ux - uxmin) * scale;
        const sy = (uy) => padT + (uymax - uy) * scale;  // canvas y-down flip

        ctx.lineWidth = 0.5; ctx.strokeStyle = '#0f1117'; ctx.lineJoin = 'round';
        for (const f of faces) {
            ctx.beginPath();
            ctx.moveTo(sx(f.pts[0][0]), sy(f.pts[0][1]));
            for (let k = 1; k < f.pts.length; k++) ctx.lineTo(sx(f.pts[k][0]), sy(f.pts[k][1]));
            ctx.closePath();
            ctx.fillStyle = f.color; ctx.fill(); ctx.stroke();
        }

        ctx.font = '9px monospace'; ctx.fillStyle = '#9aa3b2'; ctx.textAlign = 'center';
        ctx.fillText('first commit → now    depth: subsystem    stack: total lines    cap: recent churn', w / 2, h - 5);

        ctx.textAlign = 'left';
        let cx = padL;
        for (const s of subs) {
            if (cx > w - 70) break;
            ctx.fillStyle = colors[s] || '#586072'; ctx.fillRect(cx, 5, 8, 8);
            ctx.fillStyle = '#cfd3dc'; ctx.fillText(s, cx + 11, 13);
            cx += 13 + ctx.measureText(s).width + 9;
        }
    };

    draw();
    if (canvas._evoRO) canvas._evoRO.disconnect();
    canvas._evoRO = new ResizeObserver(() => draw());
    canvas._evoRO.observe(canvas.parentElement);
}

function renderMetricsCharts(data, container) {
    const runs = data.runs || [];
    const dataMetrics = data.dataMetrics || [];
    const registry = data.registry || {};
    const compositeRegistry = data.compositeRegistry || [];

    // Render header only once — subsequent calls only rebuild charts
    if (!document.getElementById('metrics-charts-area')) {
        renderMetricsHeader(container, runs);
    } else {
        updateMetricsMetadata(runs);
    }

    let chartsArea = document.getElementById('metrics-charts-area');
    if (!chartsArea) return;

    if (runs.length === 0) {
        chartsArea.innerHTML = `
            <div class="empty-state" style="margin-top: 2rem;">
                <h3>No Metrics Available</h3>
                <p>Select runs above to display metrics.</p>
            </div>
        `;
        return;
    }

    // Both scalar and composite chart configs come from the backend
    // registries - there are no hardcoded JS metric lists. Adding any
    // chart is a one-entry change in praxis/metrics/training_metrics.py.
    const allConfigs = [
        ...buildScalarConfigsFromRegistry(registry),
        ...buildCompositeConfigsFromRegistry(compositeRegistry),
    ];

    // Data-driven metric detection
    const availableMetrics = allConfigs.filter(config => {
        // Standalone cards (e.g. the repo-level evolution chart) fetch their own
        // data and are not gated on per-run metrics.
        if (config.source === 'standalone') return true;
        if (config.source === 'data_metrics') {
            return dataMetrics.some(a =>
                a.data_metrics?.[config.key] &&
                Object.keys(a.data_metrics[config.key]).length > 0
            );
        }
        if (config.keyPattern) {
            return runs.some(r =>
                Object.keys(r.metrics).some(k => config.keyPattern.test(k))
            );
        }
        return runs.some(r => r.metrics[config.key]?.some(v => v !== null));
    });

    // Destroy existing chart instances before rebuilding
    availableMetrics.forEach(config => {
        if (charts[config.canvasId]) {
            charts[config.canvasId].destroy();
            delete charts[config.canvasId];
        }
    });

    // Build chart cards as a stacked deck (card-switcher carousel)
    let chartsHTML = '<div class="chart-deck" id="chart-deck">';
    chartsHTML += '<div class="chart-deck-counter" id="chart-deck-counter"></div>';

    chartsHTML += availableMetrics.map((config, i) => {
        const stepSliderHTML = (config.type === 'expert_routing_heatmap') ?
            `<div id="layer-toggles-${config.canvasId}" class="layer-toggles" style="margin-bottom: 1rem; padding: 0.5rem; display: flex; gap: 1rem; flex-wrap: wrap; align-items: center;">
            </div>` : '';

        return `
        <div class="chart-card" data-deck-index="${i}" data-card-key="${config.key}">
            <div class="chart-title">${config.title}</div>
            ${config.description ? `<div class="chart-subtitle">${config.description}</div>` : ''}
            ${stepSliderHTML}
            <div class="deck-card-scroll">
                <div class="chart-wrapper">
                    <canvas id="${config.canvasId}"></canvas>
                </div>
            </div>
        </div>
        `;
    }).join('');

    chartsHTML += '</div>';

    chartsArea.innerHTML = chartsHTML;

    // Render charts after DOM update. Each config's `type` selects a
    // renderer from the registry; unknown types fall back to a line chart.
    // Returns a promise that settles AFTER the deferred mount, so callers
    // (and prewarmTab's off-screen layout) can hold until the deck has
    // actually measured - otherwise the hidden layout is torn down before
    // initChartDeck runs and the first visit re-lays the deck out visibly.
    return new Promise(resolve => setTimeout(async () => {
        try {
            const ctx = { runs, dataMetrics };
            // Yield between charts: building the whole deck in one chunk
            // blocks input for hundreds of ms (page-load prewarm hits this).
            for (const config of availableMetrics) {
                (METRIC_RENDERERS[config.type] || METRIC_RENDERERS.line)(config, ctx);
                await new Promise(r => setTimeout(r, 0));
            }
            initChartDeck('chart-deck');
        } finally {
            resolve();
        }
    }, 10));
}

// ============================================================================
// CARD DECK - continuous-scalar momentum carousel
// One fractional position (deck._pos) is the single source of truth. Every
// card's transform is a PURE function of (cardIndex - pos) using cached card
// heights + constants, so the drag/momentum hot path performs NO layout reads
// (no getBoundingClientRect / offsetHeight) and forces no reflow. Touch drags
// pos 1:1; release coasts pos under exponential friction and eases into the
// nearest integer slot. Wheel accumulates pos and snaps when it stops; arrows
// step one slot. On mobile the deck lifts (wallet 'A') so the active card title
// tucks just under the tab row, hiding the per-tab title strip; a downward
// swipe begun at the deck's top edge drops it back ('B') to reveal that strip.
//
// Gesture hierarchy (uniform across every deck and card):
//   1. GRIP - the top band of the head card (title chrome included). Never
//      inner-scrolls: drives the seam + A/B anchor. Exists on every card,
//      titled or not; opt out per card with class "deck-no-grip".
//   2. BODY - scrolls the card's content while there's room, then behaves
//      like the grip at the content edge.
// Anchor flips are exclusive with cycling in both paths: a B->A lift locks
// the gesture, and dropping to B takes a sustained pull (DECK_DROP_THRESHOLD).
// ============================================================================

// activeIndex and anchor state persist across the DOM rebuilds that happen on
// every metrics poll (the deck element is recreated; these maps survive).
const deckActive = {};
const deckAnchor = {};

// Compact fan + motion feel - all tunable.
const DECK_PEEK = 18;            // px each fanned card peeks past the head (compact)
const DECK_SCALE_STEP = 0.045;   // scale shrink per rank behind the head
const DECK_MAX_FAN = 3;          // cards drawn behind the head
const DECK_MIN_CHART_H = 192;    // px; smallest a chart shrinks to under mobile pressure (keeps it readable)
const DECK_SWIPE_STEP = 70;      // finger px that advance one card (1:1 during the drag)
const DECK_DROP_THRESHOLD = 200; // px of SUSTAINED downward pull before the deck drops to B
                                 // (reveals the header). Deliberately high (~two cards' worth)
                                 // so casual swipes at A cycle cards instead of dropping to B.
const DECK_SEAM = 88;            // finger px to flip one card AT the content edge (the "seam").
                                 // Eased slow-fast-slow so it pauses at the content end + anchor.
const DECK_SEAM_FLING = 0.5;     // px/ms finger speed that commits a partial seam on release
const DECK_GRIP_H = 56;          // px; the standard grip band at every card's top. Title
                                 // chrome extends it, but the band exists on EVERY card
                                 // (titled or bare), so the grip gesture is uniform deck-wide.
                                 // Per-card opt-out: class "deck-no-grip".
const DECK_WHEEL_STEP = 120;     // wheel px that advance one card
// Release motion: NO free coast. The release projects ONE target slot from the fling
// direction, then a fixed-duration easeOutCubic slide lands on it - monotonic, so it
// can never reverse (no bounce) and has no asymptotic tail (decisive, weighty).
const DECK_CYCLE_DUR = 220;      // ms; slot-slide duration on release
const DECK_WHEEL_SETTLE = 130;   // ms after the last wheel notch -> snap
const DECK_FLOOR_MARGIN = 14;    // px between the floor and the screen bottom
// A<->B anchor: a vertical lift between the floor (B) and the tab row (A), eased over
// DECK_ANCHOR_DUR with the same easeOutCubic (matches the slide's weight).
const DECK_ANCHOR_DUR = 240;     // ms; A<->B lift duration
// Inner-scroll momentum: flick a card's body and it coasts under friction, clamped
// hard at the top/bottom edge (no rubber-band, no bleed into cycling). The feel
// constants (SCROLL_TAU / SCROLL_MIN_VEL) are shared with the wheel-momentum
// module so touch and wheel coast identically everywhere.
export function initChartDeck(deck, opts = {}) {
    if (typeof deck === 'string') deck = document.getElementById(deck);
    if (!deck) return;
    deck._fanDown = !!opts.fanDown;
    deck._cycleDir = deck._cycleDir || 1;   // +1 forward / -1 backward; seats entering cards
    deck.classList.toggle('deck-fan', deck._fanDown);   // fan-down sheets cap + fade their body
    const cards = Array.from(deck.querySelectorAll('.chart-card'));
    const count = cards.length;
    if (count === 0) return;

    // Chart decks ship cards with no scroll body; give each a .deck-card-scroll
    // (idempotent) so tall cards scroll internally on mobile like the fan-down sheets.
    // Title/subtitle/toggles/number stay pinned outside; only the body content scrolls.
    // Runs before measure; on the dynamics deck it runs before Chart.js mounts (the
    // canvas is moved while still empty) and no-ops once wrapped, so a live canvas is
    // never reparented.
    if (!deck._fanDown) {
        cards.forEach(card => {
            if (card.querySelector(':scope > .deck-card-scroll')) return;
            const body = document.createElement('div');
            body.className = 'deck-card-scroll';
            Array.from(card.children).forEach(ch => {
                if (ch.classList.contains('chart-title') ||
                    ch.classList.contains('chart-subtitle') ||
                    ch.classList.contains('layer-toggles')) return;
                body.appendChild(ch);
            });
            card.appendChild(body);
        });
    }

    // Number each card on its own top-right corner so the index rides with it.
    cards.forEach((card, i) => {
        let num = card.querySelector('.chart-card-number');
        if (!num) {
            num = document.createElement('div');
            num.className = 'chart-card-number';
            card.appendChild(num);
        }
        num.textContent = `${i + 1} / ${count}`;
        card.style.transition = 'none';  // all motion is rAF-driven, never CSS
    });

    deck._cards = cards;
    // Mobile gesture pad: a transparent surface spanning the band from the
    // head card's top to the floor, so a swipe ANYWHERE in deck territory
    // (beside or below a short compact card) drives the deck - no dead
    // zones. Sits behind the cards; pointer-events only on mobile (CSS).
    let pad = deck.querySelector(':scope > .deck-gesture-pad');
    if (!pad) {
        pad = document.createElement('div');
        pad.className = 'deck-gesture-pad';
        deck.prepend(pad);
    }
    deck._pad = pad;
    // Visiting order for the loop. Default is a linear loop over card index; a
    // sort can replace this array (the loop runs over order positions, wrapping).
    deck._order = cards.map((_, i) => i);
    const start = (((deckActive[deck.id] ?? 0) % count) + count) % count;  // wrap-safe
    deck._deck = { activeIndex: deck._order[start] ?? start, count };
    deck._pos = start;   // position in order space, wraps [0, count)
    deck._raf = 0;
    deck._anchorTarget = deckAnchor[deck.id] ?? 0;   // 0 = rest on the floor (B)
    deck._anchor = deck._anchorTarget;

    if (!deck._deckBound) {
        bindDeckEvents(deck);
        deck._deckBound = true;
    }
    ensureVisibleLayout(deck);

    if (!window._deckGlobalBound) {
        window.addEventListener('resize', onDeckResize);
        window.addEventListener('keydown', onDeckKeydown);
        window._deckGlobalBound = true;
    }
}

// Lay out once the deck is actually visible/sized. On the Dynamics tab the deck
// can be built a frame before its tab content is painted, so a hidden deck
// (offsetParent null) defers to the next frame instead of measuring zero.
function ensureVisibleLayout(deck, tries = 0) {
    if (!deck || !deck._deck) return;
    if (deck.offsetParent === null && tries < 60) {
        requestAnimationFrame(() => ensureVisibleLayout(deck, tries + 1));
        return;
    }
    if (!isMobileDeck()) { deck._anchor = deck._anchorTarget = 1; }  // desktop: top-anchored
    measureDeck(deck);
    renderDeck(deck);
    applyDeckFocus(deck);  // a freshly visible/rebuilt deck consumes any pending focus
    deck._laidOut = true;             // measured while visible
    deck._laidW = window.innerWidth;  // remember the width it was laid out at
}

// On tab activation, lay out the deck ONLY if it needs it: never measured while
// visible (prefetch retry lapsed), or the viewport WIDTH changed since (e.g.
// rotation while this tab was hidden). An unchanged deck is left alone - re-
// measuring + re-rendering it on every switch flashed the cards (briefly
// vanishing) each time the tab opened. (Width, not height, so the mobile URL-bar
// height jitter never re-triggers it.)
export function relayoutDeckOnActivate(deckId) {
    const deck = document.getElementById(deckId);
    if (!deck || !deck._deck) return;
    if (deck._laidOut && deck._laidW === window.innerWidth) return;
    ensureVisibleLayout(deck);
}

// Cache card heights and (mobile) the band between the deck top and the floor.
// Measured here on layout/resize/rebuild - never in the motion hot path. Tall
// cards get capped to the band with inner-scroll so the whole card stays in
// bounds; that cap turns offsetHeight into the capped height we position with.
function measureDeck(deck) {
    const cards = deck._cards || [];
    const mobile = isMobileDeck();
    // Clear any prior lift FIRST so the rect reads the deck's true flow top, not
    // an already-lifted one - otherwise the lift would compound toward zero on
    // each resize/rebuild. Desktop never lifts; clearing it keeps cross-breakpoint
    // toggling clean.
    deck.style.marginTop = '';
    const top = deck.getBoundingClientRect().top;   // untransformed + unlifted -> natural flow top
    // Floor = the bottom of the VISIBLE viewport, clamped to the scroll region. Use
    // visualViewport (the actually-visible height, excludes the mobile URL bar) so
    // the deck never extends below the screen.
    const region = deck.closest('.tab-content');
    const regionBottom = region ? region.getBoundingClientRect().bottom : window.innerHeight;
    // A background warm measures for FUTURE display, but visualViewport reads the
    // live screen: typing on another tab (e.g. KB queries) shrinks it to the
    // keyboard's edge, and the warmed deck would keep that stunted band (the
    // width-only activation check never rechecks it). Hidden warm -> trust the
    // warm container's geometry, not the transient viewport.
    const warming = region && region.style.visibility === 'hidden';
    const visH = (!warming && window.visualViewport && window.visualViewport.height) || window.innerHeight;
    const floorY = Math.min(window.innerHeight, regionBottom, visH) - DECK_FLOOR_MARGIN;
    let lift = 0;
    if (mobile) {
        // Ceiling = bottom edge of the per-tab title TEXT itself (the .tab-header
        // <h2>: "Metrics" / "Learning Dynamics"). Lift the deck's top right up to
        // just under the title so a card pinned at slot A (headTop=0) sits a hair
        // below the title, shadowing everything else in the header - the
        // run-selector button and the metadata rows. Pulling back to B drops the
        // head down to reveal those again. Falls back to the whole header, then
        // the tab row, if no title text is present.
        const scope = deck.closest('.app-container') || document;
        const region = deck.closest('.tab-content') || scope;
        const ceil = region.querySelector('.tab-header h2')
            || region.querySelector('.tab-header')
            || scope.querySelector('.tab-nav');
        const ceilY = ceil ? ceil.getBoundingClientRect().bottom : top;
        lift = Math.max(0, Math.round(top - ceilY));
        if (lift > 0) deck.style.marginTop = `${-lift}px`;
    }
    const bandH = Math.max(140, Math.round(floorY - top + lift));   // ceiling -> floor
    deck._bandH = bandH;
    deck._lift = lift;
    // The UNIFORM slot height: the full band (ceiling -> floor) minus only the
    // fan trail. NOT minus the lift - in slot A the head sits at the ceiling, so
    // it gets the whole band; sizing to the dropped (B) height would collapse
    // cards early and force inner scrolling that needn't exist. On mobile this
    // is the SHORT-card slot: a card that needs more expands past it, consuming
    // the fan reserve up to the full band (see below). (Desktop lift is 0, so
    // this is unchanged there.)
    const fanReserve = Math.min(DECK_MAX_FAN, Math.max(0, cards.length - 1)) * DECK_PEEK;
    const usableH = Math.max(140, bandH - fanReserve);
    deck._cardH = cards.map(c => {
        const body = c.querySelector('.deck-card-scroll') || c;
        // Reset any prior cap/flex so we read the card's natural height. We just
        // cleared max-height, so drop renderDeck's _mh cache too (else it'd skip the
        // re-write thinking the cap is still applied).
        c.style.height = ''; c.style.maxHeight = ''; c.style.display = ''; c.style.flexDirection = ''; c._mh = undefined;
        body.style.maxHeight = ''; body.style.flex = ''; body.style.minHeight = '';
        body.style.overflowY = 'auto';
        const wrapper = mobile ? c.querySelector('.chart-wrapper') : null;
        if (wrapper) wrapper.style.height = '';   // reset to CSS default before measuring
        let h = c.offsetHeight || 0;
        // Mobile slot height is per-card: short cards share the uniform slot
        // (even fan bottoms); a card that overflows it EXPANDS into the fan
        // reserve, up to the full band, instead of inner-scrolling inside the
        // smaller slot. The fan previews just hide behind the expanded head -
        // a full card beats a peek strip. Desktop keeps the uniform cap.
        const capH = mobile && h > usableH ? bandH : usableH;
        // Charts shrink under pressure: when the card would overflow even its
        // expanded cap, compact the chart (down to DECK_MIN_CHART_H) so it stays
        // expressive but avoids an inner scroll. Chart.js (responsive) re-fits
        // the canvas when the wrapper resizes.
        if (wrapper && h > capH) {
            const chrome = h - wrapper.offsetHeight;   // title/subtitle/toggles/padding
            wrapper.style.height = `${Math.max(DECK_MIN_CHART_H, capH - chrome)}px`;
            h = c.offsetHeight || 0;                   // re-measure after compacting
        }
        // On mobile, pin EVERY card to its slot (not just the tall ones). Short
        // cards then end their boxes at the same height, so the fanned cards
        // behind the head share one even bottom edge - a constant-spacing
        // staircase down the floor, instead of ragged edges set by each card's
        // content. (Desktop keeps natural height: it grows the deck and peeks
        // upward, so even bottoms don't apply there.)
        // deck-compact cards (the business card) keep their natural height
        // everywhere: forcing them to the slot stretches them vertically.
        const compact = c.classList.contains('deck-compact');
        const pin = compact ? 0 : (mobile ? capH : (h > usableH ? usableH : 0));
        if (pin) {
            // Mobile forces an exact slot height (even fan bottoms); desktop
            // only caps an overflowing card (max-height), leaving natural height.
            if (mobile) c.style.height = `${pin}px`;
            c.style.maxHeight = `${pin}px`;
            if (body !== c) {
                c.style.display = 'flex';
                c.style.flexDirection = 'column';
                body.style.flex = '1 1 auto';
                body.style.minHeight = '0';
                body.style.maxHeight = 'none';   // flex governs height; override any CSS cap
            }
        }
        // _capped cards are re-sized per-frame by renderDeck (per-card slot).
        c._capped = !compact && (mobile || h > usableH);
        if (compact) return Math.min(h, usableH);
        return mobile ? capH : Math.min(h, usableH);
    });
    deck._usableH = usableH;
    if (mobile) {
        // Seat every non-head card's inner scroll at its bottom so the fan
        // strips read as natural card ends from first paint (crossings keep
        // them seated afterwards). The head keeps its own scroll position.
        const active = deck._deck ? deck._deck.activeIndex : -1;
        cards.forEach((c, i) => { if (i !== active) seatFanCard(deck, i); });
        deck.style.height = `${bandH}px`;
        // Clip at the floor (band bottom): the downward fan bottoms out here, so a card
        // taller than the floor is simply cut off at it instead of overshooting past the
        // fan. Purely visual - the scroll/seam mechanism is untouched.
        deck.style.overflow = 'hidden';
    } else {
        deck.style.overflow = '';   // desktop grows the deck + peeks upward; no clip
    }
}

// Render the deck at its current pos + anchor. Pure: cached heights + constants,
// only style writes, zero layout reads -> no forced reflow.
function renderDeck(deck) {
    const st = deck._deck;
    if (!st || !deck._cards) return;
    // First time the deck is actually positioned: fade it in (the CSS keeps it hidden
    // until now so the unpositioned card shape never flashes). rAF lets the final
    // transforms paint before the opacity transition starts.
    if (!deck._ready) {
        deck._ready = true;
        requestAnimationFrame(() => deck.classList.add('deck-ready'));
    }
    if (!isMobileDeck()) { renderDeckDesktop(deck); return; }

    const cards = deck._cards;
    const count = st.count;
    const order = deck._order || cards.map((_, i) => i);
    const pos = deck._pos;
    const bandH = deck._bandH || deck.clientHeight || 0;
    // Lift is the A/B anchor only (eased), NOT a per-card bob: the head holds a stable
    // vertical position while cycling so cards don't bounce. At rest (B) it sits below
    // the title strip (headTop = lift); engaging lifts it to the tab row (A, headTop=0).
    const a01 = deck._anchor < 0 ? 0 : deck._anchor > 1 ? 1 : deck._anchor;

    // Head is TOP-anchored, like desktop. The fan always peeks DOWN off the head toward
    // the floor, so it never rides up over the header; trailing cards fade out.
    const headTop = (deck._lift || 0) * (1 - a01);
    const availH = Math.round((bandH - headTop) / 2) * 2;
    // Per-card slot heights from measureDeck: short cards share the uniform slot
    // (which reserves the fan floor, so the fan staircase stays even); tall cards
    // expanded into the fan reserve up to the full band. The fan offset below
    // steps each card down by DECK_PEEK as it lifts into slot A.
    const H = deck._cardH || [];
    const fallbackH = Math.max(120, deck._usableH || availH);

    for (let k = 0; k < count; k++) {
        const card = cards[order[k]];
        const delta = cyclicDelta(k, pos, count);   // shortest path on the loop
        const a = delta < 0 ? -delta : delta;
        if (a > DECK_MAX_FAN + 1) {
            if (card._vis !== 'h') { card.style.visibility = 'hidden'; card._vis = 'h'; }
            if (card._pe !== 'none') { card.style.pointerEvents = 'none'; card._pe = 'none'; }
            if (card._act) { card.classList.remove('deck-active'); card._act = false; }
            continue;
        }
        const isHead = a < 0.5;
        // The fan is gated by the anchor: fully stacked in B (a01=0, every card's title
        // aligned on the same top-left point), fanning DOWN as it lifts into A (a01=1).
        // The offset clamps at the fan reserve (DECK_MAX_FAN steps): a 4th card seats
        // exactly behind the 3rd instead of stepping past the floor and leaking its
        // content below the staircase (desktop clamps the same way via peekUnits).
        const peek = delta > DECK_MAX_FAN ? DECK_MAX_FAN : delta;
        const top = headTop + peek * DECK_PEEK * a01;   // fan DOWN into the reserved floor
        // No scale shrink on mobile: scaling from the top origin would eat the DOWNWARD
        // peek. The offset alone gives the stacked fan, so each card's edge stays visible.
        const scale = 1;
        // Downward fan: upcoming cards (delta >= 0) stay opaque and peek below the
        // head as a preview; the passed card (delta < 0) fades out as it leaves.
        const opacity = delta >= 0 ? 1 : Math.max(0, 1 + delta);
        // transform + opacity change every frame; the rest flip only on boundary
        // crossings, so write them on-change only (cuts ~70% of the per-frame writes).
        card.style.transform = `translateY(${top.toFixed(2)}px) scale(${scale.toFixed(4)})`;
        card.style.opacity = opacity >= 1 ? '1' : opacity.toFixed(3);
        if (card._vis !== 'v') { card.style.visibility = 'visible'; card._vis = 'v'; }
        if (card._capped) {
            // Set both height and max-height to the card's slot (height forces
            // it; max-height alone would let short content stay short). The
            // expanded (full-band) height applies only AT THE HEAD: in the fan
            // every card renders at the uniform slot, so an expanded card never
            // overflows the staircase and covers its neighbors. The expansion
            // blends in over the last slot of approach (grow: 0 one slot away,
            // 1 seated) so the height never pops.
            const slotH = Math.max(120, H[order[k]] || fallbackH);
            const grow = a >= 1 ? 0 : 1 - a;
            const mh = `${Math.round(fallbackH + (Math.max(fallbackH, slotH) - fallbackH) * grow)}px`;
            if (card._mh !== mh) {
                card.style.height = mh;
                card.style.maxHeight = mh;
                card._mh = mh;
            }
        }
        // Z by UNCAPPED depth: a 4th card (seated at the same offset as the 3rd
        // once the peek clamps) must stack BELOW it - capping at rank tied their
        // z-indexes and let DOM order paint the deeper card's content on top.
        const zi = 100 - Math.round(a > DECK_MAX_FAN + 1 ? DECK_MAX_FAN + 1 : a);
        if (card._zi !== zi) { card.style.zIndex = String(zi); card._zi = zi; }
        const pe = isHead ? 'auto' : 'none';
        if (card._pe !== pe) { card.style.pointerEvents = pe; card._pe = pe; }
        // Only the head card animates its canvases (see deckCardParked); the rest
        // are occluded, so pausing them is the main mobile-lag win.
        if (card._act !== isHead) { card.classList.toggle('deck-active', isHead); card._act = isHead; }
    }
    if (deck._h !== bandH) { deck.style.height = `${bandH}px`; deck._h = bandH; }

    // Keep the gesture pad under the band below the head's top: never over
    // the header controls the deck's transparent top overlaps at rest (B).
    if (deck._pad) {
        const padY = Math.round(headTop);
        if (deck._padY !== padY) {
            deck._pad.style.transform = `translateY(${padY}px)`;
            deck._pad.style.height = `${Math.max(0, bandH - padY)}px`;
            deck._padY = padY;
        }
    }

    const k = ((Math.round(pos) % count) + count) % count;
    const idx = order[k];
    if (idx !== st.activeIndex) {
        const prev = st.activeIndex;
        st.activeIndex = idx;
        deckActive[deck.id] = k;   // persist the order slot (wrap-safe across rebuilds)
        seatEntering(deck, idx);
        seatFanCard(deck, prev);   // the outgoing head rests at its bottom in the fan
    }
}

// Desktop: UPWARD fan (rolodex / file cabinet). The title rides a fixed track so it
// never bounces with card size; the head's base rests on the floor (the deck bottom
// follows the head height). Upcoming cards peek their tops UP above the head, and the
// leaving card sinks below and fades.
function renderDeckDesktop(deck) {
    const st = deck._deck;
    const cards = deck._cards;
    const H = deck._cardH || [];
    const count = st.count;
    const order = deck._order || cards.map((_, i) => i);
    const pos = deck._pos;

    const below = Math.min(DECK_MAX_FAN, count - 1);   // cards fanned above the head
    // The title rides a FIXED track: headTop never depends on card height, so it
    // doesn't bounce as differently-sized cards take the head. The deck bottom then
    // hugs the interpolated head height, so the head still rests its base on the floor.
    const headTop = below * DECK_PEEK;                  // constant fan reserve above
    const k0 = ((Math.floor(pos) % count) + count) % count;
    const k1 = (k0 + 1) % count;
    const f = pos - Math.floor(pos);
    const headH = (H[order[k0]] || 0) * (1 - f) + (H[order[k1]] || 0) * f;
    const floor = headTop + headH;                     // head base sits on the floor

    for (let k = 0; k < count; k++) {
        const card = cards[order[k]];
        const delta = cyclicDelta(k, pos, count);
        const a = delta < 0 ? -delta : delta;
        if (a > DECK_MAX_FAN + 1) {
            if (card._vis !== 'h') { card.style.visibility = 'hidden'; card._vis = 'h'; }
            if (card._pe !== 'none') { card.style.pointerEvents = 'none'; card._pe = 'none'; }
            if (card._act) { card.classList.remove('deck-active'); card._act = false; }
            continue;
        }
        const capped = a > DECK_MAX_FAN ? DECK_MAX_FAN : a;
        // Upcoming cards (delta > 0) lift UP off the floor to peek above the head;
        // passed cards (delta < 0) drop below and fade out (the leaving ghost).
        // Clamp the upward peek to the reserved band so the deepest card's title
        // sits at the deck's top edge and never pops out above it - keeping the
        // title track smooth and every title aligned (the bottom may run longer).
        const peekUnits = delta < below ? delta : below;
        const y = headTop - peekUnits * DECK_PEEK;
        const scale = 1 - capped * DECK_SCALE_STEP;
        const opacity = delta >= 0 ? 1 : Math.max(0, 1 + delta);
        card.style.transform = `translateY(${y.toFixed(2)}px) scale(${scale.toFixed(4)})`;
        card.style.opacity = opacity >= 1 ? '1' : opacity.toFixed(3);
        if (card._vis !== 'v') { card.style.visibility = 'visible'; card._vis = 'v'; }
        const zi = 100 - Math.round(capped);
        if (card._zi !== zi) { card.style.zIndex = String(zi); card._zi = zi; }
        const pe = a < 0.5 ? 'auto' : 'none';
        if (card._pe !== pe) { card.style.pointerEvents = pe; card._pe = pe; }
        const isHead = a < 0.5;
        if (card._act !== isHead) { card.classList.toggle('deck-active', isHead); card._act = isHead; }
    }
    if (deck._h !== floor) { deck.style.height = `${floor.toFixed(1)}px`; deck._h = floor; }

    const k = ((Math.round(pos) % count) + count) % count;
    const idx = order[k];
    if (idx !== st.activeIndex) {
        st.activeIndex = idx;
        deckActive[deck.id] = k;
        if (deck._fanDown) seatEntering(deck, idx);
    }
}

// Seat a newly-active card's scroll body on the edge it enters from: the TOP when
// advancing forward, the BOTTOM when going backward, so reverse scrolling flows up
// through the previous card instead of jumping to its top. Runs once per slot
// crossing (not per frame), so the scrollHeight read here is off the hot path.
function seatEntering(deck, idx) {
    const card = deck._cards[idx];
    if (!card) return;
    if (deck._fanDown) {
        // Fan-down sheets live in a scrollable tab-content (desktop); pin it to the
        // top. Mobile deck tabs are overflow:hidden, so this is a no-op there.
        const sc = deck.closest('.tab-content');
        if (sc && sc.scrollTop !== 0) sc.scrollTop = 0;
    }
    const body = card.querySelector('.deck-card-scroll') || card;
    body.scrollTop = deck._cycleDir < 0 ? body.scrollHeight : 0;
}

// A fan card shows only its bottom strip below the head, so rest its inner
// scroll at the BOTTOM: the visible strip is then the content's natural end,
// not a mid-scroll crop. (The focused card is the opposite - it enters at the
// top via seatEntering/seatTarget.) Runs on slot crossings and re-measures,
// never per frame.
function seatFanCard(deck, idx) {
    const card = deck._cards && deck._cards[idx];
    if (!card) return;
    const body = card.querySelector('.deck-card-scroll') || card;
    body.scrollTop = body.scrollHeight;
}

// Reset the card at an order slot to the edge we're entering it from (TOP when going
// forward, BOTTOM when going back) - called the moment a transition BEGINS, before the
// card slides in. Looping a short carousel re-enters the same cards, so without this
// they'd peek/arrive at their stale scroll position.
function seatTarget(deck, slot, dir) {
    const st = deck._deck;
    if (!st || !deck._cards) return;
    const order = deck._order || deck._cards.map((_, i) => i);
    const idx = order[((Math.round(slot) % st.count) + st.count) % st.count];
    const card = deck._cards[idx];
    if (!card) return;
    const body = card.querySelector('.deck-card-scroll') || card;
    body.scrollTop = dir < 0 ? body.scrollHeight : 0;
}

// Signed distance from order-slot k to pos on the loop, biased toward the FORWARD
// (fan) side so small decks still fill the fan. The wrap point is the larger of n/2
// and the fan depth: on big decks this is n/2 (shortest path, unchanged); on small
// decks (e.g. the 3-sheet Identity deck) it keeps the upcoming cards on the positive
// side instead of letting them fall to the faded leaving side - so all of them show.
function cyclicDelta(k, pos, n) {
    if (n <= 1) return k - pos;
    const fan = Math.min(DECK_MAX_FAN, n - 1);
    const t = Math.max(n / 2, fan);
    const fd = ((k - pos) % n + n) % n;   // forward distance in [0, n)
    return fd > t ? fd - n : fd;
}

function setPos(deck, p) {
    const n = deck._deck.count;
    deck._pos = n > 0 ? ((p % n) + n) % n : 0;   // wrap: the loop has no ends
    renderDeck(deck);
}

function cancelMomentum(deck) {
    deck._posMode = null;                          // halt a slide
    deck._scrollMode = null; deck._scrollVel = 0;  // halt an inner-scroll coast
}

function isMobileDeck() {
    return typeof window !== 'undefined' && window.innerWidth <= 768;
}

// Smoothstep: slow at both ends, fast in the middle. Drives the card-transition seam
// so it pauses leaving the content edge and arriving at the next anchor, flipping
// quickly in between - the non-linear "wave" that couples scrolling to card-switching.
function smoothstep(t) {
    if (t <= 0) return 0;
    if (t >= 1) return 1;
    return t * t * (3 - 2 * t);
}

// Advance the card-to-card seam by `dy` finger px. pos eases base -> base+dir across
// DECK_SEAM via smoothstep; completing seats the next card, reversing returns to base.
function advanceSeam(deck, dy) {
    deck._seamAccum += dy;
    const dir = deck._seamDir;
    const mag = deck._seamAccum * dir;   // forward progress along the seam
    if (mag <= 0) { deck._seamAccum = 0; setPos(deck, deck._seamBase); return; }
    deck._cycleDir = dir;
    if (mag >= DECK_SEAM) { deck._seamAccum = 0; setPos(deck, deck._seamBase + dir); return; }
    setPos(deck, deck._seamBase + dir * smoothstep(mag / DECK_SEAM));
}

// A/B lift: cycling forward lifts to A (fan opens); a sustained backward pull drops to B.
function seamAnchor(deck, dy) {
    if (dy > 0) { deck._backAccum = 0; setAnchor(deck, 1); }
    else { deck._backAccum += -dy; if (deck._backAccum > DECK_DROP_THRESHOLD) setAnchor(deck, 0); }
}

// Monotonic ease-out: fast then firmly decelerating, no overshoot, no tail.
function easeOutCubic(t) {
    if (t <= 0) return 0;
    if (t >= 1) return 1;
    const u = 1 - t;
    return 1 - u * u * u;
}

// Slide pos to a target slot over DECK_CYCLE_DUR via easeOutCubic. The target is kept
// UNWRAPPED (signed distance from the current pos), so the interpolation is monotonic
// toward an already-correct slot - it cannot reverse. Settles exactly on the slot.
/**
 * Slide a deck to a card identified by metric `key` (preferred, exact) or, as a
 * fallback, its visible `title`. Returns false if no such card exists in the
 * deck (e.g. the metric has no data yet, so no card was rendered) - the caller
 * can then refresh and retry. Used to deep-link from a KB search result.
 * @param {string|HTMLElement} deckOrId - Deck element or its id
 * @param {{key?: string, title?: string}} target
 */
export function jumpToDeckCard(deckOrId, { key, title } = {}) {
    const deck = typeof deckOrId === 'string' ? document.getElementById(deckOrId) : deckOrId;
    if (!deck || !deck._cards || !deck._deck) return false;
    const wantKey = key != null ? String(key) : null;
    const wantTitle = (title || '').trim();
    const cardIdx = deck._cards.findIndex(c =>
        (wantKey && c.dataset.cardKey === wantKey) ||
        (wantTitle && c.querySelector('.chart-title')?.textContent.trim() === wantTitle));
    if (cardIdx < 0) return false;
    const pos = deck._order.indexOf(cardIdx);
    if (pos < 0) return false;
    setAnchor(deck, 1);
    slideTo(deck, pos);
    return true;
}

// Event-driven deck focus: a deep-link (e.g. a KB card click) registers a
// target; the deck consumes it whenever it becomes visible/measured or rebuilds
// (see ensureVisibleLayout) or on tab activation. This unifies programmatic
// focus with user swiping - both end in slideTo - and avoids racing a deck that
// is still being built. A user gesture clears any pending request.
let _pendingFocus = null;  // { deckId, key, title }

export function requestDeckFocus(deckId, { key, title } = {}) {
    _pendingFocus = { deckId, key, title };
}

export function isDeckFocusPending(deckId) {
    return !!_pendingFocus && (!deckId || _pendingFocus.deckId === deckId);
}

export function clearDeckFocus() {
    _pendingFocus = null;
}

export function applyDeckFocus(deckOrId) {
    if (!_pendingFocus) return false;
    const deck = typeof deckOrId === 'string' ? document.getElementById(deckOrId) : deckOrId;
    if (!deck || deck.id !== _pendingFocus.deckId) return false;
    if (jumpToDeckCard(deck, { key: _pendingFocus.key, title: _pendingFocus.title })) {
        _pendingFocus = null;
        return true;
    }
    return false;
}

function slideTo(deck, target) {
    deck._posFrom = deck._pos;
    const n = deck._deck.count;
    // Shortest signed path on the loop. _pos is wrapped to [0, n) but targets are
    // computed in an unwrapped frame (e.g. seamBase - 1 = -1 while _pos wrapped to
    // n - 0.3), so the raw difference can span nearly the whole stack - the slide
    // then "flings" a full wraparound to land one slot away. Normalize first.
    let dist = target - deck._pos;
    if (n > 1) {
        dist = ((dist % n) + n) % n;
        if (dist > n / 2) dist -= n;
    }
    deck._posDist = dist;
    deck._posTargetRaw = deck._pos + dist;         // unwrapped; wrap only on read/settle
    deck._cycleDir = deck._posDist >= 0 ? 1 : -1;  // seat entering cards on the right edge
    // Reset the entering card's scroll up front (only when actually changing cards) so it
    // arrives at the edge we came from, not its stale position - matters for short loops.
    const fromSlot = ((Math.round(deck._pos) % n) + n) % n;
    const toSlot = ((Math.round(target) % n) + n) % n;
    if (toSlot !== fromSlot) seatTarget(deck, target, deck._cycleDir);
    // Longer slide for chained flicks so each card still reads as it passes (the lift
    // arc pulses once per card), but bounded so it never feels sluggish.
    const mag = Math.abs(deck._posDist);
    deck._posDur = DECK_CYCLE_DUR * Math.min(2.5, Math.max(1, Math.sqrt(mag)));
    deck._posT0 = performance.now();
    deck._posMode = 'slide';
    runDeckRAF(deck);
}

// Inner-scroll momentum: coast the focused card's body from a SNAPSHOTTED scroll range
// (read once here, off the hot path), driven purely by JS state in runDeckRAF.
function startScrollMomentum(deck) {
    const b = deck._scrollBody;
    if (!b || !b.isConnected) return;
    deck._scrollMax = b.scrollHeight - b.clientHeight;   // snapshot bounds (one read)
    if (deck._scrollMax <= 0) return;
    deck._scrollPos = b.scrollTop;
    deck._scrollMode = 'coast';
    runDeckRAF(deck);
}

// ── One rAF drives every axis ───────────────────────────────────────────────
// Duration-based pos slide + anchor lift (easeOutCubic, monotonic, no overshoot/
// tail) and a friction-decel inner-scroll coast. All pure JS state + style WRITES;
// no layout reads (scroll bounds were snapshotted). renderDeck runs only when pos or
// anchor actually moved - a pure scroll coast skips it (cards are static). The loop
// self-stops when nothing is moving.
function runDeckRAF(deck) {
    if (deck._raf) return;
    let last = 0;
    const tick = (t) => {
        if (!deck.isConnected || !deck._deck) { deck._raf = 0; return; }
        if (!last) last = t;
        let dt = t - last; last = t;
        if (dt > 32) dt = 32;
        const n = deck._deck.count;
        const wrap = p => n > 0 ? ((p % n) + n) % n : 0;
        let moving = false, render = false;

        if (deck._posMode === 'slide') {
            const e = (t - deck._posT0) / (deck._posDur || DECK_CYCLE_DUR);
            if (e >= 1) {
                deck._pos = wrap(deck._posTargetRaw);
                deck._posMode = null;
            } else {
                deck._pos = wrap(deck._posFrom + deck._posDist * easeOutCubic(e));
                moving = true;
            }
            render = true;
        }

        if (isMobileDeck() && deck._anchor !== deck._anchorTarget) {
            const e = (t - deck._anchorT0) / DECK_ANCHOR_DUR;
            if (e >= 1) {
                deck._anchor = deck._anchorTarget;
            } else {
                deck._anchor = deck._anchorFrom + (deck._anchorTarget - deck._anchorFrom) * easeOutCubic(e);
                moving = true;
            }
            render = true;
        }

        if (deck._scrollMode === 'coast') {
            const b = deck._scrollBody;
            if (!b || !b.isConnected) {
                deck._scrollMode = null;
            } else {
                deck._scrollVel *= Math.exp(-dt / SCROLL_TAU);
                let sp = deck._scrollPos + deck._scrollVel * dt;
                if (sp <= 0) { sp = 0; deck._scrollMode = null; }                 // hard clamp at edges
                else if (sp >= deck._scrollMax) { sp = deck._scrollMax; deck._scrollMode = null; }
                else if (Math.abs(deck._scrollVel) < SCROLL_MIN_VEL) { deck._scrollMode = null; }
                else moving = true;
                deck._scrollPos = sp;
                b.scrollTop = sp;   // write only; bounds were snapshotted
            }
        }

        if (render) renderDeck(deck);   // skip when only the scroll body is coasting (cards static)
        deck._raf = moving ? requestAnimationFrame(tick) : 0;
    };
    deck._raf = requestAnimationFrame(tick);
}

// "needed" anchor flip. Desktop stays pinned to the top (no floor model).
function setAnchor(deck, target) {
    if (!isMobileDeck()) { deck._anchor = deck._anchorTarget = 1; return; }
    target = target ? 1 : 0;
    deckAnchor[deck.id] = target;
    // Already heading there? Let the in-flight ease finish UNINTERRUPTED.
    // seamAnchor calls this every touchmove; without this guard each frame
    // rebased _anchorFrom/_anchorT0, restarting the 240ms curve continuously so
    // the lift crept along with the finger instead of running its own ease. We
    // only (re)start the curve when the target actually flips (a direction
    // reversal past the threshold), so A<->B is a smooth curve, not input-coupled.
    if (deck._anchorTarget === target) return;
    deck._anchorTarget = target;
    deck._anchorFrom = deck._anchor;
    deck._anchorT0 = performance.now();
    runDeckRAF(deck);
}

function bindDeckEvents(deck) {
    // ── Wheel / trackpad: accumulate pos continuously, snap when it stops ──
    deck.addEventListener('wheel', (e) => {
        const st = deck._deck;
        if (!st) return;
        const dir = e.deltaY > 0 ? 1 : -1;
        deck._scrollMode = null; deck._scrollVel = 0;   // a wheel cancels any in-flight touch-scroll coast
        const body = deck._cards[st.activeIndex] && deck._cards[st.activeIndex].querySelector('.deck-card-scroll');
        const roomIn = b => !b ? 0 : (dir > 0 ? b.scrollHeight - b.clientHeight - b.scrollTop : b.scrollTop);

        // A wheel "session" runs until DECK_WHEEL_SETTLE ms after the last notch
        // (deck._wheelT pending = session live). Lock it to scroll or cycle on its
        // FIRST notch by whether the focused card can scroll in this direction, so a
        // scroll session reaches the card's edge and STOPS - it never bleeds into
        // cycling. Re-wheel from the edge to cycle.
        if (!deck._wheelT) {
            deck._wheelMode = roomIn(body) > 1 ? 'scroll' : 'cycle';
        }

        if (deck._wheelMode === 'scroll' && body) {
            const room = roomIn(body);
            if (room > 0) body.scrollTop += Math.sign(e.deltaY) * Math.min(Math.abs(e.deltaY), room);
            // Inner scroll never shifts A/B - only an outer-container cycle does.
            e.preventDefault();
            clearTimeout(deck._wheelT);
            deck._wheelT = setTimeout(() => { deck._wheelT = 0; }, DECK_WHEEL_SETTLE);
            return;
        }

        e.preventDefault();
        cancelMomentum(deck);
        clearDeckFocus();  // wheel navigation cancels pending deep-link focus
        // Desktop fans UP, so scrolling UP advances toward the cards fanned above (you
        // scroll TOWARD a card, not away from it). Mobile fans down: conventional sign.
        const cycleSign = isMobileDeck() ? 1 : -1;
        setAnchor(deck, dir > 0 ? 1 : 0);
        deck._cycleDir = cycleSign * dir;   // pos direction; seats the entering card right
        setPos(deck, deck._pos + cycleSign * e.deltaY / DECK_WHEEL_STEP);   // wraps
        clearTimeout(deck._wheelT);
        deck._wheelT = setTimeout(() => { slideTo(deck, Math.round(deck._pos)); deck._wheelT = 0; }, DECK_WHEEL_SETTLE);
    }, { passive: false });

    // ── Touch: 1:1 drag, release -> momentum -> snap; anchor eases alongside ──
    // One continuous, non-linear axis. Each move does EITHER inner scroll OR a card
    // seam - never both. While the head card has content room, the swipe scrolls it and
    // STOPS at the edge (a wall = pause at the content end). Past the edge it drives a
    // smoothstep seam to the next card (slow-flip-slow = pause, flip, pause at anchor),
    // then the new card scrolls. So the order is: scroll -> pause -> flip -> pause ->
    // scroll, all from one finger axis.
    let dragging = false, lastY = 0, lastT = 0, fvel = 0, travel = 0;
    deck.addEventListener('touchstart', (e) => {
        cancelMomentum(deck);
        clearTimeout(deck._wheelT); deck._wheelT = 0;
        dragging = true;
        lastY = e.touches[0].clientY;
        lastT = e.timeStamp;
        fvel = 0;
        travel = 0;
        deck._seamAccum = 0;       // 0 = seated on a card; non-zero = mid card-transition
        deck._backAccum = 0;       // sustained downward travel, gating the drop to B
        deck._scrollVel = 0;       // reset inner-scroll velocity for this gesture
        deck._scrollBody = null;
        deck._anchorLocked = false; // set once this gesture flips A<->B, then no cycling
        // The GRIP: every card's standard stack-navigation handle. It is the
        // title chrome where there is one, plus a fixed band at the card's top
        // (DECK_GRIP_H) so title-less cards (e.g. the Identity tab's bare
        // sheets) get the exact same handle. A grip drag never inner-scrolls -
        // it drives the seam and the A/B anchor like a body drag at its content
        // edge, so long content can be swiped past from the top. Taps are
        // unaffected (only moves are interpreted). Cards opt out of the band
        // with the "deck-no-grip" class.
        const head = deck._deck && deck._cards[deck._deck.activeIndex];
        let grip = !!e.target.closest('.chart-title, .chart-subtitle, .chart-card-number')
            && !e.target.closest('.deck-card-scroll');
        if (!grip && head && !head.classList.contains('deck-no-grip')) {
            const r = head.getBoundingClientRect();   // once per gesture, off the hot path
            grip = lastY >= r.top && lastY <= r.top + DECK_GRIP_H;
        }
        deck._gripDrag = grip;
    }, { passive: true });

    deck.addEventListener('touchmove', (e) => {
        if (!dragging) return;
        const st = deck._deck;
        if (!st) return;
        const y = e.touches[0].clientY;
        let dy = lastY - y;            // finger up -> dy > 0 -> advance / scroll content down
        lastY = y;
        if (dy === 0) return;
        travel += dy < 0 ? -dy : dy;
        const dt = Math.max(1, e.timeStamp - lastT);
        lastT = e.timeStamp;
        fvel = dy / dt;               // px/ms finger speed (for the fling-commit on release)
        e.preventDefault();           // the deck owns the gesture (touch-action: none)

        // Grip drag: the standard handle. Same anchor mechanics as a body drag
        // at its content edge - at B an upward pull slots the deck to A (and
        // locks, no cycling in the same gesture); otherwise it drives the seam
        // both ways, with a sustained downward pull dropping back to B. The
        // only difference from a body drag is that the grip never inner-scrolls.
        if (deck._gripDrag) {
            if (deck._seamAccum === 0 && deck._anchorTarget === 0 && dy > 0) {
                const before = deck._anchorTarget;
                seamAnchor(deck, dy);
                if (deck._anchorTarget !== before) deck._anchorLocked = true;
                return;
            }
            if (deck._anchorLocked) return;
            if (deck._seamAccum === 0) {
                deck._seamBase = ((Math.round(deck._pos) % st.count) + st.count) % st.count;
                deck._seamDir = dy > 0 ? 1 : -1;
                seatTarget(deck, deck._seamBase + deck._seamDir, deck._seamDir);
            }
            seamAnchor(deck, dy);
            advanceSeam(deck, dy);
            return;
        }

        // Mid card-transition: the seam owns this move (no inner scroll until it settles).
        if (deck._seamAccum !== 0) {
            seamAnchor(deck, dy);
            advanceSeam(deck, dy);
            return;
        }

        // Seated: while the head card's content has room in this direction, the
        // gesture ALWAYS scrolls it - however fast the flick. Cycling begins only
        // at the content edge, so a long document (e.g. Identity's Arguments)
        // can't be skipped past accidentally; the momentum coast makes reaching
        // its bottom cheap.
        const body = deck._cards[st.activeIndex] && deck._cards[st.activeIndex].querySelector('.deck-card-scroll');
        const room = body
            ? (dy > 0 ? Math.max(0, body.scrollHeight - body.clientHeight - body.scrollTop)
                      : Math.max(0, body.scrollTop))
            : 0;
        if (room > 0) {
            const take = dy > 0 ? Math.min(dy, room) : -Math.min(-dy, room);
            const before = body.scrollTop;
            body.scrollTop += take;
            if (body.scrollTop !== before) {   // actually scrolled -> reading the card
                deck._scrollVel = deck._scrollVel * 0.6 + (take / dt) * 0.4;   // EMA px/ms
                deck._scrollBody = body;
                return;
            }
            // Body wouldn't move despite reported room (stuck card) -> fall through to seam.
        }

        // Anchor flips are exclusive with cycling: a pull whose intent is to move
        // the A<->B anchor must NEVER scroll through cards. At A a sustained
        // downward pull (past the content edge) drops to B; at B an upward pull
        // lifts to A. Either way drive the anchor alone - no seam - so you can
        // reposition the anchor without the deck cycling underneath you.
        // At B, an upward pull lifts straight back to A (no cycling). At A,
        // downward is NOT captured here - it falls through to cycle cards in both
        // directions, and only a SUSTAINED downward pull (accumulated in
        // seamAnchor past DECK_DROP_THRESHOLD) commits the drop to B. So you can
        // scroll cards from A either way, and reaching B takes a deliberate pull.
        if (deck._anchorTarget === 0 && dy > 0) {
            const before = deck._anchorTarget;
            seamAnchor(deck, dy);
            // A completed A<->B flip HARD-GATES card cycling for the rest of this
            // gesture, so the same drag can't overshoot past the new slot into the
            // next card. Lift the finger and drag again to cycle.
            if (deck._anchorTarget !== before) deck._anchorLocked = true;
            return;
        }

        // Gate: this gesture already flipped the anchor (A<->B) - don't let the
        // same continued drag also cycle cards.
        if (deck._anchorLocked) return;

        // Content edge (or a stuck card): begin the seam to the next/previous card.
        deck._seamBase = ((Math.round(deck._pos) % st.count) + st.count) % st.count;
        deck._seamDir = dy > 0 ? 1 : -1;
        seatTarget(deck, deck._seamBase + deck._seamDir, deck._seamDir);   // reset before it slides in
        seamAnchor(deck, dy);
        advanceSeam(deck, dy);
    }, { passive: false });

    deck.addEventListener('touchend', () => {
        if (!dragging) return;
        dragging = false;
        if (deck._seamAccum !== 0) {
            // Settle the seam: commit the flip past the midpoint or on a flick, else snap back.
            const dir = deck._seamDir;
            const mag = deck._seamAccum * dir;
            const commit = mag > DECK_SEAM * 0.5 || fvel * dir > DECK_SEAM_FLING;
            deck._seamAccum = 0;
            if (commit) clearDeckFocus();  // a real swipe cancels pending deep-link focus
            slideTo(deck, deck._seamBase + (commit ? dir : 0));
        } else if (travel >= 6 && deck._scrollBody && Math.abs(deck._scrollVel) > SCROLL_MIN_VEL) {
            startScrollMomentum(deck);   // pure inner scroll -> coast the card body
        }
    }, { passive: true });
}

function visibleDeck() {
    return Array.from(document.querySelectorAll('.chart-deck')).find(d => d.offsetParent !== null) || null;
}

function onDeckResize() {
    document.querySelectorAll('.chart-deck').forEach(d => {
        if (d.offsetParent === null || !d._deck) return;
        if (!isMobileDeck()) { d._anchor = d._anchorTarget = 1; }
        measureDeck(d);
        renderDeck(d);
    });
}

// Re-measure now-visible decks. Needed when a deck was built while its tab was
// hidden (background prefetch) and ensureVisibleLayout's retry budget lapsed.
export const relayoutVisibleDecks = onDeckResize;

function onDeckKeydown(e) {
    const deck = visibleDeck();
    if (!deck || !deck._deck) return;
    const tag = (document.activeElement && document.activeElement.tagName) || '';
    if (tag === 'INPUT' || tag === 'TEXTAREA') return;
    const cur = Math.round(deck._pos);
    // Desktop fans UP: ArrowUp advances toward the cards above. Mobile fans down.
    const fwdKey = isMobileDeck() ? 'ArrowDown' : 'ArrowUp';
    const backKey = isMobileDeck() ? 'ArrowUp' : 'ArrowDown';
    let target = cur;
    if (e.key === fwdKey) target = cur + 1;
    else if (e.key === backKey) target = cur - 1;
    else return;
    e.preventDefault();
    clearDeckFocus();  // manual navigation cancels any pending deep-link focus
    deck._cycleDir = (e.key === fwdKey) ? 1 : -1;
    setAnchor(deck, deck._cycleDir > 0 ? 1 : 0);
    slideTo(deck, target);   // wraps at the ends
}

/**
 * Render the metrics header (title, selectors, refresh button) once
 */
function renderMetricsHeader(container, runs) {
    let selectorHTML = '';

    // Build run selector
    if (state.research.historicalRuns.length > 0) {
        const selectedCount = state.research.selectedHistoricalRuns.length;
        const totalCount = state.research.historicalRuns.length;

        selectorHTML = `
            <div class="run-selector-wrapper">
                <button class="run-selector-button" id="run-selector-btn">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
                        <path d="M8.515 1.019A7 7 0 0 0 8 1V0a8 8 0 0 1 .589.022l-.074.997zm2.004.45a7.003 7.003 0 0 0-.985-.299l.219-.976c.383.086.76.2 1.126.342l-.36.933zm1.37.71a7.01 7.01 0 0 0-.439-.27l.493-.87a8.025 8.025 0 0 1 .979.654l-.615.789a6.996 6.996 0 0 0-.418-.302zm1.834 1.79a6.99 6.99 0 0 0-.653-.796l.724-.69c.27.285.52.59.747.91l-.818.576zm.744 1.352a7.08 7.08 0 0 0-.214-.468l.893-.45a7.976 7.976 0 0 1 .45 1.088l-.95.313a7.023 7.023 0 0 0-.179-.483zm.53 2.507a6.991 6.991 0 0 0-.1-1.025l.985-.17c.067.386.106.778.116 1.17l-1 .025zm-.131 1.538c.033-.17.06-.339.081-.51l.993.123a7.957 7.957 0 0 1-.23 1.155l-.964-.267c.046-.165.086-.332.12-.501zm-.952 2.379c.184-.29.346-.594.486-.908l.914.405c-.16.36-.345.706-.555 1.038l-.845-.535zm-.964 1.205c.122-.122.239-.248.35-.378l.758.653a8.073 8.073 0 0 1-.401.432l-.707-.707z"/>
                        <path d="M8 1a7 7 0 1 0 4.95 11.95l.707.707A8.001 8.001 0 1 1 8 0v1z"/>
                        <path d="M7.5 3a.5.5 0 0 1 .5.5v5.21l3.248 1.856a.5.5 0 0 1-.496.868l-3.5-2A.5.5 0 0 1 7 9V3.5a.5.5 0 0 1 .5-.5z"/>
                    </svg>
                    Runs (${selectedCount}/${totalCount})
                </button>
                <div class="run-selector-dropdown" id="run-selector-dropdown" style="display: none;">
                    <div class="run-selector-header">Compare Runs</div>
                    <div class="run-selector-list">
                        ${state.research.historicalRuns.map((run, idx) => {
                            const isSelected = state.research.selectedHistoricalRuns.includes(run.hash);
                            const color = chartLineColor(idx);
                            const timeLabel = formatRelativeTime(run.metrics_updated);
                            const badges = [];
                            if (run.is_current) badges.push('active');
                            if (run.agentName) badges.push(run.agentName);
                            if (run.source === 'remote') badges.push('remote');
                            const badgeHTML = badges.length > 0
                                ? ` <span style="opacity: 0.6; font-size: 0.8em;">(${badges.join(', ')})</span>`
                                : '';
                            const stepsLabel = run.source === 'remote'
                                ? timeLabel
                                : `${run.num_steps} steps &middot; ${timeLabel}`;
                            return `
                                <label class="run-selector-item">
                                    <input type="checkbox" ${isSelected ? 'checked' : ''} data-run-hash="${run.hash}">
                                    <span class="run-color-indicator" style="background: ${color};"></span>
                                    <span class="run-label">${run.hash}${badgeHTML}</span>
                                    <span class="run-steps">${stepsLabel}</span>
                                </label>
                            `;
                        }).join('')}
                    </div>
                </div>
            </div>
        `;
    }

    const totalPoints = runs.reduce((sum, r) => sum + (r.metadata?.num_points || 0), 0);

    const metadataHTML = `
        <span id="metrics-metadata-comparing"><strong>Comparing:</strong> ${runs.length} run${runs.length !== 1 ? 's' : ''}</span>
        <span id="metrics-metadata-points"><strong>Total Points:</strong> ${totalPoints}</span>
    `;

    const headerHTML = createTabHeader({
        title: 'Metrics',
        additionalContent: selectorHTML,
        buttons: [
            pdfButton('download-pdf-research'),
        ],
        metadata: metadataHTML
    });

    container.innerHTML = headerHTML + '<div id="metrics-charts-area" style="margin-top: 2rem;"></div>';
}

/**
 * Update the metadata subtitle without rebuilding the header
 */
function updateMetricsMetadata(runs) {
    const totalPoints = runs.reduce((sum, r) => sum + (r.metadata?.num_points || 0), 0);

    const comparingEl = document.getElementById('metrics-metadata-comparing');
    const pointsEl = document.getElementById('metrics-metadata-points');
    if (comparingEl) comparingEl.innerHTML = `<strong>Comparing:</strong> ${runs.length} run${runs.length !== 1 ? 's' : ''}`;
    if (pointsEl) pointsEl.innerHTML = `<strong>Total Points:</strong> ${totalPoints}`;
}

/**
 * Create multi-agent comparison chart
 */
function createRunComparisonChart(canvasId, label, runs, metricKey) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    if (charts[canvasId]) {
        charts[canvasId].destroy();
    }

    const theme = getContextTheme(ctx);
    const { textColor, gridColor, tooltipBg } = getThemeColors(theme);

    const datasets = runs.map((run) => {
        const metrics = run.metrics;
        const steps = metrics.steps || [];
        const values = metrics[metricKey] || [];

        if (!values.some(v => v !== null)) return null;

        let data = steps.map((step, i) => ({
            x: step,
            y: values[i]
        })).filter(point => point.y !== null)
          .sort((a, b) => a.x - b.x);

        // For validation metrics, remove consecutive duplicate values.
        // ``is_validation`` is set by the backend registry, so new
        // validation metrics get the same treatment without a JS edit.
        const isVal =
            state.research.metricRegistry?.[metricKey]?.chart?.is_validation;
        if (isVal) {
            data = data.filter((point, i) => {
                if (i === 0) return true;
                return point.y !== data[i - 1].y;
            });
        }

        // Color by the run's position in the full historical list, so it
        // stays consistent with the Runs selector regardless of which subset
        // is currently selected.
        const colorIdx = state.research.historicalRuns.findIndex(r => r.hash === run.hash);
        const color = chartLineColor(colorIdx >= 0 ? colorIdx : 0);

        return {
            label: run.hash,
            data: data,
            borderColor: color,
            backgroundColor: color + '20',
            borderWidth: 2,
            pointRadius: 0,
            pointHoverRadius: 5,
            pointHoverBackgroundColor: color,
            pointHoverBorderColor: '#fff',
            pointHoverBorderWidth: 2,
            tension: 0,
            fill: false
        };
    }).filter(d => d !== null);

    charts[canvasId] = new Chart(ctx, {
        type: 'line',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    display: runs.length > 1,
                    position: 'top',
                    labels: {
                        color: textColor,
                        usePointStyle: true,
                        padding: 15,
                        font: { size: 11 }
                    }
                },
                tooltip: {
                    backgroundColor: tooltipBg,
                    titleColor: textColor,
                    bodyColor: textColor,
                    borderColor: gridColor,
                    borderWidth: 1,
                    padding: 12,
                    displayColors: true,
                    callbacks: {
                        title: (ctx) => `Step ${ctx[0].parsed.x}`,
                        label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(4)}`
                    }
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    title: {
                        display: true,
                        text: 'Training Step',
                        color: textColor,
                        font: { size: 13, weight: '500' }
                    },
                    ticks: {
                        color: textColor,
                        maxTicksLimit: 10
                    },
                    grid: { color: gridColor }
                },
                y: {
                    title: {
                        display: true,
                        text: label,
                        color: textColor,
                        font: { size: 13, weight: '500' }
                    },
                    ticks: {
                        color: textColor,
                        callback: (value) => {
                            if (Math.abs(value) >= 1000) {
                                return value.toExponential(2);
                            }
                            return value.toFixed(3);
                        }
                    },
                    grid: { color: gridColor }
                }
            }
        }
    });
}

/**
 * Create bar chart for token counts
 */
function createTokensBarChart(canvasId, label, runs, metricKey) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    if (charts[canvasId]) {
        charts[canvasId].destroy();
    }

    const theme = getContextTheme(ctx);
    const { textColor, gridColor, tooltipBg } = getThemeColors(theme);

    // Extract latest value for each run
    const data = runs.map((run) => {
        const values = run.metrics[metricKey] || [];

        let latestValue = null;
        for (let i = values.length - 1; i >= 0; i--) {
            if (values[i] !== null) {
                latestValue = values[i];
                break;
            }
        }

        // Color by the run's position in the full historical list, so it
        // stays consistent with the Runs selector regardless of which subset
        // is currently selected.
        const colorIdx = state.research.historicalRuns.findIndex(r => r.hash === run.hash);
        const color = chartLineColor(colorIdx >= 0 ? colorIdx : 0);

        return {
            label: run.hash,
            value: latestValue,
            color: color
        };
    }).filter(d => d.value !== null);

    charts[canvasId] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.map(d => d.label),
            datasets: [{
                label: label,
                data: data.map(d => d.value),
                backgroundColor: data.map(d => d.color + '80'),
                borderColor: data.map(d => d.color),
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',  // Horizontal bars
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: tooltipBg,
                    titleColor: textColor,
                    bodyColor: textColor,
                    borderColor: gridColor,
                    borderWidth: 1,
                    padding: 12,
                    callbacks: {
                        label: (ctx) => `${ctx.parsed.x.toFixed(6)} billion tokens`
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Tokens (Billions)',
                        color: textColor,
                        font: { size: 13, weight: '500' }
                    },
                    ticks: {
                        color: textColor,
                        callback: (value) => {
                            if (value >= 1) {
                                return value.toFixed(2);
                            }
                            return value.toFixed(6);
                        }
                    },
                    grid: { color: gridColor }
                },
                y: {
                    ticks: {
                        color: textColor,
                        font: { size: 12 }
                    },
                    grid: { display: false }
                }
            }
        }
    });
}

/**
 * Create sampling weights chart showing current document weights as horizontal bars
 */
function createSamplingWeightsChart(canvasId, dataMetrics) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    // Destroy existing
    if (charts[canvasId]) {
        charts[canvasId].destroy();
    }

    // Get colors for the appropriate theme context (hybrid overlay or normal)
    const theme = getContextTheme(ctx);
    const { textColor, gridColor, tooltipBg } = getThemeColors(theme);

    // Collect all documents and their latest weights
    const documentData = [];

    dataMetrics.forEach((agent, agentIdx) => {
        if (!agent.data_metrics?.sampling_weights) return;

        const samplingWeights = agent.data_metrics.sampling_weights;

        // Get latest weight for each document
        Object.keys(samplingWeights).forEach((docName, docIdx) => {
            const weights = samplingWeights[docName];

            // Find the last non-null weight
            let latestWeight = null;
            for (let i = weights.length - 1; i >= 0; i--) {
                if (weights[i] !== null) {
                    latestWeight = weights[i];
                    break;
                }
            }

            if (latestWeight !== null) {
                const agentLabel = dataMetrics.length > 1 ? ` (${agent.name})` : '';
                const color = chartLineColor(docIdx);

                documentData.push({
                    label: `${docName}${agentLabel}`,
                    value: latestWeight,
                    color: color
                });
            }
        });
    });

    // Sort by weight value (descending) for better visualization
    documentData.sort((a, b) => b.value - a.value);

    // Calculate max weight for dynamic scaling
    const maxWeight = Math.max(...documentData.map(d => d.value));
    const scaledMax = Math.min(maxWeight * 1.1, 1.0);  // Add 10% padding, cap at 1.0

    charts[canvasId] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: documentData.map(d => d.label),
            datasets: [{
                label: 'Sampling Weight',
                data: documentData.map(d => d.value),
                backgroundColor: documentData.map(d => d.color + '80'),
                borderColor: documentData.map(d => d.color),
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',  // Horizontal bars
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: tooltipBg,
                    titleColor: textColor,
                    bodyColor: textColor,
                    borderColor: gridColor,
                    borderWidth: 1,
                    padding: 12,
                    callbacks: {
                        label: (ctx) => {
                            const percentage = (ctx.parsed.x * 100).toFixed(2);
                            return `Weight: ${ctx.parsed.x.toFixed(4)} (${percentage}%)`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Sampling Weight',
                        color: textColor,
                        font: { size: 13, weight: '500' }
                    },
                    min: 0,
                    max: scaledMax,  // Dynamic max based on data
                    ticks: {
                        color: textColor,
                        callback: (value) => {
                            const percentage = (value * 100).toFixed(0);
                            return `${percentage}%`;
                        }
                    },
                    grid: { color: gridColor }
                },
                y: {
                    ticks: {
                        color: textColor,
                        font: { size: 12 }
                    },
                    grid: { display: false }
                }
            }
        }
    });
}

/**
 * Render step slider for navigating through training steps in heatmap
 * Only renders once to avoid recursion
 */
function renderStepSlider(canvasId, sortedSteps, currentStepIndex, agents, chartData) {
    const container = document.getElementById(`layer-toggles-${canvasId}`);
    if (!container) return;

    // Check if slider already exists
    const existingSlider = document.getElementById(`step-slider-${canvasId}`);
    if (existingSlider) {
        // Just update the value display
        const valueDisplay = document.getElementById(`step-value-${canvasId}`);
        const currentStep = sortedSteps[currentStepIndex];
        const maxStep = sortedSteps[sortedSteps.length - 1];
        if (valueDisplay) {
            valueDisplay.textContent = `${currentStep} / ${maxStep}`;
        }
        existingSlider.value = currentStepIndex;
        return;
    }

    const currentStep = sortedSteps[currentStepIndex];
    const minStep = sortedSteps[0];
    const maxStep = sortedSteps[sortedSteps.length - 1];

    container.innerHTML = `
        <div style="display: flex; align-items: center; gap: 1rem; width: 100%;">
            <label style="font-weight: 500; white-space: nowrap;">Training Step:</label>
            <input
                type="range"
                id="step-slider-${canvasId}"
                min="0"
                max="${sortedSteps.length - 1}"
                value="${currentStepIndex}"
                style="flex: 1; min-width: 200px;"
            />
            <span id="step-value-${canvasId}" style="font-weight: 600; min-width: 80px; text-align: right;">
                ${currentStep} / ${maxStep}
            </span>
        </div>
    `;

    // Wire up slider event (only once)
    const slider = document.getElementById(`step-slider-${canvasId}`);
    const valueDisplay = document.getElementById(`step-value-${canvasId}`);

    if (slider && valueDisplay) {
        slider.addEventListener('input', (e) => {
            const newIndex = parseInt(e.target.value);
            const newStep = sortedSteps[newIndex];
            valueDisplay.textContent = `${newStep} / ${maxStep}`;

            // Update state
            layerSelectionState[canvasId].stepIndex = newIndex;

            // Update chart data without recreating the entire chart
            updateHeatmapData(canvasId, sortedSteps, newIndex, chartData);
        });
    }
}

/**
 * Update heatmap data for a new step without recreating the chart
 */
function updateHeatmapData(canvasId, sortedSteps, stepIndex, chartData) {
    const chart = charts[canvasId];
    if (!chart) return;

    const currentStep = sortedSteps[stepIndex];
    const { layerExpertMetrics, layers, maxExperts } = chartData;

    // Recalculate scatter data for new step
    // X-axis = layers, Y-axis = experts (swapped)
    const newData = [];
    layers.forEach(layerNum => {
        const expertMetrics = layerExpertMetrics.get(layerNum);
        if (!expertMetrics) return;

        for (let expertNum = 0; expertNum < maxExperts; expertNum++) {
            const agentData = expertMetrics.get(expertNum);
            if (!agentData || agentData.length === 0) {
                newData.push({ x: layerNum, y: expertNum, v: null });  // Swapped
                continue;
            }

            let totalWeight = 0;
            let count = 0;

            agentData.forEach(({ steps, values }) => {
                const stepIdx = steps.indexOf(currentStep);
                if (stepIdx >= 0 && values[stepIdx] !== null) {
                    totalWeight += values[stepIdx];
                    count++;
                }
            });

            const avgWeight = count > 0 ? totalWeight / count : null;
            newData.push({ x: layerNum, y: expertNum, v: avgWeight });  // Swapped
        }
    });

    // Update chart data and colors
    chart.data.datasets[0].data = newData;
    chart.data.datasets[0].pointBackgroundColor = newData.map(d => getHeatmapColor(d.v));
    chart.data.datasets[0].pointBorderColor = newData.map(d => getHeatmapColor(d.v));

    // Update title with new step
    const uniformWeight = maxExperts > 0 ? 1.0 / maxExperts : 0.5;
    const uniformPct = (uniformWeight * 100).toFixed(1);
    chart.options.plugins.title.text = `Step ${currentStep} | Uniform: ${uniformPct}% per expert`;

    // Update without animation for instant feedback
    chart.update('none');
}

/**
 * Get color for heatmap cell based on routing weight value
 */
function getHeatmapColor(value) {
    if (value === null || value === undefined) {
        return 'rgba(180, 180, 180, 0.3)';  // Light grey for missing data
    }

    // Color scale: grey (0) → green (1)
    // Grey: rgb(220, 220, 220) - very low usage
    // Green: rgb(11, 154, 109) - high usage (matches CONSTANTS.RUN_COLORS green)

    const greyR = 220, greyG = 220, greyB = 220;
    const greenR = 11, greenG = 154, greenB = 109;

    // Linear interpolation from grey to green
    const r = Math.round(greyR + (greenR - greyR) * value);
    const g = Math.round(greyG + (greenG - greyG) * value);
    const b = Math.round(greyB + (greenB - greyB) * value);

    return `rgb(${r}, ${g}, ${b})`;
}

/**
 * Cache for model config to avoid repeated fetches
 */
let modelConfigCache = null;

/**
 * Fetch model config from API (returns YAML)
 */
async function fetchModelConfig() {
    if (modelConfigCache) {
        return modelConfigCache;
    }

    try {
        const response = await fetch('/api/config');
        if (!response.ok) {
            console.warn('[Heatmap] Failed to fetch model config');
            return null;
        }

        // Parse YAML as text (simple key-value extraction)
        const yamlText = await response.text();
        const config = {};

        // Extract depth and num_layers from YAML
        const depthMatch = yamlText.match(/^depth:\s*(\d+)/m);
        const numLayersMatch = yamlText.match(/^num_layers:\s*(\d+)/m);

        if (depthMatch) config.depth = parseInt(depthMatch[1]);
        if (numLayersMatch) config.num_layers = parseInt(numLayersMatch[1]);

        modelConfigCache = config;
        return config;
    } catch (error) {
        console.warn('[Heatmap] Error fetching model config:', error);
        return null;
    }
}

/**
 * Calculate reasoning steps info from model config
 * Returns { hasReasoningSteps: boolean, numActualLayers: number, numReasoningSteps: number }
 */
async function calculateReasoningSteps() {
    const config = await fetchModelConfig();

    if (!config || !config.num_layers) {
        return { hasReasoningSteps: false, numActualLayers: 0, numReasoningSteps: 1 };
    }

    const depth = config.depth;
    const numLayers = config.num_layers;

    // Check if depth is defined and greater than num_layers
    // If depth = num_layers * reasoning_steps, then reasoning_steps = depth / num_layers
    if (depth && depth > numLayers) {
        const numReasoningSteps = depth / numLayers;

        // Only use reasoning steps if it's a clean division
        if (Number.isInteger(numReasoningSteps)) {
            return {
                hasReasoningSteps: true,
                numActualLayers: numLayers,
                numReasoningSteps: numReasoningSteps
            };
        }
    }

    // No reasoning steps detected
    return {
        hasReasoningSteps: false,
        numActualLayers: numLayers,
        numReasoningSteps: 1
    };
}

/**
 * Get label for a layer index, accounting for reasoning steps
 * layerIndex: the raw layer index from metrics (e.g., 0-11)
 * Returns formatted label like "L0 R0" or "Layer 0"
 */
function getLayerLabel(layerIndex, reasoningInfo) {
    if (!reasoningInfo || !reasoningInfo.hasReasoningSteps) {
        return `Layer ${layerIndex}`;
    }

    // layerIndex = actualLayer + reasoningStep * numActualLayers
    // So: actualLayer = layerIndex % numActualLayers
    //     reasoningStep = floor(layerIndex / numActualLayers)
    const actualLayer = layerIndex % reasoningInfo.numActualLayers;
    const reasoningStep = Math.floor(layerIndex / reasoningInfo.numActualLayers);

    return `L${actualLayer} R${reasoningStep}`;
}

/**
 * Create expert routing heatmap showing convergence across layers and experts
 * Uses scatter plot with colored squares to create heatmap effect
 */
async function createExpertRoutingChart(canvasId, agents) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    const theme = getContextTheme(ctx);
    const { textColor, gridColor, tooltipBg } = getThemeColors(theme);

    // Parse metrics to detect layers, experts, and steps
    const layerExpertMetrics = new Map();
    let maxExperts = 0;
    let allSteps = new Set();

    agents.forEach((agent) => {
        const metrics = agent.metrics;
        const steps = metrics.steps || [];
        steps.forEach(s => allSteps.add(s));

        Object.keys(metrics).forEach(k => {
            const match = k.match(/^layer_(\d+)_expert_(\d+)_routing_weight$/);
            if (!match) return;

            const layerNum = parseInt(match[1]);
            const expertNum = parseInt(match[2]);

            if (!layerExpertMetrics.has(layerNum)) {
                layerExpertMetrics.set(layerNum, new Map());
            }
            if (!layerExpertMetrics.get(layerNum).has(expertNum)) {
                layerExpertMetrics.get(layerNum).set(expertNum, []);
            }
            layerExpertMetrics.get(layerNum).get(expertNum).push({
                agent: agent.name,
                metricKey: k,
                steps: steps,
                values: metrics[k]
            });
            maxExperts = Math.max(maxExperts, expertNum + 1);
        });
    });

    const layers = Array.from(layerExpertMetrics.keys()).sort((a, b) => a - b);
    const sortedSteps = Array.from(allSteps).sort((a, b) => a - b);

    if (layers.length === 0 || sortedSteps.length === 0) return;

    // Calculate reasoning steps from model config
    const reasoningInfo = await calculateReasoningSteps();
    console.log('[Heatmap] Reasoning info:', reasoningInfo);
    console.log('[Heatmap] Model config:', modelConfigCache);

    // Initialize or get current step index
    if (!layerSelectionState[canvasId]) {
        layerSelectionState[canvasId] = { stepIndex: sortedSteps.length - 1 };  // Default to latest step
    }

    const currentStepIndex = layerSelectionState[canvasId].stepIndex;
    const currentStep = sortedSteps[currentStepIndex];

    // Store chart data for slider updates
    const chartData = { layerExpertMetrics, layers, maxExperts, reasoningInfo };

    // Render step slider control (only once)
    renderStepSlider(canvasId, sortedSteps, currentStepIndex, agents, chartData);

    // Transform data for current step into scatter plot points
    // X-axis = layers, Y-axis = experts (swapped from before)
    const scatterData = [];

    layers.forEach(layerNum => {
        const expertMetrics = layerExpertMetrics.get(layerNum);
        if (!expertMetrics) return;

        for (let expertNum = 0; expertNum < maxExperts; expertNum++) {
            const agentData = expertMetrics.get(expertNum);
            if (!agentData || agentData.length === 0) {
                scatterData.push({
                    x: layerNum,  // Swapped: layer on X
                    y: expertNum,  // Swapped: expert on Y
                    v: null
                });
                continue;
            }

            // For multi-agent: average weights across agents at this step
            let totalWeight = 0;
            let count = 0;

            agentData.forEach(({ steps, values }) => {
                const stepIdx = steps.indexOf(currentStep);
                if (stepIdx >= 0 && values[stepIdx] !== null) {
                    totalWeight += values[stepIdx];
                    count++;
                }
            });

            const avgWeight = count > 0 ? totalWeight / count : null;

            scatterData.push({
                x: layerNum,  // Swapped: layer on X
                y: expertNum,  // Swapped: expert on Y
                v: avgWeight
            });
        }
    });

    // Destroy existing chart
    if (charts[canvasId]) {
        charts[canvasId].destroy();
    }

    // Calculate uniform weight reference
    const uniformWeight = maxExperts > 0 ? 1.0 / maxExperts : 0.5;
    const uniformPct = (uniformWeight * 100).toFixed(1);

    // Create heatmap using scatter plot with colored square points
    charts[canvasId] = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Routing Weight',
                data: scatterData,
                pointStyle: 'rect',
                // Dynamic cell size to fill the space
                pointRadius: (context) => {
                    const chart = context.chart;
                    const chartArea = chart.chartArea;
                    if (!chartArea) return 15;  // Default during initialization

                    // Calculate cell size to fill columns/rows completely
                    const width = chartArea.right - chartArea.left;
                    const height = chartArea.bottom - chartArea.top;
                    const cellWidth = width / layers.length;
                    const cellHeight = height / maxExperts;

                    // Use smaller of the two to ensure squares fit
                    return Math.min(cellWidth, cellHeight) / 2;  // Divide by 2 because radius
                },
                // Hover radius should match normal radius (no shrinking!)
                pointHoverRadius: (context) => {
                    const chart = context.chart;
                    const chartArea = chart.chartArea;
                    if (!chartArea) return 15;

                    const width = chartArea.right - chartArea.left;
                    const height = chartArea.bottom - chartArea.top;
                    const cellWidth = width / layers.length;
                    const cellHeight = height / maxExperts;

                    return Math.min(cellWidth, cellHeight) / 2;  // Same size as normal
                },
                pointBackgroundColor: scatterData.map(d => getHeatmapColor(d.v)),
                pointBorderColor: scatterData.map(d => getHeatmapColor(d.v)),
                pointHoverBorderColor: textColor,
                pointHoverBorderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: `Step ${currentStep} | Uniform: ${uniformPct}% per expert`,
                    color: textColor,
                    font: { size: 12, weight: '500' },
                    padding: { bottom: 10 }
                },
                tooltip: {
                    backgroundColor: tooltipBg,
                    titleColor: textColor,
                    bodyColor: textColor,
                    borderColor: gridColor,
                    borderWidth: 1,
                    padding: 12,
                    callbacks: {
                        title() {
                            return `Step ${currentStep}`;
                        },
                        label(context) {
                            const dataPoint = context.raw;
                            const layer = dataPoint.x;  // Swapped
                            const expert = dataPoint.y;  // Swapped
                            const weight = dataPoint.v;
                            const layerLabel = getLayerLabel(layer, reasoningInfo);
                            if (weight === null) {
                                return `${layerLabel}, Expert ${expert}: No data`;
                            }
                            return `${layerLabel}, Expert ${expert}: ${(weight * 100).toFixed(2)}%`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom',
                    title: {
                        display: true,
                        text: reasoningInfo.hasReasoningSteps ? 'Layers/Reasoning Steps' : 'Layer',
                        color: textColor,
                        font: { size: 13, weight: '500' }
                    },
                    min: Math.min(...layers) - 0.5,
                    max: Math.max(...layers) + 0.5,
                    ticks: {
                        color: textColor,
                        stepSize: 1,
                        callback: (value) => {
                            if (!Number.isInteger(value) || !layers.includes(value)) return '';
                            return getLayerLabel(value, reasoningInfo);
                        }
                    },
                    grid: { color: gridColor, lineWidth: 0.5 }
                },
                y: {
                    type: 'linear',
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Expert',
                        color: textColor,
                        font: { size: 13, weight: '500' }
                    },
                    min: -0.5,
                    max: maxExperts - 0.5,
                    ticks: {
                        color: textColor,
                        stepSize: 1,
                        callback: (value) => Number.isInteger(value) ? value : ''
                    },
                    grid: { color: gridColor, lineWidth: 0.5 }
                }
            }
        }
    });
}

/**
 * Create multi-expert chart for any expert-indexed metric (e.g., architecture selection)
 * Generic version that works with any keyPattern
 */
function createMultiExpertChart(canvasId, title, yAxisLabel, agents, keyPattern, options = {}) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    // Destroy existing
    if (charts[canvasId]) {
        charts[canvasId].destroy();
    }

    const theme = getContextTheme(ctx);
    const { textColor, gridColor, tooltipBg } = getThemeColors(theme);

    const allDatasets = [];
    let maxExperts = 0;

    agents.forEach((agent, agentIdx) => {
        const metrics = agent.metrics;
        const steps = metrics.steps || [];

        console.log(`[createMultiExpertChart] Agent: ${agent.name}, Steps: ${steps.length}, First: ${steps[0]}, Last: ${steps[steps.length-1]}`);

        // Find all metrics matching the pattern
        const expertKeys = Object.keys(metrics).filter(k => k.match(keyPattern));
        console.log(`[createMultiExpertChart] Found ${expertKeys.length} keys matching pattern:`, expertKeys);
        maxExperts = Math.max(maxExperts, expertKeys.length);

        expertKeys.forEach((expertKey) => {
            // Extract expert number
            const match = expertKey.match(/expert[_/](\d+)/);
            const expertNum = match ? match[1] : '0';
            const values = metrics[expertKey] || [];

            console.log(`[createMultiExpertChart] Key: ${expertKey}, Values length: ${values.length}, First: ${values[0]}, Last: ${values[values.length-1]}`);

            const data = steps.map((step, i) => ({
                x: step,
                y: values[i]
            })).filter(point => point.y !== null)
              .sort((a, b) => a.x - b.x);

            console.log(`[createMultiExpertChart] Data points after filtering: ${data.length}, First x: ${data[0]?.x}, Last x: ${data[data.length-1]?.x}`);

            const color = chartLineColor(parseInt(expertNum));
            const label = agents.length > 1 ? `${agent.name} - Expert ${expertNum}` : `Expert ${expertNum}`;

            allDatasets.push({
                label: label,
                data: data,
                borderColor: color,
                backgroundColor: color + '20',
                borderWidth: 2,
                pointRadius: 0,
                pointHoverRadius: 5,
                pointHoverBackgroundColor: color,
                pointHoverBorderColor: '#fff',
                pointHoverBorderWidth: 2,
                tension: options.stepped ? 0 : 0.3,
                stepped: options.stepped || false,
                fill: false
            });
        });
    });

    if (allDatasets.length === 0) {
        return;
    }

    // Calculate x-axis bounds from all data
    let minX = Infinity;
    let maxX = -Infinity;
    allDatasets.forEach(dataset => {
        dataset.data.forEach(point => {
            minX = Math.min(minX, point.x);
            maxX = Math.max(maxX, point.x);
        });
    });

    console.log(`[createMultiExpertChart] Creating chart with X bounds: [${minX}, ${maxX}], ${allDatasets.length} datasets`);

    charts[canvasId] = new Chart(ctx, {
        type: 'line',
        data: { datasets: allDatasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        color: textColor,
                        usePointStyle: true,
                        padding: 12,
                        font: { size: 11 }
                    }
                },
                tooltip: {
                    backgroundColor: tooltipBg,
                    titleColor: textColor,
                    bodyColor: textColor,
                    borderColor: gridColor,
                    borderWidth: 1,
                    padding: 12
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    title: {
                        display: true,
                        text: 'Training Step',
                        color: textColor,
                        font: { size: 13, weight: '500' }
                    },
                    min: minX,
                    max: maxX,
                    ticks: {
                        color: textColor,
                        maxTicksLimit: 10,
                        precision: 0
                    },
                    grid: { color: gridColor },
                    beginAtZero: false
                },
                y: {
                    type: 'linear',
                    title: {
                        display: true,
                        text: yAxisLabel,
                        color: textColor,
                        font: { size: 13, weight: '500' }
                    },
                    min: 0,
                    ticks: {
                        color: textColor,
                        callback: options.stepped ? (value) => value.toFixed(0) : (value) => `${value.toFixed(0)}%`
                    },
                    grid: { color: gridColor }
                }
            }
        }
    });
}

/**
 * Destroy all charts (for cleanup)
 */
export function destroyAllCharts() {
    Object.keys(charts).forEach(key => {
        if (charts[key]) {
            charts[key].destroy();
            delete charts[key];
        }
    });
}
