/**
 * Praxis Web - Chart.js Integration
 * Full Chart.js implementation for Research tab
 */

import { state, CONSTANTS } from './state.js';
import { fetchAPI } from './api.js';
import { createTabHeader } from './components.js';

// Chart instances storage (exported for hybrid mode)
export const charts = {};

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
 * Update all chart colors dynamically without redrawing
 * Pure functional approach: applies theme colors to existing charts
 * Much faster than destroying and recreating charts
 */
export function updateChartColors() {
    const { textColor, gridColor, tooltipBg } = getThemeColors();

    Object.values(charts).forEach(chart => {
        if (!chart) return;

        // Update scale colors (axes)
        if (chart.options.scales) {
            Object.values(chart.options.scales).forEach(scale => {
                if (scale.title) scale.title.color = textColor;
                if (scale.ticks) scale.ticks.color = textColor;
                if (scale.grid) scale.grid.color = gridColor;
            });
        }

        // Update legend colors
        if (chart.options.plugins?.legend?.labels) {
            chart.options.plugins.legend.labels.color = textColor;
        }

        // Update title colors (for charts with subtitles)
        if (chart.options.plugins?.title) {
            chart.options.plugins.title.color = textColor;
        }

        // Update tooltip colors
        if (chart.options.plugins?.tooltip) {
            chart.options.plugins.tooltip.backgroundColor = tooltipBg;
            chart.options.plugins.tooltip.titleColor = textColor;
            chart.options.plugins.tooltip.bodyColor = textColor;
            chart.options.plugins.tooltip.borderColor = gridColor;
        }

        // Apply updates instantly without animation
        chart.update('none');
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
 * Fetches run on tab activation and on explicit user-initiated refreshes
 * (refresh button, run-selector change). There is no background polling;
 * charts only rebuild when the user asks for it.
 *
 * @param {boolean} force - If true, re-fetch even if already loaded.
 */
export async function loadResearchMetricsWithCharts(force = false) {
    if (state.research.loaded && !force) return;

    const container = document.getElementById('research-container');
    if (!container) return;

    const chartsArea = document.getElementById('metrics-charts-area');
    if (chartsArea) {
        chartsArea.innerHTML = '<div class="loading-placeholder">Loading metrics...</div>';
    } else {
        container.innerHTML = '<div class="loading-placeholder">Loading metrics...</div>';
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

        renderMetricsCharts({
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
};

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
        <div class="chart-card" data-deck-index="${i}">
            <div class="chart-title">${config.title}</div>
            ${config.description ? `<div class="chart-subtitle">${config.description}</div>` : ''}
            ${stepSliderHTML}
            <div class="chart-wrapper">
                <canvas id="${config.canvasId}"></canvas>
            </div>
        </div>
        `;
    }).join('');

    chartsHTML += '</div>';

    chartsArea.innerHTML = chartsHTML;

    // Render charts after DOM update. Each config's `type` selects a
    // renderer from the registry; unknown types fall back to a line chart.
    setTimeout(() => {
        const ctx = { runs, dataMetrics };
        availableMetrics.forEach(config => {
            (METRIC_RENDERERS[config.type] || METRIC_RENDERERS.line)(config, ctx);
        });
        initChartDeck('chart-deck');
    }, 10);
}

// ============================================================================
// CARD-SWITCHER CAROUSEL
// Chart cards are stacked in place; one is active (front, full size) while the
// next few peek out behind it. Wheel/swipe/arrows cycle the deck without
// growing the page. activeIndex persists across re-renders (deck DOM is
// rebuilt on every metrics refresh).
// ============================================================================

// Per-deck state lives on the element (deck._deck). activeIndex also persists
// in this module map keyed by deck id, so a card stays selected across the DOM
// rebuilds that happen on every metrics refresh. Any number of decks can be
// live (Research, Dynamics); the visible one wins keyboard/resize.
const deckActive = {};
const deckSlot = {};  // head slot per deck (0 = top .. DECK_MAX_FAN = bottom)

const DECK_PEEK = 34;          // fan/title peek and the gap between the two head slots
const DECK_SCALE_STEP = 0.04;
const DECK_MAX_FAN = 3;        // ahead cards fanned below the head
const DECK_STEP = 100;         // wheel px per card; accumulated -> momentum
const DECK_SWIPE_STEP = 55;    // touch px per card (more direct than the wheel)
const DECK_MAX_PER_EVENT = 3;  // base per-event card cap (lifted as energy builds)
const DECK_STICK_FACTOR = 3;   // toward-top stick reserve = this * step at zero energy
const DECK_ENERGY_TAU = 220;   // ms EMA time-constant: swipe energy halves ~every 150ms idle
const DECK_ENERGY_FULL = 3;    // step-units of accumulated motion that fully melts the stick

export function initChartDeck(deck, opts = {}) {
    if (typeof deck === 'string') deck = document.getElementById(deck);
    if (!deck) return;
    // fan-down decks (long documents) keep the head on top and let sheets hang
    // below, so a tall card can't float up over the header.
    deck._fanDown = !!opts.fanDown;
    const cards = deck.querySelectorAll('.chart-card');
    const count = cards.length;
    if (count === 0) return;

    // Number each card on its own top-right corner (a "collector's" stamp) so
    // the index rides with the card instead of floating in one fixed spot.
    cards.forEach((card, i) => {
        let num = card.querySelector('.chart-card-number');
        if (!num) {
            num = document.createElement('div');
            num.className = 'chart-card-number';
            card.appendChild(num);
        }
        num.textContent = `${i + 1} / ${count}`;
    });

    const prev = deckActive[deck.id] ?? 0;
    deck._deck = { activeIndex: Math.max(0, Math.min(prev, count - 1)), count };
    deck._headSlot = deckSlot[deck.id] ?? 0;  // 0 = top .. DECK_MAX_FAN = bottom

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

// Lay out once the deck is actually visible/sized. On the Dynamics tab the
// deck can be built a frame before its tab content is painted, so a hidden
// deck (offsetParent null) defers to the next frame instead of silently
// skipping layout and never fanning (Research recovers via its metric poll;
// Dynamics inits once, so it needs this).
function ensureVisibleLayout(deck, tries = 0) {
    if (!deck || !deck._deck) return;
    if (deck.offsetParent === null && tries < 60) {
        requestAnimationFrame(() => ensureVisibleLayout(deck, tries + 1));
        return;
    }
    layoutDeck(deck);
}

function layoutDeck(deck) {
    if (!deck || deck.offsetParent === null) return;  // absent or hidden
    const st = deck._deck;
    if (!st) return;
    const cards = Array.from(deck.querySelectorAll('.chart-card'));
    if (cards.length === 0) return;

    applyWalletShift(deck);  // re-apply current latch w/ fresh header height

    // Mobile always fans strictly downward (head pinned at top): the two-slot
    // head-walk would slide the focused card down and fan earlier cards upward,
    // which fights the latched focal anchor. Body-capping only touches
    // .deck-card-scroll, so chart cards pass through untouched.
    if (deck._fanDown || window.innerWidth <= 768) {
        // Fan strictly downward: head on top, the rest hang below it as
        // top-anchored sheets. Tall documents fall down (more weight) and can
        // never rise over the header. No above-fan, no head shift.
        //
        // Cap each sheet's body from its real on-screen top down to the bottom
        // of the viewport (minus the fan peek and a little padding), so a long
        // document scrolls internally instead of running off-screen.
        const activeBody = cards[st.activeIndex].querySelector('.deck-card-scroll');
        if (activeBody) {
            const top = activeBody.getBoundingClientRect().top;
            const pad = 24 + DECK_MAX_FAN * DECK_PEEK;
            const avail = Math.max(160, Math.round(window.innerHeight - top - pad));
            cards.forEach(c => {
                const b = c.querySelector('.deck-card-scroll');
                if (b) b.style.maxHeight = `${avail}px`;
            });
        }
        const heights = cards.map(c => c.offsetHeight);
        const activeH = heights[st.activeIndex];
        let maxRank = 0;
        cards.forEach((card, i) => {
            const offset = i - st.activeIndex;
            if (offset === 0) {
                card.style.transformOrigin = 'top center';
                card.style.transform = 'translateY(0) scale(1)';
                card.style.opacity = '1';
                card.style.zIndex = '100';
                card.style.pointerEvents = 'auto';
                return;
            }
            const rank = Math.abs(offset);
            if (rank > DECK_MAX_FAN) {
                card.style.transform = 'translateY(0) scale(0.97)';
                card.style.opacity = '0';
                card.style.zIndex = '90';
                card.style.pointerEvents = 'none';
                return;
            }
            card.style.transformOrigin = 'top center';
            card.style.transform = `translateY(${rank * DECK_PEEK}px) scale(${1 - rank * DECK_SCALE_STEP})`;
            card.style.opacity = '1';
            card.style.zIndex = String(100 - rank);
            card.style.pointerEvents = 'none';
            maxRank = Math.max(maxRank, rank);
        });
        deck.style.height = `${activeH + maxRank * DECK_PEEK}px`;
        const c = deck.querySelector('.chart-deck-counter');
        if (c) c.textContent = `${st.activeIndex + 1} / ${st.count}`;
        return;
    }

    // The head rests at slot s (0 = top .. DECK_MAX_FAN = bottom); it walks one
    // slot per scroll toward the direction's end. Cards before the head peek
    // above it (titles, anchored to the head top); cards after peek below it
    // (edges, anchored to the head BOTTOM so a tall head can't swallow the fan).
    const s = Math.max(0, Math.min(deck._headSlot ?? 0, DECK_MAX_FAN));
    // Batch-read heights first to avoid per-card read/write reflow churn; the
    // below-fan needs each card's height to anchor its peek to the head bottom.
    const heights = cards.map(c => c.offsetHeight);
    const activeH = heights[st.activeIndex];
    const headY = s * DECK_PEEK;
    const headBottom = headY + activeH;
    let belowDepth = 0;

    cards.forEach((card, i) => {
        const offset = i - st.activeIndex;
        if (offset === 0) {
            card.style.transformOrigin = 'top center';
            card.style.transform = `translateY(${headY}px) scale(1)`;
            card.style.opacity = '1';
            card.style.zIndex = '100';
            card.style.pointerEvents = 'auto';
            return;
        }
        const slot = s + offset;  // behind -> above the head, ahead -> below
        if (slot < 0 || slot > DECK_MAX_FAN) {
            card.style.transformOrigin = 'top center';
            card.style.transform = `translateY(${headY}px) scale(0.97)`;
            card.style.opacity = '0';
            card.style.zIndex = '90';
            card.style.pointerEvents = 'none';
            return;
        }
        const depth = Math.abs(offset);
        let y;
        if (offset < 0) {
            // behind: peek above the head top by depth*PEEK (title shows)
            y = headY - depth * DECK_PEEK;
            card.style.transformOrigin = 'top center';
        } else {
            // ahead: peek depth*PEEK below the head BOTTOM, regardless of either
            // card's height (anchor the card's bottom, not the deck top). Clamp
            // to >= 0 so a card taller than the head can't get a negative offset
            // and rise above the deck top, overflowing over the Layers row /
            // header above it - a tall card falls downward instead.
            y = Math.max(0, headBottom + depth * DECK_PEEK - heights[i]);
            card.style.transformOrigin = 'bottom center';
            belowDepth = Math.max(belowDepth, depth);
        }
        card.style.transform = `translateY(${y}px) scale(${1 - depth * DECK_SCALE_STEP})`;
        // Full opacity for any visible fan card - depth reads from scale and
        // stacking, not transparency, so the peeking titles stay crisp white
        // instead of fading to grey.
        card.style.opacity = '1';
        card.style.zIndex = String(100 - depth);
        card.style.pointerEvents = 'none';
    });

    deck.style.height = `${headBottom + belowDepth * DECK_PEEK}px`;

    const counter = deck.querySelector('.chart-deck-counter');
    if (counter) counter.textContent = `${st.activeIndex + 1} / ${st.count}`;
}

// Coalesce rapid advances into one layout per animation frame: several wheel
// events in a frame update activeIndex synchronously but re-target the
// transition only once, which keeps fast scrolling smooth instead of janky.
function scheduleLayout(deck) {
    if (deck._layoutRAF) return;
    deck._layoutRAF = requestAnimationFrame(() => {
        deck._layoutRAF = 0;
        layoutDeck(deck);
    });
}

// Sticky-edge step: how much scroll/swipe distance it takes to leave the
// current card. A plain card uses the base step (a light stick); a card with a
// long inner document sticks harder (up to 2.5x), so you decelerate and slot
// instead of overshooting when its content loops out.
function deckStep(deck, base) {
    const st = deck._deck;
    if (!st) return base;
    const card = deck.querySelectorAll('.chart-card')[st.activeIndex];
    const body = card && card.querySelector('.deck-card-scroll');
    let factor = 1;
    if (body && body.clientHeight > 0) {
        factor = Math.max(1, Math.min(2.5, body.scrollHeight / body.clientHeight));
    }
    return base * factor;
}

// Mobile wallet shift, LATCHED: slot A (deck pulled up over the header, title
// anchored at the focal point) stays latched once you move off the top - it
// does NOT release just because cycling lands you back on the first card.
// Release to slot B (header readable) only on a deliberate return to rest:
// pulling back past the top of the first card. Desktop: no shift, no latch.
function applyWalletShift(deck) {
    if (!deck || !deck._deck) return;
    if (typeof window === 'undefined' || window.innerWidth > 768) {
        deck.style.transform = '';
        return;
    }
    if (!deck._walletA) { deck.style.transform = ''; return; }  // slot B
    const sc = deck.closest('.tab-content');
    const header = sc && sc.querySelector('.tab-header');
    const hH = header ? header.offsetHeight : 0;
    deck.style.transform = `translateY(${-hH}px)`;  // slot A
}
function walletToA(deck) { deck._walletA = true; applyWalletShift(deck); }
function walletToB(deck) { deck._walletA = false; deck.style.transform = ''; }

function advanceDeckBy(deck, delta) {
    const st = deck._deck;
    if (!st || delta === 0) return false;
    const next = Math.max(0, Math.min(st.activeIndex + delta, st.count - 1));
    if (next === st.activeIndex) return false;
    // Walk the head one slot per card toward the scroll direction's end (top on
    // up, bottom on down), so reversing direction cycles it through the
    // in-between slots instead of jumping 1 <-> N. Fan-down decks keep the head
    // pinned to the top.
    if (!deck._fanDown) {
        const steps = Math.abs(delta);
        const cur = deck._headSlot ?? 0;
        deck._headSlot = delta > 0
            ? Math.min(DECK_MAX_FAN, cur + steps)
            : Math.max(0, cur - steps);
        deckSlot[deck.id] = deck._headSlot;
    }
    st.activeIndex = next;
    deckActive[deck.id] = next;
    // Fan-down sheets are long: when the page got scrolled (reading a sheet),
    // re-root the scroll so the newly-focused sheet's title lands in the same
    // place instead of opening mid-document with only its tail in view.
    if (deck._fanDown) {
        const sc = deck.closest('.tab-content');
        if (sc && sc.scrollTop !== 0) sc.scrollTop = 0;
        // Enter the newly-focused sheet from the edge you're heading toward:
        // scrolling down opens it at the top, scrolling up opens it at the
        // bottom (you came from below) so reading flows continuously.
        const activeCard = deck.querySelectorAll('.chart-card')[next];
        const body = activeCard && activeCard.querySelector('.deck-card-scroll');
        if (body) body.scrollTop = delta < 0 ? body.scrollHeight : 0;
    }
    walletToA(deck);  // any cycle latches the focal slot (mobile; no-op desktop)
    scheduleLayout(deck);
    return true;
}

function bindDeckEvents(deck) {
    // Momentum cycling: accumulate wheel delta and advance one card per
    // DECK_STEP crossed - a fast flick rolls through several cards, a gentle
    // nudge moves one, with no per-card settle lock.
    let acc = 0, lastDir = 0;
    // Swipe energy: a time-decayed EMA of recent motion (in step-units), shared
    // by wheel and touch. It builds across a strong pull or several quick ones,
    // melts the toward-top stick, lifts the per-event cap, and shortens the card
    // transition - so effort, stick relief, and animation tempo all move
    // together instead of fighting fixed rate-limits.
    let energy = 0, energyT = 0;
    function pumpEnergy(t, magSteps) {
        if (energyT) { const dt = t - energyT; if (dt > 0) energy *= Math.exp(-dt / DECK_ENERGY_TAU); }
        energyT = t;
        energy += magSteps;
        return Math.min(1, energy / DECK_ENERGY_FULL);  // relief 0..1
    }
    // Phase the transition duration with energy: long settle when gentle, short
    // under a flick so rapid re-targets don't queue/flicker against the blur.
    function setDeckTempo(relief) {
        deck.style.setProperty('--deck-trans', (0.32 - relief * 0.24).toFixed(3) + 's');
    }
    // Advance as energy/accumulator allow: cap lifts from the base toward the
    // whole deck as energy builds, and the toward-top reserve melts to zero.
    function runAdvance(dir, base, getAcc, setAcc, relief) {
        const baseStep = deckStep(deck, base);
        const maxPer = Math.max(DECK_MAX_PER_EVENT, Math.round(relief * deck._deck.count));
        let advanced = 0;
        while (advanced < maxPer) {
            const reserve = dir < 0 ? DECK_STICK_FACTOR * baseStep * (1 - relief) : 0;
            const step = baseStep + reserve;
            if (Math.abs(getAcc()) < step) break;
            setAcc(getAcc() - dir * step);
            if (!advanceDeckBy(deck, dir)) { setAcc(0); break; }
            advanced++;
        }
    }
    deck.addEventListener('wheel', (e) => {
        const st = deck._deck;
        if (!st) return;
        const dir = e.deltaY > 0 ? 1 : -1;
        // Fan-down decks: scroll the focused sheet's own body first; only fall
        // through to cycling once it's scrolled to its top/bottom edge.
        if (deck._fanDown) {
            const activeCard = deck.querySelectorAll('.chart-card')[st.activeIndex];
            const body = activeCard && activeCard.querySelector('.deck-card-scroll');
            if (body) {
                const canDown = body.scrollTop + body.clientHeight < body.scrollHeight - 1;
                const canUp = body.scrollTop > 0;
                if ((dir > 0 && canDown) || (dir < 0 && canUp)) {
                    body.scrollTop += e.deltaY;
                    walletToA(deck);  // reading the body -> latch focal slot A
                    e.preventDefault();
                    acc = 0;
                    return;
                }
            }
        }
        // At the ends, release the wheel to normal page scroll.
        const atEnd = (dir > 0 && st.activeIndex >= st.count - 1) ||
                      (dir < 0 && st.activeIndex <= 0);
        // Pulling back past the first card is the deliberate return to rest:
        // release the latch so the header (slot B) comes back.
        if (atEnd) { if (dir < 0) walletToB(deck); acc = 0; return; }
        e.preventDefault();
        // Only a direction reversal clears the accumulator - a same-direction
        // pause must NOT, or slow steady scrolling (a notch every few hundred ms)
        // would zero out before it ever crosses a step. All effort banks.
        if (dir !== lastDir) acc = 0;
        lastDir = dir;
        acc += e.deltaY;
        // Asymmetric stick: hold only toward the top (dir < 0) so earlier cards
        // don't whip past; forward is free. The hold melts as energy builds, so
        // a strong/sustained pull rolls through with no rate-limit.
        const relief = pumpEnergy(e.timeStamp, Math.abs(e.deltaY) / deckStep(deck, DECK_STEP));
        setDeckTempo(relief);
        runAdvance(dir, DECK_STEP, () => acc, (v) => { acc = v; }, relief);
    }, { passive: false });

    // Touch: live, distance-proportional cycling (multi-card), and manual body
    // scroll for fan-down sheets - an ancestor's touch-action blocks native
    // vertical scroll, so we drive it here. Mirrors the wheel's momentum feel.
    let tActive = false, tLastY = 0, tAcc = 0, tLastDir = 0, tLastT = 0;
    deck.addEventListener('touchstart', (e) => {
        tActive = true;
        tLastY = e.touches[0].clientY;
        tLastT = e.timeStamp;
        tAcc = 0;
    }, { passive: true });
    deck.addEventListener('touchmove', (e) => {
        if (!tActive) return;
        const st = deck._deck;
        if (!st) return;
        const y = e.touches[0].clientY;
        let dy = tLastY - y;            // finger up -> dy > 0 -> next sheet
        tLastY = y;
        if (dy === 0) return;
        const dir = dy > 0 ? 1 : -1;
        tLastT = e.timeStamp;

        // fan-down: drag-scroll the focused sheet's body first; any overshoot
        // past its top/bottom edge carries straight into cycling (continuous,
        // no dead-zone "catch" at the document boundary).
        if (deck._fanDown) {
            const ac = deck.querySelectorAll('.chart-card')[st.activeIndex];
            const body = ac && ac.querySelector('.deck-card-scroll');
            if (body) {
                const room = dir > 0
                    ? (body.scrollHeight - body.clientHeight) - body.scrollTop
                    : body.scrollTop;
                if (room > 0) {
                    const used = Math.min(Math.abs(dy), room);
                    body.scrollTop += dir * used;
                    walletToA(deck);        // reading the body -> latch focal slot A
                    e.preventDefault();
                    dy -= dir * used;       // leftover continues into cycling
                    if (Math.abs(dy) < 1) return;
                }
            }
        }

        const atEnd = (dir > 0 && st.activeIndex >= st.count - 1) ||
                      (dir < 0 && st.activeIndex <= 0);
        if (atEnd) {
            // Pulling back past the first card: release to rest (slot B) and
            // swallow the gesture so it can't bubble into the page's
            // pull-to-refresh. Refresh stays available from the header area,
            // whose touches never reach this deck-scoped handler.
            if (dir < 0) { walletToB(deck); e.preventDefault(); }
            return;
        }
        e.preventDefault();
        if (dir !== tLastDir) { tAcc = 0; tLastDir = dir; }
        tAcc += dy;
        // Same energy-melted stick as the wheel (toward-top only); a strong or
        // repeated pull builds enough energy to roll through every card at once.
        const relief = pumpEnergy(e.timeStamp, Math.abs(dy) / deckStep(deck, DECK_SWIPE_STEP));
        setDeckTempo(relief);
        runAdvance(dir, DECK_SWIPE_STEP, () => tAcc, (v) => { tAcc = v; }, relief);
    }, { passive: false });
    deck.addEventListener('touchend', () => { tActive = false; }, { passive: true });
}

function visibleDeck() {
    return Array.from(document.querySelectorAll('.chart-deck'))
        .find(d => d.offsetParent !== null) || null;
}

function onDeckResize() {
    document.querySelectorAll('.chart-deck').forEach(d => {
        if (d.offsetParent !== null) layoutDeck(d);
    });
}

function onDeckKeydown(e) {
    const deck = visibleDeck();
    if (!deck) return;
    const tag = (document.activeElement && document.activeElement.tagName) || '';
    if (tag === 'INPUT' || tag === 'TEXTAREA') return;
    if (e.key === 'ArrowDown') { if (advanceDeckBy(deck, 1)) e.preventDefault(); }
    else if (e.key === 'ArrowUp') { if (advanceDeckBy(deck, -1)) e.preventDefault(); }
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
                            const color = CONSTANTS.RUN_COLORS[idx % CONSTANTS.RUN_COLORS.length];
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

    const refreshIcon = `
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
            <path fill-rule="evenodd" d="M8 3a5 5 0 1 0 4.546 2.914.5.5 0 0 1 .908-.417A6 6 0 1 1 8 2v1z"/>
            <path d="M8 4.466V.534a.25.25 0 0 1 .41-.192l2.36 1.966c.12.1.12.284 0 .384L8.41 4.658A.25.25 0 0 1 8 4.466z"/>
        </svg>
    `;

    const totalPoints = runs.reduce((sum, r) => sum + (r.metadata?.num_points || 0), 0);

    const metadataHTML = `
        <span id="metrics-metadata-comparing"><strong>Comparing:</strong> ${runs.length} run${runs.length !== 1 ? 's' : ''}</span>
        <span id="metrics-metadata-points"><strong>Total Points:</strong> ${totalPoints}</span>
    `;

    const headerHTML = createTabHeader({
        title: 'Metrics',
        additionalContent: selectorHTML,
        buttons: [{
            id: 'refresh-metrics-btn',
            label: 'Refresh',
            icon: refreshIcon,
            className: 'tab-header-button'
        }],
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
        const color = CONSTANTS.RUN_COLORS[((colorIdx >= 0 ? colorIdx : 0)) % CONSTANTS.RUN_COLORS.length];

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
        const color = CONSTANTS.RUN_COLORS[((colorIdx >= 0 ? colorIdx : 0)) % CONSTANTS.RUN_COLORS.length];

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
                const color = CONSTANTS.RUN_COLORS[docIdx % CONSTANTS.RUN_COLORS.length];

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

            const color = CONSTANTS.RUN_COLORS[parseInt(expertNum) % CONSTANTS.RUN_COLORS.length];
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
