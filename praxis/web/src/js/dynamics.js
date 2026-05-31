/**
 * Praxis Web - Learning Dynamics Visualization
 * Tracks per-layer gradient flow, update ratios, and per-expert dynamics.
 */

import { state, CONSTANTS, chartLineColor } from './state.js';
import { fetchAPI } from './api.js';
import { createTabHeader } from './components.js';
import { formatRelativeTime, initChartDeck, applyChartTheme } from './charts.js';

// Chart instances
export const dynamicsCharts = {};

// Layer toggle state
export const dynamicsLayerState = {};

/**
 * Get theme-appropriate colors
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
 * Update chart colors for theme changes
 */
export function updateDynamicsChartColors() {
    Object.values(dynamicsCharts).forEach(applyChartTheme);
}

/**
 * Load the list of runs that have dynamics data, for the picker dropdown.
 * Filters /api/runs to entries with has_dynamics=true. Idempotent.
 */
async function loadAvailableDynamicsRuns() {
    try {
        const response = await fetch('/api/runs');
        if (!response.ok) return [];
        const data = await response.json();
        const all = data.runs || [];
        state.dynamics.availableRuns = all.filter(r => r.has_dynamics);
        return state.dynamics.availableRuns;
    } catch (error) {
        console.error('[Dynamics] Failed to load run list:', error);
        return [];
    }
}

/**
 * Toggle the dropdown open/closed.
 */
export function toggleDynamicsRunSelector() {
    state.dynamics.runSelectorOpen = !state.dynamics.runSelectorOpen;
    const dropdown = document.getElementById('dynamics-run-selector-dropdown');
    if (dropdown) {
        dropdown.style.display = state.dynamics.runSelectorOpen ? 'block' : 'none';
    }
}

/**
 * Select a run (single-select); null = current run.
 */
export function selectDynamicsRun(hash) {
    state.dynamics.selectedRun = hash || null;
    state.dynamics.runSelectorOpen = false;
    state.dynamics.loaded = false;
    loadDynamicsWithCharts(true);
}

/**
 * Load and render learning dynamics
 */
export async function loadDynamicsWithCharts(force = false) {
    if (state.dynamics.loaded && !force) {
        return;
    }

    const container = document.getElementById('dynamics-container');
    if (!container) return;

    container.innerHTML = '<div class="loading-placeholder">Loading learning dynamics...</div>';

    try {
        await loadAvailableDynamicsRuns();
        const runQuery = state.dynamics.selectedRun
            ? `&runs=${encodeURIComponent(state.dynamics.selectedRun)}`
            : '';
        const response = await fetch(`/api/dynamics?since=0&limit=1000${runQuery}`);

        if (!response.ok) {
            throw new Error(`API returned ${response.status}`);
        }

        const data = await response.json();

        if (data.status === 'no_data' || !data.runs || data.runs.length === 0) {
            // No run yet: still render the tab (activation curves load on their own);
            // just show whatever cards have data, never a "no data" banner.
            renderDynamicsCharts({}, container);
            return;
        }

        state.dynamics.data = data.runs[0];
        const steps = data.runs[0].dynamics?.steps || [];
        if (steps.length > 0) {
            state.dynamics.lastStep = Math.max(...steps);
        }

        renderDynamicsCharts(data.runs[0], container);
        state.dynamics.loaded = true;

        console.log(`[Dynamics] Loaded ${steps.length} data points`);

    } catch (error) {
        console.error('[Dynamics] Failed to load:', error);
        container.innerHTML = `
            <div class="error-message">
                <h3>Error Loading Dynamics</h3>
                <p>${error.message}</p>
            </div>
        `;
    }
}

/**
 * Build the single-select runs dropdown (HTML string). Uses the same
 * .run-selector-* CSS as the Research tab, scoped via a #dynamics- prefix
 * so events can target the dynamics picker specifically.
 */
function renderDynamicsRunSelector() {
    const runs = state.dynamics.availableRuns || [];
    if (runs.length === 0) return '';

    const selected = state.dynamics.selectedRun;
    const activeRun = runs.find(r => r.hash === selected)
        || runs.find(r => r.is_current)
        || runs[0];
    const label = activeRun ? activeRun.hash : 'Run';

    return `
        <div class="run-selector-wrapper">
            <button class="run-selector-button" id="dynamics-run-selector-btn">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
                    <path d="M8 1a7 7 0 1 0 4.95 11.95l.707.707A8.001 8.001 0 1 1 8 0v1z"/>
                    <path d="M7.5 3a.5.5 0 0 1 .5.5v5.21l3.248 1.856a.5.5 0 0 1-.496.868l-3.5-2A.5.5 0 0 1 7 9V3.5a.5.5 0 0 1 .5-.5z"/>
                </svg>
                Run: ${label}
            </button>
            <div class="run-selector-dropdown" id="dynamics-run-selector-dropdown" style="display: none;">
                <div class="run-selector-header">Select Run</div>
                <div class="run-selector-list">
                    ${runs.map(run => {
                        const isActive = run.hash === (activeRun ? activeRun.hash : null);
                        const time = formatRelativeTime(run.metrics_updated);
                        const badge = run.is_current ? ' <span style="opacity: 0.6; font-size: 0.8em;">(active)</span>' : '';
                        return `
                            <label class="run-selector-item">
                                <input type="radio" name="dynamics-run" ${isActive ? 'checked' : ''} data-dynamics-run-hash="${run.hash}">
                                <span class="run-label">${run.hash}${badge}</span>
                                <span class="run-steps">${run.num_steps} steps &middot; ${time}</span>
                            </label>
                        `;
                    }).join('')}
                </div>
            </div>
        </div>
    `;
}


// ─── Metric detection helpers ───────────────────────────────────────────────

/**
 * Compile the backend Dynamics chart registry into family configs:
 * key_pattern -> RegExp, derived canvas id, sorted by order.
 */
function buildDynamicsFamilyConfigs(registry) {
    return (registry || [])
        .slice()
        .sort((a, b) => (a.order ?? 0) - (b.order ?? 0))
        .map(e => ({
            key: e.key,
            type: e.type,
            title: e.title || e.key,
            subtitle: e.subtitle || '',
            keyPattern: e.key_pattern ? new RegExp(e.key_pattern) : null,
            layerToggles: !!e.layer_toggles,
            legend: !!e.legend,
            order: e.order ?? 0,
            canvasId: `dynamics-${e.key.replace(/_/g, '-')}`,
        }));
}

/** A family is present when any dynamics key matches its pattern. */
function familyPresent(config, dynamics) {
    return !!config.keyPattern &&
        Object.keys(dynamics).some(k => config.keyPattern.test(k));
}

/**
 * Layer indices for the toggle UI: the union of layer numbers across every
 * present layer-toggle family, read from the registry patterns rather than
 * hardcoded metric names.
 */
function detectLayerIndices(dynamics, configs) {
    const layerFamilies = configs.filter(c => c.layerToggles);
    const layers = new Set();
    Object.keys(dynamics).forEach(k => {
        if (!layerFamilies.some(c => c.keyPattern.test(k))) return;
        const m = k.match(/layer_(\d+)/);
        if (m) layers.add(parseInt(m[1]));
    });
    return Array.from(layers).sort((a, b) => a - b);
}

// type -> builder adapter. Builders are declared below (hoisted); each
// takes (canvasId, dynamics). Layer-dependent ones read the live toggle
// selection from dynamicsLayerState.
const DYNAMICS_FAMILY_RENDERERS = {
    layer_grad_norms: (id, dyn) => createLayerGradNormsChart(id, dyn, dynamicsLayerState.layers),
    layer_update_ratio: (id, dyn) => createLayerUpdateRatioChart(id, dyn, dynamicsLayerState.layers),
    expert_grad_norms: (id, dyn) => createExpertGradNormsChart(id, dyn, dynamicsLayerState.layers),
    expert_grad_vars: (id, dyn) => createExpertGradVarsChart(id, dyn, dynamicsLayerState.layers),
    task_weights: (id, dyn) => createTaskWeightsChart(id, dyn, detectTaskWeightKeys(dyn)),
    halting_hist: (id, dyn) => createHaltingHistogramChart(id, dyn, detectHaltingBuckets(dyn)),
};

function buildDynamicsFamilyCard(config, desc) {
    const legendHTML = config.legend
        ? `<div class="chart-legend" id="${config.canvasId}-legend"></div>`
        : '';
    return `
        <div style="margin-top: 2rem;">
            <div class="chart-card">
                <div class="chart-title">${config.title}</div>
                <div class="chart-subtitle">${desc(config.key, config.subtitle)}</div>
                <div class="chart-wrapper" style="height: 400px;">
                    <canvas id="${config.canvasId}"></canvas>
                </div>
                ${legendHTML}
            </div>
        </div>
    `;
}

function mountDynamicsFamily(config, dynamics) {
    if (!document.getElementById(config.canvasId)) return;
    const render = DYNAMICS_FAMILY_RENDERERS[config.type];
    if (render) render(config.canvasId, dynamics);
}

/** Task-weight series arrive as `task_weight_<name>` keys. */
function detectTaskWeightKeys(dynamics) {
    return Object.keys(dynamics)
        .filter(k => k.startsWith('task_weight_'))
        .sort();
}

/**
 * Detect halting histogram buckets. Returns {rs, maxLoops} or null if
 * no halting metrics are present (non-KL strategies, or not yet logged).
 */

function detectHaltingBuckets(dynamics) {
    const rs = new Set();
    Object.keys(dynamics).forEach(k => {
        const m = k.match(/^halting\/(train|eval)_r_(\d+)$/);
        if (m) rs.add(parseInt(m[2]));
    });
    if (rs.size === 0) return null;
    const sorted = Array.from(rs).sort((a, b) => a - b);
    return { rs: sorted, maxLoops: sorted[sorted.length - 1] };
}

/**
 * Pull the latest non-null value from a metric series.
 */
// ─── Scalar-metric manifest (data-driven dashboard) ────────────────────────
//
// Entries in ``descriptions`` whose value is an object with a ``chart`` hint
// opt the metric into auto-rendering. Each chart hint may carry:
//   title:    chart title text
//   y_label:  y-axis label
//   y_scale:  'linear' (default) or 'logarithmic'
//   group:    section key used to cluster related metrics together
//   order:    integer ordering within the group (default 0)
//   series_group: metrics sharing this key render as lines on ONE chart
//                 (lowest-order member supplies title/axis/subtitle)
//   series_label: this metric's legend label within its series_group
//
// Bespoke chart types (heatmaps, histograms, stacked-per-task series) stay
// hardcoded - the manifest only handles scalar time-series.

function buildScalarMetricManifest(descriptions) {
    const groups = new Map();
    for (const [key, entry] of Object.entries(descriptions || {})) {
        if (!entry || typeof entry !== 'object') continue;
        const chart = entry.chart;
        if (!chart || typeof chart !== 'object') continue;
        const groupName = chart.group || 'misc';
        if (!groups.has(groupName)) groups.set(groupName, []);
        groups.get(groupName).push({ key, chart, description: entry.description });
    }
    for (const entries of groups.values()) {
        entries.sort((a, b) => (a.chart.order ?? 0) - (b.chart.order ?? 0));
    }
    return groups;
}

function canvasIdForMetric(key) {
    return `dynamics-${key.replace(/_/g, '-')}`;
}

// Collapse a section's entries into render items: entries sharing a
// ``series_group`` become one multi-line chart (its lowest-order member,
// which appears first since entries are pre-sorted, leads the title/axis);
// the rest stay as single-series charts.
function seriesItemsFor(entries) {
    const items = [];
    const byGroup = new Map();
    for (const entry of entries) {
        const sg = entry.chart.series_group;
        if (!sg) {
            items.push({ kind: 'single', ...entry });
            continue;
        }
        const label = entry.chart.series_label || entry.chart.title || entry.key;
        if (byGroup.has(sg)) {
            items[byGroup.get(sg)].series.push({ key: entry.key, label });
        } else {
            byGroup.set(sg, items.length);
            items.push({
                kind: 'multi',
                canvasId: `dynamics-series-${sg.replace(/_/g, '-')}`,
                lead: entry,
                series: [{ key: entry.key, label }],
            });
        }
    }
    return items;
}

function metricCardHTML(canvasId, title, subtitle) {
    return `
        <div style="margin-top: 2rem;">
            <div class="chart-card">
                <div class="chart-title">${title}</div>
                <div class="chart-subtitle">${subtitle}</div>
                <div class="chart-wrapper" style="height: 400px;">
                    <canvas id="${canvasId}"></canvas>
                </div>
            </div>
        </div>
    `;
}

function buildManifestSectionsHTML(manifest, getDesc) {
    let html = '';
    for (const [, entries] of manifest) {
        for (const item of seriesItemsFor(entries)) {
            if (item.kind === 'multi') {
                const c = item.lead.chart;
                html += metricCardHTML(
                    item.canvasId,
                    c.title || item.lead.key,
                    getDesc(item.lead.key, item.lead.description || '')
                );
            } else {
                html += metricCardHTML(
                    canvasIdForMetric(item.key),
                    item.chart.title || item.key,
                    getDesc(item.key, item.description || '')
                );
            }
        }
    }
    return html;
}

function mountManifestCharts(manifest, dynamics) {
    for (const [, entries] of manifest) {
        for (const item of seriesItemsFor(entries)) {
            if (item.kind === 'multi') {
                const c = item.lead.chart;
                if (!document.getElementById(item.canvasId)) continue;
                createMultiSeriesMetricChart(
                    item.canvasId,
                    dynamics,
                    item.series,
                    c.y_label || c.title || item.lead.key,
                    c.y_scale || 'linear'
                );
            } else {
                const canvasId = canvasIdForMetric(item.key);
                if (!document.getElementById(canvasId)) continue;
                createScalarMetricChart(
                    canvasId,
                    dynamics,
                    item.key,
                    item.chart.y_label || item.key,
                    item.chart.y_scale || 'linear'
                );
            }
        }
    }
}

// ─── Snapshot dispatcher (heatmaps, PCA grids, etc.) ───────────────────────
//
// Snapshot entries in ``descriptions`` carry a ``snapshot`` hint that picks
// a renderer from SNAPSHOT_RENDERERS. Backend payloads come from a single
// /api/head_snapshots fetch, keyed by metric name. New snapshot types add
// one entry to SNAPSHOT_RENDERERS and one entry to the head's
// ``dashboard_snapshots()``.

function snapshotEntries(descriptions) {
    const out = [];
    for (const [key, entry] of Object.entries(descriptions || {})) {
        if (!entry?.snapshot) continue;
        out.push({ key, snap: entry.snapshot, description: entry.description });
    }
    out.sort((a, b) => (a.snap.order ?? 0) - (b.snap.order ?? 0));
    return out;
}

function buildSnapshotSectionsHTML(descriptions, getDesc) {
    let html = '';
    for (const { key, snap, description } of snapshotEntries(descriptions)) {
        const canvasId = canvasIdForMetric(key);
        html += `
            <div style="margin-top: 2rem;">
                <div class="chart-card">
                    <div class="chart-title">${snap.title || key}</div>
                    <div class="chart-subtitle">${getDesc(key, description || '')}</div>
                    <div class="chart-wrapper" style="height: 400px;">
                        <canvas id="${canvasId}"></canvas>
                    </div>
                </div>
            </div>
        `;
    }
    return html;
}

async function mountSnapshotCharts(descriptions) {
    const entries = snapshotEntries(descriptions);
    if (entries.length === 0) return;

    let snapshots;
    try {
        const response = await fetch('/api/head_snapshots');
        if (!response.ok) throw new Error(`API returned ${response.status}`);
        const data = await response.json();
        if (data.status !== 'ok') return;
        snapshots = data.snapshots || {};
    } catch (e) {
        console.error('[Dynamics] Snapshots failed to load:', e);
        return;
    }

    for (const { key, snap } of entries) {
        const renderer = SNAPSHOT_RENDERERS[snap.renderer];
        if (!renderer) {
            console.warn(`[Dynamics] No renderer for snapshot "${key}" (renderer=${snap.renderer})`);
            continue;
        }
        const payload = snapshots[key];
        if (!payload) continue;
        const canvas = document.getElementById(canvasIdForMetric(key));
        if (!canvas) continue;
        try {
            renderer(canvas, payload, snap);
        } catch (e) {
            console.error(`[Dynamics] Snapshot "${key}" render failed:`, e);
        }
    }
}

function latestValue(series) {
    if (!Array.isArray(series)) return null;
    for (let i = series.length - 1; i >= 0; i--) {
        const v = series[i];
        if (v !== null && v !== undefined) return v;
    }
    return null;
}

// ─── Main render ────────────────────────────────────────────────────────────

/**
 * Render dynamics charts
 */
function renderDynamicsCharts(runData, container) {
    const dynamics = runData.dynamics || {};
    const steps = dynamics.steps || dynamics.step || [];
    // Backend-driven descriptions: each chart subtitle reads from this map
    // (key = metric or chart-group name) and falls back to the inline text.
    // Entries may be plain strings (legacy) or rich objects of shape
    // {description, chart?}; ``desc()`` accepts both forms.
    const descriptions = runData.descriptions || {};
    const desc = (key, fallback) => {
        const entry = descriptions[key];
        if (typeof entry === 'string') return entry;
        return entry?.description || fallback;
    };
    const manifest = buildScalarMetricManifest(descriptions);

    // Build the present chart families from the backend registry, and stash
    // them so the layer-toggle handler re-renders exactly this set.
    const familyConfigs = buildDynamicsFamilyConfigs(runData.chart_registry)
        .filter(c => familyPresent(c, dynamics));
    dynamicsLayerState.familyConfigs = familyConfigs;

    const allLayers = detectLayerIndices(dynamics, familyConfigs);
    const expertCount = new Set(
        Object.keys(dynamics)
            .map(k => k.match(/expert_(\d+)/))
            .filter(Boolean)
            .map(m => parseInt(m[1]))
    ).size;

    if (!dynamicsLayerState.layers) {
        dynamicsLayerState.layers = [...allLayers];
        dynamicsLayerState.allLayers = [...allLayers];
    }

    // ── Header ──────────────────────────────────────────────────────────
    const refreshIcon = `
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
            <path fill-rule="evenodd" d="M8 3a5 5 0 1 0 4.546 2.914.5.5 0 0 1 .908-.417A6 6 0 1 1 8 2v1z"/>
            <path d="M8 4.466V.534a.25.25 0 0 1 .41-.192l2.36 1.966c.12.1.12.284 0 .384L8.41 4.658A.25.25 0 0 1 8 4.466z"/>
        </svg>
    `;

    const metaParts = [`<span><strong>Points:</strong> ${steps.length}</span>`];
    if (expertCount > 0) {
        metaParts.push(`<span><strong>Experts:</strong> ${expertCount}</span>`);
    }

    const headerHTML = createTabHeader({
        title: 'Learning Dynamics',
        additionalContent: renderDynamicsRunSelector(),
        buttons: [{
            id: 'refresh-dynamics-btn',
            label: 'Refresh',
            icon: refreshIcon,
            className: 'tab-header-button'
        }],
        metadata: metaParts.join('\n')
    });

    // ── Chart cards ─────────────────────────────────────────────────────
    let chartsHTML = `
        <div style="margin-top: 2rem;">
            <div id="dynamics-layer-toggles"></div>
        </div>
    `;

    // Families ordered before the head-metric sections (gradient flow,
    // expert charts, task weights), rendered straight from the registry.
    chartsHTML += familyConfigs
        .filter(c => c.order < 100)
        .map(c => buildDynamicsFamilyCard(c, desc))
        .join('');

    // Head-driven scalar metrics (harmonic, crystal, future heads). The
    // manifest is built from descriptions whose values carry a ``chart``
    // hint - new heads opt in just by tagging their metric_descriptions.
    chartsHTML += buildManifestSectionsHTML(manifest, desc);

    // Non-scalar snapshot charts (heatmaps, PCA density grids, etc.).
    // Like the scalar manifest, snapshots are discovered via descriptions
    // whose values carry a ``snapshot`` hint.
    chartsHTML += buildSnapshotSectionsHTML(descriptions, desc);

    // Families ordered after the head sections (e.g. halting distribution).
    chartsHTML += familyConfigs
        .filter(c => c.order >= 100)
        .map(c => buildDynamicsFamilyCard(c, desc))
        .join('');

    // Activation curves (always rendered; placeholder if endpoint returns empty)
    chartsHTML += `
        <div style="margin-top: 2rem;">
            <div class="chart-card">
                <div class="chart-title">Activation Forward</div>
                <div class="chart-subtitle" id="activation-forward-subtitle">Forward curve per activation module. Line = mean across features; shaded band = 10-90 percentile spread.</div>
                <div class="chart-wrapper" style="height: 400px;">
                    <canvas id="dynamics-activation-forward"></canvas>
                </div>
                <div class="chart-legend" id="dynamics-activation-forward-legend"></div>
            </div>
        </div>

        <div style="margin-top: 2rem;">
            <div class="chart-card">
                <div class="chart-title">Activation Derivative</div>
                <div class="chart-subtitle">dy/dx per activation module via autograd. Line = mean across features; shaded band = 10-90 percentile spread.</div>
                <div class="chart-wrapper" style="height: 400px;">
                    <canvas id="dynamics-activation-backward"></canvas>
                </div>
                <div class="chart-legend" id="dynamics-activation-backward-legend"></div>
            </div>
        </div>
    `;

    container.innerHTML = headerHTML + chartsHTML;

    // Stack the dynamics charts into the same card deck as the Research tab.
    // Cards are a flat sequence (each already carries its own title), so we
    // just move them into one deck and leave the layer-toggles control above.
    const dCards = Array.from(container.querySelectorAll('.chart-card'));
    if (dCards.length) {
        const deck = document.createElement('div');
        deck.className = 'chart-deck';
        deck.id = 'dynamics-deck';
        deck.innerHTML = '<div class="chart-deck-counter"></div>';
        const firstWrapper = dCards[0].parentElement;
        firstWrapper.parentNode.insertBefore(deck, firstWrapper);
        dCards.forEach(card => deck.appendChild(card));
        // Drop the now-empty margin-top wrappers the cards left behind.
        Array.from(container.children).forEach(ch => {
            if (ch !== deck && ch.tagName === 'DIV' && ch.children.length === 0) {
                ch.remove();
            }
        });
        initChartDeck(deck);
    }

    // Layer toggles
    renderDynamicsLayerToggles();

    // Create charts after DOM is ready
    setTimeout(() => {
        try {
            familyConfigs.forEach(c => mountDynamicsFamily(c, dynamics));
            mountManifestCharts(manifest, dynamics);
            mountSnapshotCharts(descriptions);
            loadActivationCurves();
            // Re-layout once charts have real heights (deck height tracks the
            // active card); activeIndex is preserved across the re-init.
            initChartDeck('dynamics-deck');
        } catch (error) {
            console.error('[Dynamics] Chart creation failed:', error);
        }
    }, 10);
}

// ─── Layer toggles ──────────────────────────────────────────────────────────

/**
 * Render layer toggles
 */
function renderDynamicsLayerToggles() {
    const container = document.getElementById('dynamics-layer-toggles');
    if (!container) return;

    const allLayers = dynamicsLayerState.allLayers || [];
    if (allLayers.length === 0) return;

    const buttons = [
        '<button class="layer-toggle-btn" data-layer="all">All</button>',
        '<button class="layer-toggle-btn" data-layer="none">None</button>',
        ...allLayers.map(layer =>
            `<button class="layer-toggle-btn" data-layer="${layer}">L${layer}</button>`
        )
    ].join('');

    container.innerHTML = `
        <div style="margin-bottom: 1rem;">
            <strong>Layers:</strong> ${buttons}
        </div>
    `;

    updateDynamicsLayerToggles();

    container.querySelectorAll('.layer-toggle-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const layer = e.target.dataset.layer;

            if (layer === 'all') {
                dynamicsLayerState.layers = [...dynamicsLayerState.allLayers];
            } else if (layer === 'none') {
                dynamicsLayerState.layers = [];
            } else {
                const layerNum = parseInt(layer);
                const idx = dynamicsLayerState.layers.indexOf(layerNum);
                if (idx >= 0) {
                    dynamicsLayerState.layers.splice(idx, 1);
                } else {
                    dynamicsLayerState.layers.push(layerNum);
                    dynamicsLayerState.layers.sort((a, b) => a - b);
                }
            }

            updateDynamicsLayerToggles();
            rebuildAllCharts();
        });
    });
}

/**
 * Update layer toggle button states
 */
function updateDynamicsLayerToggles() {
    const container = document.getElementById('dynamics-layer-toggles');
    if (!container) return;

    const allLayers = dynamicsLayerState.allLayers || [];
    const selectedLayers = dynamicsLayerState.layers || [];

    container.querySelectorAll('.layer-toggle-btn').forEach(btn => {
        const layer = btn.dataset.layer;

        if (layer === 'all') {
            btn.classList.toggle('active', selectedLayers.length === allLayers.length);
        } else if (layer === 'none') {
            btn.classList.toggle('active', selectedLayers.length === 0);
        } else {
            const layerNum = parseInt(layer);
            btn.classList.toggle('active', selectedLayers.includes(layerNum));
        }
    });
}

/**
 * Rebuild all visible charts with current layer selection
 */
function rebuildAllCharts() {
    const dynamics = state.dynamics.data?.dynamics || {};

    // Re-render the same families that were mounted initially. Layer-toggle
    // families pick up the new selection from dynamicsLayerState; the rest
    // simply refresh.
    (dynamicsLayerState.familyConfigs || []).forEach(c => mountDynamicsFamily(c, dynamics));

    // Head scalar metrics (independent of layer selection). Source of
    // truth is the descriptions manifest in the run payload.
    const descriptions = state.dynamics.data?.descriptions || {};
    mountManifestCharts(buildScalarMetricManifest(descriptions), dynamics);
}

// ─── Shared chart helpers ───────────────────────────────────────────────────
// Line colors come from the shared, accent-anchored palette (chartLineColor) so the
// Research and Dynamics tabs match and re-tint with the hue toggle.

function baseChartOptions(yLabel, yType, textColor, gridColor, tooltipBg) {
    return {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { intersect: false, mode: 'index' },
        parsing: false,
        normalized: true,
        plugins: {
            decimation: { enabled: true, algorithm: 'lttb', samples: 500 },
            legend: {
                display: true,
                position: 'top',
                labels: { color: textColor, usePointStyle: true, padding: 12 }
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
                title: { display: true, text: 'Training Step', color: textColor },
                ticks: { color: textColor },
                grid: { color: gridColor }
            },
            y: {
                type: yType,
                title: { display: true, text: yLabel, color: textColor },
                ticks: {
                    color: textColor,
                    callback: yType === 'logarithmic'
                        ? (value) => value.toExponential(0)
                        : undefined
                },
                grid: { color: gridColor }
            }
        }
    };
}

function makeLineDataset(label, data, color) {
    return {
        label,
        data,
        borderColor: color,
        backgroundColor: color + '20',
        borderWidth: 2,
        pointRadius: 0,
        pointHoverRadius: 5,
        tension: 0.3,
        fill: false
    };
}

function renderChart(canvasId, datasets, yLabel, yType) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    if (dynamicsCharts[canvasId]) {
        dynamicsCharts[canvasId].destroy();
    }

    const { textColor, gridColor, tooltipBg } = getThemeColors();
    const options = baseChartOptions(yLabel, yType, textColor, gridColor, tooltipBg);
    // Defer legend rendering to the scrollable HTML legend so clicks don't
    // block on a Chart.js redraw and Show/Hide-all is available.
    options.plugins.legend.display = false;

    const chart = new Chart(ctx, {
        type: 'line',
        data: { datasets },
        options
    });
    dynamicsCharts[canvasId] = chart;

    renderScrollableLegend(`${canvasId}-legend`, chart);
}

// ─── Universal charts ───────────────────────────────────────────────────────

/**
 * Gradient Flow: per-layer gradient norms (log scale)
 */
function createLayerGradNormsChart(canvasId, dynamics, layers) {
    const steps = dynamics.steps || [];
    const datasets = [];

    layers.forEach(layer => {
        const key = `layer_${layer}_grad_norm`;
        if (!dynamics[key]) return;

        const values = dynamics[key];
        const data = steps.map((step, idx) => ({
            x: step, y: values[idx]
        })).filter(p => p.y !== null && p.y !== undefined);

        const color = chartLineColor(layer);
        datasets.push(makeLineDataset(`L${layer}`, data, color));
    });

    renderChart(canvasId, datasets, 'Gradient Norm (L2, Log Scale)', 'logarithmic');
}

/**
 * Update-to-Weight Ratio: per-layer (log scale)
 */
function createLayerUpdateRatioChart(canvasId, dynamics, layers) {
    const steps = dynamics.steps || [];
    const datasets = [];

    layers.forEach(layer => {
        const key = `layer_${layer}_update_ratio`;
        if (!dynamics[key]) return;

        const values = dynamics[key];
        const data = steps.map((step, idx) => ({
            x: step, y: values[idx]
        })).filter(p => p.y !== null && p.y !== undefined);

        const color = chartLineColor(layer);
        datasets.push(makeLineDataset(`L${layer}`, data, color));
    });

    renderChart(canvasId, datasets, 'Update Ratio (Log Scale)', 'logarithmic');
}

// ─── Task weights (conditional) ─────────────────────────────────────────────

/**
 * Task Loss Weights: per-task scalar weights over time.
 * Keys arrive as `task_weight_<name>` (e.g. task_weight_pretrain).
 */
function createTaskWeightsChart(canvasId, dynamics, keys) {
    const steps = dynamics.steps || [];
    const datasets = [];

    keys.forEach((key, idx) => {
        const values = dynamics[key];
        if (!values) return;

        const data = steps.map((step, i) => ({
            x: step, y: values[i]
        })).filter(p => p.y !== null && p.y !== undefined);

        const label = key.replace(/^task_weight_/, '');
        const color = chartLineColor(idx);
        datasets.push(makeLineDataset(label, data, color));
    });

    renderChart(canvasId, datasets, 'Effective Weight', 'linear');
}

// ─── Harmonic head diagnostics (conditional) ────────────────────────────────

/**
 * Magma-ish gradient: black -> purple -> red -> yellow. Cheap colormap that
 * reads well in both themes without depending on a JS color library.
 */
function magma(t) {
    t = Math.max(0, Math.min(1, t));
    const r = Math.round(255 * Math.pow(t, 0.5));
    const g = Math.round(255 * Math.pow(Math.max(0, t - 0.3) / 0.7, 1.4));
    const b = Math.round(255 * (0.4 * (1 - Math.abs(t - 0.5) * 2)));
    return [r, g, Math.max(0, b)];
}

/**
 * Generic 2D heatmap renderer for non-scalar snapshots.
 *
 * Payload contract: ``data.grid`` is a 2D array (rows x cols of numbers),
 * ``data.max_count`` sets the color-scale ceiling. Optional fields:
 * ``x_range``, ``y_range`` for axis-range labels.
 *
 * Snapshot options (from the description's ``snapshot`` hint): ``color_scale``
 * is ``'linear'`` (default) or ``'log'`` - log is right for heavy-tailed
 * distributions like PCA density counts.
 */
function renderHeatmap2D(canvas, data, options = {}) {
    const grid = data.grid;
    if (!Array.isArray(grid) || grid.length === 0) return;
    const rows = grid.length;
    const cols = Array.isArray(grid[0]) ? grid[0].length : 0;
    if (cols === 0) return;

    const peak = data.max_count || 1.0;
    const scaleFn = options.color_scale === 'log' ? (v) => Math.log1p(Math.max(0, v)) : (v) => Math.max(0, v);
    const peakScaled = Math.max(scaleFn(peak), 1e-12);

    const wrapper = canvas.parentElement;
    const w = wrapper.clientWidth || 800;
    const h = wrapper.clientHeight || 400;
    canvas.width = w;
    canvas.height = h;

    const ctx = canvas.getContext('2d');
    ctx.imageSmoothingEnabled = false;

    // Render at native grid resolution offscreen, then scale to fill.
    const off = document.createElement('canvas');
    off.width = cols;
    off.height = rows;
    const offCtx = off.getContext('2d');
    const img = offCtx.createImageData(cols, rows);

    for (let i = 0; i < rows; i++) {
        const row = grid[i];
        for (let j = 0; j < cols; j++) {
            const v = scaleFn(row[j]) / peakScaled;
            const [r, g, b] = magma(v);
            const idx = (i * cols + j) * 4;
            img.data[idx] = r;
            img.data[idx + 1] = g;
            img.data[idx + 2] = b;
            img.data[idx + 3] = 255;
        }
    }
    offCtx.putImageData(img, 0, 0);

    const { textColor, gridColor } = getThemeColors();
    ctx.fillStyle = gridColor;
    ctx.fillRect(0, 0, w, h);

    const ml = 60, mb = 30, mt = 8, mr = 16;
    const drawW = Math.max(1, w - ml - mr);
    const drawH = Math.max(1, h - mt - mb);
    ctx.drawImage(off, ml, mt, drawW, drawH);

    ctx.fillStyle = textColor;
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';

    const xLabel = formatAxisRange(data.x_range, cols);
    const yLabel = formatAxisRange(data.y_range, rows);
    ctx.fillText(xLabel, ml + drawW / 2, h - 8);
    ctx.save();
    ctx.translate(16, mt + drawH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(yLabel, 0, 0);
    ctx.restore();

    ctx.textAlign = 'right';
    const peakLabel = options.color_scale === 'log' ? `log peak ${peak.toExponential(2)}` : `peak ${peak.toExponential(2)}`;
    ctx.fillText(peakLabel, w - 4, mt + 12);
}

function formatAxisRange(range, fallback) {
    if (Array.isArray(range) && range.length === 2) {
        const lo = range[0], hi = range[1];
        const fmt = (v) => Number.isInteger(v) ? String(v) : v.toFixed(2);
        return `${fmt(lo)} .. ${fmt(hi)}`;
    }
    return `1..${fallback}`;
}

const SNAPSHOT_RENDERERS = {
    heatmap_2d: renderHeatmap2D,
};

/**
 * Single-series scalar over training steps (no legend needed).
 */
function createScalarMetricChart(canvasId, dynamics, key, yLabel, yType) {
    const steps = dynamics.steps || [];
    const values = dynamics[key];
    if (!values) return;

    const data = steps.map((step, idx) => ({
        x: step, y: values[idx]
    })).filter(p => p.y !== null && p.y !== undefined);

    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    if (dynamicsCharts[canvasId]) {
        dynamicsCharts[canvasId].destroy();
    }

    const { textColor, gridColor, tooltipBg } = getThemeColors();
    const options = baseChartOptions(yLabel, yType, textColor, gridColor, tooltipBg);
    options.plugins.legend.display = false;

    dynamicsCharts[canvasId] = new Chart(ctx, {
        type: 'line',
        data: { datasets: [makeLineDataset(yLabel, data, chartLineColor(0))] },
        options
    });
}

// Several same-scale metrics on one chart (e.g. min/mean/max), one line each.
function createMultiSeriesMetricChart(canvasId, dynamics, series, yLabel, yType) {
    const steps = dynamics.steps || [];
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    if (dynamicsCharts[canvasId]) {
        dynamicsCharts[canvasId].destroy();
    }

    const datasets = [];
    series.forEach(({ key, label }, idx) => {
        const values = dynamics[key];
        if (!values) return;
        const data = steps.map((step, i) => ({ x: step, y: values[i] }))
            .filter(p => p.y !== null && p.y !== undefined);
        datasets.push(makeLineDataset(label, data, chartLineColor(idx)));
    });
    if (datasets.length === 0) return;

    const { textColor, gridColor, tooltipBg } = getThemeColors();
    const options = baseChartOptions(yLabel, yType, textColor, gridColor, tooltipBg);

    dynamicsCharts[canvasId] = new Chart(ctx, { type: 'line', data: { datasets }, options });
}

// ─── Expert charts (conditional) ────────────────────────────────────────────

/**
 * Expert gradient norms chart (log scale)
 */
function createExpertGradNormsChart(canvasId, dynamics, layers) {
    const steps = dynamics.steps || [];
    const datasets = [];

    const metricKeys = Object.keys(dynamics).filter(k =>
        k.match(/^layer_\d+_expert_\d+_grad_norm$/)
    );
    const experts = Array.from(new Set(
        metricKeys.map(k => parseInt(k.match(/expert_(\d+)_/)[1]))
    )).sort((a, b) => a - b);

    layers.forEach(layer => {
        experts.forEach(expert => {
            const key = `layer_${layer}_expert_${expert}_grad_norm`;
            if (!dynamics[key]) return;

            const values = dynamics[key];
            const data = steps.map((step, idx) => ({
                x: step, y: values[idx]
            })).filter(p => p.y !== null && p.y !== undefined);

            const color = chartLineColor(expert);
            datasets.push(makeLineDataset(`L${layer} E${expert}`, data, color));
        });
    });

    renderChart(canvasId, datasets, 'Gradient Norm (L2, Log Scale)', 'logarithmic');
}

/**
 * Expert gradient variance chart (log scale)
 */
function createExpertGradVarsChart(canvasId, dynamics, layers) {
    const steps = dynamics.steps || [];
    const datasets = [];

    const metricKeys = Object.keys(dynamics).filter(k =>
        k.match(/^layer_\d+_expert_\d+_grad_var$/)
    );
    const experts = Array.from(new Set(
        metricKeys.map(k => parseInt(k.match(/expert_(\d+)_/)[1]))
    )).sort((a, b) => a - b);

    layers.forEach(layer => {
        experts.forEach(expert => {
            const key = `layer_${layer}_expert_${expert}_grad_var`;
            if (!dynamics[key]) return;

            const values = dynamics[key];
            const data = steps.map((step, idx) => ({
                x: step, y: values[idx]
            })).filter(p => p.y !== null && p.y !== undefined);

            const color = chartLineColor(expert);
            datasets.push(makeLineDataset(`L${layer} E${expert}`, data, color));
        });
    });

    renderChart(canvasId, datasets, 'Gradient Variance (Log Scale)', 'logarithmic');
}

// ─── Halting histogram ──────────────────────────────────────────────────────

/**
 * Halting loop-count distribution: grouped vertical bar chart. Training series
 * shows the log-normal Poisson samples; eval series shows where KL-halting
 * actually cut the loop short (or max_loops when it ran to full depth).
 */
function createHaltingHistogramChart(canvasId, dynamics, buckets) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    if (dynamicsCharts[canvasId]) {
        dynamicsCharts[canvasId].destroy();
    }

    const { textColor, gridColor, tooltipBg } = getThemeColors();

    const labels = buckets.rs.map(r => `r=${r}`);
    const trainCounts = buckets.rs.map(r => latestValue(dynamics[`halting/train_r_${r}`]) || 0);
    const evalCounts = buckets.rs.map(r => latestValue(dynamics[`halting/eval_r_${r}`]) || 0);

    const trainTotal = trainCounts.reduce((a, b) => a + b, 0);
    const evalTotal = evalCounts.reduce((a, b) => a + b, 0);
    const trainFreq = trainTotal > 0 ? trainCounts.map(c => c / trainTotal) : trainCounts;
    const evalFreq = evalTotal > 0 ? evalCounts.map(c => c / evalTotal) : evalCounts;

    const datasets = [];
    if (trainTotal > 0) {
        datasets.push({
            label: `Training (random, n=${trainTotal})`,
            data: trainFreq,
            backgroundColor: chartLineColor(0) + '80',
            borderColor: chartLineColor(0),
            borderWidth: 2
        });
    }
    if (evalTotal > 0) {
        datasets.push({
            label: `Inference (learned, n=${evalTotal})`,
            data: evalFreq,
            backgroundColor: chartLineColor(1) + '80',
            borderColor: chartLineColor(1),
            borderWidth: 2
        });
    }

    dynamicsCharts[canvasId] = new Chart(ctx, {
        type: 'bar',
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: { color: textColor, usePointStyle: true, padding: 12 }
                },
                tooltip: {
                    backgroundColor: tooltipBg,
                    titleColor: textColor,
                    bodyColor: textColor,
                    borderColor: gridColor,
                    borderWidth: 1,
                    padding: 12,
                    callbacks: {
                        label: (tctx) => `${tctx.dataset.label}: ${(tctx.parsed.y * 100).toFixed(1)}%`
                    }
                }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Recurrence Loops', color: textColor },
                    ticks: { color: textColor },
                    grid: { display: false }
                },
                y: {
                    title: { display: true, text: 'Frequency', color: textColor },
                    ticks: {
                        color: textColor,
                        callback: (v) => `${(v * 100).toFixed(0)}%`
                    },
                    grid: { color: gridColor },
                    beginAtZero: true
                }
            }
        }
    });
}

// ─── Activation curves ──────────────────────────────────────────────────────

/**
 * Fetch activation curves from the API and render forward + derivative charts.
 */
async function loadActivationCurves() {
    const forwardCanvas = document.getElementById('dynamics-activation-forward');
    const backwardCanvas = document.getElementById('dynamics-activation-backward');
    if (!forwardCanvas || !backwardCanvas) return;

    try {
        const response = await fetch('/api/activation_curves');
        if (!response.ok) throw new Error(`API returned ${response.status}`);
        const data = await response.json();

        const curves = data.curves || [];
        const subtitle = document.getElementById('activation-forward-subtitle');
        if (subtitle) {
            const uniqueTypes = [...new Set(curves.map(c => c.type).filter(Boolean))];
            const typeStr = uniqueTypes.length === 1
                ? uniqueTypes[0]
                : uniqueTypes.length > 1
                    ? `${uniqueTypes.length} types: ${uniqueTypes.join(', ')}`
                    : '';
            const suffix = 'line = representative feature (median of primary param); shaded band = 10-90 percentile across all features.';
            subtitle.textContent = typeStr
                ? `${typeStr} - ${suffix}`
                : `Per-module curves using live parameters. ${suffix}`;
        }

        if (curves.length === 0) {
            [forwardCanvas, backwardCanvas].forEach(canvas => {
                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, canvas.width, canvas.height);
            });
            return;
        }

        renderActivationChart('dynamics-activation-forward', curves, 'forward', 'Output');
        renderActivationChart('dynamics-activation-backward', curves, 'backward', 'dy/dx');
    } catch (error) {
        console.error('[Dynamics] Activation curves failed to load:', error);
    }
}

function renderActivationChart(canvasId, curves, field, yLabel) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    if (dynamicsCharts[canvasId]) {
        dynamicsCharts[canvasId].destroy();
    }

    const { textColor, gridColor, tooltipBg } = getThemeColors();

    const datasets = [];
    const lowKey = `${field}_low`;
    const highKey = `${field}_high`;

    curves.forEach((curve, idx) => {
        const color = chartLineColor(idx);
        const label = curve.type ? `${curve.name} (${curve.type})` : curve.name;
        const meanData = curve.x.map((x, i) => ({ x, y: curve[field][i] }));

        if (Array.isArray(curve[lowKey]) && Array.isArray(curve[highKey])) {
            const lowData = curve.x.map((x, i) => ({ x, y: curve[lowKey][i] }));
            const highData = curve.x.map((x, i) => ({ x, y: curve[highKey][i] }));
            datasets.push({
                label: `__band_low__${label}`,
                data: lowData,
                borderColor: 'transparent',
                backgroundColor: 'transparent',
                pointRadius: 0,
                fill: false,
                showLine: true
            });
            datasets.push({
                label: `__band_high__${label}`,
                data: highData,
                borderColor: 'transparent',
                backgroundColor: color + '25',
                pointRadius: 0,
                fill: '-1',
                showLine: true
            });
        }

        datasets.push(makeLineDataset(label, meanData, color));
    });

    const options = baseChartOptions(yLabel, 'linear', textColor, gridColor, tooltipBg);
    options.scales.x.title.text = 'Input';
    // Chart.js's built-in legend can't scroll and consumes a lot of vertical
    // space with many curves; we render a scrollable custom legend instead.
    options.plugins.legend.display = false;
    if (!options.plugins.tooltip.filter) {
        options.plugins.tooltip.filter = (tctx) => !tctx.dataset.label.startsWith('__band_');
    }

    const chart = new Chart(ctx, {
        type: 'line',
        data: { datasets },
        options
    });
    dynamicsCharts[canvasId] = chart;

    renderScrollableLegend(`${canvasId}-legend`, chart);
}

/**
 * Render a scrollable HTML legend with Show/Hide-all controls.
 *
 * - Skips `__band_*` band datasets (activation charts use these for the
 *   shaded percentile envelope) and pairs visibility toggles so hiding a
 *   row also hides its band siblings.
 * - Visibility changes flow through `chart.setDatasetVisibility` (no
 *   internal render) and are flushed once via `chart.update('none')` inside
 *   `requestAnimationFrame`, so the click handler returns immediately and
 *   the row's muted-class repaint isn't blocked by Chart.js's canvas
 *   redraw. "Hide all" of N series becomes one frame, not N.
 */
function renderScrollableLegend(legendId, chart) {
    const container = document.getElementById(legendId);
    if (!container) return;

    container.innerHTML = '';

    const controls = document.createElement('div');
    controls.className = 'chart-legend-controls';

    const items = document.createElement('div');
    items.className = 'chart-legend-items';

    const rowsByIdx = new Map();

    const flushVisibility = (changes) => {
        requestAnimationFrame(() => {
            changes.forEach(({ idx, visible }) => {
                chart.setDatasetVisibility(idx, visible);
            });
            chart.update('none');
        });
    };

    const pairedIndices = (ds, idx) => {
        const out = [idx];
        chart.data.datasets.forEach((peer, i) => {
            if (
                peer.label === `__band_low__${ds.label}` ||
                peer.label === `__band_high__${ds.label}`
            ) {
                out.push(i);
            }
        });
        return out;
    };

    chart.data.datasets.forEach((ds, idx) => {
        if (ds.label && ds.label.startsWith('__band_')) return;

        const row = document.createElement('button');
        row.type = 'button';
        row.className = 'chart-legend-item';
        row.dataset.idx = String(idx);

        const swatch = document.createElement('span');
        swatch.className = 'chart-legend-swatch';
        swatch.style.background = ds.borderColor || '#888';

        const label = document.createElement('span');
        label.className = 'chart-legend-label';
        label.textContent = ds.label;

        row.appendChild(swatch);
        row.appendChild(label);

        row.addEventListener('click', () => {
            const nextVisible = !chart.isDatasetVisible(idx);
            row.classList.toggle('muted', !nextVisible);
            flushVisibility(
                pairedIndices(ds, idx).map((i) => ({ idx: i, visible: nextVisible }))
            );
        });

        rowsByIdx.set(idx, row);
        items.appendChild(row);
    });

    const toggleAll = (visible) => {
        const changes = [];
        chart.data.datasets.forEach((_, i) => {
            changes.push({ idx: i, visible });
        });
        rowsByIdx.forEach((row) => row.classList.toggle('muted', !visible));
        flushVisibility(changes);
    };

    const makeControlButton = (text, onClick) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'chart-legend-control';
        btn.textContent = text;
        btn.addEventListener('click', onClick);
        return btn;
    };

    controls.appendChild(makeControlButton('Show all', () => toggleAll(true)));
    controls.appendChild(makeControlButton('Hide all', () => toggleAll(false)));

    container.appendChild(controls);
    container.appendChild(items);
}

// ─── Cleanup ────────────────────────────────────────────────────────────────

/**
 * Destroy all dynamics charts
 */
export function destroyAllDynamicsCharts() {
    Object.keys(dynamicsCharts).forEach(key => {
        if (dynamicsCharts[key]) {
            dynamicsCharts[key].destroy();
            delete dynamicsCharts[key];
        }
    });
}
