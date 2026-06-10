/**
 * Praxis Web - Learning Dynamics Visualization
 * Tracks per-layer gradient flow, update ratios, and per-expert dynamics.
 */

import { state, CONSTANTS, chartLineColor } from './state.js';
import { fetchAPI } from './api.js';
import { createTabHeader, pdfButton } from './components.js';
import { formatRelativeTime, initChartDeck, applyChartTheme } from './charts.js';
import { sampleColormap } from './colormaps.js';
import { dedupe, hasRealContent } from './prefetch.js';

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
    await dedupe('tab:dynamics', () => loadDynamicsInner(force));
}

async function loadDynamicsInner(force) {
    if (state.dynamics.loaded && !force) {
        return;
    }

    const container = document.getElementById('dynamics-container');
    if (!container) return;

    if (!hasRealContent(container)) {
        container.innerHTML = '<div class="loading-placeholder">Loading learning dynamics...</div>';
    }

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
            await renderDynamicsCharts({}, container);
            return;
        }

        state.dynamics.data = data.runs[0];
        const steps = data.runs[0].dynamics?.steps || [];
        if (steps.length > 0) {
            state.dynamics.lastStep = Math.max(...steps);
        }

        await renderDynamicsCharts(data.runs[0], container);
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
 * Caller tag for a card title: which component raised the metric. Takes the
 * producing class names feeding the card and renders " (Primary, et al)".
 * "et al" appears only when 2+ distinct producers contribute (or when a
 * family explicitly aggregates many instances of one class via forceEtAl).
 */
function callerTag(callers, forceEtAl = false) {
    const list = (callers || []).filter(Boolean);
    if (!list.length) return '';
    const counts = {};
    list.forEach(n => { counts[n] = (counts[n] || 0) + 1; });
    const primary = Object.keys(counts).sort((a, b) => counts[b] - counts[a])[0];
    const etAl = forceEtAl || new Set(list).size > 1;
    return ` <span class="chart-caller">(${primary}${etAl ? ', et al' : ''})</span>`;
}

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
            caller: e.caller || '',
            callerEtAl: !!e.caller_et_al,
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
    seq_mix: (id, dyn) => createSeqMixChart(id, dyn, detectSeqMixKeys(dyn)),
    halting_hist: (id, dyn) => createHaltingHistogramChart(id, dyn, detectHaltingBuckets(dyn)),
    width_profile: (id, dyn) => createWidthProfileChart(id, dyn, detectWidthDepths(dyn)),
    width_evolution: (id, dyn) => createWidthEvolutionChart(id, dyn, detectWidthDepths(dyn)),
};

/** Layer filter buttons, scoped to a single layer-aware card. */
function layerTogglesHTML(allLayers) {
    if (!allLayers.length) return '';
    const buttons = [
        '<button class="layer-toggle-btn" data-layer="all">All</button>',
        '<button class="layer-toggle-btn" data-layer="none">None</button>',
        ...allLayers.map(l => `<button class="layer-toggle-btn" data-layer="${l}">L${l}</button>`)
    ].join('');
    return `<div class="dynamics-layer-toggles"><strong>Layers:</strong> ${buttons}</div>`;
}

function buildDynamicsFamilyCard(config, desc, allLayers) {
    const legendHTML = config.legend
        ? `<div class="chart-legend" id="${config.canvasId}-legend"></div>`
        : '';
    const togglesHTML = config.layerToggles ? layerTogglesHTML(allLayers) : '';
    return `
        <div style="margin-top: 2rem;">
            <div class="chart-card" data-card-key="${config.key}">
                <div class="chart-title">${config.title}${callerTag(config.caller ? [config.caller] : [], config.callerEtAl)}</div>
                <div class="chart-subtitle">${desc(config.key, config.subtitle)}</div>
                ${togglesHTML}
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

/** Per-multiplier sampling-probability keys for the sequence-length mix,
 * sorted by multiplier (seq_prob_x1, seq_prob_x2, ...). */
function detectSeqMixKeys(dynamics) {
    return Object.keys(dynamics)
        .filter(k => /^seq_prob_x\d+$/.test(k))
        .sort((a, b) => parseInt(a.match(/\d+/)[0]) - parseInt(b.match(/\d+/)[0]));
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

/** Depth indices present in the mixture-of-widths profile series. */
function detectWidthDepths(dynamics) {
    const depths = new Set();
    Object.keys(dynamics).forEach(k => {
        const m = k.match(/^width\/active_d(\d+)$/);
        if (m) depths.add(parseInt(m[1]));
    });
    if (depths.size === 0) return null;
    return Array.from(depths).sort((a, b) => a - b);
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

// Cross-tab group ordering (lower = earlier). Keeps a producer's scalars and
// its snapshots in one contiguous block: heads, then loss/codec, then aux.
// Groups not listed sort after these, alphabetically (deterministic); the
// catch-all buckets trail last.
const GROUP_ORDER = {
    optimizer: 10, memory: 20, arc: 30,
    harmonic_head: 40, crystal_head: 50, parallel_head: 60,
    halo: 70, calm: 80, contrastive_isotropy: 90,
    misc: 900, snapshots: 950,
};
const groupOrder = (name) => GROUP_ORDER[name] ?? 500;

// One manifest for both scalar charts and non-scalar snapshots, grouped so a
// group's cards (e.g. all "halo") render together. Each value is
// ``{scalars, snaps}``; groups are returned in GROUP_ORDER then name order.
function buildMetricManifest(descriptions) {
    const groups = new Map();
    const ensure = (g) => {
        if (!groups.has(g)) groups.set(g, { scalars: [], snaps: [] });
        return groups.get(g);
    };
    for (const [key, entry] of Object.entries(descriptions || {})) {
        if (!entry || typeof entry !== 'object') continue;
        const { chart, snapshot: snap } = entry;
        if (chart && typeof chart === 'object') {
            ensure(chart.group || 'misc').scalars.push(
                { key, chart, description: entry.description, caller: entry.caller });
        }
        if (snap && typeof snap === 'object') {
            ensure(snap.group || 'snapshots').snaps.push(
                { key, snap, description: entry.description, caller: entry.caller });
        }
    }
    for (const g of groups.values()) {
        g.scalars.sort((a, b) => (a.chart.order ?? 0) - (b.chart.order ?? 0));
        g.snaps.sort((a, b) => (a.snap.order ?? 0) - (b.snap.order ?? 0));
    }
    return new Map([...groups.entries()].sort((a, b) =>
        groupOrder(a[0]) - groupOrder(b[0]) || a[0].localeCompare(b[0])));
}

// A scalar metric is "present" only if the run logged a finite value for it.
// The manifest declares more metrics than any single setup emits (e.g. the
// Adam-only second-moment optimizer cards stay empty under Lion), so prune to
// what has data before rendering rather than showing dead cards.
function metricHasData(dynamics, key) {
    const series = dynamics[key];
    return Array.isArray(series) &&
        series.some(v => v !== null && v !== undefined && Number.isFinite(v));
}

function pruneManifestToData(manifest, dynamics) {
    const pruned = new Map();
    for (const [group, g] of manifest) {
        // Scalars need logged data; snapshots are fetched live so they always
        // pass through. Keep the group if either survives.
        const scalars = g.scalars.filter(e => metricHasData(dynamics, e.key));
        if (scalars.length || g.snaps.length) pruned.set(group, { scalars, snaps: g.snaps });
    }
    return pruned;
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
            const item = items[byGroup.get(sg)];
            item.series.push({ key: entry.key, label });
            item.callers.push(entry.caller);
        } else {
            byGroup.set(sg, items.length);
            items.push({
                kind: 'multi',
                canvasId: `dynamics-series-${sg.replace(/_/g, '-')}`,
                lead: entry,
                series: [{ key: entry.key, label }],
                callers: [entry.caller],
            });
        }
    }
    return items;
}

function metricCardHTML(canvasId, title, subtitle, key) {
    return `
        <div style="margin-top: 2rem;">
            <div class="chart-card" data-card-key="${key || ''}">
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
    for (const [, g] of manifest) {
        for (const item of seriesItemsFor(g.scalars)) {
            if (item.kind === 'multi') {
                const c = item.lead.chart;
                html += metricCardHTML(
                    item.canvasId,
                    (c.title || item.lead.key) + callerTag(item.callers),
                    getDesc(item.lead.key, item.lead.description || ''),
                    item.lead.key
                );
            } else {
                html += metricCardHTML(
                    canvasIdForMetric(item.key),
                    (item.chart.title || item.key) + callerTag([item.caller]),
                    getDesc(item.key, item.description || ''),
                    item.key
                );
            }
        }
        // The group's snapshots render right after its scalars (e.g. the HALO
        // energy ring sits with the HALO metrics).
        for (const s of g.snaps) {
            html += snapshotCardHTML(s.key, s.snap, getDesc(s.key, s.description || ''), s.caller);
        }
    }
    return html;
}

function mountManifestCharts(manifest, dynamics) {
    for (const [, g] of manifest) {
        for (const item of seriesItemsFor(g.scalars)) {
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
        out.push({ key, snap: entry.snapshot, description: entry.description, caller: entry.caller });
    }
    out.sort((a, b) => (a.snap.order ?? 0) - (b.snap.order ?? 0));
    return out;
}

// ─── Harmonic sonification (Web Audio) ─────────────────────────────────────
// "Hear" the harmonic field: additive synthesis of its spectrum. Each temporal
// frequency f_t of the amp grid (summed over feature frequencies f_d) drives a
// sine partial of a harmonic series (base * (f_t+1)), amplitude from the
// spectrum - so the field's rhythm becomes a timbre. All in the browser from
// the already-fetched snapshot; no backend round-trip.
let _snapshotCache = {};
let _audioCtx = null;
let _audioStop = null;

function stopHarmonicAudio() {
    if (_audioStop) { _audioStop(); _audioStop = null; }
}

function playHarmonicAudio(spectrum, btn) {
    if (_audioStop) { stopHarmonicAudio(); return; }  // toggle off
    const grid = spectrum && spectrum.grid;
    if (!Array.isArray(grid) || !grid.length) return;
    const amps = grid.map(row => row.reduce((a, b) => a + Math.abs(b), 0));
    const amax = Math.max(...amps, 1e-9);
    try { _audioCtx = _audioCtx || new (window.AudioContext || window.webkitAudioContext)(); }
    catch (e) { return; }
    const ctx = _audioCtx;
    if (ctx.state === 'suspended') ctx.resume();

    const now = ctx.currentTime, dur = 2.6, base = 110;  // A2 fundamental
    const master = ctx.createGain();
    master.gain.value = 0;
    master.connect(ctx.destination);
    const oscs = [];
    for (let f = 1; f < amps.length; f++) {  // skip DC (f=0)
        const a = amps[f] / amax;
        if (a < 0.02) continue;
        const osc = ctx.createOscillator();
        osc.type = 'sine';
        osc.frequency.value = base * (f + 1);
        const g = ctx.createGain();
        g.gain.value = (a * 0.3) / Math.sqrt(f + 1);  // 1/sqrt rolloff so highs don't shriek
        osc.connect(g); g.connect(master);
        osc.start(now); osc.stop(now + dur);
        oscs.push(osc);
    }
    master.gain.setValueAtTime(0, now);
    master.gain.linearRampToValueAtTime(0.8, now + 0.06);
    master.gain.setValueAtTime(0.8, now + dur - 0.5);
    master.gain.linearRampToValueAtTime(0.0001, now + dur);

    if (btn) btn.classList.add('playing');
    _audioStop = () => {
        oscs.forEach(o => { try { o.stop(); } catch (e) {} });
        try { master.disconnect(); } catch (e) {}
        if (btn) btn.classList.remove('playing');
    };
    setTimeout(() => { stopHarmonicAudio(); }, dur * 1000 + 120);
}

function wireHarmonicListenButtons() {
    document.querySelectorAll('.harmonic-listen-btn').forEach(btn => {
        if (btn._wired) return;
        btn._wired = true;
        btn.addEventListener('click', () => playHarmonicAudio(_snapshotCache[btn.dataset.audioKey], btn));
    });
}

function snapshotCardHTML(key, snap, subtitle, caller) {
    // A listen control on the audible harmonic cards; it always synthesizes from
    // that branch's spectrum (real amplitudes), even from the traces card.
    const audible = /harmonic_(spectrum|traces)$/.test(key);
    const listen = audible
        ? `<button class="harmonic-listen-btn" data-audio-key="${key.replace(/harmonic_.*/, 'harmonic_spectrum')}" title="Hear the harmonic field (additive synthesis of its spectrum)"><span class="hl-icon">▶</span> listen</button>`
        : '';
    return `
        <div style="margin-top: 2rem;">
            <div class="chart-card" data-card-key="${key}">
                <div class="chart-title">${snap.title || key}${callerTag([caller])}${listen}</div>
                <div class="chart-subtitle">${subtitle}</div>
                <div class="chart-wrapper" style="height: 400px;">
                    <canvas id="${canvasIdForMetric(key)}"></canvas>
                </div>
            </div>
        </div>
    `;
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
        _snapshotCache = snapshots;  // for the harmonic listen controls
    } catch (e) {
        console.error('[Dynamics] Snapshots failed to load:', e);
        return;
    }

    wireHarmonicListenButtons();

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
    const manifest = pruneManifestToData(
        buildMetricManifest(descriptions), dynamics);

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

    dynamicsLayerState.allLayers = [...allLayers];
    if (!dynamicsLayerState.layers) {
        dynamicsLayerState.layers = [...allLayers];
    }

    // ── Header ──────────────────────────────────────────────────────────
    const metaParts = [`<span><strong>Points:</strong> ${steps.length}</span>`];
    if (expertCount > 0) {
        metaParts.push(`<span><strong>Experts:</strong> ${expertCount}</span>`);
    }

    const headerHTML = createTabHeader({
        title: 'Learning Dynamics',
        additionalContent: renderDynamicsRunSelector(),
        buttons: [
            pdfButton('download-pdf-dynamics'),
        ],
        metadata: metaParts.join('\n')
    });

    // ── Chart cards ─────────────────────────────────────────────────────
    // Families ordered before the head-metric sections (gradient flow,
    // expert charts, task weights), rendered straight from the registry.
    // Layer-aware families carry their own inline layer filter.
    let chartsHTML = familyConfigs
        .filter(c => c.order < 100)
        .map(c => buildDynamicsFamilyCard(c, desc, allLayers))
        .join('');

    // Head-driven metrics (harmonic, crystal, halo, future producers): scalar
    // charts and their non-scalar snapshots, clustered by group and ordered by
    // GROUP_ORDER. New producers opt in just by tagging metric_descriptions.
    chartsHTML += buildManifestSectionsHTML(manifest, desc);

    // Families ordered after the head sections (e.g. halting distribution).
    chartsHTML += familyConfigs
        .filter(c => c.order >= 100)
        .map(c => buildDynamicsFamilyCard(c, desc, allLayers))
        .join('');

    // Activation curves (always rendered; placeholder if endpoint returns empty)
    chartsHTML += `
        <div style="margin-top: 2rem;">
            <div class="chart-card">
                <div class="chart-title" id="activation-forward-title">Activation Forward</div>
                <div class="chart-subtitle" id="activation-forward-subtitle">Forward curve per activation module. Line = mean across features; shaded band = 10-90 percentile spread.</div>
                <div class="chart-wrapper" style="height: 400px;">
                    <canvas id="dynamics-activation-forward"></canvas>
                </div>
                <div class="chart-legend" id="dynamics-activation-forward-legend"></div>
            </div>
        </div>

        <div style="margin-top: 2rem;">
            <div class="chart-card">
                <div class="chart-title" id="activation-backward-title">Activation Derivative</div>
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
    // just move them into one deck; layer-aware cards carry their own filter.
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

    // Create charts after DOM is ready. Awaitable (same contract as the
    // Research render): the loader - and prewarmTab's off-screen layout -
    // hold until the deck has measured with real chart heights, so a visit
    // never triggers a visible re-layout.
    return new Promise(resolve => setTimeout(async () => {
        try {
            // Yield between mounts: a single synchronous pass over the whole
            // deck blocks input for hundreds of ms during page-load prewarm.
            for (const c of familyConfigs) {
                mountDynamicsFamily(c, dynamics);
                await new Promise(r => setTimeout(r, 0));
            }
            mountManifestCharts(manifest, dynamics);
            await new Promise(r => setTimeout(r, 0));
            mountSnapshotCharts(descriptions);
            loadActivationCurves();
            // Re-layout once charts have real heights (deck height tracks the
            // active card); activeIndex is preserved across the re-init.
            initChartDeck('dynamics-deck');
        } catch (error) {
            console.error('[Dynamics] Chart creation failed:', error);
        } finally {
            resolve();
        }
    }, 10));
}

// ─── Layer toggles ──────────────────────────────────────────────────────────

/**
 * Wire the per-card layer filters. The buttons are emitted inline by
 * buildDynamicsFamilyCard; here we just bind clicks and sync active states.
 * Selection is shared, so every card's filter stays in step.
 */
function renderDynamicsLayerToggles() {
    const container = document.getElementById('dynamics-container');
    if (!container) return;
    if ((dynamicsLayerState.allLayers || []).length === 0) return;

    updateDynamicsLayerToggles();

    container.querySelectorAll('.layer-toggle-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const layer = e.currentTarget.dataset.layer;

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
 * Sync active states across every card's layer filter.
 */
function updateDynamicsLayerToggles() {
    const container = document.getElementById('dynamics-container');
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
    mountManifestCharts(buildMetricManifest(descriptions), dynamics);
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

/**
 * Sequence-length curriculum mix: one line per multiplier (×1, ×2, ...)
 * tracing its learned sampling probability over training. Same shape as the
 * task-weights chart; the lines sum to ~1 (a distribution over lengths).
 */
function createSeqMixChart(canvasId, dynamics, keys) {
    const steps = dynamics.steps || [];
    const datasets = [];

    keys.forEach((key, idx) => {
        const values = dynamics[key];
        if (!values) return;

        const data = steps.map((step, i) => ({
            x: step, y: values[i]
        })).filter(p => p.y !== null && p.y !== undefined);

        const label = key.replace(/^seq_prob_x/, '×');  // seq_prob_x4 -> ×4
        const color = chartLineColor(idx);
        datasets.push(makeLineDataset(label, data, color));
    });

    renderChart(canvasId, datasets, 'Sampling Probability', 'linear');
}

// ─── Harmonic head diagnostics (conditional) ────────────────────────────────

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
            const [r, g, b] = sampleColormap('praxis_heat', v);
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

/** Apply an alpha to a hex/rgb/hsl color string for canvas strokes. */
function withAlpha(color, a) {
    color = (color || '').trim();
    if (color.startsWith('#')) {
        let h = color.slice(1);
        if (h.length === 3) h = h.split('').map(c => c + c).join('');
        const n = parseInt(h, 16);
        return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
    }
    if (color.startsWith('hsl(')) return color.replace(')', ` / ${a})`);
    if (color.startsWith('rgb(')) return color.replace('rgb(', 'rgba(').replace(')', `, ${a})`);
    return color;
}

/** Resolve the brand accent to a concrete rgb() via a throwaway probe. */
function readAccentColor() {
    const probe = document.createElement('span');
    probe.style.cssText = 'color: var(--accent); display: none;';
    document.body.appendChild(probe);
    const c = getComputedStyle(probe).color;
    probe.remove();
    return c || 'rgb(26, 161, 121)';
}

/**
 * Harmonic spiral renderer: the real signal behind the old fake "correlation"
 * animation. The head sends the field's top-2 PCA cross-section (x, y) with
 * position as the third axis (z), so the periodic loop unrolls into a rising
 * spiral; ``band`` is the energy left outside the plane, drawn as ribbon width
 * projected radially ("planes from the center"). The only motion is real - a
 * tracer climbing the sequence plus a slow camera spin for depth. No spinning
 * scaffolding, no per-frame model calls.
 */
// Pause a card's heavy canvas animation when it is a non-active card in a deck
// (stacked behind the head, occluded). Standalone cards - not inside a deck -
// always animate. Cuts the dynamics deck from ~one RAF per card down to one for
// the head card, which is the main source of mobile lag.
function deckCardParked(canvas) {
    const card = canvas.closest('.chart-card');
    if (!card || !card.closest('.chart-deck')) return false;
    return !card.classList.contains('deck-active');
}

function renderHarmonicSpiral(canvas, data) {
    if (canvas._harmonicRAF) cancelAnimationFrame(canvas._harmonicRAF);
    const path = Array.isArray(data.path) ? data.path : [];
    const band = Array.isArray(data.band) ? data.band : [];
    if (path.length < 2) return;

    const ctx = canvas.getContext('2d');
    const TILT = 26 * Math.PI / 180;  // camera elevation
    const HEIGHT = 2.4;               // tower height in world units
    const BAND_GAIN = 0.6;
    const PERIOD = 720;               // frames for the tracer to climb once
    let frame = 0, cachedTheme = null, accent = 'rgb(26, 161, 121)';

    const draw = () => {
        if (!canvas.isConnected) { canvas._harmonicRAF = null; return; }
        if (deckCardParked(canvas)) { canvas._harmonicRAF = requestAnimationFrame(draw); return; }
        if (cachedTheme !== state.theme) { cachedTheme = state.theme; accent = readAccentColor(); }

        const wrapper = canvas.parentElement;
        const w = wrapper.clientWidth || 800, h = wrapper.clientHeight || 400;
        if (w < 2 || h < 2) { canvas._harmonicRAF = requestAnimationFrame(draw); return; }
        if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }

        const { textColor, gridColor } = getThemeColors();
        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = gridColor;
        ctx.fillRect(0, 0, w, h);

        const cx = w / 2, cy = h / 2;
        const scale = Math.min(w, h) * 0.30;
        const cosA = Math.cos(frame * 0.005), sinA = Math.sin(frame * 0.005);  // slow spin
        const cosE = Math.cos(TILT), sinE = Math.sin(TILT);

        // Spin about the vertical (z) axis, tilt toward the viewer, project.
        const project = (x, y, z01) => {
            const Z = (z01 - 0.5) * HEIGHT;
            const Xs = x * cosA - y * sinA;
            const Ys = x * sinA + y * cosA;
            return {
                sx: cx + Xs * scale,
                sy: cy - (Z * cosE - Ys * sinE) * scale,
                depth: Ys * cosE + Z * sinE,
            };
        };

        // Build ribbon quads (radial offset) and spine segments, depth-sorted.
        const prims = [];
        let prev = null;
        for (let i = 0; i < path.length; i++) {
            const [x, y, z] = path[i];
            const r = Math.hypot(x, y) || 1e-6;
            const wdt = (band[i] || 0) * BAND_GAIN;
            const center = project(x, y, z);
            const cur = {
                center,
                outer: project(x + (x / r) * wdt, y + (y / r) * wdt, z),
                inner: project(x - (x / r) * wdt, y - (y / r) * wdt, z),
            };
            if (prev) {
                prims.push({
                    type: 'quad', depth: (prev.center.depth + center.depth) / 2,
                    pts: [prev.inner, prev.outer, cur.outer, cur.inner],
                });
                prims.push({
                    type: 'spine', depth: (prev.center.depth + center.depth) / 2,
                    a: prev.center, b: center,
                });
            }
            prev = cur;
        }
        let dmin = Infinity, dmax = -Infinity;
        for (const p of prims) { if (p.depth < dmin) dmin = p.depth; if (p.depth > dmax) dmax = p.depth; }
        const drange = (dmax - dmin) || 1;
        prims.sort((p, q) => p.depth - q.depth);  // painter's algorithm: far first

        for (const p of prims) {
            const near = (p.depth - dmin) / drange;  // 0 far .. 1 near
            if (p.type === 'quad') {
                ctx.beginPath();
                ctx.moveTo(p.pts[0].sx, p.pts[0].sy);
                for (let k = 1; k < 4; k++) ctx.lineTo(p.pts[k].sx, p.pts[k].sy);
                ctx.closePath();
                ctx.fillStyle = withAlpha(accent, 0.08 + 0.24 * near);
                ctx.fill();
            } else {
                ctx.beginPath();
                ctx.moveTo(p.a.sx, p.a.sy);
                ctx.lineTo(p.b.sx, p.b.sy);
                ctx.strokeStyle = withAlpha(accent, 0.4 + 0.5 * near);
                ctx.lineWidth = 1.5;
                ctx.stroke();
            }
        }

        // Tracer climbing the sequence axis.
        const ti = Math.min(path.length - 1, Math.floor(((frame % PERIOD) / PERIOD) * path.length));
        const tp = project(path[ti][0], path[ti][1], path[ti][2]);
        ctx.beginPath();
        ctx.arc(tp.sx, tp.sy, 4, 0, 2 * Math.PI);
        ctx.fillStyle = accent;
        ctx.shadowColor = accent;
        ctx.shadowBlur = 12;
        ctx.fill();
        ctx.shadowBlur = 0;

        const pr = data.participation_ratio;
        if (typeof pr === 'number') {
            ctx.fillStyle = textColor;
            ctx.font = '12px sans-serif';
            ctx.textAlign = 'right';
            ctx.fillText(`eff. dim ${pr.toFixed(1)} · band = spread off-plane`, w - 6, 16);
        }

        frame++;
        canvas._harmonicRAF = requestAnimationFrame(draw);
    };

    canvas._harmonicRAF = requestAnimationFrame(draw);
}

/**
 * Harmonic epicycle renderer: a second lens on the field. The head sends the
 * top-2 PCA loop plus its dominant Fourier modes; we redraw it as nested
 * rotating vectors whose tip traces the curve. The spinning arms are generic
 * Fourier scaffolding (true of any closed curve); the loop shape and the arm
 * lengths are the real per-model signal. Pure client-side off a cached seed.
 */
function renderHarmonicCurve(canvas, data) {
    if (canvas._harmonicRAF) cancelAnimationFrame(canvas._harmonicRAF);
    const modes = Array.isArray(data.modes) ? data.modes : [];
    const points = Array.isArray(data.points) ? data.points : [];
    if (!points.length && !modes.length) return;

    const ctx = canvas.getContext('2d');
    const PERIOD = 600; // frames per full sweep (~10s at 60fps)
    let frame = 0;
    let cachedTheme = null, accent = 'rgb(26, 161, 121)';

    const draw = () => {
        if (!canvas.isConnected) { canvas._harmonicRAF = null; return; }
        if (deckCardParked(canvas)) { canvas._harmonicRAF = requestAnimationFrame(draw); return; }
        if (cachedTheme !== state.theme) {
            cachedTheme = state.theme;
            accent = readAccentColor();
        }

        const wrapper = canvas.parentElement;
        const w = wrapper.clientWidth || 800;
        const h = wrapper.clientHeight || 400;
        if (w < 2 || h < 2) { canvas._harmonicRAF = requestAnimationFrame(draw); return; }
        if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }

        const { textColor, gridColor } = getThemeColors();
        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = gridColor;
        ctx.fillRect(0, 0, w, h);

        const cx = w / 2, cy = h / 2;
        const scale = Math.min(w, h) * 0.4;
        const tau = (frame % PERIOD) / PERIOD;

        // Faint full closed curve for context.
        if (points.length) {
            ctx.beginPath();
            points.forEach((p, i) => {
                const X = cx + p[0] * scale, Y = cy - p[1] * scale;
                i === 0 ? ctx.moveTo(X, Y) : ctx.lineTo(X, Y);
            });
            ctx.closePath();
            ctx.strokeStyle = withAlpha(accent, 0.18);
            ctx.lineWidth = 1;
            ctx.stroke();
        }

        // Epicycle scaffolding: nested rotating vectors at the current phase.
        let px = cx, py = cy;
        for (const m of modes) {
            const ang = 2 * Math.PI * m.f * tau;
            const c = Math.cos(ang), s = Math.sin(ang);
            const dx = m.re * c - m.im * s;
            const dy = m.re * s + m.im * c;
            const nx = px + dx * scale, ny = py - dy * scale;
            const r = Math.hypot(dx, dy) * scale;
            if (r > 1.5) {
                ctx.beginPath();
                ctx.arc(px, py, r, 0, 2 * Math.PI);
                ctx.strokeStyle = withAlpha(textColor, 0.08);
                ctx.lineWidth = 1;
                ctx.stroke();
                ctx.beginPath();
                ctx.moveTo(px, py);
                ctx.lineTo(nx, ny);
                ctx.strokeStyle = withAlpha(textColor, 0.22);
                ctx.stroke();
            }
            px = nx; py = ny;
        }

        // Bright arc traced so far this sweep.
        if (points.length) {
            const upto = Math.max(1, Math.floor(tau * points.length));
            ctx.beginPath();
            for (let i = 0; i <= upto; i++) {
                const X = cx + points[i][0] * scale, Y = cy - points[i][1] * scale;
                i === 0 ? ctx.moveTo(X, Y) : ctx.lineTo(X, Y);
            }
            ctx.strokeStyle = accent;
            ctx.lineWidth = 2;
            ctx.stroke();
        }

        // Glowing tip at the end of the epicycle chain.
        ctx.beginPath();
        ctx.arc(px, py, 3.5, 0, 2 * Math.PI);
        ctx.fillStyle = accent;
        ctx.shadowColor = accent;
        ctx.shadowBlur = 12;
        ctx.fill();
        ctx.shadowBlur = 0;

        const pr = data.participation_ratio;
        if (typeof pr === 'number') {
            ctx.fillStyle = textColor;
            ctx.font = '12px sans-serif';
            ctx.textAlign = 'right';
            ctx.fillText(`eff. dim ${pr.toFixed(1)} · ${modes.length} modes`, w - 6, 16);
        }

        frame++;
        canvas._harmonicRAF = requestAnimationFrame(draw);
    };

    canvas._harmonicRAF = requestAnimationFrame(draw);
}

/**
 * Field traces renderer: the time-domain view. The head sends b(t, d) sampled
 * over one period as per-feature lines; we overlay them (hue ramped around the
 * brand accent) so the harmonics' interference reads as a moiré. Static data,
 * so it draws once - the tick loop only re-renders on resize or theme change.
 */
function renderFieldTraces(canvas, data) {
    if (canvas._harmonicRAF) cancelAnimationFrame(canvas._harmonicRAF);
    const traces = Array.isArray(data.traces) ? data.traces : [];
    const nFeat = traces.length;
    const nTime = nFeat ? traces[0].length : 0;
    if (nTime < 2) return;

    const ctx = canvas.getContext('2d');
    let lastW = 0, lastH = 0, lastTheme = null;

    const render = (w, h) => {
        const { textColor, gridColor } = getThemeColors();
        const baseHue = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--accent-hue')) || 161;
        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = gridColor;
        ctx.fillRect(0, 0, w, h);

        const ml = 8, mr = 8, mt = 8, mb = 8;
        const drawW = w - ml - mr, drawH = h - mt - mb;
        const midY = mt + drawH / 2, amp = (drawH / 2) * 0.92;
        const denom = nFeat - 1 || 1;
        const xAt = (j) => ml + (j / (nTime - 1)) * drawW;
        const yAt = (v) => midY - v * amp;

        for (let i = 0; i < nFeat; i++) {
            const tr = traces[i];
            ctx.beginPath();
            for (let j = 0; j < nTime; j++) {
                const X = xAt(j), Y = yAt(tr[j]);
                j === 0 ? ctx.moveTo(X, Y) : ctx.lineTo(X, Y);
            }
            ctx.strokeStyle = `hsla(${baseHue - 70 + (i / denom) * 140}, 60%, 55%, 0.4)`;
            ctx.lineWidth = 1;
            ctx.stroke();
        }

        ctx.fillStyle = textColor;
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText(`${nFeat} features · b(t,d) over one period`, w - 6, 16);
    };

    const tick = () => {
        if (!canvas.isConnected) { canvas._harmonicRAF = null; return; }
        if (deckCardParked(canvas)) { canvas._harmonicRAF = requestAnimationFrame(tick); return; }
        const wrapper = canvas.parentElement;
        const w = wrapper.clientWidth || 800, h = wrapper.clientHeight || 400;
        if (w >= 2 && h >= 2 && (w !== lastW || h !== lastH || state.theme !== lastTheme)) {
            lastW = w; lastH = h; lastTheme = state.theme;
            if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
            render(w, h);
        }
        canvas._harmonicRAF = requestAnimationFrame(tick);
    };
    canvas._harmonicRAF = requestAnimationFrame(tick);
}

/**
 * Diverging correlation-matrix renderer. Grid values are cosine similarities
 * in [-1, 1]: red = positive, blue = negative, white = zero. Static; only
 * re-renders on resize or theme change.
 */
function renderCorrMatrix(canvas, data) {
    if (canvas._harmonicRAF) cancelAnimationFrame(canvas._harmonicRAF);
    const grid = data.grid;
    if (!Array.isArray(grid) || grid.length === 0) return;
    const rows = grid.length;
    const cols = Array.isArray(grid[0]) ? grid[0].length : 0;
    if (cols === 0) return;

    const ctx = canvas.getContext('2d');
    let lastW = 0, lastH = 0, lastTheme = null;

    // White-centered diverging map: -1 blue, 0 white, +1 red.
    const diverge = (v) => {
        const a = Math.min(1, Math.abs(v)) * 0.75;
        const lo = Math.round(255 * (1 - a));
        return v >= 0 ? [255, lo, lo] : [lo, lo, 255];
    };

    const render = (w, h) => {
        const off = document.createElement('canvas');
        off.width = cols; off.height = rows;
        const offCtx = off.getContext('2d');
        const img = offCtx.createImageData(cols, rows);
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                const [r, g, b] = diverge(grid[i][j]);
                const idx = (i * cols + j) * 4;
                img.data[idx] = r; img.data[idx + 1] = g; img.data[idx + 2] = b; img.data[idx + 3] = 255;
            }
        }
        offCtx.putImageData(img, 0, 0);

        const { textColor, gridColor } = getThemeColors();
        ctx.fillStyle = gridColor;
        ctx.fillRect(0, 0, w, h);
        ctx.imageSmoothingEnabled = false;
        const ml = 50, mb = 28, mt = 8, mr = 12;
        const drawW = Math.max(1, w - ml - mr), drawH = Math.max(1, h - mt - mb);
        ctx.drawImage(off, ml, mt, drawW, drawH);

        ctx.fillStyle = textColor;
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(`feature j (1..${cols})`, ml + drawW / 2, h - 8);
        ctx.save();
        ctx.translate(16, mt + drawH / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText(`feature i (1..${rows})`, 0, 0);
        ctx.restore();
    };

    const tick = () => {
        if (!canvas.isConnected) { canvas._harmonicRAF = null; return; }
        if (deckCardParked(canvas)) { canvas._harmonicRAF = requestAnimationFrame(tick); return; }
        const wrapper = canvas.parentElement;
        const w = wrapper.clientWidth || 800, h = wrapper.clientHeight || 400;
        if (w >= 2 && h >= 2 && (w !== lastW || h !== lastH || state.theme !== lastTheme)) {
            lastW = w; lastH = h; lastTheme = state.theme;
            if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
            render(w, h);
        }
        canvas._harmonicRAF = requestAnimationFrame(tick);
    };
    canvas._harmonicRAF = requestAnimationFrame(tick);
}

/** 2D convex hull (monotone chain) of {x, y} points - a block's silhouette. */
function convexHull2D(pts) {
    const p = pts.slice().sort((a, b) => a.x - b.x || a.y - b.y);
    if (p.length < 3) return p;
    const cross = (o, a, b) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
    const lower = [];
    for (const q of p) {
        while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], q) <= 0) lower.pop();
        lower.push(q);
    }
    const upper = [];
    for (let i = p.length - 1; i >= 0; i--) {
        const q = p[i];
        while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], q) <= 0) upper.pop();
        upper.push(q);
    }
    lower.pop(); upper.pop();
    return lower.concat(upper);
}

/**
 * Harmonic staircase renderer: the frequency-domain sibling of the spiral. The
 * head sends one block per harmonic (amplitude + Weyl phase); we stack them by
 * energy rank, fan them around a column by phase, size them by amplitude, and
 * draw shaded 3D boxes. A faint silhouette under each block keeps it from
 * blinking out edge-on. Slow camera spin for depth; no fake data motion.
 */
function renderHarmonicStaircase(canvas, data) {
    if (canvas._harmonicRAF) cancelAnimationFrame(canvas._harmonicRAF);
    const steps = Array.isArray(data.steps) ? data.steps : [];
    if (!steps.length) return;

    const ctx = canvas.getContext('2d');
    const TILT = 24 * Math.PI / 180, HEIGHT = 2.4;
    const LIGHT = [0.4, -0.4, 0.82];
    const n = steps.length;
    let frame = 0, cachedTheme = null, accent = [26, 161, 121];
    const parseRGB = (s) => { const m = (s || '').match(/\d+/g); return m ? [+m[0], +m[1], +m[2]] : [26, 161, 121]; };

    const draw = () => {
        if (!canvas.isConnected) { canvas._harmonicRAF = null; return; }
        if (deckCardParked(canvas)) { canvas._harmonicRAF = requestAnimationFrame(draw); return; }
        if (cachedTheme !== state.theme) { cachedTheme = state.theme; accent = parseRGB(readAccentColor()); }

        const wrapper = canvas.parentElement;
        const w = wrapper.clientWidth || 800, h = wrapper.clientHeight || 400;
        if (w < 2 || h < 2) { canvas._harmonicRAF = requestAnimationFrame(draw); return; }
        if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }

        const { textColor, gridColor } = getThemeColors();
        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = gridColor;
        ctx.fillRect(0, 0, w, h);

        const cx = w / 2, cy = h / 2, scale = Math.min(w, h) * 0.22;
        const cosA = Math.cos(frame * 0.0022), sinA = Math.sin(frame * 0.0022);
        const cosE = Math.cos(TILT), sinE = Math.sin(TILT);
        const project = (x, y, z) => {
            const Xs = x * cosA - y * sinA, Ys = x * sinA + y * cosA;
            return { sx: cx + Xs * scale, sy: cy - (z * cosE - Ys * sinE) * scale, depth: Ys * cosE + z * sinE, wx: Xs, wy: Ys, wz: z };
        };

        const faces = [];
        for (let i = 0; i < n; i++) {
            const s = steps[i];
            const a = Math.max(0, Math.min(1, s.a));
            const fnorm = Math.max(0, Math.min(1, s.fnorm || 0));
            const wz = (n > 1 ? i / (n - 1) : 0.5) * HEIGHT - HEIGHT / 2;
            // Position around the column by phase; radius spread by frequency
            // (a real per-plank value), with a base so nothing piles at center.
            const ang = s.phase, rad = 0.8 + 1.2 * fnorm;
            const ccx = rad * Math.cos(ang), ccy = rad * Math.sin(ang);
            // Long 2x4 plank: length dominates; width by amplitude, thickness by
            // frequency (low freq = thicker). Oriented along its frequency dir.
            const L = 0.5 + 0.25 * a;
            const hx = L / 2, hy = L * (0.08 + 0.06 * a), hz = L * (0.04 + 0.05 * (1 - fnorm));
            const alpha = s.yaw;
            const ca = Math.cos(alpha), sa = Math.sin(alpha);
            const corners = [];
            for (const sz of [-hz, hz])
                for (const sy of [-hy, hy])
                    for (const sx of [-hx, hx]) {
                        const lx = sx * ca - sy * sa, ly = sx * sa + sy * ca;
                        corners.push(project(ccx + lx, ccy + ly, wz + sz));
                    }
            let bx = 0, by = 0, bz = 0;  // block center (post-spin) for outward test
            for (const c of corners) { bx += c.wx; by += c.wy; bz += c.wz; }
            bx /= 8; by /= 8; bz /= 8;
            // Faint silhouette so the plank never blinks out when seen edge-on.
            faces.push({ type: 'hull', depth: by * cosE + bz * sinE, hull: convexHull2D(corners.map(c => ({ x: c.sx, y: c.sy }))) });
            const C = (ix, iy, iz) => corners[iz * 4 + iy * 2 + ix];
            const facedefs = [
                [[0,0,1],[1,0,1],[1,1,1],[0,1,1]], [[0,0,0],[0,1,0],[1,1,0],[1,0,0]],
                [[0,0,0],[1,0,0],[1,0,1],[0,0,1]], [[0,1,0],[0,1,1],[1,1,1],[1,1,0]],
                [[0,0,0],[0,0,1],[0,1,1],[0,1,0]], [[1,0,0],[1,1,0],[1,1,1],[1,0,1]],
            ];
            for (const fd of facedefs) {
                const p = fd.map(([ix, iy, iz]) => C(ix, iy, iz));
                const e1 = [p[1].wx - p[0].wx, p[1].wy - p[0].wy, p[1].wz - p[0].wz];
                const e2 = [p[2].wx - p[0].wx, p[2].wy - p[0].wy, p[2].wz - p[0].wz];
                let nx = e1[1]*e2[2]-e1[2]*e2[1], ny = e1[2]*e2[0]-e1[0]*e2[2], nz = e1[0]*e2[1]-e1[1]*e2[0];
                const nl = Math.hypot(nx, ny, nz) || 1;
                nx /= nl; ny /= nl; nz /= nl;
                // orient normal outward, then cull faces pointing away from camera
                const fcx = (p[0].wx+p[1].wx+p[2].wx+p[3].wx)/4;
                const fcy = (p[0].wy+p[1].wy+p[2].wy+p[3].wy)/4;
                const fcz = (p[0].wz+p[1].wz+p[2].wz+p[3].wz)/4;
                if (nx*(fcx-bx) + ny*(fcy-by) + nz*(fcz-bz) < 0) { nx = -nx; ny = -ny; nz = -nz; }
                if (ny*cosE + nz*sinE <= 0) continue;  // back-face: camera views along (0, cosE, sinE)
                const ndl = Math.max(0, nx*LIGHT[0]+ny*LIGHT[1]+nz*LIGHT[2]);
                faces.push({ type: 'face', p, bright: 0.42 + 0.58 * ndl, depth: (p[0].depth+p[1].depth+p[2].depth+p[3].depth)/4 });
            }
        }
        faces.sort((u, v) => u.depth - v.depth);  // painter's: far first

        // central column behind the blocks
        const c0 = project(0, 0, -HEIGHT/2), c1 = project(0, 0, HEIGHT/2);
        ctx.strokeStyle = withAlpha(textColor, 0.15);
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(c0.sx, c0.sy); ctx.lineTo(c1.sx, c1.sy); ctx.stroke();

        for (const f of faces) {
            if (f.type === 'hull') {
                if (f.hull.length < 2) continue;
                ctx.beginPath();
                ctx.moveTo(f.hull[0].x, f.hull[0].y);
                for (let k = 1; k < f.hull.length; k++) ctx.lineTo(f.hull[k].x, f.hull[k].y);
                ctx.closePath();
                ctx.fillStyle = withAlpha(`rgb(${accent[0]}, ${accent[1]}, ${accent[2]})`, 0.1);
                ctx.fill();
                continue;
            }
            ctx.beginPath();
            ctx.moveTo(f.p[0].sx, f.p[0].sy);
            for (let k = 1; k < 4; k++) ctx.lineTo(f.p[k].sx, f.p[k].sy);
            ctx.closePath();
            const b = f.bright;
            ctx.fillStyle = `rgb(${Math.round(accent[0]*b)}, ${Math.round(accent[1]*b)}, ${Math.round(accent[2]*b)})`;
            ctx.fill();
            ctx.strokeStyle = withAlpha(textColor, 0.12);
            ctx.lineWidth = 0.5;
            ctx.stroke();
        }

        ctx.fillStyle = textColor;
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText(`${n} harmonics · height = energy, angle = phase, plank = frequency`, w - 6, 16);

        frame++;
        canvas._harmonicRAF = requestAnimationFrame(draw);
    };
    canvas._harmonicRAF = requestAnimationFrame(draw);
}

/**
 * Bias/variance strands: a morphing cylinder of feature-particles. One flat end
 * (z=0) arranges them by phase on a ring - the static field, pure bias, white.
 * The other end (z=1) is the (bias energy, variance energy) plane: x = static
 * energy, y = input-conditional energy. Variance-dominant features lift off the
 * bias axis and take the accent color. With no input-conditional field the plane
 * stays collapsed on the bias axis - the split appearing is the trained result.
 * Endpoints are measured; the cylinder between them is interpolation.
 */
function renderHarmonicStrands(canvas, data) {
    if (canvas._strandsRAF) cancelAnimationFrame(canvas._strandsRAF);
    const angle = data.angle || [], bias = data.bias_energy || [], vr = data.var_energy || [];
    const N = Math.min(angle.length, bias.length, vr.length);
    if (N < 2) return;

    const ctx = canvas.getContext('2d');
    const TILT = 26 * Math.PI / 180, HEIGHT = 2.4, STEPS = 28, TWIST = Math.PI * 1.25;
    const SAMP = STEPS + 1, cosE = Math.cos(TILT), sinE = Math.sin(TILT);
    let frame = 0;

    // Precompute ONCE: per-feature color along the diverging bias/variance
    // spectrum (t = variance/(bias+variance); blue = pure bias, red = pure
    // variance), and the frame-independent 3D strand geometry. The hot loop then
    // only rotates+projects these flat buffers - no trig, no allocation per frame.
    // First pass: peak energy on either axis (geometry scale - so a bias-free
    // "pure" arm still spans the geometry) and the heaviest per-feature
    // variance share (color reference).
    let bmax = 1e-6, tmax = 0, biasMax = 0;
    for (let i = 0; i < N; i++) {
        const b = Math.max(bias[i], 0), v = Math.max(vr[i], 0);
        bmax = Math.max(bmax, b, v);
        biasMax = Math.max(biasMax, b);
        if (b + v > 1e-9) tmax = Math.max(tmax, v / (b + v));
    }
    // A bias-free ("pure") arm reverses the color ramp: variance pushes back
    // from the other end of the corkscrew, so red enters at the ring.
    const pure = biasMax <= 1e-9;
    // Color reference: the field's heaviest per-feature variance, floored so a
    // near-zero-variance field stays blue (we don't amplify noise to red). When
    // variance IS present, each strand's tip color is its variance share RELATIVE
    // to that heaviest feature - so per-feature specialization shows even when the
    // absolute field fraction is small (the absolute % is the readout up top).
    const tref = Math.max(tmax, 0.04);
    // Each hair ramps from the blue bias reference at its base (z=0) to its own
    // relative balance at the tip: the most variance-heavy feature reddens, a
    // half-as-heavy one tops out white, a bias-dominant one stays blue.
    const segCol = new Array(N * SAMP);
    const gx = new Float32Array(N * SAMP), gy = new Float32Array(N * SAMP), gz = new Float32Array(N * SAMP);
    for (let i = 0; i < N; i++) {
        const b = Math.max(bias[i], 0), v = Math.max(vr[i], 0);
        const r = Math.sqrt(b / bmax), px = r * 2 - 1, py = Math.sqrt(v / bmax) * 2 - 1;
        const t = b + v > 1e-9 ? v / (b + v) : 0;
        const rel = Math.min(1, t / tref);   // variance share vs the heaviest feature
        const a = angle[i], o = i * SAMP;
        for (let s = 0; s < SAMP; s++) {
            const z = s / STEPS, wgt = 1 - z, th = a + z * TWIST;
            gx[o + s] = r * Math.cos(th) * wgt + px * z;   // ring -> plane morph
            gy[o + s] = r * Math.sin(th) * wgt + py * z;
            gz[o + s] = (z - 0.5) * HEIGHT;                // pre-centered cylinder height
            const c = sampleColormap('bias_variance', rel * (pure ? 1 - z : z));   // blue base -> colormap(rel) tip (reversed for pure)
            segCol[o + s] = `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
        }
    }
    // Reusable per-frame buffers (allocated once -> no GC churn in the hot loop).
    const psx = new Float32Array(N * SAMP), psy = new Float32Array(N * SAMP);
    const dmean = new Float32Array(N), order = new Int32Array(N);
    for (let i = 0; i < N; i++) order[i] = i;

    const draw = () => {
        if (!canvas.isConnected) { canvas._strandsRAF = null; return; }
        if (deckCardParked(canvas)) { canvas._strandsRAF = requestAnimationFrame(draw); return; }
        const wrapper = canvas.parentElement;
        const w = wrapper.clientWidth || 800, h = wrapper.clientHeight || 400;
        if (w < 2 || h < 2) { canvas._strandsRAF = requestAnimationFrame(draw); return; }
        if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }

        const { textColor, gridColor } = getThemeColors();
        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = gridColor;
        ctx.fillRect(0, 0, w, h);

        const cx = w / 2, cy = h / 2, scale = Math.min(w, h) * 0.30;
        const cosA = Math.cos(frame * 0.005), sinA = Math.sin(frame * 0.005);

        // Rotate + project every precomputed point (branchless), and accumulate
        // each strand's mean depth for back-to-front ordering.
        for (let i = 0; i < N; i++) {
            const o = i * SAMP;
            let dsum = 0;
            for (let s = 0; s < SAMP; s++) {
                const j = o + s, X = gx[j], Y = gy[j], Z = gz[j];
                const Xs = X * cosA - Y * sinA, Ys = X * sinA + Y * cosA;
                psx[j] = cx + Xs * scale;
                psy[j] = cy - (Z * cosE - Ys * sinE) * scale;
                dsum += Ys * cosE + Z * sinE;
            }
            dmean[i] = dsum / SAMP;
        }
        order.sort((a, b) => dmean[a] - dmean[b]);  // back-to-front

        // One continuous hair per feature - a full warp in X, Y, Z, colored by
        // its bias/variance balance.
        ctx.lineWidth = 1.3;
        ctx.lineJoin = 'round';
        for (let k = 0; k < N; k++) {
            const i = order[k], o = i * SAMP;
            ctx.globalAlpha = 0.2 + 0.5 * ((dmean[i] + 1.5) / 3);
            // Draw the gradient as runs of constant (colormap-quantised) color so
            // it stays a smooth blue->red gradient at a few strokes per hair, not
            // one per segment. Adjacent runs share their junction point.
            let s = 0;
            while (s < STEPS) {
                const c = segCol[o + s + 1];   // color of segment s -> s+1
                ctx.strokeStyle = c;
                ctx.beginPath();
                ctx.moveTo(psx[o + s], psy[o + s]);
                let e = s + 1;
                ctx.lineTo(psx[o + e], psy[o + e]);
                while (e < STEPS && segCol[o + e + 1] === c) {
                    e++;
                    ctx.lineTo(psx[o + e], psy[o + e]);
                }
                ctx.stroke();
                s = e;
            }
        }
        ctx.globalAlpha = 1;

        const sep = data.separated || 0;
        ctx.fillStyle = textColor;
        ctx.font = '11px monospace';
        ctx.fillText(`variance ${(100 * sep).toFixed(0)}% of field energy`, 10, 16);
        ctx.fillText(
            sep < 0.01
                ? 'pure bias - strands not separated yet'
                : 'hue = per-feature share, relative to the heaviest',
            10, 30
        );

        // Spectrum key: each hair runs blue (pure bias) -> red (pure variance).
        const lo = sampleColormap('bias_variance', 0), hi = sampleColormap('bias_variance', 1);
        ctx.fillStyle = `rgb(${lo[0]}, ${lo[1]}, ${lo[2]})`;
        ctx.fillText('bias', 10, h - 10);
        ctx.fillStyle = textColor;
        ctx.fillText('-', 38, h - 10);
        ctx.fillStyle = `rgb(${hi[0]}, ${hi[1]}, ${hi[2]})`;
        ctx.fillText('variance', 48, h - 10);

        frame++;
        canvas._strandsRAF = requestAnimationFrame(draw);
    };
    canvas._strandsRAF = requestAnimationFrame(draw);
}

/**
 * HALO energy ring: a polar map of the batch's radial energy. Each token's
 * embedding has a mean-square radius; HALO drives them onto a shell of radius
 * sqrt(1 - 2/D). We paint the radii histogram as concentric rings - bright
 * where consensus embeddings pile onto the shell (the ring of bias), dark
 * toward the origin (the abstain sink) and beyond (variance, no structure).
 */
function renderHaloRing(canvas, data) {
    if (canvas._haloRAF) cancelAnimationFrame(canvas._haloRAF);
    const radii = data.radii || [];
    const BINS = radii.length;
    if (BINS < 2) return;

    const ctx = canvas.getContext('2d');
    const rMax = data.r_max || 1, shell = data.shell_r || 0;
    let frame = 0;

    const draw = () => {
        if (!canvas.isConnected) { canvas._haloRAF = null; return; }
        if (deckCardParked(canvas)) { canvas._haloRAF = requestAnimationFrame(draw); return; }
        const wrapper = canvas.parentElement;
        const w = wrapper.clientWidth || 800, h = wrapper.clientHeight || 400;
        if (w < 2 || h < 2) { canvas._haloRAF = requestAnimationFrame(draw); return; }
        if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }

        const { textColor } = getThemeColors();
        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = '#000';            // the void: pure variance has no structure
        ctx.fillRect(0, 0, w, h);

        const cx = w / 2, cy = h / 2, maxR = Math.min(w, h) * 0.42;
        const thick = (maxR / BINS) + 1.2;

        // Outer -> inner so the bright shell sits cleanly over the dark interior.
        for (let b = BINS - 1; b >= 0; b--) {
            const rData = ((b + 0.5) / BINS) * rMax;
            const cr = (rData / rMax) * maxR;
            const dens = Math.min(1, Math.max(0, radii[b]));
            if (dens <= 0.002) continue;
            const c = sampleColormap('halo_energy', dens);
            ctx.strokeStyle = `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
            ctx.lineWidth = thick;
            ctx.beginPath();
            ctx.arc(cx, cy, cr, 0, 2 * Math.PI);
            ctx.stroke();
        }

        // Faint marker for the theoretical shell radius sqrt(1 - 2/D).
        if (shell > 0) {
            ctx.strokeStyle = 'rgba(120, 200, 255, 0.35)';
            ctx.lineWidth = 1;
            ctx.setLineDash([4, 4]);
            ctx.beginPath();
            ctx.arc(cx, cy, (shell / rMax) * maxR, 0, 2 * Math.PI);
            ctx.stroke();
            ctx.setLineDash([]);
        }

        ctx.fillStyle = textColor;
        ctx.font = '11px monospace';
        ctx.fillText(`ring of bias @ r=${shell.toFixed(3)} - inner/outer void = variance`, 10, 16);
        ctx.fillText(`${data.n || 0} tokens`, 10, h - 10);

        frame++;
        canvas._haloRAF = requestAnimationFrame(draw);
    };
    canvas._haloRAF = requestAnimationFrame(draw);
}

/**
 * Sequence snake on the dial: the field over a single sequence as 12 blocks on a
 * clock-face of four quadrants. Angle = the field's PCA phase per sample, radius
 * = time sinking toward the origin (newest at center). The badge reports the
 * winding number - whether the snake actually circles the origin, from the data.
 */
function renderHarmonicSnake(canvas, data) {
    const pts = (data && data.points) || [];
    if (pts.length < 2) return;

    const draw = () => {
        if (!canvas.isConnected) return;
        const wrap = canvas.parentElement;
        const w = wrap.clientWidth || 600, h = wrap.clientHeight || 400;
        if (w < 2 || h < 2) return;
        const dpr = window.devicePixelRatio || 1;
        canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
        canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
        const ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        const { textColor } = getThemeColors();
        ctx.fillStyle = '#0f1117'; ctx.fillRect(0, 0, w, h);

        const cx = w / 2, cy = h / 2, R = Math.min(w, h) * 0.40;
        // Clock-face frame: outer ring, quadrant cross, 12 hour ticks.
        ctx.strokeStyle = 'rgba(150,160,180,0.22)'; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.arc(cx, cy, R, 0, 2 * Math.PI); ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(cx - R, cy); ctx.lineTo(cx + R, cy);
        ctx.moveTo(cx, cy - R); ctx.lineTo(cx, cy + R);
        ctx.stroke();
        for (let i = 0; i < 12; i++) {
            const a = Math.PI / 2 - i * Math.PI / 6;
            ctx.beginPath();
            ctx.moveTo(cx + R * 0.93 * Math.cos(a), cy - R * 0.93 * Math.sin(a));
            ctx.lineTo(cx + R * Math.cos(a), cy - R * Math.sin(a));
            ctx.stroke();
        }
        ctx.fillStyle = 'rgba(150,160,180,0.5)';
        ctx.beginPath(); ctx.arc(cx, cy, 2.5, 0, 2 * Math.PI); ctx.fill();

        // The snake: 12 blocks at (angle, radius=time), tail(old)->head(now).
        const P = pts.map(p => [cx + R * p.radius * Math.cos(p.angle),
                                cy - R * p.radius * Math.sin(p.angle)]);
        const mid = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
        const col = (t) => `rgba(${90 + 150 * t}, ${150 + 90 * t}, 245, 0.9)`;
        // Body curves AROUND each point (quadratic anchored to the segment
        // midpoints, control = the point) rather than straight through, with each
        // control warped perpendicular by a cosine * magnitude factor so the
        // snake undulates - the "one additional factor" from the per-point PCA
        // magnitude. Blocks stay at the true points; the line bends around them.
        const ctrl = P.map((p, k) => {
            const a = P[Math.max(0, k - 1)], b = P[Math.min(P.length - 1, k + 1)];
            const dx = b[0] - a[0], dy = b[1] - a[1], L = Math.hypot(dx, dy) || 1;
            const warp = R * 0.13 * (pts[k].mag || 0) * Math.cos(k * Math.PI);
            return [p[0] + (-dy / L) * warp, p[1] + (dx / L) * warp];
        });
        ctx.lineWidth = 2.2; ctx.lineJoin = 'round'; ctx.lineCap = 'round';
        for (let k = 0; k < P.length - 1; k++) {
            const start = k === 0 ? P[0] : mid(P[k - 1], P[k]);
            const end = mid(P[k], P[k + 1]);
            ctx.strokeStyle = col(k / (P.length - 1));
            ctx.beginPath();
            ctx.moveTo(start[0], start[1]);
            ctx.quadraticCurveTo(ctrl[k][0], ctrl[k][1], end[0], end[1]);
            ctx.stroke();
        }
        {  // final stub: last midpoint -> head, bowing around the last control
            const k = P.length - 1;
            ctx.strokeStyle = col(1);
            ctx.beginPath();
            const s = mid(P[k - 1], P[k]);
            ctx.moveTo(s[0], s[1]);
            ctx.quadraticCurveTo(ctrl[k][0], ctrl[k][1], P[k][0], P[k][1]);
            ctx.stroke();
        }
        for (let k = 0; k < P.length; k++) {
            const t = k / (P.length - 1);
            const sz = 5 + 5 * (pts[k].mag || 0);
            ctx.fillStyle = `rgb(${90 + 150 * t}, ${150 + 90 * t}, 245)`;
            ctx.fillRect(P[k][0] - sz / 2, P[k][1] - sz / 2, sz, sz);
        }

        const ok = !!data.circles_origin;
        ctx.font = '11px monospace'; ctx.textAlign = 'left';
        ctx.fillStyle = ok ? '#5fd08a' : '#9aa3b2';
        ctx.fillText(`circles the origin: ${ok ? 'yes' : 'no'}  (winding ${(data.winding ?? 0).toFixed(2)})`, 10, 18);
        ctx.fillStyle = textColor; ctx.font = '9px monospace';
        ctx.fillText('12 samples · radius = time → origin · angle = field phase', 10, h - 8);
    };

    draw();
    if (canvas._snakeRO) canvas._snakeRO.disconnect();
    canvas._snakeRO = new ResizeObserver(() => draw());
    canvas._snakeRO.observe(canvas.parentElement);
}

const SNAPSHOT_RENDERERS = {
    heatmap_2d: renderHeatmap2D,
    halo_ring: renderHaloRing,
    harmonic_spiral: renderHarmonicSpiral,
    harmonic_curve: renderHarmonicCurve,
    field_traces: renderFieldTraces,
    corr_matrix: renderCorrMatrix,
    harmonic_staircase: renderHarmonicStaircase,
    harmonic_strands: renderHarmonicStrands,
    harmonic_snake: renderHarmonicSnake,
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

// Mixture-of-widths profile: active inner-width fraction per recurrent depth.
// A filled arch over depth - inflating early, decaying through the tail - so you
// can see at a glance that features grow at the front of the stack and thin out
// late. (When the schedule is learned, this arch will move over training.)
function createWidthProfileChart(canvasId, dynamics, depths) {
    const ctx = document.getElementById(canvasId);
    if (!ctx || !depths) return;

    if (dynamicsCharts[canvasId]) {
        dynamicsCharts[canvasId].destroy();
    }

    const { textColor, gridColor, tooltipBg } = getThemeColors();

    const labels = depths.map(d => `d${d}`);
    const fracs = depths.map(d => latestValue(dynamics[`width/active_d${d}`]) || 0);

    dynamicsCharts[canvasId] = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [{
                label: 'Active width',
                data: fracs,
                borderColor: chartLineColor(0),
                backgroundColor: chartLineColor(0) + '33',
                borderWidth: 2,
                fill: true,
                tension: 0.35,
                pointRadius: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
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
                        label: (tctx) => `${(tctx.parsed.y * 100).toFixed(1)}% of inner width`
                    }
                }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Recurrent Depth', color: textColor },
                    ticks: { color: textColor },
                    grid: { display: false }
                },
                y: {
                    title: { display: true, text: 'Active Width', color: textColor },
                    ticks: {
                        color: textColor,
                        callback: (v) => `${(v * 100).toFixed(0)}%`
                    },
                    grid: { color: gridColor },
                    beginAtZero: true,
                    max: 1
                }
            }
        }
    });
}

// Mixture-of-widths over training: faint per-depth strata (the arch held over
// time) plus a bold realized-mean line that wanders as halting changes how deep
// the loop runs. Together: the distribution shape AND its movement over time.
function createWidthEvolutionChart(canvasId, dynamics, depths) {
    if (!depths) return;
    const steps = dynamics.steps || [];
    const series = (key) => steps
        .map((step, idx) => ({ x: step, y: (dynamics[key] || [])[idx] }))
        .filter(p => p.y !== null && p.y !== undefined);

    const datasets = depths.map(d => {
        const ds = makeLineDataset(`d${d}`, series(`width/active_d${d}`), chartLineColor(d));
        ds.borderWidth = 1;
        ds.borderColor = chartLineColor(d) + '66';  // faint strata
        return ds;
    });

    const realized = series('width/realized_mean');
    if (realized.length) {
        const { textColor } = getThemeColors();
        const bold = makeLineDataset('realized mean', realized, textColor);
        bold.borderWidth = 2.5;
        datasets.push(bold);
    }

    renderChart(canvasId, datasets, 'Active Width', 'linear');
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
        // Caller tag: the activation module class(es) that produced these curves.
        const callers = curves.map(c => c.type).filter(Boolean);
        ['activation-forward-title', 'activation-backward-title'].forEach((id, i) => {
            const el = document.getElementById(id);
            if (!el) return;
            const base = i === 0 ? 'Activation Forward' : 'Activation Derivative';
            el.innerHTML = base + callerTag(callers, callers.length > 1);
        });

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
