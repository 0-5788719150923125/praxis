/**
 * Praxis Web - Learning Dynamics Visualization
 * Tracks per-layer gradient flow, update ratios, and per-expert dynamics.
 */

import { state, CONSTANTS } from './state.js';
import { fetchAPI } from './api.js';
import { createTabHeader } from './components.js';

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
    const { textColor, gridColor, tooltipBg } = getThemeColors();

    Object.values(dynamicsCharts).forEach(chart => {
        if (!chart) return;

        if (chart.options.scales) {
            Object.values(chart.options.scales).forEach(scale => {
                if (scale.title) scale.title.color = textColor;
                if (scale.ticks) scale.ticks.color = textColor;
                if (scale.grid) scale.grid.color = gridColor;
            });
        }

        if (chart.options.plugins?.legend?.labels) {
            chart.options.plugins.legend.labels.color = textColor;
        }

        if (chart.options.plugins?.tooltip) {
            chart.options.plugins.tooltip.backgroundColor = tooltipBg;
            chart.options.plugins.tooltip.titleColor = textColor;
            chart.options.plugins.tooltip.bodyColor = textColor;
            chart.options.plugins.tooltip.borderColor = gridColor;
        }

        chart.update('none');
    });
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
        const response = await fetch(`/api/dynamics?since=0&limit=1000`);

        if (!response.ok) {
            throw new Error(`API returned ${response.status}`);
        }

        const data = await response.json();

        if (data.status === 'no_data' || !data.runs || data.runs.length === 0) {
            renderEmptyState(container, data.message);
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
 * Render empty state
 */
function renderEmptyState(container, message) {
    const refreshIcon = `
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
            <path fill-rule="evenodd" d="M8 3a5 5 0 1 0 4.546 2.914.5.5 0 0 1 .908-.417A6 6 0 1 1 8 2v1z"/>
            <path d="M8 4.466V.534a.25.25 0 0 1 .41-.192l2.36 1.966c.12.1.12.284 0 .384L8.41 4.658A.25.25 0 0 1 8 4.466z"/>
        </svg>
    `;

    const headerHTML = createTabHeader({
        title: 'Learning Dynamics',
        buttons: [{
            id: 'refresh-dynamics-btn',
            label: 'Refresh',
            icon: refreshIcon,
            className: 'tab-header-button'
        }],
        metadata: '<span><strong>Status:</strong> No data</span>'
    });

    container.innerHTML = `
        ${headerHTML}
        <div class="empty-state" style="margin-top: 2rem;">
            <h3>No Learning Dynamics Yet</h3>
            <p>${message || 'Start training to see learning dynamics'}</p>
        </div>

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
                <div class="chart-subtitle">dy/dx per activation module via autograd (mean across features)</div>
                <div class="chart-wrapper" style="height: 400px;">
                    <canvas id="dynamics-activation-backward"></canvas>
                </div>
                <div class="chart-legend" id="dynamics-activation-backward-legend"></div>
            </div>
        </div>
    `;

    setTimeout(() => loadActivationCurves(), 10);
}

// ─── Metric detection helpers ───────────────────────────────────────────────

/**
 * Detect universal per-layer metrics (layer_X_grad_norm, without "expert")
 */
function detectUniversalLayers(dynamics) {
    const keys = Object.keys(dynamics).filter(k =>
        k.match(/^layer_\d+_grad_norm$/)
    );
    return Array.from(new Set(
        keys.map(k => parseInt(k.match(/^layer_(\d+)_/)[1]))
    )).sort((a, b) => a - b);
}

/**
 * Detect expert metrics (layer_X_expert_Y_grad_norm)
 */
function detectExpertLayers(dynamics) {
    const keys = Object.keys(dynamics).filter(k =>
        k.match(/^layer_\d+_expert_\d+_grad_norm$/)
    );
    const layers = new Set(keys.map(k => parseInt(k.match(/layer_(\d+)_/)[1])));
    const experts = new Set(keys.map(k => parseInt(k.match(/expert_(\d+)_/)[1])));
    return {
        layers: Array.from(layers).sort((a, b) => a - b),
        experts: Array.from(experts).sort((a, b) => a - b)
    };
}

/**
 * Detect halting histogram buckets. Returns {rs, maxLoops} or null if
 * no halting metrics are present (non-KL strategies, or not yet logged).
 */
function detectTaskWeightKeys(dynamics) {
    return Object.keys(dynamics)
        .filter(k => k.startsWith('task_weight_'))
        .sort();
}

function detectHarmonicKeys(dynamics) {
    return Object.keys(dynamics)
        .filter(k => k.startsWith('harmonic_'))
        .sort();
}

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
    const descriptions = runData.descriptions || {};
    const desc = (key, fallback) => descriptions[key] || fallback;

    if (steps.length === 0) {
        renderEmptyState(container, "No dynamics data points found");
        return;
    }

    // Detect what data is available
    const universalLayers = detectUniversalLayers(dynamics);
    const { layers: expertLayers, experts: expertsList } = detectExpertLayers(dynamics);
    const hasUniversal = universalLayers.length > 0;
    const hasExperts = expertLayers.length > 0 && expertsList.length > 0;
    const haltingBuckets = detectHaltingBuckets(dynamics);
    const hasHalting = haltingBuckets !== null;
    const taskWeightKeys = detectTaskWeightKeys(dynamics);
    const hasTaskWeights = taskWeightKeys.length > 0;
    const harmonicKeys = detectHarmonicKeys(dynamics);
    const hasHarmonic = harmonicKeys.length > 0;

    // Layer toggle state — use universal layers, fall back to expert layers
    const allLayers = hasUniversal ? universalLayers :
                      hasExperts ? expertLayers : [];
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
    metaParts.push(`<span><strong>Layers:</strong> ${allLayers.length}</span>`);
    if (hasExperts) {
        metaParts.push(`<span><strong>Experts:</strong> ${expertsList.length}</span>`);
    }

    const headerHTML = createTabHeader({
        title: 'Learning Dynamics',
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

    // Universal charts (always first when available)
    if (hasUniversal) {
        chartsHTML += `
            <div style="margin-top: 2rem;">
                <div class="chart-card">
                    <div class="chart-title">Gradient Flow</div>
                    <div class="chart-subtitle">${desc('layer_grad_norms', 'L2 norm of gradients per decoder layer')}</div>
                    <div class="chart-wrapper" style="height: 400px;">
                        <canvas id="dynamics-layer-grad-norms"></canvas>
                    </div>
                    <div class="chart-legend" id="dynamics-layer-grad-norms-legend"></div>
                </div>
            </div>

            <div style="margin-top: 2rem;">
                <div class="chart-card">
                    <div class="chart-title">Update-to-Weight Ratio</div>
                    <div class="chart-subtitle">${desc('layer_update_ratio', 'Relative update magnitude per layer (||grad|| &times; lr / ||weight||)')}</div>
                    <div class="chart-wrapper" style="height: 400px;">
                        <canvas id="dynamics-layer-update-ratio"></canvas>
                    </div>
                    <div class="chart-legend" id="dynamics-layer-update-ratio-legend"></div>
                </div>
            </div>
        `;
    }

    // Expert charts (conditional, after universal)
    if (hasExperts) {
        chartsHTML += `
            <div style="margin-top: 2rem;">
                <div class="chart-card">
                    <div class="chart-title">Gradient Norms per Expert</div>
                    <div class="chart-subtitle">L2 norm of gradients across all parameters</div>
                    <div class="chart-wrapper" style="height: 400px;">
                        <canvas id="dynamics-grad-norms"></canvas>
                    </div>
                    <div class="chart-legend" id="dynamics-grad-norms-legend"></div>
                </div>
            </div>

            <div style="margin-top: 2rem;">
                <div class="chart-card">
                    <div class="chart-title">Gradient Variance per Expert</div>
                    <div class="chart-subtitle">Variance of gradient values across all parameters</div>
                    <div class="chart-wrapper" style="height: 400px;">
                        <canvas id="dynamics-grad-vars"></canvas>
                    </div>
                    <div class="chart-legend" id="dynamics-grad-vars-legend"></div>
                </div>
            </div>
        `;
    }

    // Task weights (conditional - only when a learnable task weighter is in use)
    if (hasTaskWeights) {
        chartsHTML += `
            <div style="margin-top: 2rem;">
                <div class="chart-card">
                    <div class="chart-title">Task Loss Weights</div>
                    <div class="chart-subtitle">${desc('task_weights', 'Per-task scalar multipliers applied to the loss.')}</div>
                    <div class="chart-wrapper" style="height: 400px;">
                        <canvas id="dynamics-task-weights"></canvas>
                    </div>
                    <div class="chart-legend" id="dynamics-task-weights-legend"></div>
                </div>
            </div>
        `;
    }

    // Harmonic head diagnostics (conditional - only when the field is present)
    if (hasHarmonic) {
        if (harmonicKeys.includes('harmonic_amplitudes_norm')) {
            chartsHTML += `
                <div style="margin-top: 2rem;">
                    <div class="chart-card">
                        <div class="chart-title">Harmonic Field Amplitudes</div>
                        <div class="chart-subtitle">${desc('harmonic_amplitudes_norm', 'L2 norm of the 2D amplitude grid.')}</div>
                        <div class="chart-wrapper" style="height: 400px;">
                            <canvas id="dynamics-harmonic-amplitudes"></canvas>
                        </div>
                    </div>
                </div>
            `;
        }
        if (harmonicKeys.includes('harmonic_grad_ratio')) {
            chartsHTML += `
                <div style="margin-top: 2rem;">
                    <div class="chart-card">
                        <div class="chart-title">Harmonic Gradient Ratio</div>
                        <div class="chart-subtitle">${desc('harmonic_grad_ratio', '||grad(amplitudes)|| / ||grad(lm_head)||')}</div>
                        <div class="chart-wrapper" style="height: 400px;">
                            <canvas id="dynamics-harmonic-grad-ratio"></canvas>
                        </div>
                    </div>
                </div>
            `;
        }
        // Live spectrum heatmap pulled from /api/harmonic_spectrum on render.
        chartsHTML += `
            <div style="margin-top: 2rem;">
                <div class="chart-card">
                    <div class="chart-title">Harmonic Spectrum</div>
                    <div class="chart-subtitle">${desc('harmonic_spectrum', 'Heatmap of |amp[f_t, f_d]|.')}</div>
                    <div class="chart-wrapper" style="height: 400px;">
                        <canvas id="dynamics-harmonic-spectrum"></canvas>
                    </div>
                </div>
            </div>
        `;
    }

    // Halting distribution (conditional - only when a halting strategy is in use)
    if (hasHalting) {
        chartsHTML += `
            <div style="margin-top: 2rem;">
                <div class="chart-card">
                    <div class="chart-title">Halting Distribution</div>
                    <div class="chart-subtitle">Loop counts used per forward pass. Training = random samples (log-normal Poisson); inference = where KL-halting actually fired.</div>
                    <div class="chart-wrapper" style="height: 400px;">
                        <canvas id="dynamics-halting-hist"></canvas>
                    </div>
                </div>
            </div>
        `;
    }

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

    // Layer toggles
    renderDynamicsLayerToggles();

    // Create charts after DOM is ready
    setTimeout(() => {
        try {
            if (hasUniversal) {
                createLayerGradNormsChart('dynamics-layer-grad-norms', dynamics, dynamicsLayerState.layers);
                createLayerUpdateRatioChart('dynamics-layer-update-ratio', dynamics, dynamicsLayerState.layers);
            }
            if (hasExperts) {
                createExpertGradNormsChart('dynamics-grad-norms', dynamics, dynamicsLayerState.layers);
                createExpertGradVarsChart('dynamics-grad-vars', dynamics, dynamicsLayerState.layers);
            }
            if (hasTaskWeights) {
                createTaskWeightsChart('dynamics-task-weights', dynamics, taskWeightKeys);
            }
            if (hasHalting) {
                createHaltingHistogramChart('dynamics-halting-hist', dynamics, haltingBuckets);
            }
            if (hasHarmonic) {
                if (document.getElementById('dynamics-harmonic-amplitudes')) {
                    createHarmonicScalarChart(
                        'dynamics-harmonic-amplitudes', dynamics,
                        'harmonic_amplitudes_norm', 'Amplitudes ||L2||', 'logarithmic'
                    );
                }
                if (document.getElementById('dynamics-harmonic-grad-ratio')) {
                    createHarmonicScalarChart(
                        'dynamics-harmonic-grad-ratio', dynamics,
                        'harmonic_grad_ratio', 'Grad Ratio (Log Scale)', 'logarithmic'
                    );
                }
                if (document.getElementById('dynamics-harmonic-spectrum')) {
                    loadHarmonicSpectrum('dynamics-harmonic-spectrum');
                }
            }
            loadActivationCurves();
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
    const layers = dynamicsLayerState.layers;

    // Universal
    if (document.getElementById('dynamics-layer-grad-norms')) {
        createLayerGradNormsChart('dynamics-layer-grad-norms', dynamics, layers);
    }
    if (document.getElementById('dynamics-layer-update-ratio')) {
        createLayerUpdateRatioChart('dynamics-layer-update-ratio', dynamics, layers);
    }
    // Expert
    if (document.getElementById('dynamics-grad-norms')) {
        createExpertGradNormsChart('dynamics-grad-norms', dynamics, layers);
    }
    if (document.getElementById('dynamics-grad-vars')) {
        createExpertGradVarsChart('dynamics-grad-vars', dynamics, layers);
    }
    // Halting (doesn't depend on layer toggles, but refresh on rebuild)
    const halting = detectHaltingBuckets(dynamics);
    if (halting && document.getElementById('dynamics-halting-hist')) {
        createHaltingHistogramChart('dynamics-halting-hist', dynamics, halting);
    }
    // Task weights (independent of layer selection)
    const taskWeightKeys = detectTaskWeightKeys(dynamics);
    if (taskWeightKeys.length > 0 && document.getElementById('dynamics-task-weights')) {
        createTaskWeightsChart('dynamics-task-weights', dynamics, taskWeightKeys);
    }
    // Harmonic head diagnostics (independent of layer selection)
    if (document.getElementById('dynamics-harmonic-amplitudes')) {
        createHarmonicScalarChart(
            'dynamics-harmonic-amplitudes', dynamics,
            'harmonic_amplitudes_norm', 'Amplitudes ||L2||', 'logarithmic'
        );
    }
    if (document.getElementById('dynamics-harmonic-grad-ratio')) {
        createHarmonicScalarChart(
            'dynamics-harmonic-grad-ratio', dynamics,
            'harmonic_grad_ratio', 'Grad Ratio (Log Scale)', 'logarithmic'
        );
    }
}

// ─── Shared chart helpers ───────────────────────────────────────────────────

const LAYER_COLORS = [
    '#4A90E2', '#FF6B6B', '#00D9FF', '#FFD700', '#00FF9F', '#FF6B9D',
    '#B388FF', '#FF8A65', '#81C784', '#4DD0E1', '#FFB74D', '#CE93D8'
];

const EXPERT_COLORS = [
    '#4A90E2', '#FF6B6B', '#00D9FF', '#FFD700', '#00FF9F', '#FF6B9D'
];

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

        const color = LAYER_COLORS[layer % LAYER_COLORS.length];
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

        const color = LAYER_COLORS[layer % LAYER_COLORS.length];
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
        const color = EXPERT_COLORS[idx % EXPERT_COLORS.length];
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
 * Fetch the live amplitude spectrum and render as a canvas heatmap.
 */
async function loadHarmonicSpectrum(canvasId) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    try {
        const response = await fetch('/api/harmonic_spectrum');
        if (!response.ok) throw new Error(`API returned ${response.status}`);
        const data = await response.json();
        if (data.status !== 'ok' || !Array.isArray(data.spectrum)) return;

        renderHarmonicSpectrum(canvas, data);
    } catch (error) {
        console.error('[Dynamics] Spectrum failed to load:', error);
    }
}

function renderHarmonicSpectrum(canvas, data) {
    const F_t = data.F_t, F_d = data.F_d;
    const spectrum = data.spectrum;
    const peak = data.max || 1.0;

    // The chart wrapper is 400px tall; size the canvas backing store to match
    // the wrapper while drawing a F_t x F_d image scaled to fill it.
    const wrapper = canvas.parentElement;
    const w = wrapper.clientWidth || 800;
    const h = wrapper.clientHeight || 400;
    canvas.width = w;
    canvas.height = h;

    const ctx = canvas.getContext('2d');
    ctx.imageSmoothingEnabled = false;

    // Build an offscreen ImageData at native (F_t, F_d) resolution. Rows are
    // f_t (time-axis frequencies) drawn vertically, columns are f_d (feature
    // -axis frequencies). We then draw the offscreen canvas scaled to fill.
    const off = document.createElement('canvas');
    off.width = F_d;
    off.height = F_t;
    const offCtx = off.getContext('2d');
    const img = offCtx.createImageData(F_d, F_t);

    for (let i = 0; i < F_t; i++) {
        const row = spectrum[i];
        for (let j = 0; j < F_d; j++) {
            const v = peak > 0 ? row[j] / peak : 0;
            const [r, g, b] = magma(v);
            const idx = (i * F_d + j) * 4;
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

    // Reserve a left/bottom margin for axis labels.
    const ml = 60, mb = 30, mt = 8, mr = 16;
    const drawW = Math.max(1, w - ml - mr);
    const drawH = Math.max(1, h - mt - mb);
    ctx.drawImage(off, ml, mt, drawW, drawH);

    ctx.fillStyle = textColor;
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(`f_d (1..${F_d})`, ml + drawW / 2, h - 8);
    ctx.save();
    ctx.translate(16, mt + drawH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(`f_t (1..${F_t})`, 0, 0);
    ctx.restore();
    ctx.textAlign = 'right';
    ctx.fillText(`peak ${peak.toExponential(2)}`, w - 4, mt + 12);
}

/**
 * Single-series scalar over training steps (no legend needed).
 */
function createHarmonicScalarChart(canvasId, dynamics, key, yLabel, yType) {
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
        data: { datasets: [makeLineDataset(yLabel, data, '#B388FF')] },
        options
    });
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

            const color = EXPERT_COLORS[expert % EXPERT_COLORS.length];
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

            const color = EXPERT_COLORS[expert % EXPERT_COLORS.length];
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
            backgroundColor: '#4A90E280',
            borderColor: '#4A90E2',
            borderWidth: 2
        });
    }
    if (evalTotal > 0) {
        datasets.push({
            label: `Inference (learned, n=${evalTotal})`,
            data: evalFreq,
            backgroundColor: '#FF6B6B80',
            borderColor: '#FF6B6B',
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
        const color = LAYER_COLORS[idx % LAYER_COLORS.length];
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
