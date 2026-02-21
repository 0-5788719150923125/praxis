/**
 * Praxis Web - Gradient Dynamics Visualization
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
        textColor: isDark ? '#e0e0e0' : '#1a1a1a',
        gridColor: isDark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.15)',
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
 * Load and render gradient dynamics
 */
export async function loadDynamicsWithCharts(force = false) {
    if (state.dynamics.loaded && !force) {
        return;
    }

    const container = document.getElementById('dynamics-container');
    if (!container) return;

    container.innerHTML = '<div class="loading-placeholder">Loading gradient dynamics...</div>';

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
        title: 'Gradient Dynamics',
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
            <h3>No Gradient Dynamics Yet</h3>
            <p>${message || 'Start training to see gradient dynamics'}</p>
        </div>
    `;
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

// ─── Main render ────────────────────────────────────────────────────────────

/**
 * Render dynamics charts
 */
function renderDynamicsCharts(runData, container) {
    const dynamics = runData.dynamics || {};
    const steps = dynamics.steps || dynamics.step || [];

    if (steps.length === 0) {
        renderEmptyState(container, "No dynamics data points found");
        return;
    }

    // Detect what data is available
    const universalLayers = detectUniversalLayers(dynamics);
    const { layers: expertLayers, experts: expertsList } = detectExpertLayers(dynamics);
    const hasUniversal = universalLayers.length > 0;
    const hasExperts = expertLayers.length > 0 && expertsList.length > 0;

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
        title: 'Gradient Dynamics',
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
                    <div class="chart-subtitle">L2 norm of gradients per decoder layer</div>
                    <div class="chart-wrapper" style="height: 400px;">
                        <canvas id="dynamics-layer-grad-norms"></canvas>
                    </div>
                </div>
            </div>

            <div style="margin-top: 2rem;">
                <div class="chart-card">
                    <div class="chart-title">Update-to-Weight Ratio</div>
                    <div class="chart-subtitle">Relative update magnitude per layer (||grad|| &times; lr / ||weight||)</div>
                    <div class="chart-wrapper" style="height: 400px;">
                        <canvas id="dynamics-layer-update-ratio"></canvas>
                    </div>
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
                </div>
            </div>

            <div style="margin-top: 2rem;">
                <div class="chart-card">
                    <div class="chart-title">Gradient Variance per Expert</div>
                    <div class="chart-subtitle">Variance of gradient values across all parameters</div>
                    <div class="chart-wrapper" style="height: 400px;">
                        <canvas id="dynamics-grad-vars"></canvas>
                    </div>
                </div>
            </div>
        `;
    }

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

    dynamicsCharts[canvasId] = new Chart(ctx, {
        type: 'line',
        data: { datasets },
        options: baseChartOptions(yLabel, yType, textColor, gridColor, tooltipBg)
    });
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
