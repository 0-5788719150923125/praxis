/**
 * Praxis Web - Gradient Dynamics Visualization
 * Tracks gradient norms and variance per expert during training
 */

import { state, CONSTANTS } from './state.js';
import { fetchAPI } from './api.js';
import { createTabHeader } from './components.js';

// Chart instances
export const dynamicsCharts = {};

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
        const response = await fetch(`/api/dynamics?since=0&limit=10000`);

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

    // Detect number of experts
    const expertKeys = Object.keys(dynamics).filter(k => k.match(/^expert_\d+_grad_norm$/));
    const numExperts = new Set(expertKeys.map(k => parseInt(k.match(/expert_(\d+)_/)[1]))).size;

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
        metadata: `
            <span><strong>Points:</strong> ${steps.length}</span>
            <span><strong>Experts:</strong> ${numExperts}</span>
        `
    });

    const chartsHTML = `
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

    container.innerHTML = headerHTML + chartsHTML;

    // Render charts
    setTimeout(() => {
        try {
            createGradientNormsChart('dynamics-grad-norms', dynamics, numExperts);
            createGradientVarsChart('dynamics-grad-vars', dynamics, numExperts);
        } catch (error) {
            console.error('[Dynamics] Chart creation failed:', error);
        }
    }, 10);
}

/**
 * Create gradient norms chart
 */
function createGradientNormsChart(canvasId, dynamics, numExperts) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    if (dynamicsCharts[canvasId]) {
        dynamicsCharts[canvasId].destroy();
    }

    const { textColor, gridColor, tooltipBg } = getThemeColors();
    const steps = dynamics.steps || [];

    const datasets = [];
    const colors = ['#4A90E2', '#FF6B6B', '#00D9FF', '#FFD700', '#00FF9F', '#FF6B9D'];

    for (let i = 0; i < numExperts; i++) {
        const key = `expert_${i}_grad_norm`;
        if (!dynamics[key]) continue;

        const values = dynamics[key];
        const data = steps.map((step, idx) => ({
            x: step,
            y: values[idx]
        })).filter(p => p.y !== null && p.y !== undefined);

        datasets.push({
            label: `Expert ${i}`,
            data: data,
            borderColor: colors[i % colors.length],
            backgroundColor: colors[i % colors.length] + '20',
            borderWidth: 2,
            pointRadius: 0,
            pointHoverRadius: 5,
            tension: 0.3,
            fill: false
        });
    }

    dynamicsCharts[canvasId] = new Chart(ctx, {
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
                    display: true,
                    position: 'top',
                    labels: {
                        color: textColor,
                        usePointStyle: true,
                        padding: 12
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
                        color: textColor
                    },
                    ticks: { color: textColor },
                    grid: { color: gridColor }
                },
                y: {
                    type: 'logarithmic',
                    title: {
                        display: true,
                        text: 'Gradient Norm (L2, Log Scale)',
                        color: textColor
                    },
                    ticks: {
                        color: textColor,
                        callback: (value) => value.toExponential(0)
                    },
                    grid: { color: gridColor }
                }
            }
        }
    });
}

/**
 * Create gradient variance chart
 */
function createGradientVarsChart(canvasId, dynamics, numExperts) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    if (dynamicsCharts[canvasId]) {
        dynamicsCharts[canvasId].destroy();
    }

    const { textColor, gridColor, tooltipBg } = getThemeColors();
    const steps = dynamics.steps || [];

    const datasets = [];
    const colors = ['#4A90E2', '#FF6B6B', '#00D9FF', '#FFD700', '#00FF9F', '#FF6B9D'];

    for (let i = 0; i < numExperts; i++) {
        const key = `expert_${i}_grad_var`;
        if (!dynamics[key]) continue;

        const values = dynamics[key];
        const data = steps.map((step, idx) => ({
            x: step,
            y: values[idx]
        })).filter(p => p.y !== null && p.y !== undefined);

        datasets.push({
            label: `Expert ${i}`,
            data: data,
            borderColor: colors[i % colors.length],
            backgroundColor: colors[i % colors.length] + '20',
            borderWidth: 2,
            pointRadius: 0,
            pointHoverRadius: 5,
            tension: 0.3,
            fill: false
        });
    }

    dynamicsCharts[canvasId] = new Chart(ctx, {
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
                    display: true,
                    position: 'top',
                    labels: {
                        color: textColor,
                        usePointStyle: true,
                        padding: 12
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
                        color: textColor
                    },
                    ticks: { color: textColor },
                    grid: { color: gridColor }
                },
                y: {
                    type: 'logarithmic',
                    title: {
                        display: true,
                        text: 'Gradient Variance (Log Scale)',
                        color: textColor
                    },
                    ticks: {
                        color: textColor,
                        callback: (value) => value.toExponential(0)
                    },
                    grid: { color: gridColor }
                }
            }
        }
    });
}

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
