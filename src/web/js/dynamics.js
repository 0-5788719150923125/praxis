/**
 * Praxis Web - Gradient Dynamics Visualization
 * Expert Gradient Divergence Tracker - validates dual-sided perturbation hypothesis
 */

import { state, CONSTANTS } from './state.js';
import { fetchAPI } from './api.js';
import { createTabHeader } from './components.js';

// Chart instances for dynamics
const dynamicsCharts = {};

/**
 * Get theme-appropriate colors (reuse from charts.js pattern)
 */
function getThemeColors() {
    const isDark = state.theme === 'dark';
    return {
        textColor: isDark ? '#e0e0e0' : '#1a1a1a',
        gridColor: isDark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.15)',
        tooltipBg: isDark ? '#1e1e1e' : '#ffffff'
    };
}

/**
 * Update dynamics chart colors for theme changes
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

        if (chart.options.plugins?.title) {
            chart.options.plugins.title.color = textColor;
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
    if (state.dynamics.loaded && !force) return;

    const container = document.getElementById('dynamics-container');
    if (!container) return;

    container.innerHTML = '<div class="loading-placeholder">Loading gradient dynamics...</div>';

    try {
        // Fetch dynamics data
        console.log('[Dynamics] Fetching from /api/dynamics...');
        const response = await fetch(`/api/dynamics?since=0&limit=1000`);

        console.log('[Dynamics] Response status:', response.status);

        if (!response.ok) {
            throw new Error(`API returned ${response.status}`);
        }

        const data = await response.json();
        console.log('[Dynamics] API response:', data);

        if (data.status === 'no_data' || !data.runs || data.runs.length === 0) {
            // Show helpful empty state
            console.log('[Dynamics] No data available, message:', data.message);
            renderEmptyState(container, data.message);
            return;
        }

        console.log('[Dynamics] Rendering charts with', data.runs.length, 'runs');
        // Render charts
        renderDynamicsCharts(data.runs[0], container);
        state.dynamics.loaded = true;

    } catch (error) {
        console.error('[Dynamics] Error loading:', error);
        container.innerHTML = `
            <div class="error-message">
                <h3>Error Loading Dynamics</h3>
                <p>${error.message}</p>
                <p style="margin-top: 1rem; font-size: 0.9em; opacity: 0.7;">
                    Gradient dynamics require logging during training.
                    See docs/gradient_visualization_proposals.md for details.
                </p>
            </div>
        `;
    }
}

/**
 * Render empty state when no dynamics data available
 */
function renderEmptyState(container, message) {
    const refreshIcon = `
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
            <path fill-rule="evenodd" d="M8 3a5 5 0 1 0 4.546 2.914.5.5 0 0 1 .908-.417A6 6 0 1 1 8 2v1z"/>
            <path d="M8 4.466V.534a.25.25 0 0 1 .41-.192l2.36 1.966c.12.1.12.284 0 .384L8.41 4.658A.25.25 0 0 1 8 4.466z"/>
        </svg>
    `;

    const headerHTML = createTabHeader({
        title: 'Learning',
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
            <p>${message || 'Waiting for training data from Prismatic router...'}</p>
            <div style="margin-top: 2rem; padding: 1.5rem; background: rgba(100, 150, 200, 0.1); border-radius: 8px; text-align: left;">
                <h4 style="margin-top: 0;">How Gradient Dynamics Work</h4>
                <p style="margin: 0.5rem 0;">
                    When you train with <code>router_type: prismatic</code>, gradient dynamics are <strong>automatically logged</strong>
                    every 10 steps via the <code>DynamicsLoggerCallback</code>.
                </p>
                <p style="margin: 0.5rem 0;">
                    The chart will show whether aggressive dual-sided perturbations (scale=1.0, top 5% + bottom 5%)
                    force genuinely different learning dynamics between clean and perturbed experts.
                </p>
                <p style="margin-top: 1.5rem; font-size: 0.9em; opacity: 0.8;">
                    <strong>Key question:</strong> Are bottom weights waking up under ±100% perturbations?
                </p>
                <p style="margin: 0.5rem 0; font-size: 0.9em; opacity: 0.8;">
                    Start training with Prismatic to populate this chart.
                </p>
            </div>
        </div>
    `;
}

/**
 * Render dynamics charts
 */
function renderDynamicsCharts(runData, container) {
    console.log('[Dynamics] renderDynamicsCharts called with runData:', runData);

    const dynamics = runData.dynamics || {};
    console.log('[Dynamics] dynamics object:', dynamics);

    // Debug: Check if values are null
    if (dynamics.expert_0_bottom_norm) {
        const hasValues = dynamics.expert_0_bottom_norm.some(v => v !== null);
        console.log('[Dynamics] expert_0_bottom_norm has non-null values:', hasValues);
        console.log('[Dynamics] expert_0_bottom_norm sample:', dynamics.expert_0_bottom_norm.slice(0, 3));
    }

    // API returns 'step' (singular), frontend expects 'steps' (plural)
    const steps = dynamics.steps || dynamics.step || [];
    console.log('[Dynamics] steps array:', steps, 'length:', steps.length);

    if (steps.length === 0) {
        console.log('[Dynamics] No steps found, showing empty state');
        renderEmptyState(container, "No dynamics data points found");
        return;
    }

    // Detect number of experts from dynamics keys
    const expertKeys = Object.keys(dynamics).filter(k => k.match(/^expert_\d+_/));
    console.log('[Dynamics] expertKeys:', expertKeys);
    const numExperts = new Set(expertKeys.map(k => parseInt(k.match(/expert_(\d+)_/)[1]))).size;
    console.log('[Dynamics] numExperts:', numExperts);

    // Build refresh icon
    const refreshIcon = `
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
            <path fill-rule="evenodd" d="M8 3a5 5 0 1 0 4.546 2.914.5.5 0 0 1 .908-.417A6 6 0 1 1 8 2v1z"/>
            <path d="M8 4.466V.534a.25.25 0 0 1 .41-.192l2.36 1.966c.12.1.12.284 0 .384L8.41 4.658A.25.25 0 0 1 8 4.466z"/>
        </svg>
    `;

    // Calculate divergence score (latest value)
    let divergenceScore = 'N/A';
    if (dynamics.expert_1_divergence && dynamics.expert_1_divergence.length > 0) {
        const latest = dynamics.expert_1_divergence[dynamics.expert_1_divergence.length - 1];
        divergenceScore = latest ? latest.toFixed(4) : 'N/A';
    }

    const headerHTML = createTabHeader({
        title: 'Learning',
        buttons: [{
            id: 'refresh-dynamics-btn',
            label: 'Refresh',
            icon: refreshIcon,
            className: 'tab-header-button'
        }],
        metadata: `
            <span><strong>Points:</strong> ${steps.length}</span>
            <span><strong>Experts:</strong> ${numExperts}</span>
            <span><strong>Divergence:</strong> ${divergenceScore}</span>
        `
    });

    // Build controls for toggling weight tiers
    const controlsHTML = `
        <div class="dynamics-controls" style="margin-top: 1.5rem; padding: 1rem; background: rgba(100, 150, 200, 0.05); border-radius: 8px; display: flex; gap: 2rem; align-items: center;">
            <div style="font-weight: 600;">Show Weight Tiers:</div>
            <label style="display: flex; align-items: center; gap: 0.5rem; cursor: pointer;">
                <input type="checkbox" id="show-top-weights" checked>
                <span>Top 5% (Coarse-grained)</span>
            </label>
            <label style="display: flex; align-items: center; gap: 0.5rem; cursor: pointer;">
                <input type="checkbox" id="show-bottom-weights" checked>
                <span>Bottom 5% (Fine-grained)</span>
            </label>
            <label style="display: flex; align-items: center; gap: 0.5rem; cursor: pointer;">
                <input type="checkbox" id="show-middle-weights" checked>
                <span>Middle 90% (Unperturbed)</span>
            </label>
        </div>
    `;

    // Build chart container
    const chartsHTML = `
        <div style="margin-top: 2rem;">
            <div class="chart-card">
                <div class="chart-title">Expert Gradient Norms: Clean vs Perturbed</div>
                <div class="chart-subtitle">Are bottom weights waking up? Comparing Expert 0 (clean) with Expert 1+ (dual-sided perturbed)</div>
                <div class="chart-wrapper" style="height: 400px;">
                    <canvas id="dynamics-expert-comparison"></canvas>
                </div>
            </div>
        </div>
    `;

    container.innerHTML = headerHTML + controlsHTML + chartsHTML;

    console.log('[Dynamics] DOM updated, scheduling chart creation...');

    // Render chart after DOM update
    setTimeout(() => {
        console.log('[Dynamics] Creating expert comparison chart...');
        try {
            createExpertComparisonChart('dynamics-expert-comparison', dynamics, numExperts);
            console.log('[Dynamics] Chart created successfully');
        } catch (error) {
            console.error('[Dynamics] Error creating chart:', error);
        }

        // Attach event listeners to controls
        ['show-top-weights', 'show-bottom-weights', 'show-middle-weights'].forEach(id => {
            const checkbox = document.getElementById(id);
            if (checkbox) {
                checkbox.addEventListener('change', () => {
                    updateChartVisibility();
                });
            }
        });
    }, 10);
}

/**
 * Create expert comparison chart
 */
function createExpertComparisonChart(canvasId, dynamics, numExperts) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    // Destroy existing
    if (dynamicsCharts[canvasId]) {
        dynamicsCharts[canvasId].destroy();
    }

    const { textColor, gridColor, tooltipBg } = getThemeColors();

    const steps = dynamics.steps || [];

    // Build datasets: one line per expert per tier
    const datasets = [];
    const tiers = ['top', 'bottom', 'middle'];
    const tierLabels = {
        'top': 'Top 5% (Coarse)',
        'bottom': 'Bottom 5% (Fine)',
        'middle': 'Middle 90% (Stable)'
    };

    for (let expertIdx = 0; expertIdx < numExperts; expertIdx++) {
        tiers.forEach((tier, tierIdx) => {
            const key = `expert_${expertIdx}_${tier}_norm`;
            if (!dynamics[key]) {
                console.log(`[Dynamics] Missing key: ${key}`);
                return;
            }

            const values = dynamics[key];
            console.log(`[Dynamics] ${key}:`, values);

            const data = steps.map((step, i) => ({
                x: step,
                y: values[i]
            })).filter(point => point.y !== null);

            console.log(`[Dynamics] ${key} filtered data:`, data.length, 'points');

            // Color scheme:
            // Expert 0 (clean) = blue shades
            // Expert 1+ (perturbed) = orange/red shades
            const baseColor = expertIdx === 0
                ? ['#4A90E2', '#7EB2F5', '#B3D4FF'][tierIdx]  // Blues
                : ['#FF6B6B', '#FFA07A', '#FFB399'][tierIdx]; // Reds/Oranges

            const label = expertIdx === 0
                ? `Expert 0 (Clean) - ${tierLabels[tier]}`
                : `Expert ${expertIdx} (Perturbed) - ${tierLabels[tier]}`;

            datasets.push({
                label: label,
                data: data,
                borderColor: baseColor,
                backgroundColor: baseColor + '20',
                borderWidth: 2,
                pointRadius: 0,
                pointHoverRadius: 5,
                pointHoverBackgroundColor: baseColor,
                pointHoverBorderColor: '#fff',
                pointHoverBorderWidth: 2,
                tension: 0.3,
                fill: false,
                hidden: false,  // Visibility controlled by checkboxes
                expertIdx: expertIdx,
                tier: tier
            });
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
                        padding: 12,
                        font: { size: 10 },
                        filter: (item) => {
                            // Filter out hidden datasets from legend
                            return !item.hidden;
                        }
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
                        label: (ctx) => {
                            const value = ctx.parsed.y;
                            const label = ctx.dataset.label;
                            return `${label}: ${value.toExponential(3)}`;
                        }
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
                    type: 'logarithmic',  // Log scale for gradient norms
                    title: {
                        display: true,
                        text: 'Gradient Norm (L2, Log Scale)',
                        color: textColor,
                        font: { size: 13, weight: '500' }
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
 * Update chart visibility based on checkbox state
 */
function updateChartVisibility() {
    const showTop = document.getElementById('show-top-weights')?.checked ?? true;
    const showBottom = document.getElementById('show-bottom-weights')?.checked ?? true;
    const showMiddle = document.getElementById('show-middle-weights')?.checked ?? true;

    Object.values(dynamicsCharts).forEach(chart => {
        if (!chart) return;

        chart.data.datasets.forEach(dataset => {
            if (dataset.tier === 'top') {
                dataset.hidden = !showTop;
            } else if (dataset.tier === 'bottom') {
                dataset.hidden = !showBottom;
            } else if (dataset.tier === 'middle') {
                dataset.hidden = !showMiddle;
            }
        });

        chart.update('none');
    });
}

/**
 * Destroy all dynamics charts (cleanup)
 */
export function destroyAllDynamicsCharts() {
    Object.keys(dynamicsCharts).forEach(key => {
        if (dynamicsCharts[key]) {
            dynamicsCharts[key].destroy();
            delete dynamicsCharts[key];
        }
    });
}
