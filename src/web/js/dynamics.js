/**
 * Praxis Web - Gradient Dynamics Visualization
 * Expert Gradient Divergence Tracker - validates dual-sided perturbation hypothesis
 */

import { state, CONSTANTS } from './state.js';
import { fetchAPI } from './api.js';
import { createTabHeader } from './components.js';

// Chart instances for dynamics (exported for hybrid mode)
export const dynamicsCharts = {};

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
 * Get theme-appropriate colors (reuse from charts.js pattern)
 * @param {string} [forceTheme] - Optional theme override ('light' or 'dark')
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
 * Downsample data points using Largest Triangle Three Buckets (LTTB) algorithm
 * Preserves visual shape while reducing point count for performance
 * @param {Array} data - Array of {x, y, ...} points
 * @param {number} threshold - Target number of points
 * @returns {Array} Downsampled array
 */
function downsampleLTTB(data, threshold) {
    if (data.length <= threshold || threshold <= 2) {
        return data;
    }

    const sampled = [];
    const bucketSize = (data.length - 2) / (threshold - 2);

    // Always keep first point
    sampled.push(data[0]);

    for (let i = 0; i < threshold - 2; i++) {
        const avgRangeStart = Math.floor((i + 1) * bucketSize) + 1;
        const avgRangeEnd = Math.floor((i + 2) * bucketSize) + 1;
        const avgRangeEnd2 = avgRangeEnd < data.length ? avgRangeEnd : data.length;

        // Calculate average point in next bucket
        let avgX = 0, avgY = 0;
        let avgRangeLength = avgRangeEnd2 - avgRangeStart;

        for (let j = avgRangeStart; j < avgRangeEnd2; j++) {
            avgX += data[j].x;
            avgY += data[j].y;
        }
        avgX /= avgRangeLength;
        avgY /= avgRangeLength;

        // Get current bucket range
        const rangeStart = Math.floor(i * bucketSize) + 1;
        const rangeEnd = Math.floor((i + 1) * bucketSize) + 1;

        // Point in previous bucket
        const prevPoint = sampled[sampled.length - 1];

        // Find point in current bucket with largest triangle area
        let maxArea = -1;
        let maxAreaPoint = null;

        for (let j = rangeStart; j < rangeEnd; j++) {
            if (j >= data.length) break;

            // Calculate triangle area
            const area = Math.abs(
                (prevPoint.x - avgX) * (data[j].y - prevPoint.y) -
                (prevPoint.x - data[j].x) * (avgY - prevPoint.y)
            ) * 0.5;

            if (area > maxArea) {
                maxArea = area;
                maxAreaPoint = data[j];
            }
        }

        if (maxAreaPoint) {
            sampled.push(maxAreaPoint);
        }
    }

    // Always keep last point
    sampled.push(data[data.length - 1]);

    return sampled;
}

/**
 * Load and render gradient dynamics (initial load)
 */
export async function loadDynamicsWithCharts(force = false) {
    // If already loaded and user forces refresh, do incremental fetch
    if (state.dynamics.loaded && force) {
        console.log('[Dynamics] Forcing incremental update...');
        await fetchIncrementalDynamics();
        return;
    }

    // If already loaded and not forcing, do nothing
    if (state.dynamics.loaded && !force) {
        return;
    }

    const container = document.getElementById('dynamics-container');
    if (!container) return;

    container.innerHTML = '<div class="loading-placeholder">Loading gradient dynamics...</div>';

    try {
        // Initial load: fetch all data from beginning
        const response = await fetch(`/api/dynamics?since=0&limit=10000`);

        if (!response.ok) {
            throw new Error(`API returned ${response.status}`);
        }

        const data = await response.json();

        if (data.status === 'no_data' || !data.runs || data.runs.length === 0) {
            // Show helpful empty state
            renderEmptyState(container, data.message);
            return;
        }

        // Store data and update last step
        state.dynamics.data = data.runs[0];
        const steps = data.runs[0].dynamics?.steps || [];
        if (steps.length > 0) {
            state.dynamics.lastStep = Math.max(...steps);
        }

        // Render charts
        renderDynamicsCharts(data.runs[0], container);
        state.dynamics.loaded = true;

    } catch (error) {
        console.error('[Dynamics] Failed to load:', error);
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
 * Fetch and append incremental dynamics data
 */
async function fetchIncrementalDynamics() {
    if (!state.dynamics.loaded || !state.dynamics.data) return;

    try {
        // Fetch only new data since last step
        const response = await fetch(`/api/dynamics?since=${state.dynamics.lastStep + 1}&limit=1000`);

        if (!response.ok) {
            console.warn('[Dynamics] Incremental fetch failed:', response.status);
            return;
        }

        const data = await response.json();

        if (data.status === 'no_data' || !data.runs || data.runs.length === 0) {
            // No new data yet
            return;
        }

        const newRunData = data.runs[0];
        const newSteps = newRunData.dynamics?.steps || [];

        if (newSteps.length === 0) {
            // No new steps
            return;
        }

        // Update last step
        state.dynamics.lastStep = Math.max(...newSteps);

        // Append new data to existing data
        appendDynamicsData(newRunData);

        // Update charts with new data
        appendDynamicsToCharts(newRunData);

        console.log(`[Dynamics] Loaded ${newSteps.length} new data points (up to step ${state.dynamics.lastStep})`);

    } catch (error) {
        console.error('[Dynamics] Incremental fetch error:', error);
    }
}

/**
 * Append new dynamics data to existing stored data
 */
function appendDynamicsData(newRunData) {
    if (!state.dynamics.data || !state.dynamics.data.dynamics) return;

    const existingDynamics = state.dynamics.data.dynamics;
    const newDynamics = newRunData.dynamics;

    // Append all numeric arrays
    Object.keys(newDynamics).forEach(key => {
        if (Array.isArray(newDynamics[key]) && Array.isArray(existingDynamics[key])) {
            existingDynamics[key] = existingDynamics[key].concat(newDynamics[key]);
        } else if (Array.isArray(newDynamics[key])) {
            // New key, add it
            existingDynamics[key] = newDynamics[key];
        }
    });
}


/**
 * Append new dynamics data to existing charts
 * When new data arrives, we need to re-downsample the entire dataset from state
 */
function appendDynamicsToCharts(newRunData) {
    const newDynamics = newRunData.dynamics || {};
    const newSteps = newDynamics.steps || [];

    if (newSteps.length === 0) return;

    // Use the full dataset from state (which now includes new data)
    const fullDynamics = state.dynamics.data.dynamics;
    const fullSteps = fullDynamics.steps || [];

    // Update radial phase map chart
    const phaseChart = dynamicsCharts['dynamics-phase-radial'];
    if (phaseChart && phaseChart.data && phaseChart.data.datasets) {
        const metadata = state.dynamics.data.metadata || {};
        const phase_offsets = metadata.phase_offsets || [];
        const numExperts = metadata.num_experts || 0;

        if (numExperts > 0) {
            // Color palette (same as creation)
            const colorPalette = ['#00D9FF', '#FF6B9D', '#00FF9F', '#FFD700'];

            // For each perturbed expert
            for (let expertIdx = 1; expertIdx < numExperts; expertIdx++) {
                const phase = phase_offsets[expertIdx] || 0;

                const tendrils = [
                    { key: `expert_${expertIdx}_top_norm`, angleOffset: 0, color: colorPalette[0] },
                    { key: `expert_${expertIdx}_bottom_norm`, angleOffset: Math.PI / 4, color: colorPalette[1] },
                    { key: `expert_${expertIdx}_weight_angle`, angleOffset: Math.PI / 2, color: colorPalette[2] }
                ];

                for (const tendril of tendrils) {
                    const values = fullDynamics[tendril.key];
                    if (!values || values.length === 0) continue;

                    // Find matching dataset
                    const dataset = phaseChart.data.datasets.find(ds =>
                        ds.expertIdx === expertIdx && ds.tendrilType?.includes(tendril.key.includes('top') ? 'Top' : tendril.key.includes('bottom') ? 'Bottom' : 'Divergence')
                    );

                    if (dataset) {
                        // Rebuild full dataset with all points
                        const rawRadialPoints = fullSteps.map((step, i) => {
                            const value = values[i];
                            if (value === null || value === undefined) return null;

                            const radius = step;
                            const angle = phase + tendril.angleOffset;
                            const x = radius * Math.cos(angle);
                            const y = radius * Math.sin(angle);
                            const pointRadius = 3 + Math.log(value + 1) * 1.5;
                            const normalizedValue = Math.min(value / 100, 1.0);
                            const opacity = 0.4 + normalizedValue * 0.5;

                            return { x, y, step, value, angle, pointRadius, opacity };
                        }).filter(p => p !== null);

                        // Re-downsample entire dataset
                        const downsampledPoints = downsampleLTTB(rawRadialPoints, 800);

                        // Replace dataset
                        dataset.data = downsampledPoints;

                        // Update styling arrays
                        dataset.backgroundColor = downsampledPoints.map(p => {
                            const hex = tendril.color.replace('#', '');
                            const r = parseInt(hex.substr(0, 2), 16);
                            const g = parseInt(hex.substr(2, 2), 16);
                            const b = parseInt(hex.substr(4, 2), 16);
                            return `rgba(${r}, ${g}, ${b}, ${p.opacity})`;
                        });
                        dataset.pointRadius = downsampledPoints.map(p => p.pointRadius);
                        dataset.pointHoverRadius = downsampledPoints.map(p => p.pointRadius * 1.8);
                    }
                }
            }

            phaseChart.update('none');
        }
    }

    // Update expert comparison chart
    const comparisonChart = dynamicsCharts['dynamics-expert-comparison'];
    if (comparisonChart && comparisonChart.data && comparisonChart.data.datasets) {
        comparisonChart.data.datasets.forEach(dataset => {
            const { expertIdx, tier } = dataset;
            const key = `expert_${expertIdx}_${tier}_norm`;
            const values = fullDynamics[key];

            if (values && values.length > 0) {
                // Rebuild full dataset with all points
                const rawData = fullSteps.map((step, i) => ({
                    x: step,
                    y: values[i]
                })).filter(point => point.y !== null);

                // Re-downsample entire dataset
                dataset.data = downsampleLTTB(rawData, 1000);
            }
        });

        comparisonChart.update('none');
    }

    // Update metadata display (divergence score)
    const headerMetadata = document.querySelector('.tab-header-metadata');
    if (headerMetadata && fullDynamics.expert_1_divergence) {
        const latest = fullDynamics.expert_1_divergence[fullDynamics.expert_1_divergence.length - 1];
        const divergenceScore = latest ? latest.toFixed(4) : 'N/A';
        const totalPoints = fullSteps.length;

        headerMetadata.innerHTML = `
            <span><strong>Points:</strong> ${totalPoints}</span>
            <span><strong>Experts:</strong> ${numExperts || 'N/A'}</span>
            <span><strong>Divergence:</strong> ${divergenceScore}</span>
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
    const dynamics = runData.dynamics || {};

    // API returns 'step' (singular), frontend expects 'steps' (plural)
    const steps = dynamics.steps || dynamics.step || [];

    if (steps.length === 0) {
        renderEmptyState(container, "No dynamics data points found");
        return;
    }

    // Detect number of experts from dynamics keys
    const expertKeys = Object.keys(dynamics).filter(k => k.match(/^expert_\d+_/));
    const numExperts = new Set(expertKeys.map(k => parseInt(k.match(/expert_(\d+)_/)[1]))).size;

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

    // Build chart containers
    const chartsHTML = `
        <div style="margin-top: 2rem;">
            <div class="chart-card">
                <div class="chart-title">Helical Phase Map: Expert Harmonic Relationships</div>
                <div class="chart-subtitle">Radial expansion from origin - each expert at different phase offset (Euler's formula modulation)</div>
                <div class="chart-wrapper" style="height: 600px;">
                    <canvas id="dynamics-phase-radial"></canvas>
                </div>
            </div>
        </div>

        <div style="margin-top: 2rem;">
            <div class="dynamics-controls" style="margin-bottom: 1rem; padding: 1rem; background: rgba(100, 150, 200, 0.05); border-radius: 8px; display: flex; gap: 2rem; align-items: center; overflow-x: auto; -webkit-overflow-scrolling: touch;">
                <div style="font-weight: 600; white-space: nowrap;">Granularity:</div>
                <label style="display: flex; align-items: center; gap: 0.5rem; cursor: pointer; white-space: nowrap;">
                    <input type="checkbox" id="show-top-weights" checked>
                    <span>Coarse</span>
                </label>
                <label style="display: flex; align-items: center; gap: 0.5rem; cursor: pointer; white-space: nowrap;">
                    <input type="checkbox" id="show-bottom-weights" checked>
                    <span>Fine</span>
                </label>
                <label style="display: flex; align-items: center; gap: 0.5rem; cursor: pointer; white-space: nowrap;">
                    <input type="checkbox" id="show-middle-weights" checked>
                    <span>Unperturbed</span>
                </label>
            </div>

            <div class="chart-card">
                <div class="chart-title">Expert Gradient Norms: Clean vs Perturbed</div>
                <div class="chart-subtitle">Are bottom weights waking up? Comparing Expert 0 (clean) with Expert 1+ (dual-sided perturbed)</div>
                <div class="chart-wrapper" style="height: 400px;">
                    <canvas id="dynamics-expert-comparison"></canvas>
                </div>
            </div>
        </div>
    `;

    container.innerHTML = headerHTML + chartsHTML;

    // Render charts after DOM update
    setTimeout(() => {
        try {
            // Create helical phase radial map
            const metadata = runData.metadata || {};
            createPiResonanceMap('dynamics-phase-radial', dynamics, metadata);

            // Create gradient comparison chart (existing)
            createExpertComparisonChart('dynamics-expert-comparison', dynamics, numExperts);
        } catch (error) {
            console.error('[Dynamics] Chart creation failed:', error);
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
 * Create Helical Phase Map (Radial Polar Visualization)
 *
 * Plots measurements expanding radially from origin, with angular position
 * determined by helical phase offset. Reveals emergent patterns and whether
 * harmonic relationships between experts create stable computational structures.
 *
 * Concept: Time expands outward from center. Each expert positioned at different
 * phase angle based on Euler's formula: phase = expert_idx * 2π / num_experts.
 * If patterns/clustering emerge, it suggests helical structure transfers to learned features.
 */
function createPiResonanceMap(canvasId, dynamics, metadata) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    // Destroy existing
    if (dynamicsCharts[canvasId]) {
        dynamicsCharts[canvasId].destroy();
    }

    // Get theme colors
    const theme = getContextTheme(ctx);
    const { textColor, gridColor, tooltipBg } = getThemeColors(theme);

    const steps = dynamics.steps || [];
    if (steps.length === 0) return;

    const phase_offsets = metadata.phase_offsets || [];
    const numExperts = metadata.num_experts || 0;

    if (numExperts === 0) return;

    // Color palette for different measurement types (tendrils)
    const colorPalette = [
        '#00D9FF',  // Cyan - Top gradients
        '#FF6B9D',  // Pink - Bottom gradients
        '#00FF9F',  // Green - Weight angle
        '#FFD700',  // Gold - Routing weight
    ];

    const datasets = [];

    // For each perturbed expert, create multiple tendrils (measurement types)
    for (let expertIdx = 1; expertIdx < numExperts; expertIdx++) {
        const phase = phase_offsets[expertIdx] || 0;  // Actual helical phase offset

        // Define measurement tendrils for this expert
        const tendrils = [
            {
                key: `expert_${expertIdx}_top_norm`,
                label: `E${expertIdx} Top`,
                angleOffset: 0,
                color: colorPalette[0]
            },
            {
                key: `expert_${expertIdx}_bottom_norm`,
                label: `E${expertIdx} Bottom`,
                angleOffset: Math.PI / 4,  // 45° offset
                color: colorPalette[1]
            },
            {
                key: `expert_${expertIdx}_weight_angle`,
                label: `E${expertIdx} Divergence`,
                angleOffset: Math.PI / 2,  // 90° offset
                color: colorPalette[2]
            },
        ];

        for (const tendril of tendrils) {
            const values = dynamics[tendril.key];
            if (!values || values.length === 0) continue;

            // Generate points expanding radially from origin
            const rawRadialPoints = steps.map((step, i) => {
                const value = values[i];
                if (value === null || value === undefined) return null;

                // Polar coordinates
                const radius = step;  // Time expands outward
                const angle = phase + tendril.angleOffset;  // Helical phase + tendril offset

                // Convert polar → cartesian
                const x = radius * Math.cos(angle);
                const y = radius * Math.sin(angle);

                // Point size based on value magnitude
                const pointRadius = 3 + Math.log(value + 1) * 1.5;

                // Opacity based on value (higher = more opaque)
                const normalizedValue = Math.min(value / 100, 1.0);
                const opacity = 0.4 + normalizedValue * 0.5;

                return {
                    x: x,
                    y: y,
                    step: step,
                    value: value,
                    angle: angle,
                    pointRadius: pointRadius,
                    opacity: opacity
                };
            }).filter(p => p !== null);

            if (rawRadialPoints.length === 0) continue;

            // Downsample to max 800 points for performance (scatter charts need fewer points)
            const radialPoints = downsampleLTTB(rawRadialPoints, 800);

            datasets.push({
                label: `${tendril.label} (φ=${(phase * 180 / Math.PI).toFixed(0)}°)`,  // Show actual phase angle
                data: radialPoints,
                backgroundColor: radialPoints.map(p => {
                    const hex = tendril.color.replace('#', '');
                    const r = parseInt(hex.substr(0, 2), 16);
                    const g = parseInt(hex.substr(2, 2), 16);
                    const b = parseInt(hex.substr(4, 2), 16);
                    return `rgba(${r}, ${g}, ${b}, ${p.opacity})`;
                }),
                borderColor: tendril.color,
                borderWidth: 1,
                pointRadius: radialPoints.map(p => p.pointRadius),
                pointHoverRadius: radialPoints.map(p => p.pointRadius * 1.8),
                pointHoverBorderWidth: 2,
                pointHoverBorderColor: '#fff',
                expertIdx: expertIdx,
                phase_offset: phase,  // Actual helical phase
                tendrilType: tendril.label,
                showLine: false
            });
        }
    }

    if (datasets.length === 0) {
        const message = document.createElement('div');
        message.className = 'empty-state';
        message.style.padding = '2rem';
        message.style.textAlign = 'center';
        message.innerHTML = `
            <h3>No Data Available</h3>
            <p>Weight divergence measurements will appear here as training progresses.</p>
            <p style="margin-top: 1rem; font-size: 0.9em; opacity: 0.7;">
                This chart reveals if helical modulation (Euler's formula) creates stable patterns.
            </p>
        `;
        ctx.parentElement.appendChild(message);
        return;
    }

    // Create radial scatter chart
    dynamicsCharts[canvasId] = new Chart(ctx, {
        type: 'scatter',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'point'
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        color: textColor,
                        usePointStyle: true,
                        padding: 10,
                        font: { size: 10 }
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
                        title: (ctx) => {
                            const point = ctx[0].raw;
                            return `Step ${point.step}`;
                        },
                        label: (ctx) => {
                            const point = ctx.raw;
                            const dataset = ctx.dataset;
                            return [
                                `${dataset.label}`,
                                `Value: ${point.value.toExponential(3)}`,
                                `Position: (${point.x.toFixed(1)}, ${point.y.toFixed(1)})`,
                                `Angle: ${(point.angle * 180 / Math.PI).toFixed(1)}°`
                            ];
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    title: {
                        display: true,
                        text: 'X (Radial Projection)',
                        color: textColor,
                        font: { size: 13, weight: '500' }
                    },
                    ticks: { color: textColor },
                    grid: {
                        color: gridColor,
                        drawTicks: false
                    },
                    // Center the chart by using symmetric bounds
                    min: undefined,
                    max: undefined
                },
                y: {
                    type: 'linear',
                    title: {
                        display: true,
                        text: 'Y (Radial Projection)',
                        color: textColor,
                        font: { size: 13, weight: '500' }
                    },
                    ticks: { color: textColor },
                    grid: {
                        color: gridColor,
                        drawTicks: false
                    },
                    // Center the chart
                    min: undefined,
                    max: undefined
                }
            }
        }
    });
}

/**
 * (Deprecated helix and cascade functions removed - replaced by radial map)
 */

/**
 * Create expert comparison chart
 */

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

    // Get colors for the appropriate theme context (hybrid overlay or normal)
    const theme = getContextTheme(ctx);
    const { textColor, gridColor, tooltipBg } = getThemeColors(theme);

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
                return;
            }

            const values = dynamics[key];

            const rawData = steps.map((step, i) => ({
                x: step,
                y: values[i]
            })).filter(point => point.y !== null);

            // Downsample to max 1000 points for performance
            const data = downsampleLTTB(rawData, 1000);

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
