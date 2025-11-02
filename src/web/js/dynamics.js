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
 * Load and render gradient dynamics
 */
export async function loadDynamicsWithCharts(force = false) {
    if (state.dynamics.loaded && !force) return;

    const container = document.getElementById('dynamics-container');
    if (!container) return;

    container.innerHTML = '<div class="loading-placeholder">Loading gradient dynamics...</div>';

    try {
        // Fetch dynamics data
        const response = await fetch(`/api/dynamics?since=0&limit=1000`);

        if (!response.ok) {
            throw new Error(`API returned ${response.status}`);
        }

        const data = await response.json();

        if (data.status === 'no_data' || !data.runs || data.runs.length === 0) {
            // Show helpful empty state
            renderEmptyState(container, data.message);
            return;
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

    // Build controls for toggling weight tiers
    const controlsHTML = `
        <div class="dynamics-controls" style="margin-top: 1.5rem; padding: 1rem; background: rgba(100, 150, 200, 0.05); border-radius: 8px; display: flex; gap: 2rem; align-items: center; overflow-x: auto; -webkit-overflow-scrolling: touch;">
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
    `;

    // Build chart containers
    const chartsHTML = `
        <div style="margin-top: 2rem;">
            <div class="chart-card">
                <div class="chart-title">Pi-Phase Helix: Expert Routing Trajectories</div>
                <div class="chart-subtitle">Quantum Echoes - each expert seeded by walking backwards through π</div>
                <div class="chart-wrapper" style="height: 500px;">
                    <canvas id="dynamics-pi-helix"></canvas>
                </div>
            </div>
        </div>

        <div style="margin-top: 2rem;">
            <div class="chart-card">
                <div class="chart-title">Pi-Divergence Cascade: Pattern Discovery Events</div>
                <div class="chart-subtitle">Swirling scatter of gradient measurements mapped to π-phase space - revealing computational resonance</div>
                <div class="chart-wrapper" style="height: 500px;">
                    <canvas id="dynamics-pi-cascade"></canvas>
                </div>
            </div>
        </div>

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

    // Render charts after DOM update
    setTimeout(() => {
        try {
            // Create pi-helix visualization
            const metadata = runData.metadata || {};
            createPiHelixChart('dynamics-pi-helix', dynamics, metadata);

            // Create pi-divergence cascade (scatter plot)
            createPiDivergenceCascade('dynamics-pi-cascade', dynamics, metadata);

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
 * Create Pi-Phase Helix visualization
 *
 * Maps expert routing trajectories to 3D helices with pi-digit phase offsets.
 * Creates pseudo-3D effect through depth-based opacity and line width.
 */
function createPiHelixChart(canvasId, dynamics, metadata) {
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

    // Get pi-phase metadata
    const pi_phases = metadata.pi_phases || [];
    const pi_seeds = metadata.pi_seeds || [];
    const numExperts = metadata.num_experts || 0;

    if (numExperts === 0) {
        // No experts, show placeholder
        return;
    }

    // Helix parameters
    const windings = 4;  // Number of full rotations
    const maxSteps = steps[steps.length - 1];

    // Build datasets - one helix per expert
    const datasets = [];
    const colorPalette = [
        '#4A90E2',  // Expert 0 - Blue (clean)
        '#00D9FF',  // Expert 1 - Cyan
        '#00FF9F',  // Expert 2 - Green
        '#FFD700',  // Expert 3 - Gold
        '#FF6B9D',  // Expert 4 - Pink
        '#FF4757',  // Expert 5 - Red
    ];

    for (let expertIdx = 0; expertIdx < numExperts; expertIdx++) {
        const routing_key = `expert_${expertIdx}_routing_weight`;
        const routing_weights = dynamics[routing_key];

        // Skip if no routing data for this expert
        if (!routing_weights || routing_weights.length === 0) {
            console.warn(`[Pi-Helix] No routing weights found for ${routing_key}`);
            continue;
        }

        // Get pi-phase for this expert
        const phase = pi_phases[expertIdx] || 0;
        const pi_digit = pi_seeds[expertIdx];

        // Generate helix path data points (filter out None/null values)
        const helixData = steps
            .map((step, i) => {
                // Skip if routing weight is null/undefined
                const amplitude = routing_weights[i];
                if (amplitude === null || amplitude === undefined) {
                    return null;
                }

                // Normalize step to [0, 1]
                const t = step / maxSteps;

                // Parametric helix equations with pi-phase offset
                const angle = t * windings * 2 * Math.PI + phase;

                // X: progress through time (with slight perspective tilt)
                const x = t * 100;  // 0 to 100 for percentage

                // Y: helix in 2D (combining sin/cos for pseudo-3D projection)
                // Using isometric-style projection: y = radius * sin, z-depth affects y
                const y_component = amplitude * Math.sin(angle);
                const z_component = amplitude * Math.cos(angle);

                // Isometric projection: y_screen = y + z * 0.5
                const y = y_component + z_component * 0.5;

                // Store z for depth-based opacity
                return {
                    x: x,
                    y: y,
                    z: z_component,  // depth (for opacity)
                    step: step,
                    amplitude: amplitude,
                    angle: angle
                };
            })
            .filter(point => point !== null);  // Remove null entries

        // Skip this expert if no valid data points
        if (helixData.length === 0) {
            console.warn(`[Pi-Helix] Expert ${expertIdx} has no valid routing data points`);
            continue;
        }

        // Create dataset with depth-based visual effects
        const baseColor = colorPalette[expertIdx % colorPalette.length];

        const label = expertIdx === 0
            ? `Expert 0 (Clean)`
            : `Expert ${expertIdx} (π[${99999 - expertIdx + 1}] = ${pi_digit ?? '?'})`;

        datasets.push({
            label: label,
            data: helixData,
            borderColor: baseColor,
            backgroundColor: baseColor + '20',
            borderWidth: 2,
            pointRadius: 0,
            pointHoverRadius: 6,
            pointHoverBackgroundColor: baseColor,
            pointHoverBorderColor: '#fff',
            pointHoverBorderWidth: 2,
            tension: 0.4,  // Smooth curves
            fill: false,
            segment: {
                // Apply pseudo-3D depth effect via opacity
                borderWidth: (ctx) => {
                    if (!ctx.p0 || !ctx.p0.raw) return 2;
                    // Thicker when closer (z > 0), thinner when farther (z < 0)
                    const z = ctx.p0.raw.z || 0;
                    return 2 + z * 1.5;  // Range: 0.5 to 3.5
                },
                borderColor: (ctx) => {
                    if (!ctx.p0 || !ctx.p0.raw) return baseColor;
                    // More opaque when closer, more transparent when farther
                    const z = ctx.p0.raw.z || 0;
                    const opacity = 0.4 + (z + 1) * 0.3;  // Range: 0.1 to 1.0
                    const hex = baseColor.replace('#', '');
                    const r = parseInt(hex.substr(0, 2), 16);
                    const g = parseInt(hex.substr(2, 2), 16);
                    const b = parseInt(hex.substr(4, 2), 16);
                    return `rgba(${r}, ${g}, ${b}, ${opacity})`;
                }
            },
            expertIdx: expertIdx,
            pi_digit: pi_digit
        });
    }

    // If no datasets were created, show a message
    if (datasets.length === 0) {
        const message = document.createElement('div');
        message.className = 'empty-state';
        message.style.padding = '2rem';
        message.style.textAlign = 'center';
        message.innerHTML = `
            <h3>No Routing Data Available</h3>
            <p>Routing weights are logged in metrics.db but not found for these training steps.</p>
            <p style="margin-top: 1rem; font-size: 0.9em; opacity: 0.7;">
                The Prismatic router automatically logs routing weights via _log_routing_metrics().
                This data will appear once training progresses.
            </p>
        `;
        ctx.parentElement.appendChild(message);
        return;
    }

    // Create the chart
    dynamicsCharts[canvasId] = new Chart(ctx, {
        type: 'line',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'nearest'
            },
            parsing: {
                xAxisKey: 'x',
                yAxisKey: 'y'
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
                                `Routing: ${(point.amplitude * 100).toFixed(1)}%`,
                                `Phase: ${(point.angle % (2 * Math.PI)).toFixed(2)} rad`,
                                `Depth: ${point.z > 0 ? 'near' : 'far'}`
                            ];
                        }
                    }
                },
                title: {
                    display: false
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    title: {
                        display: true,
                        text: 'Training Progress (%)',
                        color: textColor,
                        font: { size: 13, weight: '500' }
                    },
                    ticks: {
                        color: textColor,
                        callback: (value) => `${value.toFixed(0)}%`
                    },
                    grid: { color: gridColor }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Routing Amplitude (Helix Projection)',
                        color: textColor,
                        font: { size: 13, weight: '500' }
                    },
                    ticks: {
                        color: textColor
                    },
                    grid: { color: gridColor }
                }
            }
        }
    });
}

/**
 * Create Pi-Divergence Cascade visualization
 *
 * Scatter plot of divergence measurements mapped to pi-phase helix positions.
 * Each point represents expert divergence at a training step, creating a
 * swirling cascade pattern that may reveal computational resonance with pi structure.
 */
function createPiDivergenceCascade(canvasId, dynamics, metadata) {
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

    const pi_phases = metadata.pi_phases || [];
    const pi_seeds = metadata.pi_seeds || [];
    const numExperts = metadata.num_experts || 0;

    if (numExperts === 0) return;

    // Helix parameters (matching the helix chart)
    const windings = 4;
    const maxSteps = steps[steps.length - 1];

    // Build scatter datasets - show multiple tiers per expert for richer visualization
    const datasets = [];
    const colorPalette = [
        '#00D9FF',  // Cyan (top tier, expert 1)
        '#FF6B9D',  // Pink (bottom tier, expert 1)
        '#00FF9F',  // Green (top tier, expert 2)
        '#FFD700',  // Gold (bottom tier, expert 2)
        '#7B68EE',  // Purple (additional experts)
        '#FF4757',  // Red
    ];

    let datasetIdx = 0;

    // For each perturbed expert (skip expert 0 - it's clean/baseline)
    for (let expertIdx = 1; expertIdx < numExperts; expertIdx++) {
        // Get pi-phase for this expert
        const phase = pi_phases[expertIdx] || 0;
        const pi_digit = pi_seeds[expertIdx];

        // Show two cascades per expert: top tier and bottom tier gradients
        const tiers = [
            { key: `expert_${expertIdx}_top_norm`, label: 'Top 5%' },
            { key: `expert_${expertIdx}_bottom_norm`, label: 'Bottom 5%' }
        ];

        for (const tier of tiers) {
            const values = dynamics[tier.key];

            if (!values || values.length === 0) {
                console.warn(`[Pi-Cascade] No data for ${tier.key}`);
                continue;
            }

            // Generate scatter points - one per training step
            const scatterPoints = steps.map((step, i) => {
                const amplitude = values[i];
                if (amplitude === null || amplitude === undefined) return null;

                // Normalize step to [0, 1]
                const t = step / maxSteps;

                // Parametric helix equations with pi-phase offset
                const angle = t * windings * 2 * Math.PI + phase;

                // X: progress through time
                const x = t * 100;  // 0 to 100%

                // Y: helix projection (isometric)
                const y_component = amplitude * Math.sin(angle);
                const z_component = amplitude * Math.cos(angle);
                const y = y_component + z_component * 0.5;  // Isometric projection

                // Point size based on amplitude magnitude
                const pointRadius = 3 + Math.sqrt(amplitude) * 0.3;  // Range: 3-8

                // Opacity based on depth (z-component)
                const normalizedZ = (z_component / amplitude) || 0;  // -1 to 1
                const opacity = 0.4 + (normalizedZ + 1) * 0.3;  // Range: 0.4 to 1.0

                return {
                    x: x,
                    y: y,
                    z: z_component,
                    step: step,
                    value: amplitude,
                    angle: angle,
                    pointRadius: pointRadius,
                    opacity: opacity
                };
            }).filter(p => p !== null);

            if (scatterPoints.length === 0) continue;

            // Solid color per tier (no time gradient)
            const baseColor = colorPalette[datasetIdx % colorPalette.length];
            datasetIdx++;

            const label = `Expert ${expertIdx} ${tier.label} (π[${99999 - expertIdx + 1}] = ${pi_digit ?? '?'})`;

            datasets.push({
                label: label,
                data: scatterPoints,
                backgroundColor: scatterPoints.map(p => {
                    // Solid expert color with depth-based opacity only
                    const hex = baseColor.replace('#', '');
                    const r = parseInt(hex.substr(0, 2), 16);
                    const g = parseInt(hex.substr(2, 2), 16);
                    const b = parseInt(hex.substr(4, 2), 16);
                    return `rgba(${r}, ${g}, ${b}, ${p.opacity})`;
                }),
                borderColor: baseColor,
                borderWidth: 1,
                pointRadius: scatterPoints.map(p => p.pointRadius),
                pointHoverRadius: scatterPoints.map(p => p.pointRadius * 1.5),
                pointHoverBorderWidth: 2,
                pointHoverBorderColor: '#fff',
                expertIdx: expertIdx,
                pi_digit: pi_digit,
                tier: tier.label,
                showLine: false  // Scatter only, no connecting lines
            });
        }
    }

    if (datasets.length === 0) {
        const message = document.createElement('div');
        message.className = 'empty-state';
        message.style.padding = '2rem';
        message.style.textAlign = 'center';
        message.innerHTML = `
            <h3>No Gradient Data Available</h3>
            <p>Expert gradient measurements will appear here during training.</p>
            <p style="margin-top: 1rem; font-size: 0.9em; opacity: 0.7;">
                This chart reveals whether π-digit seeding creates computational patterns.
            </p>
        `;
        ctx.parentElement.appendChild(message);
        return;
    }

    // Create the scatter chart
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
                                `Phase: ${(point.angle % (2 * Math.PI)).toFixed(2)} rad`,
                                `Depth: ${point.z > 0 ? 'near (+)' : 'far (-)'}`
                            ];
                        }
                    }
                },
                title: {
                    display: false
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    title: {
                        display: true,
                        text: 'Training Progress (%)',
                        color: textColor,
                        font: { size: 13, weight: '500' }
                    },
                    ticks: {
                        color: textColor,
                        callback: (value) => `${value.toFixed(0)}%`
                    },
                    grid: { color: gridColor }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Divergence Amplitude (π-Phase Projection)',
                        color: textColor,
                        font: { size: 13, weight: '500' }
                    },
                    ticks: {
                        color: textColor
                    },
                    grid: { color: gridColor }
                }
            }
        }
    });
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

            const data = steps.map((step, i) => ({
                x: step,
                y: values[i]
            })).filter(point => point.y !== null);

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
