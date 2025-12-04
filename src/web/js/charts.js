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
        textColor: isDark ? '#e0e0e0' : '#1a1a1a',
        gridColor: isDark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.15)',
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
 * Load available agents for multi-agent comparison
 */
export async function loadAvailableAgents() {
    try {
        const data = await fetchAPI('agents');

        if (data.agents) {
            // Filter to only online agents (active instances)
            state.agents.availableAgents = data.agents.filter(a => a.status === 'online');

            // Auto-select all online agents by default
            if (state.agents.selectedAgents.length === 0) {
                state.agents.selectedAgents = state.agents.availableAgents.map(a => a.name);
            }
        }

        return state.agents.availableAgents;
    } catch (error) {
        console.error('[Charts] Error loading agents:', error);
        return [];
    }
}

/**
 * Load data metrics for selected agents (sampling weights, etc.)
 */
async function loadAgentDataMetrics(agentName) {
    const agent = state.agents.availableAgents.find(a => a.name === agentName);
    if (!agent || agent.status === 'archived') return null;

    try {
        let baseUrl = agent.url.replace(/\/praxis(\.git)?$/, '');
        const response = await fetch(`${baseUrl}/api/data-metrics?since=0&limit=1000&downsample=lttb`);

        if (!response.ok) return null;

        const data = await response.json();

        if (data.status === 'no_data' || !data.runs || data.runs.length === 0) {
            return null;
        }

        return {
            name: agentName,
            url: agent.url,
            data_metrics: data.runs[0].data_metrics,
            metadata: data.runs[0].metadata
        };
    } catch (error) {
        // Silently handle errors (CORS, network, etc.)
        return null;
    }
}

/**
 * Toggle agent selector dropdown
 */
export function toggleAgentSelector() {
    state.agents.selectorOpen = !state.agents.selectorOpen;
    const dropdown = document.getElementById('agent-selector-dropdown');
    if (dropdown) {
        dropdown.style.display = state.agents.selectorOpen ? 'block' : 'none';
    }
}

/**
 * Toggle agent selection
 */
export function toggleAgentSelection(name) {
    const index = state.agents.selectedAgents.indexOf(name);

    if (index > -1) {
        state.agents.selectedAgents.splice(index, 1);
    } else {
        state.agents.selectedAgents.push(name);
    }

    // Update checkbox state
    const checkbox = document.querySelector(`input[data-agent-name="${name}"]`);
    if (checkbox) {
        checkbox.checked = state.agents.selectedAgents.includes(name);
    }

    // Reload metrics with new selection
    loadResearchMetricsWithCharts(true);
}

/**
 * Load and render research metrics with full Chart.js integration
 */
export async function loadResearchMetricsWithCharts(force = false) {
    if (state.research.loaded && !force) return;

    const container = document.getElementById('research-container');
    if (!container) return;

    container.innerHTML = '<div class="loading-placeholder">Loading metrics...</div>';

    try {
        // Load available agents first
        await loadAvailableAgents();

        if (state.agents.selectedAgents.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <h3>No Agents Selected</h3>
                    <p>Select at least one agent to display metrics.</p>
                </div>
            `;
            return;
        }

        // Fetch metrics from each selected agent
        const agentMetricsPromises = state.agents.selectedAgents.map(async (agentName) => {
            const agent = state.agents.availableAgents.find(a => a.name === agentName);
            if (!agent) return null;

            // Skip archived agents - backend already categorizes git hosts as "archived"
            // This trusts backend's single source of truth instead of duplicating logic
            if (agent.status === 'archived') {
                return null;
            }

            // Try to fetch metrics - errors are caught gracefully below
            // No need to preemptively filter URLs; let CORS/network errors happen naturally
            try {
                let baseUrl = agent.url.replace(/\/praxis(\.git)?$/, '');
                const response = await fetch(`${baseUrl}/api/metrics?since=0&limit=1000&downsample=lttb`);

                if (!response.ok) {
                    console.warn(`[Charts] Metrics fetch failed for ${agentName}: ${response.status}`);
                    return null;
                }

                const data = await response.json();

                if (data.status === 'no_data' || !data.runs || data.runs.length === 0) {
                    console.warn(`[Charts] No metrics data for ${agentName}`);
                    return null;
                }

                console.log(`[Charts] Loaded ${data.runs[0].metadata?.num_points || 0} metrics for ${agentName}`);

                return {
                    name: agentName,
                    url: agent.url,
                    metrics: data.runs[0].metrics,
                    metadata: data.runs[0].metadata
                };
            } catch (error) {
                // Log errors for debugging large dataset issues
                console.error(`[Charts] Error loading metrics for ${agentName}:`, error);
                return null;
            }
        });

        const results = await Promise.all(agentMetricsPromises);
        const agentMetrics = results.filter(r => r !== null);

        if (agentMetrics.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <h3>No Metrics Available</h3>
                    <p>Selected agents have no training metrics.</p>
                </div>
            `;
            return;
        }

        // Also fetch data metrics (sampling weights, etc.) for each agent
        const dataMetricsPromises = state.agents.selectedAgents.map(agentName =>
            loadAgentDataMetrics(agentName)
        );
        const dataMetricsResults = await Promise.all(dataMetricsPromises);
        const agentDataMetrics = dataMetricsResults.filter(r => r !== null);

        renderMetricsCharts({ agents: agentMetrics, dataMetrics: agentDataMetrics }, container);
        state.research.loaded = true;

        // Set up refresh button event listener (manual reload)
        const refreshBtn = document.getElementById('refresh-metrics-btn');
        if (refreshBtn) {
            refreshBtn.onclick = () => {
                console.log('[Charts] Manual refresh triggered');
                loadResearchMetricsWithCharts(true);  // Force reload
            };
        }

    } catch (error) {
        console.error('[Charts] Error loading metrics:', error);
        container.innerHTML = `
            <div class="error-message">
                <h3>Error Loading Metrics</h3>
                <p>${error.message}</p>
            </div>
        `;
    }
}

/**
 * Render full metrics charts
 */
function renderMetricsCharts(data, container) {
    const agents = data.agents || [];
    const dataMetrics = data.dataMetrics || [];

    if (agents.length === 0) {
        container.innerHTML = `<div class="empty-state"><p>No data available</p></div>`;
        return;
    }

    // Build agent selector
    let selectorHTML = '';
    if (state.agents.availableAgents.length > 0) {
        selectorHTML = `
            <div class="run-selector-wrapper">
                <button class="run-selector-button" id="agent-selector-btn">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
                        <path d="M3 9.5a1.5 1.5 0 1 1 0-3 1.5 1.5 0 0 1 0 3zm5 0a1.5 1.5 0 1 1 0-3 1.5 1.5 0 0 1 0 3zm5 0a1.5 1.5 0 1 1 0-3 1.5 1.5 0 0 1 0 3z"/>
                    </svg>
                    Agents (${state.agents.selectedAgents.length}/${state.agents.availableAgents.length})
                </button>
                <div class="run-selector-dropdown" id="agent-selector-dropdown" style="display: none;">
                    <div class="run-selector-header">Select Agents to Compare</div>
                    <div class="run-selector-list">
                        ${state.agents.availableAgents.map((agent, idx) => {
                            const isSelected = state.agents.selectedAgents.includes(agent.name);
                            const color = CONSTANTS.RUN_COLORS[idx % CONSTANTS.RUN_COLORS.length];
                            return `
                                <label class="run-selector-item">
                                    <input type="checkbox" ${isSelected ? 'checked' : ''} data-agent-name="${agent.name}">
                                    <span class="run-color-indicator" style="background: ${color};"></span>
                                    <span class="run-label">${agent.name}</span>
                                    <span class="run-steps">online</span>
                                </label>
                            `;
                        }).join('')}
                    </div>
                </div>
            </div>
        `;
    }

    // Build refresh button icon
    const refreshIcon = `
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
            <path fill-rule="evenodd" d="M8 3a5 5 0 1 0 4.546 2.914.5.5 0 0 1 .908-.417A6 6 0 1 1 8 2v1z"/>
            <path d="M8 4.466V.534a.25.25 0 0 1 .41-.192l2.36 1.966c.12.1.12.284 0 .384L8.41 4.658A.25.25 0 0 1 8 4.466z"/>
        </svg>
    `;

    // Create metadata subtitle
    const metadataHTML = `
        <span><strong>Comparing:</strong> ${agents.length} agent${agents.length > 1 ? 's' : ''}</span>
        <span><strong>Total Points:</strong> ${agents.reduce((sum, a) => sum + (a.metadata?.num_points || 0), 0)}</span>
    `;

    // Header using new component
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

    // Data-driven metric detection - filter configs to find available metrics
    const availableMetrics = CONSTANTS.METRIC_CONFIGS.filter(config => {
        if (config.source === 'data_metrics') {
            // Check data metrics (e.g., sampling weights)
            return dataMetrics.some(a =>
                a.data_metrics?.[config.key] &&
                Object.keys(a.data_metrics[config.key]).length > 0
            );
        } else if (config.isComposite && config.key === 'expert_routing_weights') {
            // Check for expert routing weight metrics: layer_*_expert_*_routing_weight
            return agents.some(a =>
                Object.keys(a.metrics).some(k =>
                    k.match(/^layer_\d+_expert_\d+_routing_weight$/)
                )
            );
        } else if (config.isComposite && config.keyPattern) {
            // Check if ANY metrics match the keyPattern
            return agents.some(a =>
                Object.keys(a.metrics).some(k => k.match(config.keyPattern))
            );
        } else {
            // Check regular agent metrics
            return agents.some(a => a.metrics[config.key]?.some(v => v !== null));
        }
    });

    // Build chart cards with spacing - data-driven loop
    let chartsHTML = '<div style="display: flex; flex-direction: column; gap: 2rem; margin-top: 2rem;">';

    chartsHTML += availableMetrics.map(config => {
        // Add step slider container for expert routing heatmap
        const stepSliderHTML = (config.key === 'expert_routing_weights') ?
            `<div id="layer-toggles-${config.canvasId}" class="layer-toggles" style="margin-bottom: 1rem; padding: 0.5rem; display: flex; gap: 1rem; flex-wrap: wrap; align-items: center;">
            </div>` : '';

        return `
        <div class="chart-card">
            <div class="chart-title">${config.title}</div>
            ${stepSliderHTML}
            <div class="chart-wrapper">
                <canvas id="${config.canvasId}"></canvas>
            </div>
        </div>
        `;
    }).join('');

    chartsHTML += '</div>';

    container.innerHTML = headerHTML + chartsHTML;

    // Render charts after DOM update - data-driven loop
    setTimeout(() => {
        availableMetrics.forEach(config => {
            if (config.type === 'bar') {
                createTokensBarChart(config.canvasId, config.label, agents, config.key);
            } else if (config.type === 'sampling') {
                createSamplingWeightsChart(config.canvasId, dataMetrics);
            } else if (config.type === 'multi_expert_line') {
                // Expert routing or architecture selection chart
                if (config.keyPattern) {
                    createMultiExpertChart(config.canvasId, config.title, config.label, agents, config.keyPattern, config);
                } else {
                    createExpertRoutingChart(config.canvasId, agents);
                }
            } else {
                // Default: line chart
                createMultiAgentChart(config.canvasId, config.label, agents, config.key);
            }
        });
    }, 10);
}

/**
 * Create multi-agent comparison chart
 */
function createMultiAgentChart(canvasId, label, agents, metricKey) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    // Destroy existing
    if (charts[canvasId]) {
        charts[canvasId].destroy();
    }

    // Get colors for the appropriate theme context (hybrid overlay or normal)
    const theme = getContextTheme(ctx);
    const { textColor, gridColor, tooltipBg } = getThemeColors(theme);

    // Build datasets
    const datasets = agents.map((agent) => {
        const metrics = agent.metrics;
        const steps = metrics.steps || [];
        const values = metrics[metricKey] || [];

        let data = steps.map((step, i) => ({
            x: step,
            y: values[i]
        })).filter(point => point.y !== null)
          .sort((a, b) => a.x - b.x);  // Ensure monotonic x-values

        // For validation metrics, remove consecutive duplicate values
        // This prevents "staircase" rendering when validation only updates every N steps
        if (metricKey === 'val_loss' || metricKey === 'val_perplexity') {
            data = data.filter((point, i) => {
                if (i === 0) return true;  // Always keep first point
                return point.y !== data[i - 1].y;  // Keep only when value changes
            });
        }

        const agentIdx = state.agents.availableAgents.findIndex(a => a.name === agent.name);
        const color = CONSTANTS.RUN_COLORS[agentIdx % CONSTANTS.RUN_COLORS.length];

        return {
            label: agent.name,
            data: data,
            borderColor: color,
            backgroundColor: color + '20',
            borderWidth: 2,
            pointRadius: 0,
            pointHoverRadius: 5,
            pointHoverBackgroundColor: color,
            pointHoverBorderColor: '#fff',
            pointHoverBorderWidth: 2,
            tension: 0,  // Use straight lines, no Bezier interpolation
            fill: false
        };
    });

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
                    display: agents.length > 1,
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
function createTokensBarChart(canvasId, label, agents, metricKey) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    // Destroy existing
    if (charts[canvasId]) {
        charts[canvasId].destroy();
    }

    // Get colors for the appropriate theme context (hybrid overlay or normal)
    const theme = getContextTheme(ctx);
    const { textColor, gridColor, tooltipBg } = getThemeColors(theme);

    // Extract latest token count for each agent
    const data = agents.map((agent) => {
        const metrics = agent.metrics;
        const values = metrics[metricKey] || [];

        // Get the last non-null value (latest token count)
        let latestValue = null;
        for (let i = values.length - 1; i >= 0; i--) {
            if (values[i] !== null) {
                latestValue = values[i];
                break;
            }
        }

        const agentIdx = state.agents.availableAgents.findIndex(a => a.name === agent.name);
        const color = CONSTANTS.RUN_COLORS[agentIdx % CONSTANTS.RUN_COLORS.length];

        return {
            label: agent.name,
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
    const newData = [];
    layers.forEach(layerNum => {
        const expertMetrics = layerExpertMetrics.get(layerNum);
        if (!expertMetrics) return;

        for (let expertNum = 0; expertNum < maxExperts; expertNum++) {
            const agentData = expertMetrics.get(expertNum);
            if (!agentData || agentData.length === 0) {
                newData.push({ x: expertNum, y: layerNum, v: null });
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
            newData.push({ x: expertNum, y: layerNum, v: avgWeight });
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
    const scatterData = [];

    layers.forEach(layerNum => {
        const expertMetrics = layerExpertMetrics.get(layerNum);
        if (!expertMetrics) return;

        for (let expertNum = 0; expertNum < maxExperts; expertNum++) {
            const agentData = expertMetrics.get(expertNum);
            if (!agentData || agentData.length === 0) {
                scatterData.push({
                    x: expertNum,
                    y: layerNum,
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
                x: expertNum,
                y: layerNum,
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

    // Calculate cell size for heatmap effect
    const cellSize = 20;  // Point radius for square cells

    // Create heatmap using scatter plot with colored square points
    charts[canvasId] = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Routing Weight',
                data: scatterData,
                pointStyle: 'rect',
                pointRadius: cellSize,
                pointHoverRadius: cellSize + 2,
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
                            const layer = dataPoint.y;
                            const expert = dataPoint.x;
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
                },
                y: {
                    type: 'linear',
                    position: 'left',
                    title: {
                        display: true,
                        text: reasoningInfo.hasReasoningSteps ? 'Layer (Reasoning Step)' : 'Layer',
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
