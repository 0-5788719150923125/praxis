/**
 * Praxis Web - Chart.js Integration
 * Full Chart.js implementation for Research tab
 */

import { state, CONSTANTS } from './state.js';
import { fetchAPI } from './api.js';

// Chart instances storage
const charts = {};

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

                if (!response.ok) return null;

                const data = await response.json();

                if (data.status === 'no_data' || !data.runs || data.runs.length === 0) {
                    return null;
                }

                return {
                    name: agentName,
                    url: agent.url,
                    metrics: data.runs[0].metrics,
                    metadata: data.runs[0].metadata
                };
            } catch (error) {
                // Silently handle all errors (CORS, network, timeout, etc.)
                // This gracefully handles any URL type without hardcoded filtering
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

        renderMetricsCharts({ agents: agentMetrics }, container);
        state.research.loaded = true;

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

    // Header
    let html = `
        <div class="research-header">
            <div class="research-title-row">
                <h2>Metrics</h2>
                <div class="research-controls">
                    ${selectorHTML}
                    <button class="refresh-button" id="refresh-metrics-btn">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
                            <path fill-rule="evenodd" d="M8 3a5 5 0 1 0 4.546 2.914.5.5 0 0 1 .908-.417A6 6 0 1 1 8 2v1z"/>
                            <path d="M8 4.466V.534a.25.25 0 0 1 .41-.192l2.36 1.966c.12.1.12.284 0 .384L8.41 4.658A.25.25 0 0 1 8 4.466z"/>
                        </svg>
                        Refresh
                    </button>
                </div>
            </div>
            <div class="metadata">
                <span><strong>Comparing:</strong> ${agents.length} agent${agents.length > 1 ? 's' : ''}</span>
                <span><strong>Total Points:</strong> ${agents.reduce((sum, a) => sum + (a.metadata?.num_points || 0), 0)}</span>
            </div>
        </div>
    `;

    // Check which metrics exist
    const hasLoss = agents.some(a => a.metrics.loss?.some(v => v !== null));
    const hasValLoss = agents.some(a => a.metrics.val_loss?.some(v => v !== null));
    const hasPerplexity = agents.some(a => a.metrics.val_perplexity?.some(v => v !== null));
    const hasLR = agents.some(a => a.metrics.learning_rate?.some(v => v !== null));
    const hasTokens = agents.some(a => a.metrics.num_tokens?.some(v => v !== null));
    const hasSoftmax = agents.some(a => a.metrics.softmax_collapse?.some(v => v !== null));
    const hasAvgStepTime = agents.some(a => a.metrics.avg_step_time?.some(v => v !== null));

    // Build chart cards with spacing
    html += '<div style="display: flex; flex-direction: column; gap: 2rem; margin-top: 2rem;">';

    if (hasLoss) html += '<div class="chart-card"><div class="chart-title">Training Loss</div><div class="chart-wrapper"><canvas id="chart-train-loss"></canvas></div></div>';
    if (hasValLoss) html += '<div class="chart-card"><div class="chart-title">Validation Loss</div><div class="chart-wrapper"><canvas id="chart-val-loss"></canvas></div></div>';
    if (hasPerplexity) html += '<div class="chart-card"><div class="chart-title">Perplexity</div><div class="chart-wrapper"><canvas id="chart-perplexity"></canvas></div></div>';
    if (hasLR) html += '<div class="chart-card"><div class="chart-title">Learning Rate</div><div class="chart-wrapper"><canvas id="chart-lr"></canvas></div></div>';
    if (hasTokens) html += '<div class="chart-card"><div class="chart-title">Tokens (Billions)</div><div class="chart-wrapper"><canvas id="chart-tokens"></canvas></div></div>';
    if (hasAvgStepTime) html += '<div class="chart-card"><div class="chart-title">Average Step Time</div><div class="chart-wrapper"><canvas id="chart-avg-step-time"></canvas></div></div>';
    if (hasSoftmax) html += '<div class="chart-card"><div class="chart-title">Softmax Collapse</div><div class="chart-wrapper"><canvas id="chart-softmax"></canvas></div></div>';

    html += '</div>';

    container.innerHTML = html;

    // Render charts after DOM update
    setTimeout(() => {
        if (hasLoss) createMultiAgentChart('chart-train-loss', 'Training Loss', agents, 'loss');
        if (hasValLoss) createMultiAgentChart('chart-val-loss', 'Validation Loss', agents, 'val_loss');
        if (hasPerplexity) createMultiAgentChart('chart-perplexity', 'Perplexity', agents, 'val_perplexity');
        if (hasLR) createMultiAgentChart('chart-lr', 'Learning Rate', agents, 'learning_rate');
        if (hasTokens) createTokensBarChart('chart-tokens', 'Tokens (B)', agents, 'num_tokens');
        if (hasAvgStepTime) createMultiAgentChart('chart-avg-step-time', 'Avg Step Time (s)', agents, 'avg_step_time');
        if (hasSoftmax) createMultiAgentChart('chart-softmax', 'Softmax Collapse', agents, 'softmax_collapse');
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

    const isDark = state.theme === 'dark';
    const textColor = isDark ? '#e0e0e0' : '#333333';
    const gridColor = isDark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';

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
                    backgroundColor: isDark ? '#1e1e1e' : '#ffffff',
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

    const isDark = state.theme === 'dark';
    const textColor = isDark ? '#e0e0e0' : '#333333';
    const gridColor = isDark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';

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
                    backgroundColor: isDark ? '#1e1e1e' : '#ffffff',
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
