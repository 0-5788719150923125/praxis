/**
 * Praxis Web - Tab Loading and Rendering
 * Lazy-load tab content when tabs are switched
 */

import { state } from './state.js';
import { fetchSpec, fetchAgents } from './api.js';
import { loadResearchMetricsWithCharts } from './charts.js';

/**
 * Load Spec tab content
 */
export async function loadSpec() {
    if (state.spec.loaded) return;

    const container = document.getElementById('spec-container');
    if (!container) return;

    container.innerHTML = '<div class="loading-placeholder">Loading specification...</div>';

    try {
        const data = await fetchSpec();
        state.spec.data = data;
        state.spec.loaded = true;
        state.spec.error = null;

        renderSpec(data, container);
    } catch (error) {
        state.spec.error = error.message;
        container.innerHTML = `<div class="loading-placeholder" style="color: #cc0000;">Error: ${error.message}</div>`;
    }
}

/**
 * Render spec tab content
 */
function renderSpec(data, container) {
    let html = '';

    // Hashes section
    if (data.full_hash && data.truncated_hash) {
        html += '<div class="spec-section">';
        html += '<div class="spec-title">Hashes</div>';

        const truncLen = data.truncated_hash.length;
        const truncPart = data.full_hash.substring(0, truncLen);
        const restPart = data.full_hash.substring(truncLen);
        html += '<div class="spec-hash">';
        html += `<a href="#args-section" style="color: #0B9A6D; font-weight: 600; text-decoration: none;">${truncPart}</a>`;
        html += `<span style="color: var(--text);">${restPart}</span>`;
        html += '</div>';
        html += '</div>';
    }

    // Commands section
    if (data.git_url) {
        html += '<div class="spec-section">';
        html += '<div class="spec-title">Commands</div>';
        html += '<div style="margin-bottom: 1rem;">';
        html += '<div class="spec-metadata">Clone from source:</div>';
        html += `<div class="spec-code" style="margin-bottom: 1rem;">git clone ${data.git_url}</div>`;
        html += '<div class="spec-metadata">Move into directory:</div>';
        html += '<div class="spec-code" style="margin-bottom: 1rem;">cd praxis</div>';
        html += '<div class="spec-metadata">Reproduce experiment:</div>';
        const command = data.command ? data.command.replace('python main.py', './launch') : './launch';
        html += `<div class="spec-code">${escapeHtml(command)}</div>`;
        html += '</div>';
        html += '</div>';
    }

    // Parameters section
    if (data.param_stats) {
        html += '<div class="spec-section">';
        html += '<div class="spec-title">Parameters</div>';

        if (data.param_stats.model_parameters) {
            html += `<div class="spec-metadata">Model Parameters: <span style="color: #0B9A6D; font-weight: 600;">${data.param_stats.model_parameters.toLocaleString()}</span></div>`;
        }

        if (data.param_stats.optimizer_parameters) {
            html += `<div class="spec-metadata">Optimizer Parameters: <span style="color: #0B9A6D; font-weight: 600;">${data.param_stats.optimizer_parameters.toLocaleString()}</span></div>`;
        }

        html += '</div>';
    }

    // Model architecture
    if (data.model_architecture) {
        html += '<div class="spec-section">';
        html += '<div class="spec-title">Architecture</div>';
        html += `<pre class="spec-code">${escapeHtml(data.model_architecture)}</pre>`;
        html += '</div>';
    }

    // Arguments section
    html += '<div id="args-section" class="spec-section">';
    html += '<div class="spec-title">Arguments</div>';

    if (data.timestamp) {
        html += `<div class="spec-metadata">Created: ${data.timestamp}</div>`;
    }

    if (data.args) {
        html += '<pre class="spec-code">' + escapeHtml(JSON.stringify(data.args, null, 2)) + '</pre>';
    }

    html += '</div>';

    container.innerHTML = html;
}

/**
 * Load Agents tab content
 */
export async function loadAgents() {
    if (state.agents.loaded) return;

    const container = document.getElementById('agents-container');
    if (!container) return;

    container.innerHTML = '<div class="loading-placeholder">Loading agents...</div>';

    try {
        const data = await fetchAgents();
        state.agents.availableAgents = data.agents || [];
        state.agents.loaded = true;
        state.agents.error = null;

        renderAgents(data.agents || [], container);
    } catch (error) {
        state.agents.error = error.message;
        container.innerHTML = `<div class="loading-placeholder" style="color: #cc0000;">Error: ${error.message}</div>`;
    }
}

/**
 * Render agents tab content
 */
function renderAgents(agents, container) {
    if (!agents || agents.length === 0) {
        container.innerHTML = '<div class="agents-empty">No agents found.</div>';
        return;
    }

    let html = '<div class="agents-section">';
    html += '<div class="agents-title">Available Agents</div>';
    html += '<div class="agents-table"><div class="agents-list">';

    agents.forEach(agent => {
        const statusClass = agent.status || 'offline';
        const statusText = statusClass.charAt(0).toUpperCase() + statusClass.slice(1);

        html += '<div class="agent-row">';
        html += '<div class="agent-info">';
        html += `<div class="agent-name">${escapeHtml(agent.name || 'Unknown')}</div>`;
        if (agent.url) {
            html += `<div class="agent-url">${escapeHtml(agent.url)}</div>`;
        }
        html += '</div>';
        html += `<div class="agent-status ${statusClass}">`;
        html += `<span class="status-dot ${statusClass}"></span>`;
        html += `<span>${statusText}</span>`;
        html += '</div>';
        html += '</div>';
    });

    html += '</div></div></div>';
    container.innerHTML = html;
}

/**
 * Load Research/Metrics tab content
 * Delegates to charts.js for full implementation
 */
export async function loadResearchMetrics(force = false) {
    await loadResearchMetricsWithCharts(force);
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(str) {
    if (typeof str !== 'string') return str;
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}
