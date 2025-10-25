/**
 * Praxis Web - Tab Loading and Rendering
 * Lazy-load tab content when tabs are switched
 */

import { state } from './state.js';
import { fetchAPI } from './api.js';
import { loadResearchMetricsWithCharts } from './charts.js';

/**
 * Generic tab data loader - DRY pattern for all tab loading
 * @param {Object} config - Tab loading configuration
 * @param {string} config.stateKey - Key in state object (e.g., 'spec', 'agents')
 * @param {string} config.containerId - DOM container ID
 * @param {string} config.loadingMessage - Message to show while loading
 * @param {Function} config.fetchFn - Async function to fetch data
 * @param {Function} config.renderFn - Function to render data
 * @param {Function} [config.processData] - Optional function to process fetched data before storing
 */
async function loadTabData(config) {
    const stateObj = state[config.stateKey];
    if (stateObj.loaded) return;

    const container = document.getElementById(config.containerId);
    if (!container) return;

    container.innerHTML = `<div class="loading-placeholder">${config.loadingMessage}</div>`;

    try {
        const data = await config.fetchFn();

        // Process data if processor provided, otherwise use raw data
        const processedData = config.processData ? config.processData(data) : data;

        // Store data in state
        if (config.processData) {
            Object.assign(stateObj, processedData);
        } else {
            stateObj.data = data;
        }

        stateObj.loaded = true;
        stateObj.error = null;

        config.renderFn(data, container);
    } catch (error) {
        stateObj.error = error.message;
        container.innerHTML = `<div class="loading-placeholder" style="color: #cc0000;">Error: ${error.message}</div>`;
    }
}

/**
 * Load Spec tab content
 */
export async function loadSpec() {
    await loadTabData({
        stateKey: 'spec',
        containerId: 'spec-container',
        loadingMessage: 'Loading specification...',
        fetchFn: () => fetchAPI('spec'),
        renderFn: renderSpec
    });
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

        // Step 1: Clone
        html += '<div class="spec-metadata">1. Clone from source:</div>';
        html += `<div class="spec-code" style="margin-bottom: 1rem;">git clone ${data.git_url}</div>`;

        // Step 2: Move into directory
        html += '<div class="spec-metadata">2. Move into directory:</div>';
        html += '<div class="spec-code" style="margin-bottom: 1rem;">cd praxis</div>';

        // Step 3: Download experiment config from API
        // Extract experiment name from command to create matching filename
        const command = data.command ? data.command.replace('python main.py', './launch') : './launch';
        const expMatch = command.match(/--([a-z0-9\-]+)/);
        const expName = expMatch ? expMatch[1] : 'reproduce';
        const configFilename = `experiments/${expName}.yml`;

        // Link to API endpoint for config download
        const configUrl = '/api/config';

        html += `<div class="spec-metadata" style="margin-bottom: 1rem;">3. <a href="${configUrl}" download="${expName}.yml" style="color: #0B9A6D; font-weight: 600; text-decoration: none;">Download</a> config, save it to: <code style="background: rgba(11, 154, 109, 0.1); padding: 2px 6px; border-radius: 3px;">${escapeHtml(configFilename)}</code></div>`;

        // Step 4: Reproduce
        html += '<div class="spec-metadata">4. Reproduce experiment:</div>';
        const reproduceCommand = command + (command.includes('--reset') ? '' : ' --reset');
        html += `<div class="spec-code">${escapeHtml(reproduceCommand)}</div>`;

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
    await loadTabData({
        stateKey: 'agents',
        containerId: 'agents-container',
        loadingMessage: 'Loading agents...',
        fetchFn: () => fetchAPI('agents'),
        processData: (data) => ({ availableAgents: data.agents || [] }),
        renderFn: (data, container) => renderAgents(data.agents || [], container)
    });
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

        // Build inline display: masked_url | short_hash | url
        let infoLine = '';
        const displayUrl = agent.masked_url || agent.url;
        if (displayUrl) {
            infoLine += escapeHtml(displayUrl);
        }
        if (agent.short_hash) {
            infoLine += ` | ${escapeHtml(agent.short_hash)}`;
        }
        if (agent.url) {
            infoLine += ` | ${escapeHtml(agent.url)}`;
        }
        if (infoLine) {
            html += `<div class="agent-url">${infoLine}</div>`;
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
