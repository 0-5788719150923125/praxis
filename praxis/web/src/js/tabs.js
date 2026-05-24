/**
 * Praxis Web - Tab Loading and Rendering
 * Lazy-load tab content when tabs are switched
 */

import { state, getAgentFreshnessColor } from './state.js';
import { fetchAPI } from './api.js';
import { loadResearchMetricsWithCharts } from './charts.js';
import { formatRelativeTime } from './charts.js';
import {
    createSection,
    createCodeBlock,
    createPreBlock,
    createMetadata,
    createKeyValue,
    createStepsList,
    createButton,
    createWrapper,
    createHashDisplay,
    createTabHeader,
    renderIf,
    formatNumber,
    formatJSON,
    escapeHtml
} from './components.js';
import { SPEC_CONFIG, extractCommandInfo, AGENT_DISPLAY_FIELDS } from './config.js';

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
 * Load the list of runs (for the Identity-tab run picker dropdown).
 * Idempotent; safe to call on every spec load.
 */
async function loadAvailableSpecRuns() {
    try {
        const response = await fetch('/api/runs');
        if (!response.ok) return [];
        const data = await response.json();
        state.spec.availableRuns = data.runs || [];
        return state.spec.availableRuns;
    } catch (error) {
        console.error('[Spec] Failed to load run list:', error);
        return [];
    }
}

/**
 * Toggle the Identity-tab run picker dropdown.
 */
export function toggleSpecRunSelector() {
    state.spec.runSelectorOpen = !state.spec.runSelectorOpen;
    const dropdown = document.getElementById('spec-run-selector-dropdown');
    if (dropdown) {
        dropdown.style.display = state.spec.runSelectorOpen ? 'block' : 'none';
    }
}

/**
 * Select a run for the Identity tab (single-select; null = current run).
 */
export function selectSpecRun(hash) {
    state.spec.selectedRun = hash || null;
    state.spec.runSelectorOpen = false;
    state.spec.loaded = false;
    loadSpec(true);
}

/**
 * Load Spec tab content. When a non-current run is selected we fetch the
 * persisted snapshot via /api/spec?runs=<hash>.
 */
export async function loadSpec(force = false) {
    if (state.spec.loaded && !force) return;

    const container = document.getElementById('spec-container');
    if (!container) return;

    container.innerHTML = '<div class="loading-placeholder">Loading specification...</div>';

    try {
        await loadAvailableSpecRuns();
        const runQuery = state.spec.selectedRun
            ? `?runs=${encodeURIComponent(state.spec.selectedRun)}`
            : '';
        const response = await fetch(`/api/spec${runQuery}`);
        if (!response.ok) throw new Error(`API returned ${response.status}`);
        const data = await response.json();

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
 * Spec section renderers - pure functions using generic components
 * Map section IDs to render functions
 */
const renderSpecSections = {
    // Note: peer-button is now in the header, kept here for backwards compat
    'peer-button': (data) => {
        return ''; // Moved to header
    },

    'hashes': (data) => {
        const hash = createHashDisplay(data.full_hash, data.truncated_hash.length, '#args-section');
        return createSection('Hashes', hash);
    },

    'commands': (data) => {
        const cmdInfo = extractCommandInfo(data);
        const downloadLink = `<a href="/api/config" download="${cmdInfo.expName}.yml" class="spec-link">Download</a>`;

        const steps = [
            { instruction: 'Clone from source:', code: `git clone ${data.git_url}` },
            { instruction: 'Move into directory:', code: 'cd praxis' },
            { instruction: `${downloadLink} config, save it to:`, code: cmdInfo.configFilename },
            { instruction: 'Reproduce experiment:', code: cmdInfo.reproduceCommand }
        ];

        const stepsHtml = createStepsList(steps);
        return createSection('Commands', createWrapper(stepsHtml, 'spec-steps'));
    },

    'parameters': (data) => {
        const params = [];
        if (data.param_stats.model_parameters) {
            params.push(createKeyValue('Model Parameters', formatNumber(data.param_stats.model_parameters)));
        }
        if (data.param_stats.optimizer_parameters) {
            params.push(createKeyValue('Optimizer Parameters', formatNumber(data.param_stats.optimizer_parameters)));
        }
        return params.length > 0 ? createSection('Parameters', params.join('')) : '';
    },

    'architecture': (data) => {
        return createSection('Architecture', createPreBlock(data.model_architecture));
    },

    'arguments': (data) => {
        const content = [
            renderIf(data.timestamp, () => createMetadata(`Created: ${data.timestamp}`)),
            renderIf(data.args, () => createPreBlock(formatJSON(data.args)))
        ].filter(Boolean).join('');

        return createSection('Arguments', content, 'args-section');
    }
};

/**
 * Build the run selector dropdown (HTML string) for the Identity tab.
 * Mirrors the Dynamics-tab picker; events target #spec-* IDs and the
 * data-spec-run-hash attribute on inputs.
 */
function renderSpecRunSelector() {
    const runs = state.spec.availableRuns || [];
    if (runs.length === 0) return '';

    const selected = state.spec.selectedRun;
    const activeRun = runs.find(r => r.hash === selected)
        || runs.find(r => r.is_current)
        || runs[0];
    const label = activeRun ? activeRun.hash : 'Run';

    return `
        <div class="run-selector-wrapper">
            <button class="run-selector-button" id="spec-run-selector-btn">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
                    <path d="M8 1a7 7 0 1 0 4.95 11.95l.707.707A8.001 8.001 0 1 1 8 0v1z"/>
                    <path d="M7.5 3a.5.5 0 0 1 .5.5v5.21l3.248 1.856a.5.5 0 0 1-.496.868l-3.5-2A.5.5 0 0 1 7 9V3.5a.5.5 0 0 1 .5-.5z"/>
                </svg>
                Run: ${label}
            </button>
            <div class="run-selector-dropdown" id="spec-run-selector-dropdown" style="display: none;">
                <div class="run-selector-header">Select Run</div>
                <div class="run-selector-list">
                    ${runs.map(run => {
                        const isActive = run.hash === (activeRun ? activeRun.hash : null);
                        const time = formatRelativeTime(run.metrics_updated);
                        const badge = run.is_current ? ' <span style="opacity: 0.6; font-size: 0.8em;">(active)</span>' : '';
                        return `
                            <label class="run-selector-item">
                                <input type="radio" name="spec-run" ${isActive ? 'checked' : ''} data-spec-run-hash="${run.hash}">
                                <span class="run-label">${run.hash}${badge}</span>
                                <span class="run-steps">${run.num_steps} steps &middot; ${time}</span>
                            </label>
                        `;
                    }).join('')}
                </div>
            </div>
        </div>
    `;
}

/**
 * Render spec tab content using generic components
 * Pure functional approach - configuration drives rendering
 */
function renderSpec(data, container) {
    if (!data) {
        container.innerHTML = '<div class="loading-placeholder">No specification data available.</div>';
        return;
    }

    // Peer button only makes sense for the live run (snapshots lack git_url).
    const buttons = [];
    if (data.git_url) {
        const gitRemoteCmd = `git remote add ${data.truncated_hash} ${data.git_url}`;
        buttons.push({
            id: 'peer-button',
            label: 'Peer with agent',
            icon: '',
            className: 'tab-header-button copy-git-remote-btn',
            dataAttrs: `data-command="${escapeHtml(gitRemoteCmd)}"`
        });
    }

    const metaParts = [];
    if (data.timestamp) {
        metaParts.push(`<span><strong>Created:</strong> ${data.timestamp}</span>`);
    }
    if (data.is_snapshot) {
        metaParts.push('<span><strong>Source:</strong> snapshot</span>');
    }

    const headerHTML = createTabHeader({
        title: 'Configuration',
        additionalContent: renderSpecRunSelector(),
        buttons,
        metadata: metaParts.join('\n')
    });

    if (data.snapshot_missing) {
        container.innerHTML = headerHTML + `
            <div class="empty-state" style="margin-top: 2rem;">
                <h3>No Snapshot Available</h3>
                <p>This run was created before per-run spec snapshots were captured. Re-launch the run to populate its identity.</p>
            </div>
        `;
        return;
    }

    // Filter and sort sections based on configuration
    const visibleSections = SPEC_CONFIG.sections
        .filter(section => section.condition(data))
        .sort((a, b) => a.order - b.order);

    // Render all visible sections using generic components
    const sectionsHTML = visibleSections
        .map(section => renderSpecSections[section.id](data))
        .join('');

    container.innerHTML = headerHTML + sectionsHTML;
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
/**
 * Render agent card - pure component with commit freshness colors
 * @param {Object} agent - Agent data
 * @param {Array<Object>} allAgents - All agents for color calculation
 * @returns {string} HTML string
 */
const renderAgentCard = (agent, allAgents) => {
    const statusClass = agent.status || 'offline';
    const statusText = statusClass.charAt(0).toUpperCase() + statusClass.slice(1);

    // Build info line using configuration
    const infoLine = AGENT_DISPLAY_FIELDS
        .filter(field => field.condition(agent))
        .map(field => `${field.label}: ${escapeHtml(field.getValue(agent))}`)
        .join(' | ');

    const infoHtml = infoLine ? `<div class="agent-url">${infoLine}</div>` : '';

    // Calculate freshness-based colors for ALL agents
    // Each status type gets its own base color modulated by commit age
    const colors = getAgentFreshnessColor(agent, allAgents, state.theme);

    // Apply dynamic colors via inline styles
    // background with 0.1 opacity, text color, dot color
    const statusStyle = `background-color: ${colors.background.replace('rgb', 'rgba').replace(')', ', 0.1)')}; color: ${colors.text};`;
    const dotStyle = `background-color: ${colors.dot};`;

    return `
        <div class="agent-row">
            <div class="agent-info">
                <div class="agent-name">${escapeHtml(agent.name || 'Unknown')}</div>
                ${infoHtml}
            </div>
            <div class="agent-status ${statusClass}" style="${statusStyle}">
                <span class="status-dot ${statusClass}" style="${dotStyle}"></span>
                <span>${statusText}</span>
            </div>
        </div>
    `;
};

/**
 * Render agents tab content using generic components
 */
function renderAgents(agents, container) {
    if (!agents || agents.length === 0) {
        container.innerHTML = '<div class="loading-placeholder">No agents available.</div>';
        return;
    }

    // Create header with theme-aware title
    const title = state.theme === 'dark' ? 'Hangar' : 'Wire';
    const headerHTML = createTabHeader({
        title: title,
        metadata: `<span><strong>Discovered:</strong> ${agents.length} actor${agents.length !== 1 ? 's' : ''}</span>`
    });

    // Pass all agents to each card for color calculation
    const agentsHtml = agents.map(agent => renderAgentCard(agent, agents)).join('');

    container.innerHTML = `
        ${headerHTML}
        <div class="agents-list">${agentsHtml}</div>
    `;
}

/**
 * Load Research/Metrics tab content
 * Delegates to charts.js for full implementation
 */
export async function loadResearchMetrics(force = false) {
    await loadResearchMetricsWithCharts(force);
}

/**
 * Load Dynamics/Gradient tab content
 * Delegates to dynamics.js for full implementation
 */
export async function loadDynamics(force = false) {
    const { loadDynamicsWithCharts } = await import('./dynamics.js');
    await loadDynamicsWithCharts(force);
}

