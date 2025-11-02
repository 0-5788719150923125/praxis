/**
 * Praxis Web - Tab Loading and Rendering
 * Lazy-load tab content when tabs are switched
 */

import { state, getAgentFreshnessColor } from './state.js';
import { fetchAPI } from './api.js';
import { loadResearchMetricsWithCharts } from './charts.js';
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
 * Render spec tab content using generic components
 * Pure functional approach - configuration drives rendering
 */
function renderSpec(data, container) {
    if (!data) {
        container.innerHTML = '<div class="loading-placeholder">No specification data available.</div>';
        return;
    }

    // Create peer button for header
    const gitRemoteCmd = `git remote add ${data.truncated_hash} ${data.git_url}`;
    const peerButton = {
        id: 'peer-button',
        label: 'Peer with agent',
        icon: '',
        className: 'tab-header-button copy-git-remote-btn',
        dataAttrs: `data-command="${escapeHtml(gitRemoteCmd)}"`
    };

    // Create header with button
    const headerHTML = createTabHeader({
        title: 'Configuration',
        buttons: [peerButton],
        metadata: data.timestamp ? `<span><strong>Created:</strong> ${data.timestamp}</span>` : ''
    });

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

    // Create header
    const headerHTML = createTabHeader({
        title: 'Hangar',
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

