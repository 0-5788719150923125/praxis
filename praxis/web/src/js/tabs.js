/**
 * Praxis Web - Tab Loading and Rendering
 * Lazy-load tab content when tabs are switched
 */

import { state, getAgentFreshnessColor } from './state.js';
import { fetchAPI } from './api.js';
import { loadResearchMetricsWithCharts, initChartDeck } from './charts.js';
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
import { spawnAgent, agentViews, severAgent } from './swarm.js';

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

    // Filter sections based on configuration, then group them into three
    // "sheets" so the Identity tab ripples through them as a card deck (same
    // carousel as Research/Dynamics).
    const present = id =>
        SPEC_CONFIG.sections.some(s => s.id === id && s.condition(data));
    // Each sheet gets a card title + a brief subtitle. Single-section sheets
    // (Architecture, Arguments) render their content directly so the card
    // title isn't duplicated by an identical inner section title.
    const sheets = [
        {
            title: 'Identity & Commands',
            subtitle: 'Run hashes, reproduce commands, parameter counts',
            build: () => ['hashes', 'commands', 'parameters']
                .filter(present)
                .map(id => renderSpecSections[id](data))
                .join(''),
        },
        {
            title: 'Architecture',
            subtitle: 'Instantiated model module tree',
            build: () => present('architecture') ? createPreBlock(data.model_architecture) : '',
        },
        {
            title: 'Arguments',
            subtitle: 'Resolved run configuration',
            build: () => present('arguments') && data.args ? createPreBlock(formatJSON(data.args)) : '',
        },
    ];
    const cardsHTML = sheets
        .map(sheet => {
            const inner = sheet.build();
            if (!inner || !inner.trim()) return '';
            return `<div class="chart-card">
                <div class="chart-title">${sheet.title}</div>
                <div class="chart-subtitle">${sheet.subtitle}</div>
                <div class="deck-card-scroll">${inner}</div>
            </div>`;
        })
        .filter(Boolean)
        .join('');

    container.innerHTML = headerHTML +
        `<div class="chart-deck" id="spec-deck"><div class="chart-deck-counter"></div>${cardsHTML}</div>`;
    initChartDeck('spec-deck', { fanDown: true });
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
 * Toggle the slide-out CONTRACTS panel open/closed. Animates via a class on the
 * panel + button instead of re-rendering, so the fleet list below stays put.
 */
export function toggleContractsView() {
    state.contracts.open = !state.contracts.open;
    const panel = document.getElementById('contracts-panel');
    const btn = document.getElementById('contracts-toggle');
    if (panel) panel.classList.toggle('open', state.contracts.open);
    if (btn) btn.classList.toggle('active', state.contracts.open);
}

/**
 * Agree to a contract: spawn a browser ship locally AND join the backend pool
 * (so the expert-pool count / remote_layers grow), then re-render the fleet.
 * Leaves the contracts panel open.
 */
export async function agreeContract(contractId) {
    const contract = state.contracts.available.find(c => c.id === contractId);
    if (!contract) return;
    spawnAgent(contract);
    renderAgents(state.agents.availableAgents, document.getElementById('agents-container'));
    // Join the live backend pool. Best-effort: if no pool is active (404) the
    // local browser ship still stands on its own.
    try {
        const res = await fetch('/api/swarm/join', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ count: 1 }),
        });
        if (res.ok) {
            const data = await fetchAPI('agents');
            renderAgents(data.agents || [], document.getElementById('agents-container'));
        }
    } catch (e) {
        /* offline / no pool - the local ship is enough */
    }
}

/**
 * Sever a spawned agent's connection and re-render the Stage fleet.
 */
export function severSwarmAgent(agentId) {
    if (!severAgent(agentId)) return;
    renderAgents(state.agents.availableAgents, document.getElementById('agents-container'));
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
    // Browser-spawned ships render with their own info line (dim/layers/passes)
    // and a SEVER button instead of the freshness-colored remote-actor badge.
    if (agent.kind === 'browser') return renderBrowserShipCard(agent);
    // Backend sidecar experts: same unified tiny transformer, but the app's own
    // (no SEVER) - a plain OBSERVE/IDLE status badge.
    if (agent.kind === 'backend') return renderBackendExpertCard(agent);

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

/** The live info line for a ship (dim / layers / passes). */
const shipInfoLine = (agent) =>
    `dim: ${agent.hidden}x${agent.hidden} | layers: ${agent.layers} | passes: ${agent.passes}`;

/**
 * Render a browser-spawned ship: a first-class actor reporting its dims, layer
 * count, and live forward-pass count. The status badge is a button that flips to
 * a red SEVER on hover/focus (clicking tears down the agent). The info line has
 * a stable id so the heartbeat refresh updates its text in place (no rebuild).
 */
const renderBrowserShipCard = (agent) => {
    const statusText = agent.status.toUpperCase();
    return `
        <div class="agent-row">
            <div class="agent-info">
                <div class="agent-name">${escapeHtml(agent.name)} <span class="agent-kind-tag">browser</span></div>
                <div class="agent-url" id="ship-info-${escapeHtml(agent.id)}">${shipInfoLine(agent)}</div>
            </div>
            <button class="agent-status agent-sever ${agent.status}" data-agent-id="${escapeHtml(agent.id)}"
                    title="Sever this agent's connection" aria-label="Sever agent ${escapeHtml(agent.name)}">
                <span class="status-dot ${agent.status}"></span>
                <span class="agent-status-label">${statusText}</span>
                <span class="agent-sever-label">SEVER</span>
            </button>
        </div>
    `;
};

/**
 * Render a backend sidecar expert: the unified tiny transformer hosted by the
 * app itself (not severable). Reports its rank + forward-pass count and an
 * OBSERVE (blue) / IDLE status badge.
 */
const renderBackendExpertCard = (agent) => {
    const statusText = (agent.status || 'idle').toUpperCase();
    const rank = agent.rank != null ? `${agent.rank}x${agent.rank}` : '?';
    const info = `dim: ${rank} | layers: 1 | passes: ${agent.passes ?? 0}`;
    return `
        <div class="agent-row">
            <div class="agent-info">
                <div class="agent-name">${escapeHtml(agent.name)} <span class="agent-kind-tag">backend</span></div>
                <div class="agent-url">${info}</div>
            </div>
            <div class="agent-status ${agent.status || 'idle'}">
                <span class="status-dot ${agent.status || 'idle'}"></span>
                <span>${statusText}</span>
            </div>
        </div>
    `;
};

/**
 * Render a single contract row: description + a single AGREE button.
 */
const renderContractCard = (contract) => `
    <div class="contract-row">
        <div class="contract-info">
            <div class="contract-name">${escapeHtml(contract.title)}${
                contract.guarantee ? ` <span class="contract-guarantee">(${escapeHtml(contract.guarantee)} guarantee)</span>` : ''
            }</div>
            <div class="contract-desc">${escapeHtml(contract.description)}</div>
        </div>
        <button class="contract-agree-btn" data-contract-id="${escapeHtml(contract.id)}">AGREE</button>
    </div>
`;

/**
 * The CONTRACTS toggle button for the Stage header, with a (count) badge. It
 * opens/closes the slide-out contracts panel (it doesn't swap the view).
 */
function contractsToggleButton() {
    const count = state.contracts.available.length;
    const open = state.contracts.open;
    return {
        id: 'contracts-toggle',
        label: `CONTRACTS (${count})`,
        action: 'TOGGLE_CONTRACTS_VIEW',
        className: `tab-header-button contracts-toggle${open ? ' active' : ''}`,
    };
}

/**
 * Render the Stage tab: header + a slide-out contracts panel + the fleet list.
 * Browser-spawned ships are first-class actors and sit in the same list as the
 * discovered remote agents; the contracts panel slides out above them.
 */
export function renderAgents(agents, container) {
    container = container || document.getElementById('agents-container');
    if (!container) return;
    state.agents.availableAgents = agents || state.agents.availableAgents || [];

    const title = state.theme === 'dark' ? 'Hangar' : 'Wire';
    const ships = agentViews();
    const fleet = [...ships, ...state.agents.availableAgents]; // own ships first
    // Apply the per-type naming convention here (the frontend owns names; they
    // label the agent *type*, not identity, so they repeat across sources). All
    // unified tiny-transformer experts - browser ships and backend sidecar
    // experts alike - are arc-1, arc-2, ... in list order.
    let _arc = 0;
    for (const a of fleet) {
        if (a.kind === 'browser' || a.kind === 'backend') {
            a.name = `arc-${++_arc}`;
        }
    }
    const headerHTML = createTabHeader({
        title: title,
        buttons: [contractsToggleButton()],
        metadata: `<span><strong>Fleet:</strong> ${fleet.length} actor${fleet.length !== 1 ? 's' : ''}${
            ships.length ? ` (${ships.length} local)` : ''
        }</span>`
    });

    const contractsHtml = state.contracts.available.map(renderContractCard).join('');
    const agentsHtml = fleet.length
        ? fleet.map(agent => renderAgentCard(agent, fleet)).join('')
        : '<div class="loading-placeholder">No actors discovered.</div>';

    container.innerHTML = `
        ${headerHTML}
        <div class="contracts-panel${state.contracts.open ? ' open' : ''}" id="contracts-panel">
            <div class="contracts-panel-inner">
                <div class="contracts-panel-label">Offer your compute to the swarm.</div>
                <div class="contracts-list">${contractsHtml}</div>
            </div>
        </div>
        <div class="agents-list">${agentsHtml}</div>
    `;

    ensureFleetRefresh();
}

// Keep the Stage fleet live while it's on screen. Two cadences:
//   - every tick: surgically update browser-ship counters in place (no rebuild,
//     so a hovered row never flickers to the heartbeat).
//   - slower: re-fetch /api/agents so backend-spawned experts (arc-N) appear and
//     update their status/count as the swarm grows.
// Runs whenever the Stage tab is active (a backend swarm can exist with zero
// browser ships); stops itself when you leave the tab.
let _fleetRefreshTimer = null;
let _fleetTicks = 0;
function ensureFleetRefresh() {
    if (_fleetRefreshTimer) return;
    _fleetTicks = 0;
    _fleetRefreshTimer = setInterval(async () => {
        if (state.currentTab !== 'agents') {
            clearInterval(_fleetRefreshTimer);
            _fleetRefreshTimer = null;
            return;
        }
        _fleetTicks += 1;
        // Surgical browser-ship counter update (cheap, every 2s).
        for (const agent of agentViews()) {
            const el = document.getElementById(`ship-info-${agent.id}`);
            if (el) el.textContent = shipInfoLine(agent);
        }
        // Re-fetch the discovered fleet (backend experts) every ~6s and re-render
        // if the roster changed, so arc-N experts show up as they spawn.
        if (_fleetTicks % 3 === 0) {
            try {
                const data = await fetchAPI('agents');
                const next = data.agents || [];
                const prev = state.agents.availableAgents || [];
                // Compare on identity-ish fields (uid/url + status), since names
                // are assigned at render time and would always look "changed".
                const key = (a) => `${a.uid || a.url || ''}:${a.status || ''}`;
                const changed =
                    next.length !== prev.length ||
                    next.some((a, i) => !prev[i] || key(prev[i]) !== key(a));
                if (changed) {
                    renderAgents(next, document.getElementById('agents-container'));
                }
            } catch (e) {
                /* transient fetch error - try again next tick */
            }
        }
    }, 2000);
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

