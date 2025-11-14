/**
 * Praxis Web - State Management
 * Single source of truth - everything is data
 */

import {
    createSimpleTabContent,
    createContainerWithContent,
    createChatTabContent,
    createIframeTabContent
} from './components.js';

// Application state - all data lives here
export const state = {
    // Theme
    theme: 'dark',
    isHybridMode: false, // Prismatic split-reality theme mode

    // Conversation
    messages: [],
    isThinking: false,

    // Input state
    isShowingPlaceholder: true,

    // Tabs - fully data-driven configuration
    currentTab: 'chat',  // Track current active tab
    tabs: [
        {
            id: 'chat',
            label: 'Gymnasium',
            active: true,
            containerClass: 'chat-container',
            customClasses: [],
            template: () => createChatTabContent({
                chatContainerId: 'chat-container',
                inputId: 'message-input',
                inputRows: 1
            }),
            onActivate: null,
            onDeactivate: null
        },
        {
            id: 'terminal',
            label: 'Terminal',
            active: false,
            containerClass: 'terminal-container',
            customClasses: ['has-dashboard'],
            template: () => createContainerWithContent(
                'terminal-container',
                'terminal-display',
                '<div class="terminal-line">Terminal ready. Dashboard will connect automatically when available.</div>'
            ),
            onActivate: 'recalculateDashboardScale',
            activateDelay: 0,  // Delay before calling onActivate (ms)
            activateParams: [],  // Parameters to pass to activation function
            onDeactivate: null
        },
        {
            id: 'agents',
            label: 'Agency',
            active: false,
            containerClass: 'agents-container',
            customClasses: [],
            template: () => createSimpleTabContent(
                'agents-container',
                'agents-container',
                'Loading agents...'
            ),
            onActivate: 'loadAgents',
            activateDelay: 0,
            activateParams: [],
            onDeactivate: null
        },
        // {
        //     id: 'books',
        //     label: 'Booking',
        //     active: false,
        //     containerClass: 'iframe-container',
        //     customClasses: ['iframe-view'],
        //     template: () => createIframeTabContent({
        //         url: 'https://try.axe.eco/basic/note.html',
        //         containerClass: 'iframe-container',
        //         containerId: 'books-container',
        //         title: 'Books'
        //     }),
        //     onActivate: null,
        //     activateDelay: 0,
        //     activateParams: [],
        //     onDeactivate: null
        // },
        {
            id: 'research',
            label: 'Research',
            active: false,
            containerClass: 'research-container',
            customClasses: ['metrics-view'],
            hasCharts: true,  // Feature flag: this tab contains Chart.js charts
            template: () => createSimpleTabContent(
                'research-container',
                'research-container',
                'Loading metrics...'
            ),
            onActivate: 'loadResearchMetrics',
            activateDelay: 0,
            activateParams: [false],  // Don't force refresh - user clicks reload button
            onDeactivate: null
        },
        {
            id: 'dynamics',
            label: 'Dynamics',
            active: false,
            containerClass: 'dynamics-container',
            customClasses: ['dynamics-view'],
            hasCharts: true,  // Uses Chart.js for gradient visualizations
            template: () => createSimpleTabContent(
                'dynamics-container',
                'dynamics-container',
                'Loading gradient dynamics...'
            ),
            onActivate: 'loadDynamics',
            activateDelay: 0,
            activateParams: [false],  // Don't force refresh by default
            onDeactivate: null
        },
        {
            id: 'spec',
            label: 'Identity',
            active: false,
            containerClass: 'spec-container',
            customClasses: [],
            template: () => createSimpleTabContent(
                'spec-container',
                'spec-container',
                'Loading specification...'
            ),
            onActivate: 'loadSpec',
            activateDelay: 0,
            activateParams: [],
            onDeactivate: null
        }
    ],

    // Terminal
    terminal: {
        connected: false,
        lines: ['Terminal ready. Dashboard will connect automatically when available.']
    },

    // Settings/Generation params
    settings: {
        systemPrompt: 'Write thy wrong.',
        apiUrl: '',  // Will be set on init
        maxTokens: 256,
        temperature: 0.5,
        repetitionPenalty: 1.2,
        doSample: true,
        useCache: false,
        debugLogging: false
    },

    // Modal state
    modals: {
        settingsOpen: false
    },

    // Spec data (loaded async)
    spec: {
        loaded: false,
        data: null,
        error: null
    },

    // Agents data (loaded async)
    agents: {
        loaded: false,
        availableAgents: [],
        selectedAgents: [],
        selectorOpen: false,
        error: null
    },

    // Research/metrics data (loaded async)
    research: {
        loaded: false,
        runs: [],
        selectedRuns: [],
        charts: {},
        etag: null,
        lastStep: 0,
        error: null
    },

    // Dynamics/gradient data (loaded async)
    dynamics: {
        loaded: false,
        data: null,
        lastStep: 0,  // Track last loaded step for incremental updates
        error: null
    }
};

// ============================================================================
// COLOR UTILITIES - Pure Functional Color Manipulation
// ============================================================================

/**
 * Convert RGB color to greyscale using luminance formula
 * Pure function: (r, g, b) → (grey, grey, grey)
 * @param {number} r - Red (0-255)
 * @param {number} g - Green (0-255)
 * @param {number} b - Blue (0-255)
 * @returns {Array<number>} [grey, grey, grey]
 */
export function rgbToGreyscale(r, g, b) {
    // Standard luminance formula (ITU-R BT.709)
    const grey = Math.round(0.2126 * r + 0.7152 * g + 0.0722 * b);
    return [grey, grey, grey];
}

/**
 * Linear interpolation between two colors
 * Pure function: (color1, color2, t) → interpolated color
 * @param {Array<number>} color1 - RGB array [r, g, b]
 * @param {Array<number>} color2 - RGB array [r, g, b]
 * @param {number} t - Interpolation factor (0 = color1, 1 = color2)
 * @returns {Array<number>} Interpolated RGB array
 */
export function lerpColor(color1, color2, t) {
    // Clamp t to [0, 1]
    t = Math.max(0, Math.min(1, t));
    return color1.map((c1, i) => Math.round(c1 + (color2[i] - c1) * t));
}

/**
 * Convert RGB array to CSS rgb() string
 * Pure function: [r, g, b] → "rgb(r, g, b)"
 * @param {Array<number>} rgb - RGB array [r, g, b]
 * @returns {string} CSS color string
 */
export function rgbToString(rgb) {
    return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
}

/**
 * Convert RGB array to CSS rgba() string with alpha
 * Pure function: [r, g, b], alpha → "rgba(r, g, b, alpha)"
 * @param {Array<number>} rgb - RGB array [r, g, b]
 * @param {number} alpha - Alpha value (0-1)
 * @returns {string} CSS color string
 */
export function rgbToStringAlpha(rgb, alpha) {
    return `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${alpha})`;
}

/**
 * Convert hex color to RGB array
 * Pure function: "#RRGGBB" → [r, g, b]
 * @param {string} hex - Hex color string
 * @returns {Array<number>} RGB array
 */
export function hexToRgb(hex) {
    const cleaned = hex.replace('#', '');
    const r = parseInt(cleaned.substring(0, 2), 16);
    const g = parseInt(cleaned.substring(2, 4), 16);
    const b = parseInt(cleaned.substring(4, 6), 16);
    return [r, g, b];
}

/**
 * Calculate commit freshness from timestamp
 * Pure function: (timestamp, oldest, newest) → freshness (0-1)
 * @param {number} timestamp - Unix timestamp of commit
 * @param {number} oldestTimestamp - Oldest commit timestamp
 * @param {number} newestTimestamp - Newest commit timestamp
 * @returns {number} Freshness factor (0 = oldest/dead, 1 = newest/alive)
 */
export function calculateCommitFreshness(timestamp, oldestTimestamp, newestTimestamp) {
    if (!timestamp) return 0;
    if (newestTimestamp === oldestTimestamp) return 1;
    return (timestamp - oldestTimestamp) / (newestTimestamp - oldestTimestamp);
}

/**
 * Get base color for agent status type
 * Pure function: (status, theme) → [r, g, b]
 * @param {string} status - Agent status (online/archived/offline/ambiguous)
 * @param {string} theme - Current theme (light/dark)
 * @returns {Array<number>} RGB array for base color
 */
export function getStatusBaseColor(status, theme) {
    const isDark = theme === 'dark';

    switch (status) {
        case 'online':
            return hexToRgb('#0B9A6D'); // green
        case 'archived':
            return isDark ? hexToRgb('#5b8fc9') : hexToRgb('#00274c'); // blue
        case 'offline':
            return isDark ? hexToRgb('#b0b0b0') : hexToRgb('#666666'); // grey
        case 'ambiguous':
            return hexToRgb('#FFCB05'); // yellow
        default:
            return hexToRgb('#666666'); // fallback grey
    }
}

/**
 * Get color for agent based on commit step position
 * Pure function: (agent, allAgents, theme) → { bg, text, dot }
 * Modulates the status color towards greyscale based on commit position
 * ALL agents participate - each unique commit gets evenly-spaced color
 * @param {Object} agent - Agent data with commit_timestamp and status
 * @param {Array<Object>} allAgents - All agents for comparison
 * @param {string} theme - Current theme (light/dark)
 * @returns {Object} Color palette { background, text, dot }
 */
export function getAgentFreshnessColor(agent, allAgents, theme) {
    // Get base color for this agent's status type
    const baseColor = getStatusBaseColor(agent.status, theme);
    const greyColor = rgbToGreyscale(...baseColor);

    // Get all unique commit timestamps (regardless of status)
    const allTimestamps = allAgents
        .filter(a => a.commit_timestamp)
        .map(a => a.commit_timestamp);

    if (allTimestamps.length === 0 || !agent.commit_timestamp) {
        // No timestamp data - use greyscale version
        return {
            background: rgbToString(greyColor),
            text: rgbToString(greyColor),
            dot: rgbToString(greyColor),
        };
    }

    // Get unique timestamps and sort them (oldest to newest)
    const uniqueTimestamps = [...new Set(allTimestamps)].sort((a, b) => a - b);

    // Find this agent's position in the sorted list
    const position = uniqueTimestamps.indexOf(agent.commit_timestamp);

    // Calculate freshness based on step position (0 to 1)
    // If only one commit, freshness = 1.0
    const freshness = uniqueTimestamps.length === 1
        ? 1.0
        : position / (uniqueTimestamps.length - 1);

    // Interpolate from grey (oldest) to full color (newest)
    const interpolated = lerpColor(greyColor, baseColor, freshness);

    return {
        background: rgbToString(interpolated),
        text: rgbToString(interpolated),
        dot: rgbToString(interpolated),
    };
}

// Constants
export const CONSTANTS = {
    MAX_HISTORY_LENGTH: 21,
    PREFIX: '> ',
    PLACEHOLDER_TEXT: 'Shoot',
    RUN_COLORS: [
        '#0B9A6D', '#FF6B6B', '#4ECDC4', '#FFD93D',
        '#A8E6CF', '#C77DFF', '#FF9A8B', '#6A89CC',
        '#F8B500', '#95E1D3'
    ],
    THEME_ICONS: {
        sun: `<path d="M8 11a3 3 0 1 1 0-6 3 3 0 0 1 0 6zm0 1a4 4 0 1 0 0-8 4 4 0 0 0 0 8zM8 0a.5.5 0 0 1 .5.5v2a.5.5 0 0 1-1 0v-2A.5.5 0 0 1 8 0zm0 13a.5.5 0 0 1 .5.5v2a.5.5 0 0 1-1 0v-2A.5.5 0 0 1 8 13zm8-5a.5.5 0 0 1-.5.5h-2a.5.5 0 0 1 0-1h2a.5.5 0 0 1 .5.5zM3 8a.5.5 0 0 1-.5.5h-2a.5.5 0 0 1 0-1h2A.5.5 0 0 1 3 8zm10.657-5.657a.5.5 0 0 1 0 .707l-1.414 1.415a.5.5 0 1 1-.707-.708l1.414-1.414a.5.5 0 0 1 .707 0zm-9.193 9.193a.5.5 0 0 1 0 .707L3.05 13.657a.5.5 0 0 1-.707-.707l1.414-1.414a.5.5 0 0 1 .707 0zm9.193 2.121a.5.5 0 0 1-.707 0l-1.414-1.414a.5.5 0 0 1 .707-.707l1.414 1.414a.5.5 0 0 1 0 .707zM4.464 4.465a.5.5 0 0 1-.707 0L2.343 3.05a.5.5 0 1 1 .707-.707l1.414 1.414a.5.5 0 0 1 0 .708z"/>`,
        moon: `<path d="M6 .278a.768.768 0 0 1 .08.858 7.208 7.208 0 0 0-.878 3.46c0 4.021 3.278 7.277 7.318 7.277.527 0 1.04-.055 1.533-.16a.787.787 0 0 1 .81.316.733.733 0 0 1-.031.893A8.349 8.349 0 0 1 8.344 16C3.734 16 0 12.286 0 7.71 0 4.266 2.114 1.312 5.124.06A.752.752 0 0 1 6 .278z"/>`
    },
    // Chart metrics configuration - data-driven chart rendering
    METRIC_CONFIGS: [
        { key: 'loss', canvasId: 'chart-train-loss', title: 'Training Loss', label: 'Training Loss', type: 'line' },
        { key: 'val_loss', canvasId: 'chart-val-loss', title: 'Validation Loss', label: 'Validation Loss', type: 'line' },
        { key: 'val_perplexity', canvasId: 'chart-perplexity', title: 'Perplexity', label: 'Perplexity', type: 'line' },
        { key: 'learning_rate', canvasId: 'chart-lr', title: 'Learning Rate', label: 'Learning Rate', type: 'line' },
        { key: 'num_tokens', canvasId: 'chart-tokens', title: 'Tokens (Billions)', label: 'Tokens (B)', type: 'bar' },
        { key: 'avg_step_time', canvasId: 'chart-avg-step-time', title: 'Average Step Time', label: 'Avg Step Time (s)', type: 'line' },
        { key: 'softmax_collapse', canvasId: 'chart-softmax', title: 'Softmax Collapse', label: 'Softmax Collapse', type: 'line' },
        { key: 'sampling_weights', canvasId: 'chart-sampling-weights', title: 'Task Sampling Weights', label: 'Sampling Weights', type: 'sampling', source: 'data_metrics' },
        // Expert convergence metrics (SMEAR & Prismatic routers)
        { key: 'expert_routing_weights', canvasId: 'chart-expert-routing', title: 'Expert Routing Weights (Convergence)', label: 'Routing Weight', type: 'multi_expert_line', isComposite: true },
        { key: 'expert_selection', canvasId: 'chart-expert-selection', title: 'Expert Selection (Actual k_experts Usage)', label: 'Selection Count', type: 'multi_expert_line', isComposite: true, keyPattern: /^expert_selection\/expert_\d+_count$/, stepped: true },
        { key: 'routing/entropy', canvasId: 'chart-routing-entropy', title: 'Routing Entropy (Balance)', label: 'Entropy', type: 'line' },
        { key: 'routing/concentration', canvasId: 'chart-routing-concentration', title: 'Routing Concentration (Collapse)', label: 'Max Weight', type: 'line' },
        { key: 'routing/variance', canvasId: 'chart-routing-variance', title: 'Routing Variance (Stability)', label: 'Variance', type: 'line' },
        { key: 'routing/balance', canvasId: 'chart-routing-balance', title: 'Routing Balance', label: 'Balance', type: 'line' },
        // Prismatic v8.1 - Switch Transformers loss metrics
        { key: 'expert_importance', canvasId: 'chart-expert-importance', title: 'Expert Importance (Soft Routing Probabilities)', label: 'Importance', type: 'multi_expert_line', isComposite: true, keyPattern: /^routing\/expert_\d+_importance$/ },
        { key: 'expert_load', canvasId: 'chart-expert-load', title: 'Expert Load (Hard Routing Decisions)', label: 'Load', type: 'multi_expert_line', isComposite: true, keyPattern: /^routing\/expert_\d+_load$/ },
        // Distance router - Parameter diversity loss
        { key: 'routing/diversity_loss', canvasId: 'chart-diversity-loss', title: 'Parameter Diversity Loss (Distance Router)', label: 'Diversity Loss', type: 'line' },
        // Architecture selection metrics (Prismatic v3.0)
        { key: 'arch_selection', canvasId: 'chart-arch-selection', title: 'Architecture Selection (Cumulative Counts)', label: 'Tool Call Count', type: 'multi_expert_line', isComposite: true, keyPattern: /^arch\/expert_\d+_count$/, stepped: true }
    ]
};
