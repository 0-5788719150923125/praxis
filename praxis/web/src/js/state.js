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

    // Gymnasium mode: 'read' = live KB search (default), 'evaluate' = AI chat.
    conversationMode: 'read',
    kbResults: [],
    kbSearching: false,
    kbOpenItem: null,  // {type, title, uri, body, meta} when a result is opened

    // Print: the model leads with a question (the environment-level hook for
    // online learning). The button is conditional - it only appears once the
    // model has actually been queried for and produced a question.
    print: {
        available: false,        // a model-led question is pending -> show button
        question: null,
        id: null,
        awaitingResponse: false, // question presented; next user message answers it
        lastReward: null,        // {recall, activation, energy, predicted_answer}
        energy: null             // live channel snapshot {energy, count} for the badge
    },

    // Loop: coupled to Print (must be in Print mode to enable). Repeats one task
    // (default "joke") on a timer, replacing the response each cycle - independent
    // challenges the model re-rolls until the user intervenes.
    loop: {
        enabled: false,
        generating: false
    },

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
            customClasses: [],
            template: () => createContainerWithContent(
                'terminal-container',
                'terminal-display',
                '<div class="terminal-line">Waiting for metrics stream...</div>'
            ),
            onActivate: 'renderCurrentMetrics',
            activateDelay: 0,
            activateParams: [],
            onDeactivate: null
        },
        {
            id: 'agents',
            label: 'Stage',
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
            activateParams: [false],  // Don't force refresh - invalidation events keep it fresh
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
                'Loading learning dynamics...'
            ),
            onActivate: 'loadDynamics',
            activateDelay: 0,
            activateParams: [false],  // Don't force refresh by default
            onDeactivate: null
        },
        {
            id: 'spec',
            label: 'Customs',
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

    // Live metrics data (streamed via WebSocket)
    liveMetrics: {
        connected: false,
        data: null
    },

    // Backend event feed (stage transitions, milestones) shown in the header
    // notification bell. Items arrive via the metrics_snapshot stream and are
    // deduped by their monotonic id.
    notifications: {
        items: [],
        unread: 0,
        panelOpen: false
    },

    // Settings/Generation params
    settings: {
        systemPrompt: 'Write thy wrong.',
        apiUrl: '',  // Will be set on init
        maxTokens: 256,
        temperature: 0.5,
        repetitionPenalty: 1.2,
        doSample: true,
        useCache: true,
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
        error: null,
        // Run picker (single-select; current run by default)
        availableRuns: [],
        selectedRun: null,
        runSelectorOpen: false
    },

    // Agents data (loaded async)
    agents: {
        loaded: false,
        availableAgents: [],
        selectedAgents: [],
        selectorOpen: false,
        error: null
    },

    // Swarm contracts: the Stage tab shows a slide-out CONTRACTS panel (offering
    // compute to the swarm) above the always-present fleet list. Agreeing to a
    // contract spawns an in-page SwarmAgent (see swarm.js). The list is static
    // for now - eventually contracts arrive from the mesh.
    contracts: {
        open: false,      // is the slide-out contracts panel open?
        available: [
            {
                id: 'donate-browser-agent',
                title: 'Donate a browser agent',
                guarantee: 'max security',
                description:
                    'Run a tiny transformer agent in this browser tab, sandboxed and ' +
                    'gated behind this app. It performs online updates as one expert in ' +
                    'a distributed swarm. No native code, no central server - peers sync ' +
                    'directly. Opt-in, idle until work arrives, and revocable at any time.',
            },
        ],
        agents: [],       // live SwarmAgent instances (not view objects)
    },

    // Research/metrics data (loaded async)
    research: {
        loaded: false,
        runs: [],
        selectedRuns: [],
        charts: {},
        etag: null,
        lastStep: 0,
        error: null,
        // Cached data from the most recent successful fetch. Used by the
        // auto-refresh path to keep charts up when the server returns 304.
        lastRuns: [],
        lastDataMetrics: [],
        // Scalar-metric registry (key -> {description, chart}) served by
        // /api/metrics. Source of truth for chart titles/labels/scale;
        // adding a metric to praxis.metrics.training_metrics makes it
        // appear here automatically.
        metricRegistry: {},
        // Composite/specialty chart registry (multi-expert, sampling,
        // heatmap) served by /api/metrics alongside metricRegistry. Same
        // deal: declare a chart in praxis.metrics.training_metrics and it
        // renders here, no JS edit.
        compositeRegistry: [],
        // Historical run comparison
        historicalRuns: [],       // All available runs from /api/runs
        selectedHistoricalRuns: [], // Hashes the user has checked
        runSelectorOpen: false    // Dropdown open/closed
    },

    // Dynamics/gradient data (loaded async)
    dynamics: {
        loaded: false,
        data: null,
        lastStep: 0,  // Track last loaded step for incremental updates
        error: null,
        // Run picker (single-select; current run by default)
        availableRuns: [],     // runs with has_dynamics=true
        selectedRun: null,     // hash; null => current run
        runSelectorOpen: false
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
 * Convert HSL (h in degrees, s/l in 0-1) to an [r, g, b] array.
 */
export function hslToRgb(h, s, l) {
    h = (((h % 360) + 360) % 360) / 360;
    if (s === 0) { const v = Math.round(l * 255); return [v, v, v]; }
    const hue2rgb = (p, q, t) => {
        if (t < 0) t += 1;
        if (t > 1) t -= 1;
        if (t < 1 / 6) return p + (q - p) * 6 * t;
        if (t < 1 / 2) return q;
        if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
        return p;
    };
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    return [
        Math.round(hue2rgb(p, q, h + 1 / 3) * 255),
        Math.round(hue2rgb(p, q, h) * 255),
        Math.round(hue2rgb(p, q, h - 1 / 3) * 255),
    ];
}

/**
 * The brand accent as [r, g, b], following the central --accent-hue (green, or blue
 * in logs mode). Reads the same sat/lum the CSS --accent uses, so inline-styled
 * badges never drift paler than CSS-styled ones. Falls back to the green default off-DOM.
 */
export function accentRgb() {
    const [, sat, lum] = accentHsl();
    return hslToRgb(currentAccentHue(), sat, lum);
}

/** The central accent as [hue°, sat 0-1, lum 0-1], read off <html> to match CSS --accent. */
function accentHsl() {
    const hue = currentAccentHue();
    const css = (name, fallback) => {
        if (typeof document === 'undefined') return fallback;
        const v = parseFloat(getComputedStyle(document.documentElement).getPropertyValue(name));
        return Number.isNaN(v) ? fallback : v / 100;
    };
    return [hue, css('--accent-sat', 0.87), css('--accent-lum', 0.32)];
}

/** Read the central accent hue (degrees) off <html>; 161 (green) by default. */
export function currentAccentHue() {
    if (typeof document !== 'undefined') {
        const v = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--accent-hue'));
        if (!Number.isNaN(v)) return v;
    }
    return CHART_PALETTE_REF_HUE;
}

// One shared chart-line palette for BOTH the Research and Dynamics tabs: an ordered,
// distinct set of colors spread around the wheel and anchored on the brand green
// (index 0 = the accent). The whole scheme rotates with --accent-hue, so it follows
// the green->blue logs toggle. Index cycles continuously (modulo) for any series count.
const CHART_PALETTE_REF_HUE = 161;  // hue the palette is authored at (matches green accent)
const CHART_PALETTE_BASE = [
    [161, 68, 42],  // green (accent anchor)
    [205, 70, 54],  // blue
    [38, 85, 55],   // amber
    [330, 68, 62],  // magenta
    [188, 62, 44],  // teal
    [272, 58, 66],  // purple
    [20, 82, 58],   // orange
    [104, 52, 47],  // leaf green
    [242, 56, 65],  // indigo
    [312, 54, 60],  // violet
];

/** The i-th chart-line color as a hex string (wraps; rotates with the accent hue). */
export function chartLineColor(index) {
    const n = CHART_PALETTE_BASE.length;
    const [h, s, l] = CHART_PALETTE_BASE[((index % n) + n) % n];
    const rot = currentAccentHue() - CHART_PALETTE_REF_HUE;
    const [r, g, b] = hslToRgb(h + rot, s / 100, l / 100);
    return '#' + [r, g, b].map(x => x.toString(16).padStart(2, '0')).join('');
}

/** RGB (0-255) to HSL ([h degrees, s 0-1, l 0-1]). */
export function rgbToHsl(r, g, b) {
    r /= 255; g /= 255; b /= 255;
    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    const l = (max + min) / 2, d = max - min;
    let h = 0, s = 0;
    if (d !== 0) {
        s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
        if (max === r) h = (g - b) / d + (g < b ? 6 : 0);
        else if (max === g) h = (b - r) / d + 2;
        else h = (r - g) / d + 4;
        h *= 60;
    }
    return [h, s, l];
}

/** Rotate the hue of a "#rrggbb" or "#rrggbbaa" color by `deg`, preserving alpha.
 *  Non-hex values (arrays, "transparent", gradients) pass through unchanged. */
export function rotateHexHue(color, deg) {
    if (!deg || typeof color !== 'string' || color[0] !== '#') return color;
    const hex = color.slice(1);
    if (hex.length !== 6 && hex.length !== 8) return color;
    const r = parseInt(hex.slice(0, 2), 16);
    const g = parseInt(hex.slice(2, 4), 16);
    const b = parseInt(hex.slice(4, 6), 16);
    const a = hex.length === 8 ? hex.slice(6, 8) : '';
    const [h, s, l] = rgbToHsl(r, g, b);
    const [nr, ng, nb] = hslToRgb(h + deg, s, l);
    return '#' + [nr, ng, nb].map(x => x.toString(16).padStart(2, '0')).join('') + a;
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
            return accentRgb(); // brand accent (green, or blue in logs mode)
        case 'archived':
            return isDark ? hexToRgb('#5b8fc9') : hexToRgb('#00274c'); // blue
        case 'observe':
            return accentRgb(); // green (active, matches online)
        case 'offline':
        case 'idle':
            return isDark ? hexToRgb('#b0b0b0') : hexToRgb('#666666'); // grey
        case 'ambiguous':
            return hexToRgb('#FFCB05'); // yellow
        default:
            return hexToRgb('#666666'); // fallback grey
    }
}

/**
 * The single source of truth for an agent badge's colors. Returns the status
 * color, shaded toward grey by commit freshness (oldest commit = greyest,
 * newest = full color). An agent with no commit of its own shows the full status
 * color - never a greyscale wash - so every same-status badge reads consistently.
 * Pure function: (agent, allAgents, theme) → { background, text, dot }
 */
export function getAgentFreshnessColor(agent, allAgents, theme, status = agent.status) {
    // Get base color for this agent's status type
    const baseColor = getStatusBaseColor(status, theme);
    const greyColor = rgbToGreyscale(...baseColor);

    // Get all unique commit timestamps (regardless of status)
    const allTimestamps = allAgents
        .filter(a => a.commit_timestamp)
        .map(a => a.commit_timestamp);

    if (allTimestamps.length === 0 || !agent.commit_timestamp) {
        // No commit age to place it on the freshness ramp - show the full status
        // color (the active, current end of the ramp), not a greyscale wash.
        return {
            background: rgbToString(baseColor),
            text: rgbToString(baseColor),
            dot: rgbToString(baseColor),
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
export const DEFAULT_SYSTEM_PROMPT = 'Write thy wrong.';

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
    }
};

/**
 * Central training-status label for the UI.
 * Prefers the semantic stage (preflight/pretrain/...) and falls back to the
 * Lightning mode (train/validation). Single source of truth so the terminal
 * dashboard and any future status display agree on what to show.
 * @param {Object} data - metrics snapshot
 * @returns {string} uppercase status label
 */
export function statusLabel(data) {
    return (((data && (data.stage || data.mode)) || 'train')).toUpperCase();
}
