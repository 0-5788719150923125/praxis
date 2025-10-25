/**
 * Praxis Web - State Management
 * Single source of truth - everything is data
 */

// Application state - all data lives here
export const state = {
    // Theme
    theme: 'light',

    // Conversation
    messages: [],
    isThinking: false,

    // Input state
    isShowingPlaceholder: true,

    // Tabs
    currentTab: 'chat',  // Track current active tab
    tabs: [
        { id: 'chat', label: 'Gymnasium', active: true },
        { id: 'terminal', label: 'Terminal', active: false },
        { id: 'agents', label: 'Agents', active: false },
        { id: 'research', label: 'Research', active: false },
        { id: 'spec', label: 'Identity', active: false }
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
    }
};

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
    }
};
