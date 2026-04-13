/**
 * Praxis Web - Unified Configuration
 * All application configuration and utilities in one place
 * Pure data and pure functions - no side effects
 */

// ============================================================================
// STORAGE CONFIGURATION
// ============================================================================

/**
 * Storage keys - single source of truth
 */
export const STORAGE_KEYS = {
    theme: 'praxis_theme',
    developerPrompt: 'praxis_developer_prompt',
    apiUrl: 'praxis_api_url',
    genParams: 'praxis_gen_params',
    debugLogging: 'praxis_debug_logging',
    chatHistory: 'chatHistory'
};

/**
 * Storage utilities
 */
export const storage = {
    get: (key) => {
        try {
            const storageKey = STORAGE_KEYS[key];
            if (!storageKey) {
                console.warn(`Unknown storage key: ${key}`);
                return null;
            }

            const value = localStorage.getItem(storageKey);
            if (value === null) return null;

            try {
                return JSON.parse(value);
            } catch {
                return value;
            }
        } catch (err) {
            console.error(`[Storage] Failed to get ${key}:`, err);
            return null;
        }
    },

    set: (key, value) => {
        try {
            const storageKey = STORAGE_KEYS[key];
            if (!storageKey) {
                console.warn(`Unknown storage key: ${key}`);
                return false;
            }

            const serialized = typeof value === 'object'
                ? JSON.stringify(value)
                : String(value);

            localStorage.setItem(storageKey, serialized);
            return true;
        } catch (err) {
            console.error(`[Storage] Failed to set ${key}:`, err);
            return false;
        }
    },

    remove: (key) => {
        try {
            const storageKey = STORAGE_KEYS[key];
            if (!storageKey) {
                console.warn(`Unknown storage key: ${key}`);
                return false;
            }

            localStorage.removeItem(storageKey);
            return true;
        } catch (err) {
            console.error(`[Storage] Failed to remove ${key}:`, err);
            return false;
        }
    },

    clear: () => {
        const keysToRemove = Object.values(STORAGE_KEYS);
        keysToRemove.forEach(key => {
            try {
                localStorage.removeItem(key);
            } catch (err) {
                console.error(`[Storage] Failed to remove ${key}:`, err);
            }
        });
    }
};

// ============================================================================
// FORM CONFIGURATION
// ============================================================================

/**
 * Form field configurations
 */
export const FORM_FIELDS = {
    settings: [
        {
            id: 'api-url',
            stateKey: 'settings.apiUrl',
            type: 'value',
            parse: String
        },
        {
            id: 'max-tokens',
            stateKey: 'settings.maxTokens',
            type: 'value',
            parse: parseInt,
            validate: (v) => v >= 50 && v <= 1000
        },
        {
            id: 'temperature',
            stateKey: 'settings.temperature',
            type: 'value',
            parse: parseFloat,
            displayId: 'temperature-value'
        },
        {
            id: 'repetition-penalty',
            stateKey: 'settings.repetitionPenalty',
            type: 'value',
            parse: parseFloat,
            displayId: 'repetition-penalty-value'
        },
        {
            id: 'do-sample',
            stateKey: 'settings.doSample',
            type: 'checked'
        },
        {
            id: 'debug-logging',
            stateKey: 'settings.debugLogging',
            type: 'checked'
        }
    ]
};

/**
 * Get nested property from object using dot notation
 */
const getNestedValue = (obj, path) => {
    const keys = path.split('.');
    return keys.reduce((acc, key) => acc?.[key], obj);
};

/**
 * Set nested property in object using dot notation (immutable)
 */
const setNestedValue = (obj, path, value) => {
    const keys = path.split('.');

    if (keys.length === 1) {
        return { ...obj, [keys[0]]: value };
    }

    const [first, ...rest] = keys;
    return {
        ...obj,
        [first]: setNestedValue(obj[first] || {}, rest.join('.'), value)
    };
};

/**
 * Pure function to read form values from DOM
 */
export const readFormValues = (fieldConfigs) =>
    fieldConfigs.reduce((acc, field) => {
        const element = document.getElementById(field.id);
        if (!element) return acc;

        const rawValue = field.type === 'checked' ? element.checked : element.value;
        const parsedValue = field.parse ? field.parse(rawValue) : rawValue;

        if (field.validate && !field.validate(parsedValue)) {
            console.warn(`[Form] Validation failed for ${field.id}, skipping update`);
            return acc;
        }

        return setNestedValue(acc, field.stateKey, parsedValue);
    }, {});

/**
 * Pure function to write form values to DOM
 */
export const writeFormValues = (fieldConfigs, state) => {
    fieldConfigs.forEach(field => {
        const element = document.getElementById(field.id);
        if (!element) return;

        const value = getNestedValue(state, field.stateKey);

        if (field.type === 'checked') {
            element.checked = Boolean(value);
        } else {
            element.value = value ?? '';
        }

        if (field.displayId) {
            const display = document.getElementById(field.displayId);
            if (display) display.textContent = value;
        }
    });
};

/**
 * Find field configuration by element ID
 */
export const findFieldById = (fieldConfigs, id) =>
    fieldConfigs.find(field => field.id === id) || null;

/**
 * Update range input display value
 */
export const updateRangeDisplay = (fieldConfigs, id, value) => {
    const field = findFieldById(fieldConfigs, id);
    if (!field?.displayId) return false;

    const display = document.getElementById(field.displayId);
    if (!display) return false;

    display.textContent = value;
    return true;
};

// ============================================================================
// SPEC TAB CONFIGURATION
// ============================================================================

/**
 * Extract command information - pure data transformation
 */
export const extractCommandInfo = (data) => {
    const command = data.command
        ? data.command.replace('python main.py', './launch')
        : './launch';

    const expMatch = command.match(/--([a-z0-9\-]+)/);
    const expName = expMatch ? expMatch[1] : 'reproduce';
    const configFilename = `experiments/${expName}.yml`;
    const reproduceCommand = command + (command.includes('--reset') ? '' : ' --reset');

    return {
        command,
        expName,
        configFilename,
        reproduceCommand
    };
};

/**
 * Spec section rendering configuration
 * Pure data describing what to render and when
 */
export const SPEC_CONFIG = {
    sections: [
        {
            id: 'peer-button',
            condition: (data) => data.git_url && data.truncated_hash,
            order: 0
        },
        {
            id: 'hashes',
            title: 'Hashes',
            condition: (data) => data.full_hash && data.truncated_hash,
            order: 1
        },
        {
            id: 'commands',
            title: 'Commands',
            condition: (data) => data.git_url,
            order: 2
        },
        {
            id: 'parameters',
            title: 'Parameters',
            condition: (data) => data.param_stats,
            order: 3
        },
        {
            id: 'architecture',
            title: 'Architecture',
            condition: (data) => data.model_architecture,
            order: 4
        },
        {
            id: 'arguments',
            title: 'Arguments',
            condition: () => true,
            order: 5
        }
    ]
};

// ============================================================================
// AGENT TAB CONFIGURATION
// ============================================================================

/**
 * Agent display fields configuration
 */
export const AGENT_DISPLAY_FIELDS = [
    {
        key: 'url',
        label: 'repo',
        getValue: (agent) => agent.masked_url || agent.url,
        condition: (agent) => agent.masked_url || agent.url
    },
    {
        key: 'short_hash',
        label: 'head',
        getValue: (agent) => agent.short_hash,
        condition: (agent) => agent.short_hash
    }
];

// ============================================================================
// MOBILE CONFIGURATION
// ============================================================================

export const MOBILE_CONFIG = {
    breakpoint: 768,
    scrollThreshold: 5
};
