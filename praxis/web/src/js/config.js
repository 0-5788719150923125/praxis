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
    generationKwargs: 'praxis_generation_kwargs',
    debugLogging: 'praxis_debug_logging',
    chatHistory: 'chatHistory',
    researchXAxis: 'praxis_research_x_axis',
    // What the RUN said a value should be, recorded beside the live value so
    // `resolveDefault` can tell an edit from an untouched default: a stored
    // value that still equals its seed yields to a new one, an edited value
    // does not. Registered here like any other key - `storage` silently
    // no-ops on a key it does not know, which is exactly how the Settings
    // form stopped persisting at all.
    'developerPrompt:default': 'praxis_developer_prompt_default',
    'generationKwargs:default': 'praxis_generation_kwargs_default'
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
            id: 'generation-kwargs',
            stateKey: 'settings.generationKwargs',
            type: 'value',
            parse: String
        },
        {
            id: 'debug-logging',
            stateKey: 'settings.debugLogging',
            type: 'checked'
        }
    ]
};

/**
 * The non-empty, non-comment lines of the generation-kwargs form, as the
 * server's own `key=value` list. Not parsed here: `parse_generation_kwargs`
 * on the server reads the values as YAML and rejects unknown keys, and having
 * one parser rather than two is what keeps the form honest about what a run
 * would actually decode with.
 * @param {string} text - Raw textarea contents
 * @returns {string[]} One `key=value` entry per meaningful line
 */
export const generationKwargLines = (text) =>
    String(text || '')
        .split('\n')
        .map(line => line.trim())
        .filter(line => line && !line.startsWith('#'));

/**
 * Render a `{key: value}` mapping as the `key=value` lines the form shows.
 * @param {Object} mapping - Parameters as sent by the server
 * @returns {string} Textarea contents
 */
export const generationKwargText = (mapping) =>
    Object.entries(mapping || {})
        // JSON for anything that is not a scalar. Template interpolation turns
        // [64, 1.03] into "64,1.03" - brackets gone - and the server reads that
        // back as a STRING, which then blows up inside the logits processor and
        // surfaces as an empty reply. A scalar is written bare so the common
        // line stays `temperature=0.7` rather than a quoted value.
        .map(([key, value]) => {
            const rendered = (value !== null && typeof value === 'object')
                ? JSON.stringify(value)
                : String(value);
            return `${key}=${rendered}`;
        })
        .join('\n');

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
 * Read form values and deep-merge them into target. readFormValues builds a
 * fresh nested object holding only the form's keys, so a shallow
 * Object.assign would replace whole sub-objects (state.settings) and drop
 * their non-form keys (systemPrompt, useCache).
 */
export const applyFormValues = (fieldConfigs, target) => {
    const updates = readFormValues(fieldConfigs);
    Object.entries(updates).forEach(([key, value]) => {
        if (
            value && typeof value === 'object' && !Array.isArray(value)
            && target[key] && typeof target[key] === 'object'
        ) {
            Object.assign(target[key], value);
        } else {
            target[key] = value;
        }
    });
};

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

    // The experiment loader records the matched experiments/*.yml in
    // args.config_file - the only authoritative source. Guessing from the
    // first --flag mislabeled runs whose first flag was not an experiment
    // (e.g. `./launch compose --publish-snapshot --abstractinator-f` became
    // "experiments/publish-snapshot.yml"). No config_file means no valid
    // experiment config: report none rather than inventing one.
    const configFilename = (data.args && data.args.config_file) || null;
    const expName = configFilename
        ? configFilename.split('/').pop().replace(/\.ya?ml$/, '')
        : null;
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
            title: 'Blueprint',
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
