/**
 * Praxis Web - API Service
 * Handle all API communication
 */

import { state } from './state.js';

/**
 * Send a chat message to the API
 * @param {Array} messages - Conversation history
 * @returns {Promise<Object>} API response
 */
export async function sendMessage(messages, opts = {}) {
    const payload = {
        messages: messages.map(m => ({
            role: m.role,
            content: m.content
        })),
        max_new_tokens: opts.maxNewTokens ?? state.settings.maxTokens,
        temperature: state.settings.temperature,
        repetition_penalty: state.settings.repetitionPenalty,
        do_sample: state.settings.doSample,
        use_cache: state.settings.useCache
    };
    // Loop/short turns can cap generation time so "thinking" doesn't drag.
    if (opts.timeout) payload.timeout = opts.timeout;

    if (state.settings.debugLogging) {
        console.log('[API] Sending:', payload);
    }

    const response = await fetch(`${state.settings.apiUrl}/messages/`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
    });

    if (!response.ok) {
        throw new Error(`API error: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();

    if (state.settings.debugLogging) {
        console.log('[API] Response:', data);
    }

    return data;
}

/**
 * Search the knowledge base (Read mode). Ranked, search-as-you-type.
 * @param {string} query - Raw query text
 * @param {string[]} [types] - Optional type filter (doc/run/note/link)
 * @returns {Promise<Array>} Ranked hits
 */
export async function kbSearch(query, types) {
    const params = new URLSearchParams({ q: query });
    if (types && types.length) params.set('types', types.join(','));

    const response = await fetch(`${state.settings.apiUrl}/api/kb/search?${params}`);
    if (!response.ok) {
        throw new Error(`KB search error: ${response.status} ${response.statusText}`);
    }
    const data = await response.json();
    return data.hits || [];
}

/**
 * Fetch one KB item's full body for inline rendering.
 * @param {string} id - Item id
 * @returns {Promise<Object|null>} The item, or null if not found
 */
export async function kbFetchItem(id) {
    const response = await fetch(`${state.settings.apiUrl}/api/kb/item?id=${encodeURIComponent(id)}`);
    if (!response.ok) {
        throw new Error(`KB item error: ${response.status} ${response.statusText}`);
    }
    const data = await response.json();
    return data.item || null;
}

/**
 * Print: ask the model to lead with a question (the environment-level hook).
 * Returns {available, id, question}. available=false until the model produces one.
 * @returns {Promise<Object>}
 */
export async function printAsk(reroll = false) {
    const response = await fetch(`${state.settings.apiUrl}/api/print/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            ...(reroll ? { reroll: true } : {}),
            max_new_tokens: state.settings.maxTokens,
        })
    });
    if (!response.ok) throw new Error(`Print ask error: ${response.status}`);
    return response.json();
}

/**
 * Print: submit the user's response to a model-led question. The backend scores
 * it against the model's stashed predicted answer and returns the reward.
 * @param {string} id - pending question id
 * @param {string} responseText - the user's typed answer
 * @returns {Promise<Object>} {recall, activation, energy, predicted_answer}
 */
export async function printRespond(id, responseText) {
    const response = await fetch(`${state.settings.apiUrl}/api/print/respond`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id, response: responseText })
    });
    if (!response.ok && response.status !== 409) {
        throw new Error(`Print respond error: ${response.status}`);
    }
    return response.json();
}

/**
 * Print: live engagement-energy snapshot for the badge.
 * @returns {Promise<Object>} {energy, count, buffered, last}
 */
export async function printEnergy() {
    const response = await fetch(`${state.settings.apiUrl}/api/print/energy`);
    if (!response.ok) throw new Error(`Print energy error: ${response.status}`);
    return response.json();
}

/**
 * Loop: run one looped task through the active loop mode. The backend builds
 * the prompt, generates, and parses off any self-predicted want->need score.
 * @param {string} task - task keyword or literal prompt
 * @returns {Promise<Object>} {available, id, text, predicted, mode}
 */
export async function loopGenerate(task) {
    const response = await fetch(`${state.settings.apiUrl}/api/loop/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task, max_new_tokens: state.settings.maxTokens })
    });
    if (!response.ok) throw new Error(`Loop generate error: ${response.status}`);
    return response.json();
}

/**
 * Loop: record the human's signed want->need score (-1..1) for a looped
 * section - the live joke reward fed to training. `id` ties the score to the
 * generated section so calibration mode can compare against the model's
 * self-prediction.
 * @param {number} score - -1..1
 * @param {string|null} id - section id from loopGenerate
 * @returns {Promise<Object>} {activation, reward, energy, correction?, ...}
 */
export async function loopApprove(score, id = null) {
    const response = await fetch(`${state.settings.apiUrl}/api/loop/approve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(id ? { score, id } : { score })
    });
    if (!response.ok) throw new Error(`Loop approve error: ${response.status}`);
    return response.json();
}

/**
 * API endpoint configuration - pure data
 */
const ENDPOINTS = {
    spec: { path: '/api/spec', error: 'Failed to load specification' },
    agents: { path: '/api/agents', error: 'Failed to load agents' },
    metrics: { path: '/api/metrics', error: 'Failed to load metrics' },
    ping: { path: '/api/ping', error: 'Failed to ping API' }
};

/**
 * Generic API fetch - fully data-driven
 * @param {string} endpointKey - Key from ENDPOINTS config
 * @returns {Promise<Object>} Parsed JSON response
 */
export async function fetchAPI(endpointKey) {
    const endpoint = ENDPOINTS[endpointKey];
    if (!endpoint) {
        throw new Error(`Unknown endpoint: ${endpointKey}`);
    }

    const response = await fetch(`${state.settings.apiUrl}${endpoint.path}`);
    if (!response.ok) {
        throw new Error(endpoint.error);
    }
    return response.json();
}

/**
 * Ping the API to check if it's alive
 * @returns {Promise<boolean>} True if API is reachable
 */
export async function ping() {
    try {
        await fetchAPI('ping');
        return true;
    } catch (e) {
        return false;
    }
}

/**
 * Test API connection with detailed status
 * @param {string} url - URL to test
 * @returns {Promise<Object>} Connection test result
 */
export async function testApiConnection(url) {
    try {
        const response = await fetch(`${url}/api/ping`, {
            method: 'GET',
            headers: { 'Content-Type': 'application/json' }
        });

        if (response.ok) {
            return { success: true, message: 'Connection successful!' };
        } else {
            return { success: false, message: `Server returned ${response.status}` };
        }
    } catch (error) {
        return { success: false, message: `Connection failed: ${error.message}` };
    }
}
