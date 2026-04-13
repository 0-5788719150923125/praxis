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
export async function sendMessage(messages) {
    const payload = {
        messages: messages.map(m => ({
            role: m.role,
            content: m.content
        })),
        max_new_tokens: state.settings.maxTokens,
        temperature: state.settings.temperature,
        repetition_penalty: state.settings.repetitionPenalty,
        do_sample: state.settings.doSample,
        use_cache: state.settings.useCache
    };

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
