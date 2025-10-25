/**
 * Praxis Web - Diagnostic Tool
 * Run this in browser console to check what's working
 */

console.log('=== Praxis Web Diagnostic ===');

// Check if modules loaded
try {
    console.log('✓ diagnostic.js loaded');

    // Check DOM elements
    console.log('\n--- DOM Elements ---');
    console.log('app-container:', document.querySelector('.app-container') ? '✓' : '✗');
    console.log('tab-buttons:', document.querySelector('.tab-buttons') ? '✓' : '✗');
    console.log('chat-container:', document.getElementById('chat-container') ? '✓' : '✗');
    console.log('message-input:', document.getElementById('message-input') ? '✓' : '✗');

    // Check if main.js loaded
    console.log('\n--- Module Check ---');
    console.log('Type "state" in console to check if state object exists');
    console.log('Type "render" in console to check if render function exists');

    // Try to access state (this will fail if module didn't load)
    if (typeof state !== 'undefined') {
        console.log('✓ State object accessible');
    } else {
        console.log('✗ State object NOT accessible (modules may not be loading)');
    }

} catch (error) {
    console.error('Diagnostic error:', error);
}
