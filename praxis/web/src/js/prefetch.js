/**
 * Praxis Web - shared loader dedupe
 *
 * Tab loaders are reachable from several places at once (init prewarm, tab
 * activation, run selectors, the background refresh loop). Without dedupe a
 * click during a prewarm runs the same loader twice: two fetches, two DOM
 * rewrites - the visible "renders, then re-renders" flash. Routing every
 * loader through dedupe() collapses concurrent calls into one shared promise,
 * so whoever asks second just awaits the first run.
 */

const inflight = new Map();

/**
 * True when a container holds rendered content (not just a placeholder), so
 * refreshes keep it painted - stale-while-revalidate - instead of flashing a
 * loading message.
 */
export function hasRealContent(container) {
    const first = container && container.firstElementChild;
    return !!first && !first.classList.contains('loading-placeholder');
}

/**
 * Run `fn` unless a run under the same key is already in flight, in which
 * case return that run's promise instead of starting another.
 */
export function dedupe(key, fn) {
    if (inflight.has(key)) return inflight.get(key);
    const p = Promise.resolve()
        .then(fn)
        .finally(() => inflight.delete(key));
    inflight.set(key, p);
    return p;
}
