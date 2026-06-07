/**
 * Praxis Web - KB sliding-window prefetch
 *
 * Treats the KB results list as a carousel and prefetches item bodies in a
 * window that slides with the viewport: items just ahead of the focused row are
 * pre-cached, items just behind stay post-cached, and everything outside is
 * evicted by an LRU cap. So scrolling/opening across hundreds of results is
 * instant without ever holding them all in memory. Fetches run through a small
 * concurrency-limited queue (the "rolling search queue") that is rebuilt - and
 * its now-stale entries dropped - each time the window slides.
 */

import { state } from './state.js';
import { kbFetchItem } from './api.js';
import { renderMarkdown, renderJson } from './markdown.js';

const AHEAD = 30;       // precache this many rows past the focus (read direction)
const BEHIND = 10;      // postcache this many already-passed rows
const RETAIN = 96;      // LRU cap; > window so a long scroll-back stays warm
const CONCURRENCY = 6;  // max simultaneous prefetches (browser caps ~6/host)

// Only these types have a fetchable body (link/card/agent navigate instead).
const FETCHABLE = new Set(['doc', 'note', 'run', 'page']);

const cache = new Map();     // id -> hydrated item; Map order = LRU (oldest first)
const inflight = new Map();  // id -> Promise, dedupes concurrent fetches
let queue = [];              // ordered ids awaiting prefetch
let active = 0;              // in-flight prefetch count

/** Attach the rendered body HTML so an opened item paints with no extra work. */
function hydrate(item) {
    item.html = item.type === 'run' ? renderJson(item.body) : renderMarkdown(item.body);
    return item;
}

function store(id, item) {
    cache.set(id, item);
    while (cache.size > RETAIN) {
        cache.delete(cache.keys().next().value);  // evict LRU
    }
}

/** Cached item (bumping its LRU recency), or null. */
export function kbCacheGet(id) {
    const item = cache.get(id);
    if (!item) return null;
    cache.delete(id);
    cache.set(id, item);
    return item;
}

/** Cached item if present, else fetch + hydrate + cache it (deduped). */
export async function kbCacheFetch(id) {
    const hit = kbCacheGet(id);
    if (hit) return hit;
    if (inflight.has(id)) return inflight.get(id);
    const p = (async () => {
        const item = await kbFetchItem(id);
        if (item) store(id, hydrate(item));
        return item;
    })().finally(() => inflight.delete(id));
    inflight.set(id, p);
    return p;
}

function pump() {
    while (active < CONCURRENCY && queue.length) {
        const id = queue.shift();
        if (cache.has(id) || inflight.has(id)) continue;
        active++;
        kbCacheFetch(id).catch(() => {}).finally(() => { active--; pump(); });
    }
}

/**
 * Slide the prefetch window to center on `focusIndex` in state.kbResults. Rebuilds
 * the queue (forward/precache first, then behind/postcache), which drops any
 * still-queued ids that have rolled out of the window.
 */
export function kbSlideWindow(focusIndex) {
    const results = state.kbResults || [];
    if (!results.length) { queue = []; return; }
    const i = Math.max(0, Math.min(focusIndex | 0, results.length - 1));
    const start = Math.max(0, i - BEHIND);
    const end = Math.min(results.length, i + AHEAD + 1);

    const want = [];
    for (let k = i; k < end; k++) want.push(results[k]);        // precache ahead
    for (let k = i - 1; k >= start; k--) want.push(results[k]); // postcache behind

    queue = want
        .filter(r => r && FETCHABLE.has(r.type) && !cache.has(r.id) && !inflight.has(r.id))
        .map(r => r.id);
    pump();
}

/** Last row whose top is at or above the scroll edge = the focused row. Binary
 *  search over offsetTop (normalized to the first row), so it stays O(log n)
 *  even with hundreds of rows. */
function focusIndexFromScroll(container) {
    const rows = container.children;
    if (!rows.length) return 0;
    const base = rows[0].offsetTop;
    const st = container.scrollTop;
    let lo = 0, hi = rows.length - 1, ans = 0;
    while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        if (rows[mid].offsetTop - base <= st) { ans = mid; lo = mid + 1; }
        else hi = mid - 1;
    }
    return ans;
}

/**
 * Bind the sliding window to the results list's scroll (rAF-throttled). Bound
 * once - #kb-results persists across re-renders. Inert while an item is open
 * (then the list isn't shown) and outside Read mode.
 */
export function setupKbPrefetch() {
    const container = document.getElementById('kb-results');
    if (!container || container._kbPrefetchBound) return;
    container._kbPrefetchBound = true;
    let pending = false;
    container.addEventListener('scroll', () => {
        if (pending || state.kbOpenItem) return;
        pending = true;
        requestAnimationFrame(() => {
            pending = false;
            if (state.kbOpenItem) return;
            kbSlideWindow(focusIndexFromScroll(container));
        });
    }, { passive: true });
}
