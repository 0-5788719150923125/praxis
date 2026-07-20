/**
 * Praxis Web - Offline-accent registry
 *
 * A frozen/dead export (no live server behind it - e.g. the Cloudflare Pages
 * snapshot) flags itself visually by pushing an accent name onto
 * `window.PRAXIS_THEME_REGISTRY.offlineAccent` before this module runs (it's
 * a plain classic <script>, injected ahead of the deferred `type="module"`
 * scripts, so the write always lands first). This module is the only thing
 * that reads the registry - nothing else in the frontend hardcodes "offline"
 * or "red"; a normal live run leaves the registry empty and nothing changes.
 */

export function getOfflineAccent() {
    if (typeof window === 'undefined') return null;
    const registry = window.PRAXIS_THEME_REGISTRY;
    return (registry && registry.offlineAccent) || null;
}

export function applyOfflineAccent() {
    const accent = getOfflineAccent();
    if (accent && typeof document !== 'undefined') {
        document.documentElement.setAttribute('data-accent', accent);
    }
}
