/*
 * static_mode.js - offline-snapshot shim for the Praxis dashboard.
 *
 * Injected (only) into the Cloudflare Pages export, right after the socket.io
 * CDN tag and before main.js. It makes the built frontend run with no backend:
 *
 *   1. Stubs window.io and replays a frozen metrics_snapshot on connect, so the
 *      Terminal tab and the notification bell populate (the bell carries the
 *      "you're viewing an offline snapshot" warning as its newest event).
 *   2. Patches window.fetch to redirect same-origin /api/* reads onto the dumped
 *      static files under /data/, filters KB search client-side, and returns an
 *      offline stub for every write/interactive endpoint.
 *   3. Greys out the controls that can't work offline (chat, swarm, downloads).
 *   4. Pushes the "red" accent into the frontend's offline-theme registry
 *      (js/theme.js) - a dead page should look dead, not the live green.
 *
 * None of this touches the real frontend source - it lives entirely here.
 */
(function () {
  'use strict';

  var DATA = '/data/';

  // --- 0. flag this export as offline in the theme registry ---------------
  // Runs synchronously, before main.js (a deferred module) ever executes, so
  // theme.js's applyOfflineAccent() sees it on its very first read - no flash
  // of the live green accent before the switch to red.
  window.PRAXIS_THEME_REGISTRY = window.PRAXIS_THEME_REGISTRY || {};
  window.PRAXIS_THEME_REGISTRY.offlineAccent = 'red';

  // --- 1. websocket stub with frozen-snapshot replay ----------------------

  var frozenPromise = fetch(DATA + 'metrics_live.json')
    .then(function (r) { return r.ok ? r.json() : {}; })
    .catch(function () { return {}; });

  function makeSocket() {
    var handlers = {};
    var sock = {
      connected: false,
      on: function (ev, cb) {
        (handlers[ev] = handlers[ev] || []).push(cb);
        if (!sock._scheduled) {
          sock._scheduled = true;
          // Defer past the synchronous run of connectMetricsLive(), so every
          // handler (connect, metrics_snapshot, ...) is registered first.
          setTimeout(function () {
            sock.connected = true;
            (handlers['connect'] || []).forEach(function (fn) {
              try { fn(); } catch (e) {}
            });
            frozenPromise.then(function (data) {
              (handlers['metrics_snapshot'] || []).forEach(function (fn) {
                try { fn(data); } catch (e) {}
              });
            });
          }, 0);
        }
        return sock;
      },
      emit: function () { return sock; },
      off: function () { return sock; },
      disconnect: function () { sock.connected = false; return sock; },
    };
    return sock;
  }
  var io = function () { return makeSocket(); };
  io.connect = function () { return makeSocket(); };
  window.io = io;

  // --- 2. fetch patch ------------------------------------------------------

  var origFetch = window.fetch.bind(window);

  // Direct path -> dumped file.
  var MAP = {
    '/api/ping': DATA + 'ping.json',
    '/api/runs': DATA + 'runs.json',
    '/api/spec': DATA + 'spec.json',
    '/api/config': DATA + 'config.yaml',
    '/api/metrics': DATA + 'metrics.json',
    '/api/data-metrics': DATA + 'data-metrics.json',
    '/api/dynamics': DATA + 'dynamics.json',
    '/api/head_snapshots': DATA + 'head_snapshots.json',
    '/api/activation_curves': DATA + 'activation_curves.json',
    '/api/evolution': DATA + 'evolution.json',
    '/api/spider': DATA + 'spider.json',
    '/api/agents': DATA + 'agents.json',
    '/api/print/energy': DATA + 'print_energy.json',
    '/api/print/pending': DATA + 'print_pending.json',
    '/api/loop/energy': DATA + 'loop_energy.json',
    '/api/paper.pdf': '/paper.pdf',
  };

  var kbFeed = null;
  var kbIndex = null;
  function loadKbFeed() {
    if (!kbFeed) {
      kbFeed = origFetch(DATA + 'kb_feed.json')
        .then(function (r) { return r.json(); })
        .catch(function () { return { status: 'ok', hits: [] }; });
    }
    return kbFeed;
  }
  function loadKbIndex() {
    if (!kbIndex) {
      kbIndex = origFetch(DATA + 'kb_item_index.json')
        .then(function (r) { return r.json(); })
        .catch(function () { return {}; });
    }
    return kbIndex;
  }

  function jsonResponse(obj, status) {
    return new Response(JSON.stringify(obj), {
      status: status || 200,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  function methodOf(input, init) {
    if (init && init.method) return init.method.toUpperCase();
    if (input && typeof input !== 'string' && input.method) {
      return input.method.toUpperCase();
    }
    return 'GET';
  }

  window.fetch = function (input, init) {
    var raw = typeof input === 'string' ? input : (input && input.url) || '';
    var url;
    try {
      url = new URL(raw, location.origin);
    } catch (e) {
      return origFetch(input, init);
    }

    // Cross-origin (e.g. a remote peer agent) - let it hit the network and
    // fail on its own; the frontend already tolerates remote failures.
    if (url.origin !== location.origin) return origFetch(input, init);

    var p = url.pathname;
    var method = methodOf(input, init);

    if (method !== 'GET') {
      if (p.indexOf('/api/') === 0 || p === '/messages/' || p === '/messages' || p === '/input') {
        return Promise.resolve(
          jsonResponse({ status: 'offline', message: 'Offline snapshot: live features are disabled.' }, 503)
        );
      }
      return origFetch(input, init);
    }

    // KB search: filter the dumped doc+code feed client-side.
    if (p === '/api/kb/search') {
      var q = (url.searchParams.get('q') || '').trim().toLowerCase();
      return loadKbFeed().then(function (feed) {
        if (!q) return jsonResponse(feed);
        var hits = (feed.hits || []).filter(function (h) {
          var hay = ((h.title || '') + ' ' + (h.label || '') + ' ' +
            (h.summary || '') + ' ' + (h.snippet || '') + ' ' +
            (h.uri || '')).toLowerCase();
          return hay.indexOf(q) !== -1;
        });
        return jsonResponse({ status: 'ok', query: q, hits: hits });
      });
    }

    // KB item: id -> dumped file via the index map.
    if (p === '/api/kb/item') {
      var id = url.searchParams.get('id') || '';
      return loadKbIndex().then(function (idx) {
        var file = idx[id];
        if (!file) return jsonResponse({ status: 'error', message: 'not found' }, 404);
        return origFetch(DATA + file);
      });
    }

    // Business card: fixed seed, one file per side/theme.
    if (p === '/api/card/preview.svg') {
      var side = url.searchParams.get('side') || 'front';
      var theme = url.searchParams.get('theme') || 'light';
      return origFetch(DATA + 'card_' + side + '_' + theme + '.svg');
    }

    // Swarm batch poll: no live training -> null batch keeps the arena in
    // heartbeat mode instead of erroring.
    if (p === '/api/swarm/batch') {
      return Promise.resolve(jsonResponse({ status: 'ok', batch: null }));
    }

    if (MAP[p]) return origFetch(MAP[p]);

    // Any other /api/* read we didn't dump -> offline stub.
    if (p.indexOf('/api/') === 0) {
      return Promise.resolve(jsonResponse({ status: 'offline' }, 503));
    }

    return origFetch(input, init);
  };

  // --- 3. rewrite business-card <img> requests ----------------------------
  // The card preview is an <img src>, not a fetch, so the patch above misses
  // it. Static hosts ignore the query string, so we can't vary by seed anyway;
  // redirect each card image to the pre-rendered file for its side/theme.

  function rewriteCardImg(img) {
    var s = img.getAttribute('src') || '';
    if (s.indexOf('/api/card/preview.svg') === -1) return;
    try {
      var u = new URL(s, location.origin);
      var side = u.searchParams.get('side') || 'front';
      var theme = u.searchParams.get('theme') || 'light';
      var target = DATA + 'card_' + side + '_' + theme + '.svg';
      if (img.getAttribute('src') !== target) img.setAttribute('src', target);
    } catch (e) {}
  }

  function startCardObserver() {
    document.querySelectorAll('img').forEach(rewriteCardImg);
    var mo = new MutationObserver(function (muts) {
      muts.forEach(function (m) {
        if (m.type === 'attributes' && m.target.tagName === 'IMG') {
          rewriteCardImg(m.target);
        } else if (m.type === 'childList') {
          m.addedNodes.forEach(function (n) {
            if (n.tagName === 'IMG') rewriteCardImg(n);
            else if (n.querySelectorAll) n.querySelectorAll('img').forEach(rewriteCardImg);
          });
        }
      });
    });
    mo.observe(document.documentElement, {
      subtree: true,
      childList: true,
      attributes: true,
      attributeFilter: ['src'],
    });
  }
  if (document.body) startCardObserver();
  else document.addEventListener('DOMContentLoaded', startCardObserver);

  // --- 4. grey out offline-only controls ----------------------------------

  function injectStyle() {
    var css =
      '.tool-toggle[data-tool="evaluate"],' +
      '.tool-toggle[data-tool="print"],' +
      '.tool-toggle[data-tool="loop"],' +
      '.contract-agree-btn,' +
      '.biz-btn[data-dl],' +
      '#biz-card-download {' +
      'opacity:.4 !important;pointer-events:none !important;cursor:not-allowed !important;}';
    var style = document.createElement('style');
    style.setAttribute('data-static-mode', '');
    style.textContent = css;
    (document.head || document.documentElement).appendChild(style);
    document.documentElement.setAttribute('data-offline', '1');
  }
  if (document.head) injectStyle();
  else document.addEventListener('DOMContentLoaded', injectStyle);
})();
