# Peer Bridge: cross-instance pairing for the swarm, without becoming a relay

Status: design sketch (2026-06-07). Captures a Stage-tab idea: a new contract
type that pairs two Praxis instances and lets compute or batches flow between
them. The full peer-relay form is deliberately ruled out here; the open question
is the discovery + transport that gets us a usable bridge without it. Sibling to
[world_models.md](world_models.md) (the swarm/expert-pool vision this extends).

## What's already in place

- `praxis/web/src/js/swarm.js` - browser-resident compute agents ("ships") that
  join a pool via Stage-tab contracts. `agreeContract()` -> `spawnAgent()`. The
  header comment already commits the intended transport: peer discovery, when
  wired, goes through GUN over remote STUN/TURN.
- `praxis/web/src/js/sidecar.js` - the backend's own expert host. Same
  nanoformer math as the browser ships, served over HTTP (`/experts`,
  `/expert/:uid/forward`, `/expert/:uid/train`). The pool has peers the moment
  the app is online, before any browser connects.
- `praxis/layers/remote.py` - the `RemoteLayer` / `ExpertPoolCallback` seam for
  folding a remote ship's vote back into the decoder forward. Stubbed today;
  every driven agent is a passive OBSERVE-mode learner.

So a peer bridge is an extension of the contract/ship abstraction, not a new
subsystem.

## The one hard constraint: there is no browser API for peer discovery

The intuition was a "browser window API" that detects compatible Praxis
instances. That capability does not exist:

- `window.postMessage` / `BroadcastChannel` are same-origin, same-browser only.
  They pair tabs on one machine, not instances across a network.
- Browsers have no mDNS / LAN scan / port probe from page JS, by design.

Cross-machine browser P2P means WebRTC (`RTCPeerConnection` + data channels),
which always needs: a signaling rendezvous to swap SDP offers/answers (GUN, or a
tiny backend `/api/signal` route for a first cut), STUN for NAT traversal, and
TURN as a relay fallback (must be hosted; ~10-20% of peer pairs need it).

## What's ruled out: the full relay peer

You-as-proxy - hosting another peer's contracts, routing their traffic/compute
under your machine's identity - is a non-starter for now. It carries exit-node
liability, needs a real consent + sandboxing + rate-cap story, and is the most
abusable surface. Park it.

## The promising direction: a discovery-then-API sidecar bridge

Instead of a live relay, treat a peer as a *discovered remote API* and read from
it. The existing `sidecar.js` already exposes exactly such an HTTP surface
(`/capacity`, `/experts`, `/expert/:uid/forward`). The bridge becomes:

1. **Discover** a compatible instance via some local mechanism (the open part -
   options below), yielding a base URL / endpoint.
2. **Bridge** by having the local ship read batches/votes from the peer's
   sidecar API and run them in OBSERVE mode. No backprop fed home, no traffic
   routed through anyone. A dumb consumer of a remote API, not a relay.

This keeps the existing security model intact: pure in-page JS or local sidecar,
opt-in per contract, revocable, no compute executed on anyone's behalf.

### Discovery options, least to most invasive

- **Manual/paste**: user enters a peer's sidecar URL (or a short pairing code
  that resolves through a rendezvous). Trivial, ships first, proves the loop.
- **Backend-mediated rendezvous**: instances register with a small shared
  signaling/registry endpoint and list available peers. This is the GUN-over-
  STUN/TURN path, minus the relay.
- **Cross-origin tab bridge (the open idea worth testing)**: whether two browser
  tabs on different origins can be coaxed into a channel. Honest read: not via
  `postMessage` alone (needs a window reference and target origin), and
  `BroadcastChannel` is same-origin. A shared hidden iframe pointing at one
  common rendezvous origin can relay `postMessage` between otherwise-isolated
  tabs - that is the realistic "bridge tabs cross-origin" mechanism, and it
  still leans on a common origin both sides load. Worth a spike to confirm the
  ergonomics before committing.
- **Browser extension as the tab bridge (the clean cross-origin answer)**: an
  extension runs in a privileged context not bound to any single origin, which
  is exactly what the cross-origin wall needs. `swarm.js` already keeps the ships
  framework-free so they "can be lifted into a browser extension (../fvn
  pattern)." Two strengths: (1) the background service worker can `fetch()` any
  peer's `sidecar.js` endpoints with `host_permissions`, no CORS and no common
  rendezvous origin needed - this alone delivers the read-from-remote-API path
  with zero backend infra; (2) content scripts injected into both Praxis tabs
  (any origin) talk to the shared background worker via `chrome.runtime`
  messaging, so the worker becomes the bridge between two otherwise-isolated
  origins - the hidden-iframe hack, done properly. Limits: it removes the
  *origin* barrier, not the *network* one (it bridges tabs in your browser, not
  another machine - cross-machine still needs WebRTC or HTTP to their sidecar);
  it still can't LAN-scan (discovery stays paste/registry/probe, just without
  CORS friction); MV3 service workers are ephemeral, so bridge state lives in
  `chrome.storage` with reconnect-on-wake, not an assumed-live socket. Costs are
  distribution (store review or dev-mode sideload) and the optics of a broad
  `host_permissions` - scope it as tightly as the design allows. The distinct
  win over plain JS: bridging two *already-open* Praxis tabs on different origins
  with no shared rendezvous, the one case nothing in page JS can do.
- **Python app inspects local browser activity (localhost-native discovery)**:
  the strongest assumption in our favor is that Praxis mostly runs on localhost,
  on home GPUs. So the backend can discover *its own machine's* browser activity
  - find open Praxis-related pages and use any Praxis-enabled tab to proxy
  communication between them. This sidesteps both the origin and the "how do I
  even find a peer" problems by reading the local environment directly. It feels
  invasive, and the honest framing is that it is reading your own browser, not
  the network: enumerate locally-open pages (e.g. via the browser's debugging
  protocol / a cooperating local endpoint), match Praxis origins, and hand the
  ship a same-machine bridge target. Safety rests on a real assumption worth
  stating plainly: if you are not loading shady Praxis forks, the pages you have
  open are ones you already trust. Guardrails to keep it honest: localhost-only
  by default, an allowlist of Praxis origins (never a blanket "scan everything"),
  explicit opt-in per the contract model, and read-of-presence only - the
  backend learns *that* a Praxis tab is open and its endpoint, not arbitrary
  browsing content. Open question: the discovery channel itself (a remote-
  debugging port is powerful but a large trust grant; a tiny cooperating
  loopback endpoint that Praxis pages voluntarily register with is narrower and
  probably the safer first cut).

## First milestone

Add a `bridge` (or `peer`) contract type on the Stage tab. Wire the
manual/paste discovery path to a peer's existing `sidecar.js` HTTP surface.
Drive a local OBSERVE-mode ship from the peer's batches. This proves the whole
loop with no TURN hosting, no signaling infra, and no relay liability - then the
rendezvous and cross-origin-tab experiments layer on top.
