/**
 * Praxis Web - Swarm Agent Host
 *
 * A browser-resident compute agent the user donates to the swarm (see
 * next/world_models.md). Each agent owns a tiny transformer (dim 14, 1 layer)
 * and, while idle, runs a lightweight "heartbeat" forward pass over dummy
 * weights - no backprop, just enough to prove it is alive and to count passes.
 * Eventually real Praxis components drive these passes and the agent performs
 * online Mono-Forward updates (the JS PoC lives at staging/js-transformer/).
 *
 * Browser agents are first-class ships: they join the remote actors in the
 * Hangar/Wire list with a consistent name and report dim / layers / passes.
 *
 * Security model: pure in-page JS, gated behind this web app, opt-in per
 * contract, revocable. No native code, no central relay - peer discovery (when
 * wired) goes through GUN over remote STUN/TURN.
 */

import { state } from './state.js';
import { Nanoformer, Adam } from './nanoformer.js';

// Agent lifecycle states.
//   IDLE    - spawned and heartbeating only (no real work). Grey.
//   OBSERVE - training on real backend batches, but passively: it does the
//             computation and learns locally, yet does NOT contribute back to
//             the main model's activations. The current "observer" mode. Green.
//   TRAINING- reserved: contributing back into the model forward (the seam is
//             stubbed; see ExpertPoolCallback / RemoteLayer). Not reachable yet.
export const AGENT_STATUS = {
    IDLE: 'idle',
    OBSERVE: 'observe',
    CONNECTING: 'connecting',
    TRAINING: 'training',
    SERVING: 'serving',
    PAUSED: 'paused',
    ERROR: 'error',
};

// Ship config. Trivially small by design: a 14-dim, single-layer Mono-Forward
// block is enough to be a real expert in the swarm (see next/world_models.md).
const HIDDEN = 14;
const HIDDEN_FFN = 28;
const LAYERS = 1;
const VOCAB = 16;
const HEARTBEAT_MS = 2000; // how often an idle agent runs a heartbeat pass
const TRAIN_MS = 1500;     // how often a joined agent runs an observer update
const SEQ = 8;             // sequence length for the heartbeat / train pass
const TRAIN_DECAY_MS = 4000; // no training within this window -> decay to IDLE

let _seq = 0;

// A fixed dummy token sequence for the idle heartbeat - a real forward pass over
// the model, but not learned-from (no backprop) until the swarm drives work.
const HEARTBEAT_IDS = Array.from({ length: SEQ }, (_, i) => i % VOCAB);

/**
 * A single donated compute agent. Owns a real (tiny) Nanoformer + per-layer Adam
 * optimizers, its heartbeat, and the counters the Hangar/Wire list reports.
 * Framework-free so it can be lifted into a browser extension (../fvn pattern)
 * without dragging a dependency along.
 */
export class SwarmAgent {
    constructor(contract) {
        const n = _seq++;
        this.id = `agent-${Date.now().toString(36)}-${n.toString(36)}`;
        this.name = shipName(n);
        this.contractId = contract.id;
        this.contractTitle = contract.title;
        this.status = AGENT_STATUS.IDLE;
        this.spawnedAt = Date.now();
        this.hidden = HIDDEN;
        this.layers = LAYERS;
        this.passes = 0;          // forward passes counted (heartbeat + future work)
        this.lastBeat = null;     // timestamp of the most recent pass
        this.steps = 0;           // online-update (backprop) steps - 0 while idle
        this.lastLoss = null;     // most recent layer-wise loss (once training)
        this.lastTrained = null;  // timestamp of the most recent tick() (for decay)
        // A real transformer; same model used by the Mono-Forward training path.
        this.model = new Nanoformer({
            vocab: VOCAB, d: HIDDEN, hidden: HIDDEN_FFN, layers: LAYERS, maxT: SEQ,
            seed: (Date.now() ^ (n * 2654435761)) >>> 0,
        });
        // One local optimizer per layer (Mono-Forward: each layer trains itself).
        this.opts = this.model.blocks.map(() => new Adam(0.01));
        this._timer = null;
    }

    /**
     * Begin participating. The agent pulls the backend's REAL training batches
     * (downsampled to the swarm's tiny vocab) and runs observer-mode Mono-Forward
     * updates on them - detached, not contributing back - so it reports OBSERVE
     * (blue) on real data, like its backend siblings.
     *
     * Backpressure / drop policy: the backend buffer is depth 1 (newest wins),
     * and here a `_busy` guard skips any tick whose fetch or training step is
     * still running. So if the model trains faster than this tab can keep up,
     * stale batches are simply dropped - the queue can never grow unbounded.
     * If no new batch is available (or the backend has no pool), it falls back to
     * a local heartbeat so the agent stays alive.
     */
    start() {
        if (this._timer) return;
        this._busy = false;
        this._lastSeq = -1;
        const step = async () => {
            if (this._busy) return; // drop: previous step still in flight
            // trainLayerWise runs synchronous matrix math on the main thread.
            // Skip it while the page is backgrounded - the user can't see the
            // agent's status and a blocked main thread there just wastes battery
            // and can stall the gesture queue on return. (Browsers also throttle
            // this interval when hidden; this makes the skip explicit.)
            if (document.hidden) return;
            this._busy = true;
            try {
                const res = await fetch(`/api/swarm/batch?since=${this._lastSeq}`);
                const data = res.ok ? await res.json() : null;
                const b = data && data.batch;
                if (b && Array.isArray(b.ids) && b.ids.length >= 1) {
                    this._lastSeq = b.seq;
                    this.tick(b.ids, b.targets); // train on the REAL batch -> OBSERVE
                } else {
                    this.heartbeat(); // nothing new -> stay alive, don't flip OBSERVE
                }
            } catch (e) {
                this.heartbeat(); // offline / no backend -> heartbeat only
            } finally {
                this._busy = false;
            }
        };
        step();
        this._timer = setInterval(step, TRAIN_MS);
    }

    /**
     * One heartbeat: a real forward pass through the model (greedy 1-token
     * continuation), with NO backprop - proof of life when no real batch is
     * available. Keeps the agent alive (counts a pass) but does not flip it to
     * OBSERVE; only training on a real batch (tick) does that.
     */
    heartbeat() {
        this.model.generate(HEARTBEAT_IDS, 1); // forward only; result discarded
        this.passes++;
        this.lastBeat = Date.now();
    }

    /**
     * One observer-mode update: a real Mono-Forward layer-wise step over a batch
     * of (ids, targets) sent by the backend. The agent learns locally but its
     * output does NOT feed back into the main model's activations (passive
     * observer). Counts as a forward pass; drives the OBSERVE (green) status.
     *
     * STUB (next step): a contribute() that folds the agent's vote back into the
     * decoder forward - the RemoteLayer seam in praxis/layers/remote.py - would
     * raise the agent to TRAINING. Not wired yet; today every driven agent is an
     * observer.
     */
    tick(ids, targets) {
        this.lastLoss = this.model.trainLayerWise(ids, targets, this.opts);
        this.steps++;
        this.passes++;
        this.lastBeat = Date.now();
        this.lastTrained = this.lastBeat;
        return this.lastLoss;
    }

    /**
     * Effective status: OBSERVE while it has been fed a real batch within the
     * decay window (passively training), otherwise it decays back to IDLE. (The
     * backend keeps calling tick(); when it stops sending batches, the agent
     * goes quiet again.)
     */
    effectiveStatus() {
        if (this.status === AGENT_STATUS.PAUSED || this.status === AGENT_STATUS.ERROR) {
            return this.status;
        }
        if (this.lastTrained && Date.now() - this.lastTrained < TRAIN_DECAY_MS) {
            return AGENT_STATUS.OBSERVE;
        }
        return AGENT_STATUS.IDLE;
    }

    stop() {
        if (this._timer) { clearInterval(this._timer); this._timer = null; }
        this.status = AGENT_STATUS.PAUSED;
    }

    /**
     * Plain snapshot shaped like a Hangar/Wire actor so browser agents render in
     * the same list. `kind: 'browser'` lets the renderer tag them and skip the
     * git-repo info fields that only apply to remote actors.
     */
    toView() {
        return {
            kind: 'browser',
            id: this.id,
            name: this.name,
            status: this.effectiveStatus(),
            contract: this.contractTitle,
            hidden: this.hidden,
            layers: this.layers,
            passes: this.passes,
            steps: this.steps,
            lastBeat: this.lastBeat,
            spawnedAt: this.spawnedAt,
        };
    }
}

// Consistent ship name for browser agents: a single tag + incrementing index
// (e.g. "arc-1", "arc-2"), mirroring the "self-1" convention the discovered
// actors use, so they read as one fleet.
const SHIP_TAG = 'arc';
function shipName(n) {
    return `${SHIP_TAG}-${n + 1}`;
}

/** Spawn an agent for a contract into the live registry and start its heartbeat. */
export function spawnAgent(contract) {
    const agent = new SwarmAgent(contract);
    agent.start();
    state.contracts.agents.push(agent);
    return agent.toView();
}

/** All spawned browser agents as view objects (for rendering). */
export function agentViews() {
    return state.contracts.agents.map((a) => a.toView());
}

/**
 * Sever an agent: stop its heartbeat and remove it from the registry. Returns
 * true if an agent was found and removed.
 */
export function severAgent(agentId) {
    const i = state.contracts.agents.findIndex((a) => a.id === agentId);
    if (i === -1) return false;
    state.contracts.agents[i].stop();
    state.contracts.agents.splice(i, 1);
    return true;
}
