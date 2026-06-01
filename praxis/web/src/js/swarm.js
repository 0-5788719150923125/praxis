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

// Agent lifecycle states. IDLE = spawned and heartbeating, but not yet wired
// into a live swarm. The rest become reachable once a transport drives work.
export const AGENT_STATUS = {
    IDLE: 'idle',             // spawned, heartbeating, awaiting real work
    CONNECTING: 'connecting', // joining the mesh (future)
    TRAINING: 'training',     // running online updates (future)
    SERVING: 'serving',       // answering activation requests (future)
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
const SEQ = 8;             // sequence length for the heartbeat pass

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
        // A real transformer; same model used by the Mono-Forward training path.
        this.model = new Nanoformer({
            vocab: VOCAB, d: HIDDEN, hidden: HIDDEN_FFN, layers: LAYERS, maxT: SEQ,
            seed: (Date.now() ^ (n * 2654435761)) >>> 0,
        });
        // One local optimizer per layer (Mono-Forward: each layer trains itself).
        this.opts = this.model.blocks.map(() => new Adam(0.01));
        this._timer = null;
    }

    /** Begin heartbeating. The agent stays IDLE but counts live forward passes. */
    start() {
        if (this._timer) return;
        const beat = () => this.heartbeat();
        beat(); // one immediately so the list shows a pass right away
        this._timer = setInterval(beat, HEARTBEAT_MS);
    }

    /**
     * One heartbeat: a real forward pass through the model (greedy 1-token
     * continuation), with NO backprop - proof of life and a pass to count. When
     * a transport starts driving the agent, work flows through the same counter
     * and tick() adds the layer-wise update.
     */
    heartbeat() {
        this.model.generate(HEARTBEAT_IDS, 1); // forward only; result discarded
        this.passes++;
        this.lastBeat = Date.now();
    }

    /**
     * One online-update step: a real Mono-Forward layer-wise update over a batch
     * of (ids, targets). Counts as a forward pass too. This is the hook a live
     * swarm transport calls once it has work; idle agents never reach it.
     */
    tick(ids, targets) {
        this.status = AGENT_STATUS.TRAINING;
        this.lastLoss = this.model.trainLayerWise(ids, targets, this.opts);
        this.steps++;
        this.passes++;
        this.lastBeat = Date.now();
        return this.lastLoss;
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
            status: this.status,
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

// Consistent ship name for browser agents. Two-letter NATO-ish prefix + index,
// so they read as a fleet alongside the discovered actors (e.g. "SHIP-AB-03").
const SHIP_TAGS = [
    'ARC', 'BIT', 'CAL', 'DYN', 'ECHO', 'FVN', 'GUN', 'HEX', 'ION', 'JET',
    'KEL', 'LUX', 'MON', 'NOVA', 'ORB', 'PRX', 'QBIT', 'RHO', 'SOL', 'TAU',
];
function shipName(n) {
    const tag = SHIP_TAGS[n % SHIP_TAGS.length];
    const idx = String(Math.floor(n / SHIP_TAGS.length) + 1).padStart(2, '0');
    return `SHIP-${tag}-${idx}`;
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
