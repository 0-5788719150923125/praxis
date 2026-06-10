/**
 * The spider, raised - a seeded creature living in a large 3D room,
 * watched from above like a fish tank (the Arena card on the Identity
 * tab). Pure mechanics, no backend: all randomness derives from the
 * model hash, so each model is one creature in one room.
 *
 * Everything is a coupled spectrum, the business-card move in 3D. A
 * seeded latent genome passes through a random tanh net, so body plan
 * and temperament co-vary: spine length, leg count, gait wave, hind-leg
 * bias, undulation - the same parameters that, pushed to corners, read
 * as spider, centipede, frog, lizard, or snake. Ineffective morphs still
 * wriggle and try. The room has its own genome: color lean, noise,
 * cohesion, decay.
 *
 * Layers:
 *   1. Behavior - probabilistic FSM (idle/run/sleep/jump/bump/climb)
 *      plus a voter pool of seeded slow oscillators pooled into
 *      continuous control channels.
 *   2. Skeleton - a chain of spring-follower spine segments trailing the
 *      driver; legs are assigned along the spine and PLANT their feet:
 *      each foot stays put until its rest pose drifts too far, then
 *      swings independently to a predicted landing. Toes conform to the
 *      gripped surface.
 *   3. Ink - jagged double-strokes whose jitter re-rolls a few times a
 *      second; the world renders painter-sorted so the creature passes
 *      behind and in front of furniture correctly.
 */

import { state } from './state.js';

// ---------------------------------------------------------------- seeding

function fnv1a(str) {
    let h = 0x811c9dc5;
    for (let i = 0; i < str.length; i++) {
        h ^= str.charCodeAt(i);
        h = Math.imul(h, 0x01000193);
    }
    return h >>> 0;
}

function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
        a |= 0; a = (a + 0x6d2b79f5) | 0;
        let t = Math.imul(a ^ (a >>> 15), 1 | a);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

/* Population genetics, not a single genome: seed a pool of latent
   genomes, draw a handful of parents, and blend PER TRAIT - every trait
   samples its own mixture of the parents through a shared random tanh
   encoder. The hybrid inherits coordinates no single parent occupied,
   and traits still co-vary through the shared hidden space. */
function genome(rng, latentDim = 6, hiddenDim = 10, poolSize = 100, parents = 10) {
    const pool = Array.from({ length: poolSize }, () =>
        Array.from({ length: latentDim }, () => rng() * 2 - 1));
    // Shared encoder: one set of random weights for the whole population.
    const W1 = Array.from({ length: hiddenDim }, () =>
        Array.from({ length: latentDim }, () => rng() * 2 - 1));
    const b1 = Array.from({ length: hiddenDim }, () => (rng() * 2 - 1) * 0.5);
    // Candidates: a sampled subset of the pool become this creature's parents.
    const hs = Array.from({ length: parents }, () => {
        const z = pool[Math.floor(rng() * poolSize)];
        return W1.map((row, j) => {
            let s = b1[j];
            for (let i = 0; i < latentDim; i++) s += row[i] * z[i];
            return Math.tanh(s * 1.2);
        });
    });
    return () => {
        // Per-trait mixture over the parents: softmax of seeded noise.
        const wts = hs.map(() => Math.exp((rng() * 2 - 1) * 2));
        const wSum = wts.reduce((a2, b2) => a2 + b2, 0);
        let s = (rng() * 2 - 1) * 0.4;
        for (let j = 0; j < hiddenDim; j++) {
            let hb = 0;
            for (let i = 0; i < hs.length; i++) hb += hs[i][j] * wts[i] / wSum;
            s += (rng() * 2 - 1) * hb;
        }
        return Math.tanh(s) * 0.5 + 0.5;
    };
}

// -------------------------------------------------------------- morphology

function sampleMorphology(rng) {
    const t = genome(rng);
    const u = (lo, hi) => lo + (hi - lo) * t();

    const legT = t();
    const pairs = legT < 0.1 ? 0 : 1 + Math.floor(((legT - 0.1) / 0.9) * 5.999);
    return {
        mech: t(),                    // master spectrum: organic .. machine
        // Body plan.
        scale: u(0.36, 0.6),
        elong: u(1.0, 2.2),
        square: t(),
        chunk: u(0.7, 1.2),
        spine: 1 + Math.floor(t() * 5.0),  // 1..5 trailing segments
        taper: u(0.1, 0.6),                // how fast the chain shrinks
        tail: t(),
        legPairs: pairs,                   // 0 (snake) .. 6 (centipede)
        segs: t() > 0.62 ? 3 : 2,          // elbows per leg
        heads: t() > 0.78 ? 2 : 1,
        // Machinery limbs: 0 line, 1 triangle, 2 diamond, 3 scissor -
        // only expressed when the morph leans mechanical.
        limbStyle: Math.floor(t() * 3.999),
        bladeW: u(0.5, 1.2),               // blade width of shaped limbs
        legSpan: u(0.95, 1.5),
        legSeg: u(0.6, 0.84),
        arch: u(0.55, 1.5),
        sweepBase: u(0.3, 0.65),
        sweepRange: u(1.0, 2.0),
        neck: u(1.1, 2.0),
        headSize: u(0.4, 0.72),
        stance: u(0.28, 0.5),
        // Eyes - level with the gripped plane, several layouts.
        eyeCount: 1 + Math.floor(t() * 3.999),  // 1..4 across the brow
        eyeSpread: u(0.35, 0.95),
        eyeSize: u(0.10, 0.26),
        stalky: t(),                  // > 0.6: eyes ride short stalks
        // Head dynamics.
        gazey: t(),                   // how far the head swivels to look
        pecky: u(0.02, 0.14),         // rate of head-dips to the ground (eat)
        // Locomotion archetype dials - corners of this space read as
        // spider / centipede / frog / lizard / snake.
        waveK: u(0.25, Math.PI),      // gait phase per leg: ripple .. alternate
        hindBias: t(),                // rear legs longer, leaps harder (frog)
        undulate: t(),                // lateral spine wave (snake, swimmer)
        sprawl: t(),                  // legs out to the side, low slung (lizard)
        playful: u(0.04, 0.22),       // P(run circles around the pen)
        // Temperament.
        speed: u(1.8, 3.6),
        restlessness: u(0.5, 2.2),
        jumpiness: u(0.12, 0.42),
        sleepiness: u(0.04, 0.16),
        clumsiness: u(0.05, 0.3),
        kick: u(0.25, 0.85),
        climby: u(0.25, 0.85),
        stride: u(0.8, 1.5),
        fluid: t(),
        bumpy: t(),
        floaty: t(),
        speedy: t(),
        plasticity: t(),              // how far traits FLUCTUATE at runtime
        alert: t(),                   // posture bias: skulker .. sentinel
        // Ink.
        sketch: u(0.5, 1.3),
        boilHz: u(5, 9),
        stiffness: u(45, 110),
        damping: u(7, 13),
    };
}

/* Arena archetypes: discrete prototypes whose FEATURES get blended.
   Each habitat draws sharpened mixture weights over these through its
   genome, so an arena can be 60% rock tunnel / 30% thicket - hybrids
   with furniture no single archetype carries. */
const ARCHETYPES = {
    gym: { seams: 3, ring: 1, rail: 1, grid: 0, stalks: 1, arcs: 0, rocks: 1, pebbles: 0, glass: 0, trees: 0.6, tufts: 4, mounds: 1.5, mold: 0.6, beams: 0.4, pylons: 0, faces: 0, treeline: 0.3, shell: 0.5 },
    tank: { seams: 0, ring: 0, rail: 0, grid: 0, stalks: 2, arcs: 0, rocks: 2, pebbles: 26, glass: 1, trees: 0.3, tufts: 3, mounds: 2.5, mold: 0.8, beams: 0, pylons: 0, faces: 0.3, treeline: 0, shell: 1.2 },
    tunnel: { seams: 0, ring: 0, rail: 0, grid: 0, stalks: 0, arcs: 5, rocks: 5, pebbles: 9, glass: 0, trees: 0, tufts: 1, mounds: 3.5, mold: 3, beams: 0, pylons: 0, faces: 3, treeline: 0, shell: 0.12 },
    thicket: { seams: 0, ring: 0, rail: 0, grid: 0, stalks: 14, arcs: 1, rocks: 1, pebbles: 0, glass: 0, trees: 4, tufts: 18, mounds: 3, mold: 1.5, beams: 0, pylons: 0, faces: 0.4, treeline: 1.5, shell: 0 },
    void: { seams: 0, ring: 0, rail: 0, grid: 7, stalks: 0, arcs: 0, rocks: 0, pebbles: 0, glass: 0, trees: 0, tufts: 0, mounds: 0, mold: 0, beams: 0, pylons: 0, faces: 0, treeline: 0.4, shell: 0 },
    machine: { seams: 1, ring: 0, rail: 0.5, grid: 1.5, stalks: 0, arcs: 0.5, rocks: 0.5, pebbles: 0, glass: 0.1, trees: 0, tufts: 0.5, mounds: 0.6, mold: 1.2, beams: 5, pylons: 3, faces: 0.5, treeline: 0, shell: 0.35 },
};

/* The room's own genome: archetype mixture, scale, color lean, noise,
   cohesion, decay - coupled the same way the creature is. */
function sampleHabitat(rng) {
    const t = genome(rng, 4, 6);
    const names = Object.keys(ARCHETYPES);
    const raw = names.map(() => Math.exp((t() - 0.5) * 5));
    const wSum = raw.reduce((a, b) => a + b, 0);
    const weights = {};
    names.forEach((n, i) => { weights[n] = raw[i] / wSum; });
    const feat = {};
    for (const f of Object.keys(ARCHETYPES.gym)) {
        feat[f] = names.reduce((acc, n) => acc + ARCHETYPES[n][f] * weights[n], 0);
    }
    return {
        weights, feat,
        // Every arena gets its own scale: cathedrals, crawlspaces, long
        // galleries. A low ceiling makes ceiling-grip jumps routine.
        roomW: 10 + t() * 10,
        roomD: 5 + t() * 4,
        roomH: 2.4 + t() * 3.1,
        hueOff: (t() - 0.5) * 90,
        noise: 0.4 + t() * 1.3,
        cohesion: 0.55 + t() * 0.45,
        decay: t() * 0.45,
        density: 0.6 + t(),
    };
}

// ------------------------------------------------------------ 3D helpers

let ROOM_W = 14, ROOM_D = 7, ROOM_H = 4;
const EDGE = 0.4;
const POSTURES = [0.45, 0.65, 0.85, 1.0, 1.2];   // quantized stance levels

const SURFACES = {};

/* One arena lives at a time, so room scale is module state: set the
   dimensions, rebuild the surface frames. */
function setRoom(w, d, h) {
    ROOM_W = w; ROOM_D = d; ROOM_H = h;
    Object.assign(SURFACES, {
        floor: { o: [0, 0, 0], ea: [1, 0, 0], eb: [0, 0, 1], n: [0, 1, 0], A: w, B: d },
        ceiling: { o: [0, h, 0], ea: [1, 0, 0], eb: [0, 0, 1], n: [0, -1, 0], A: w, B: d },
        wallB: { o: [0, 0, d], ea: [1, 0, 0], eb: [0, 1, 0], n: [0, 0, -1], A: w, B: h },
        wallL: { o: [0, 0, 0], ea: [0, 0, 1], eb: [0, 1, 0], n: [1, 0, 0], A: d, B: h },
        wallR: { o: [w, 0, 0], ea: [0, 0, 1], eb: [0, 1, 0], n: [-1, 0, 0], A: d, B: h },
    });
}
setRoom(14, 7, 4);

function toWorld(S, a, b, h) {
    return {
        x: S.o[0] + S.ea[0] * a + S.eb[0] * b + S.n[0] * h,
        y: S.o[1] + S.ea[1] * a + S.eb[1] * b + S.n[1] * h,
        z: S.o[2] + S.ea[2] * a + S.eb[2] * b + S.n[2] * h,
    };
}

function fromWorld(S, w) {
    const dx = w.x - S.o[0], dy = w.y - S.o[1], dz = w.z - S.o[2];
    return {
        a: dx * S.ea[0] + dy * S.ea[1] + dz * S.ea[2],
        b: dx * S.eb[0] + dy * S.eb[1] + dz * S.eb[2],
        h: dx * S.n[0] + dy * S.n[1] + dz * S.n[2],
    };
}

/* 2-bone IK in surface-local space, knee arched away from the surface. */
function ik2local(root, foot, l1, l2, arch) {
    let da = foot.a - root.a, db = foot.b - root.b, dh = foot.h - root.h;
    let d = Math.hypot(da, db, dh);
    const max = l1 + l2 - 1e-4;
    if (d > max) {
        const s = max / d;
        da *= s; db *= s; dh *= s; d = max;
    }
    const w = (l1 * l1 - l2 * l2 + d * d) / (2 * d);
    const ht = Math.sqrt(Math.max(0, l1 * l1 - w * w)) * arch;
    const ua = da / d, ub = db / d, uh = dh / d;
    let pa = -ua * uh, pb = -ub * uh, ph = 1 - uh * uh;
    const pl = Math.hypot(pa, pb, ph) || 1;
    pa /= pl; pb /= pl; ph /= pl;
    return {
        knee: { a: root.a + ua * w + pa * ht, b: root.b + ub * w + pb * ht, h: root.h + uh * w + ph * ht },
        foot: { a: root.a + da, b: root.b + db, h: root.h + dh },
    };
}

// --------------------------------------------------------------- creature

class Creature {
    constructor(seedStr) {
        this.rng = mulberry32(fnv1a(seedStr));
        this.live = mulberry32(fnv1a(seedStr + ':live'));
        this.p = sampleMorphology(this.rng);

        // Voter pool: seeded slow oscillators pooled per channel - the
        // sample-voter loop the generative process will one day drive.
        const N = 12;
        this.voters = Array.from({ length: N }, () => ({
            w1: 0.3 + this.rng() * 2.2, p1: this.rng() * Math.PI * 2,
            w2: 0.05 + this.rng() * 0.6, p2: this.rng() * Math.PI * 2,
        }));
        this.weights = {};
        for (const c of ['energy', 'buoy', 'tempo', 'swayA', 'swayB', 'gaze']) {
            this.weights[c] = Array.from(
                { length: N }, () => ((this.rng() * 2 - 1) * 2.2) / Math.sqrt(N),
            );
        }
        this.ch = { energy: 0, buoy: 0, tempo: 0, swayA: 0, swayB: 0, gaze: 0 };

        // Runtime plasticity: sampled traits are not frozen at birth.
        // Each of these gets its own nonlinear projection of the voter
        // pool, depth gated by the plasticity gene - so sprawl, arch,
        // stance, undulation, stride all breathe, and downstream
        // nonlinearities (IK, gait windows, spring physics) amplify the
        // fluctuations into behaviors no static genome shows.
        this.modW = {};
        for (const m of ['stance', 'arch', 'sprawl', 'undulate', 'stride']) {
            this.modW[m] = Array.from({ length: N },
                () => ((this.rng() * 2 - 1) * 2.0) / Math.sqrt(N));
        }
        this.mods = { stance: 0, arch: 0, sprawl: 0, undulate: 0, stride: 0 };

        // Posture ladder: stance height moves in DISCRETE levels, one
        // step per pause - three steps, crouch, hold, rise, rise, full
        // height. The quantization is the sequential nonlinearity.
        this.postureLvl = 3;          // index into POSTURES
        this.postureGoal = 3;
        this.postureT = 1 + this.rng();

        this.terrainRef = null;       // set by the Arena; floor geometry
        this.enclosure = 1;           // open worlds have no walls to climb

        this.surface = 'floor';
        this.a = ROOM_W * (0.25 + 0.5 * this.rng());
        this.b = ROOM_D * (0.25 + 0.5 * this.rng());
        this.va = 0; this.vb = 0;
        this.p.stance *= 1 - 0.35 * this.p.sprawl;   // lizards run low
        this.h = this.p.stance * this.p.scale;
        this.vh = 0;
        this.heading = this.rng() * Math.PI * 2;
        this.spin = 0;
        this.phase = this.rng() * Math.PI * 2;
        this.lie = 0;
        this.stagger = 0;
        this.state = 'idle';
        this.timer = this.expo(this.p.restlessness);
        this.ta = this.a; this.tb = this.b;

        // Locomotor effectiveness derives from the body: legless or
        // overloaded morphs are slower, but they still try - the wriggle.
        const P = this.p.legPairs;
        this.effSpeed = this.p.speed
            * (P === 0 ? 0.45 : 0.55 + 0.45 * Math.min(1, P / 3))
            * (0.7 + 0.6 * this.p.legSpan * this.p.legSeg);

        // The spine: world-space spring masses trailing the driver.
        const w0 = toWorld(SURFACES.floor, this.a, this.b, this.h);
        const M = this.p.spine;
        this.chain = Array.from({ length: M }, (_, i) => ({
            x: w0.x - (i + 1) * 0.1, y: w0.y, z: w0.z, vx: 0, vy: 0, vz: 0,
        }));
        this.headM = { ...w0, vx: 0, vy: 0, vz: 0 };
        this.headYaw = 0;     // looking around, relative to heading
        this.peck = 0;        // 0..1 head-dip envelope (eating)
        this.peckCool = 2;

        // Legs distributed along the spine (driver = segment -1, then the
        // chain), mirrored pairs, each with its own planted foot.
        this.legs = [];
        const hosts = 1 + M;
        for (let i = 0; i < P; i++) {
            const host = Math.min(hosts - 1, Math.floor((i / Math.max(1, P)) * hosts)) - 1;
            const fr = P > 1 ? i / (P - 1) : 0.5;
            const sweep = this.p.sweepBase + fr * this.p.sweepRange;
            for (const side of [1, -1]) {
                this.legs.push({
                    host,                  // -1 = driver body, 0.. = chain seg
                    side,
                    sweep: sweep * side,
                    hind: i === P - 1 && P > 1,
                    thresh: 0.6 + this.rng() * 0.5,   // independent tempo
                    gait: (this.legs.length % 2) * Math.PI
                        + i * this.p.waveK,           // ripple .. alternate
                    planted: null,         // surface coords {a,b}
                    swing: null,           // {fa,fb,ta2,tb2,t}
                });
            }
        }

        this.ink = new Map();
    }

    expo(mean) { return -Math.log(1 - this.live() + 1e-9) * mean; }

    /* True ground level under a surface point - only the floor rolls. */
    groundAt(a, b) {
        if (this.surface !== 'floor' || !this.terrainRef) return 0;
        return groundHeight(this.terrainRef, a, b);
    }

    pool(t) {
        const vals = this.voters.map(
            v => Math.sin(v.w1 * t + v.p1) * 0.6 + Math.sin(v.w2 * t + v.p2) * 0.4,
        );
        for (const c in this.weights) {
            let s = 0;
            const w = this.weights[c];
            for (let i = 0; i < w.length; i++) s += w[i] * vals[i];
            this.ch[c] = Math.tanh(s);
        }
        const depth = 0.35 * this.p.plasticity;
        for (const m in this.modW) {
            let s = 0;
            const w = this.modW[m];
            for (let i = 0; i < w.length; i++) s += w[i] * vals[i];
            this.mods[m] = Math.tanh(1.5 * s) * depth;
        }
        return this.ch;
    }

    act() {
        const r = this.live();
        const p = this.p;
        if (r < p.sleepiness) {
            this.state = 'sleep';
            this.timer = 3.0 + this.expo(4.0);
        } else if (r < p.sleepiness + p.playful) {
            // Play: fast circles around a spot in the pen, like a horse.
            this.state = 'play';
            this.timer = 4.0 + this.expo(6.0);
            const S = SURFACES[this.surface];
            this.playR = 0.8 + this.live() * 1.8;
            this.playCa = Math.min(S.A - EDGE - this.playR,
                Math.max(EDGE + this.playR, this.a));
            this.playCb = Math.min(S.B - EDGE - this.playR,
                Math.max(EDGE + this.playR, this.b));
            this.playPhase = Math.atan2(this.b - this.playCb, this.a - this.playCa);
            this.playDir = this.live() < 0.5 ? 1 : -1;
        } else if (r < p.sleepiness + p.playful + p.jumpiness) {
            // A real leap: crouch first, then launch up AND forward.
            this.state = 'crouch';
            this.timer = 0.12 + 0.1 * this.live();
            const hard = this.surface === 'floor' && this.live() < 0.35;
            this.jumpVh = (hard ? 4.6 : 2.2 + this.live() * 1.6)
                * (0.75 + 0.5 * p.floaty) * (1 + 0.5 * p.hindBias);
            this.jumpFwd = (1.0 + 2.6 * p.hindBias) * (0.6 + 0.6 * this.live());
        } else {
            this.state = 'run';
            const S = SURFACES[this.surface];
            const over = this.live() < p.clumsiness;
            this.ta = over && this.live() < 0.5
                ? (this.live() < 0.5 ? -0.6 : S.A + 0.6)
                : S.A * (0.06 + 0.88 * this.live());
            this.tb = over && this.live() >= 0.5
                ? S.B + 0.6
                : S.B * (0.06 + 0.88 * this.live());
        }
    }

    rest() {
        this.state = 'idle';
        this.timer = this.expo(this.p.restlessness);
        // Pulling up from a run often ends in a wary crouch, re-risen
        // step by step on the posture ladder.
        if (this.live() < 0.45) this.postureGoal = this.live() < 0.5 ? 1 : 0;
    }

    unplant() { for (const leg of this.legs) { leg.planted = null; leg.swing = null; } }

    transfer(name, a, b, va, vb) {
        this.surface = name;
        const S = SURFACES[name];
        this.a = Math.min(S.A - EDGE, Math.max(EDGE, a));
        this.b = Math.min(S.B - EDGE, Math.max(EDGE, b));
        this.va = va; this.vb = vb;
        this.h = this.p.stance * this.p.scale;
        this.vh = 0;
        this.heading = Math.atan2(vb, va) || this.heading;
        this.unplant();                 // landing scramble on the new surface
        if (this.state === 'run') {
            this.ta = S.A * (0.1 + 0.8 * this.live());
            this.tb = S.B * (0.1 + 0.8 * this.live());
        }
    }

    hitEdge(which) {
        const p = this.p;
        const sp = Math.hypot(this.va, this.vb);
        const S = SURFACES[this.surface];
        const fast = sp > 0.4 * this.effSpeed;

        const routes = {
            floor: { a0: ['wallL', () => [this.b, EDGE, this.vb, Math.abs(this.va)]],
                     a1: ['wallR', () => [this.b, EDGE, this.vb, Math.abs(this.va)]],
                     b1: ['wallB', () => [this.a, EDGE, this.va, Math.abs(this.vb)]] },
            wallL: { b1: ['ceiling', () => [EDGE, this.a, Math.abs(this.vb), this.va]],
                     b0: ['floor', () => [EDGE, this.a, Math.abs(this.vb), this.va]] },
            wallR: { b1: ['ceiling', () => [ROOM_W - EDGE, this.a, -Math.abs(this.vb), this.va]],
                     b0: ['floor', () => [ROOM_W - EDGE, this.a, -Math.abs(this.vb), this.va]] },
            wallB: { b1: ['ceiling', () => [this.a, ROOM_D - EDGE, this.va, -Math.abs(this.vb)]],
                     b0: ['floor', () => [this.a, ROOM_D - EDGE, this.va, -Math.abs(this.vb)]] },
            ceiling: { a0: ['wallL', () => [this.b, ROOM_H - EDGE, this.vb, -Math.abs(this.va)]],
                       a1: ['wallR', () => [this.b, ROOM_H - EDGE, this.vb, -Math.abs(this.va)]],
                       b1: ['wallB', () => [this.a, ROOM_H - EDGE, this.va, -Math.abs(this.vb)]] },
        };
        const route = routes[this.surface]?.[which];

        if (route && fast && this.enclosure > 0.3 && this.live() < p.climby) {
            this.transfer(route[0], ...route[1]());
            return;
        }
        if (fast && this.live() < p.kick) {
            if (which[0] === 'a') this.va = -this.va * 1.25;
            else this.vb = -this.vb * 1.25;
            this.vh += 0.6;
            this.stagger = 0.35;
            if (this.state === 'run') {
                this.ta = S.A - this.ta; this.tb = S.B - this.tb;
            }
            return;
        }
        if (fast) {
            if (which[0] === 'a') this.va = -this.va * 0.4;
            else this.vb = -this.vb * 0.4;
            this.vh += 0.4 * p.bumpy;
            this.stagger = 0.9;
            this.state = 'bump';
            this.timer = 0.6 + this.expo(0.4);
            this.postureLvl = 0;       // knocked flat; the ladder rebuilds
            this.postureGoal = 0;
            return;
        }
        if (which[0] === 'a') this.va = 0; else this.vb = 0;
    }

    step(dt, t) {
        const p = this.p;
        const ch = this.pool(t);
        const S = SURFACES[this.surface];
        this.timer -= dt;
        this.stagger = Math.max(0, this.stagger - dt);

        let dva = 0, dvb = 0;
        switch (this.state) {
            case 'idle': {
                const drift = 0.18 * p.bumpy;
                dva = ch.swayA * drift; dvb = ch.swayB * drift;
                this.lie = Math.max(0, this.lie - 1.6 * dt);
                if (this.timer <= 0) this.act();
                break;
            }
            case 'run': {
                const da = this.ta - this.a, db = this.tb - this.b;
                const d = Math.hypot(da, db);
                if (d < 0.18) { this.rest(); break; }
                const sp = this.effSpeed * (1 + 0.45 * ch.energy);
                dva = (da / d) * sp; dvb = (db / d) * sp;
                break;
            }
            case 'sleep':
                this.lie = Math.min(1, this.lie + (0.8 + 1.2 * p.fluid) * dt);
                if (this.timer <= 0) this.state = 'rise';
                break;
            case 'rise':
                this.lie -= (0.5 + 0.8 * p.fluid) * dt;
                if (this.lie <= 0) { this.lie = 0; this.rest(); }
                break;
            case 'crouch':
                // Load the spring: the body sinks before the leap.
                if (this.timer <= 0) {
                    this.state = 'jump';
                    this.vh += this.jumpVh;
                    this.va += Math.cos(this.heading) * this.jumpFwd;
                    this.vb += Math.sin(this.heading) * this.jumpFwd;
                }
                break;
            case 'jump':
                if (this.vh <= 0 && this.h <= p.stance * p.scale * 1.15) this.rest();
                break;
            case 'play': {
                // Orbit the pen: chase a point ahead on the circle.
                const w = (this.effSpeed * 1.15) / this.playR;
                this.playPhase += w * this.playDir * dt;
                const aim = this.playPhase + this.playDir * 0.55;
                const sp2 = this.effSpeed * 1.15;
                const ga = this.playCa + Math.cos(aim) * this.playR - this.a;
                const gb = this.playCb + Math.sin(aim) * this.playR - this.b;
                const gd = Math.hypot(ga, gb) || 1;
                dva = (ga / gd) * sp2; dvb = (gb / gd) * sp2;
                if (this.timer <= 0) this.rest();
                break;
            }
            case 'bump':
                if (this.timer <= 0) this.rest();
                break;
        }

        const agility = (2.2 + 7 * p.speedy) * (this.state === 'bump' ? 0.4 : 1);
        const flying = this.state === 'jump' && this.h > p.stance * p.scale * 1.3;
        const drag = (1.2 + 3.5 * (1 - p.fluid)) * (flying ? 0.15 : 1);
        this.va += ((dva - this.va) * agility + ch.swayA * 2.2 * p.bumpy) * dt;
        this.vb += ((dvb - this.vb) * agility + ch.swayB * 2.2 * p.bumpy) * dt;
        this.va -= this.va * drag * dt * (dva ? 0 : 1);
        this.vb -= this.vb * drag * dt * (dvb ? 0 : 1);

        this.a += this.va * dt;
        this.b += this.vb * dt;

        if (this.a < EDGE) { this.a = EDGE; this.hitEdge('a0'); }
        else if (this.a > S.A - EDGE) { this.a = S.A - EDGE; this.hitEdge('a1'); }
        if (this.b < EDGE) { this.b = EDGE; this.hitEdge('b0'); }
        else if (this.b > SURFACES[this.surface].B - EDGE) {
            this.b = SURFACES[this.surface].B - EDGE; this.hitEdge('b1');
        }

        // Solid trunks: trees push the body out and around, never through.
        if (this.surface === 'floor' && this.terrainRef) {
            for (const tr of this.terrainRef.trees) {
                const dx2 = this.a - tr.x, dz2 = this.b - tr.z;
                const d2 = Math.hypot(dx2, dz2);
                const rT = 0.22 + 0.18 * p.scale;
                if (d2 < rT && d2 > 1e-4) {
                    this.a = tr.x + (dx2 / d2) * rT;
                    this.b = tr.z + (dz2 / d2) * rT;
                    const vn = (this.va * dx2 + this.vb * dz2) / d2;
                    if (vn < 0) {            // kill the inward component
                        this.va -= (dx2 / d2) * vn;
                        this.vb -= (dz2 / d2) * vn;
                    }
                }
            }
        }
        const ground = this.groundAt(this.a, this.b);

        const standH = p.stance * p.scale;
        const lieH = 0.08 * p.scale;
        // The posture ladder ticks on its own clock: one quantized step
        // toward the goal per pause, new goals chosen at the top.
        this.postureT -= dt;
        if (this.postureT <= 0) {
            this.postureT = 0.25 + this.expo(0.7);
            if (this.postureLvl !== this.postureGoal) {
                this.postureLvl += Math.sign(this.postureGoal - this.postureLvl);
            } else if (this.state === 'idle' && this.live() < 0.5) {
                // Mostly stand tall (alert biases how tall); sometimes
                // drop low and ladder back up later.
                const r2 = this.live();
                this.postureGoal = r2 < 0.2 ? 0
                    : r2 < 0.35 ? 1
                    : 2 + Math.round((1 + p.alert) * this.live());
                this.postureGoal = Math.min(POSTURES.length - 1, this.postureGoal);
            }
        }
        const posture = POSTURES[this.postureLvl];

        const crouch = this.state === 'crouch' ? 0.55 : 0;
        const standLive = standH * posture * (1 + this.mods.stance);
        const targetH = ground + (standLive * (1 - this.lie) + lieH * this.lie)
            * (1 + 0.12 * this.ch.buoy * p.floaty) * (1 - crouch);
        const airborne = this.h > targetH + 0.12;
        const onWall = this.surface.startsWith('wall');
        if (airborne && !onWall) {
            this.vh -= (9.0 - 5.5 * p.floaty) * dt;
            if (this.legs.some(l => l.planted)) this.unplant();
        } else {
            const k = 60 - 38 * p.floaty;
            const damp = 6 + 6 * p.fluid;
            this.vh += (k * (targetH - this.h) - damp * this.vh) * dt;
        }
        this.h += this.vh * dt;
        const hMin = ground + lieH * 0.6;
        if (this.h < hMin) { this.h = hMin; this.vh = Math.abs(this.vh) * 0.3; }

        if (this.surface === 'floor' && this.enclosure > 0.3
            && this.vh > 0 && this.h >= ROOM_H - standH * 1.4) {
            this.surface = 'ceiling';
            this.h = ROOM_H - this.h;
            this.vh = -this.vh * 0.25;
            this.unplant();
            this.rest();
        } else if (this.surface === 'ceiling' && this.h >= ROOM_H - standH * 1.2) {
            this.surface = 'floor';
            this.h = ROOM_H - this.h;
            this.vh = -Math.abs(this.vh) * 0.2;
            this.unplant();
        }

        const sp = Math.hypot(this.va, this.vb);
        if (sp > 0.08) {
            const want = Math.atan2(this.vb, this.va);
            let diff = want - this.heading;
            while (diff > Math.PI) diff -= 2 * Math.PI;
            while (diff < -Math.PI) diff += 2 * Math.PI;
            const authority = Math.min(1, sp / (this.effSpeed * 0.5));
            this.spin += (diff * (6 + 16 * p.speedy) * authority
                - this.spin * (5 + 4 * p.fluid)) * dt;
        } else {
            this.spin -= this.spin * 6 * dt;
        }
        this.heading += this.spin * dt;

        // ---- spine: each segment chases a point behind the previous,
        // plus a traveling lateral wave - the snake/centipede undulation.
        const s = p.scale;
        const bodyR = 0.32 * s * p.chunk;
        const spacing = bodyR * 1.3;
        const speedF = Math.min(1, sp / Math.max(0.5, this.effSpeed) + 0.15);
        const kF = 30 + 60 * p.speedy, dF = 6 + 5 * p.fluid;
        let prevW = toWorld(S, this.a, this.b, this.h);
        let prevDirA = Math.cos(this.heading), prevDirB = Math.sin(this.heading);
        for (let i = 0; i < this.chain.length; i++) {
            const seg = this.chain[i];
            // Direction away from the leader, in surface coords.
            const segL = fromWorld(S, seg);
            const prevL = fromWorld(S, prevW);
            let da = prevL.a - segL.a, db = prevL.b - segL.b;
            const dl = Math.hypot(da, db) || 1;
            da /= dl; db /= dl;
            const wave = Math.sin(this.phase * 0.8 - i * 1.15)
                * p.undulate * (1 + this.mods.undulate) * bodyR * 1.2 * speedF;
            const anchor = toWorld(S,
                prevL.a - da * spacing + (-db) * wave,
                prevL.b - db * spacing + da * wave,
                Math.max(0.05, prevL.h * (1 - 0.08 * i)));
            this.follow(seg, anchor, kF * 0.7, dF, dt);
            prevW = seg;
            prevDirA = da; prevDirB = db;
        }

        // Head dynamics: the gaze voter swivels the head (wide when idle,
        // a lead-glance when running), and idle heads sometimes dip to the
        // ground - pecking at something only they can see.
        const yawMax = p.gazey * (this.state === 'idle' ? 0.9 : 0.3);
        this.headYaw += (this.ch.gaze * yawMax - this.headYaw) * Math.min(1, 4 * dt);
        this.peckCool -= dt;
        if (this.peck > 0) {
            this.peck = Math.max(0, this.peck - dt / 0.9);
        } else if (this.state === 'idle' && this.peckCool <= 0
            && this.live() < p.pecky) {
            this.peck = 1;
            this.peckCool = 1.5 + this.expo(3.0);
        }
        const peckDip = this.peck > 0 ? Math.sin(Math.PI * this.peck) : 0;

        // Head follower rides the nose, swiveled by the yaw, dipped by
        // the peck.
        const face = this.heading + this.headYaw;
        const fa2 = Math.cos(face), fb2 = Math.sin(face);
        const nose = toWorld(S,
            this.a + fa2 * bodyR * p.neck * p.elong * (1 + 0.25 * peckDip),
            this.b + fb2 * bodyR * p.neck * p.elong * (1 + 0.25 * peckDip),
            Math.max(0.04, this.h + 0.12 * s * (1 - this.lie) - peckDip * this.h * 0.85));
        this.follow(this.headM, nose, kF, dF, dt);

        // ---- feet: planted until the rest pose drifts, then an
        // independent swing to a predicted landing. No sliding.
        this.stepLegs(dt, sp, S);

        this.phase += (2.0 + 3.0 * sp) * p.stride * (1 + 0.3 * ch.tempo + this.mods.stride) * dt;
    }

    stepLegs(dt, sp, S) {
        const p = this.p;
        const s = p.scale;
        const g0 = this.groundAt(this.a, this.b);
        const grounded = this.h - g0 < p.stance * s * 1.3 && this.lie < 0.7;
        const reachBase = 0.95 * p.legSpan * s;
        const swinging = this.legs.filter(l => l.swing).length;
        const maxSwing = Math.max(1, Math.ceil(this.legs.length / 2));
        const mva = sp > 0.05 ? this.va / sp : Math.cos(this.heading);
        const mvb = sp > 0.05 ? this.vb / sp : Math.sin(this.heading);

        for (const leg of this.legs) {
            const host = this.hostLocal(leg.host, S);
            const reach = reachBase * (leg.hind ? 1 + 0.6 * p.hindBias : 1);
            const ang = host.dir + leg.sweep;
            const ra = host.a + Math.cos(ang) * reach * 0.7;
            const rb = host.b + Math.sin(ang) * reach * 0.7;

            if (!grounded) { leg.planted = null; leg.swing = null; continue; }
            if (!leg.planted && !leg.swing) {
                leg.planted = { a: ra, b: rb };
                continue;
            }
            if (leg.swing) {
                const sw = leg.swing;
                sw.t += dt / (0.16 + 0.10 * (1 - p.speedy));
                if (sw.t >= 1) {
                    leg.planted = { a: sw.ta2, b: sw.tb2 };
                    leg.swing = null;
                }
                continue;
            }
            // Planted: trigger a swing when drift exceeds this leg's own
            // threshold AND its gait window is open AND the body has spare
            // support - legs move independently, not on a metronome.
            const drift = Math.hypot(leg.planted.a - ra, leg.planted.b - rb);
            const window2 = Math.sin(this.phase + leg.gait) > -0.2;
            if (drift > leg.thresh * reach * 0.45 && window2 && swinging < maxSwing) {
                const lead = 0.22 * sp;
                leg.swing = {
                    fa: leg.planted.a, fb: leg.planted.b,
                    ta2: ra + mva * lead, tb2: rb + mvb * lead,
                    t: 0,
                };
                leg.planted = null;
            }
        }
    }

    /* Surface-local position and facing of a leg's host segment. */
    hostLocal(host, S) {
        if (host < 0) {
            return { a: this.a, b: this.b, h: this.h, dir: this.heading };
        }
        const seg = this.chain[host];
        const segL = fromWorld(S, seg);
        const ahead = host === 0
            ? { a: this.a, b: this.b }
            : fromWorld(S, this.chain[host - 1]);
        return {
            a: segL.a, b: segL.b, h: Math.max(0.05, segL.h),
            dir: Math.atan2(ahead.b - segL.b, ahead.a - segL.a),
        };
    }

    follow(m, w, k, damp, dt) {
        m.vx += (k * (w.x - m.x) - damp * m.vx) * dt;
        m.vy += (k * (w.y - m.y) - damp * m.vy) * dt;
        m.vz += (k * (w.z - m.z) - damp * m.vz) * dt;
        m.x += m.vx * dt; m.y += m.vy * dt; m.z += m.vz * dt;
    }

    /* ---------------- skeleton solve ---------------- */

    solve(t) {
        const p = this.p;
        const S = SURFACES[this.surface];
        const s = p.scale;
        const bodyR = 0.32 * s * p.chunk;
        const reachBase = 0.95 * p.legSpan * s;
        const sp = Math.hypot(this.va, this.vb);
        const speed = sp / Math.max(0.5, this.effSpeed);

        const bob = Math.sin(this.phase * 2) * 0.03 * s * Math.min(1, speed + 0.1);
        const wob = this.stagger > 0 ? Math.sin(t * 22) * 0.10 * this.stagger : 0;
        const bodyH = this.h + bob + wob;

        const ha = Math.cos(this.heading), hb = Math.sin(this.heading);
        const J = {};
        J.body = toWorld(S, this.a, this.b, bodyH);
        J.chain = this.chain.map(seg => ({ x: seg.x, y: seg.y, z: seg.z }));

        // Everything about the head is built in the SURFACE plane: facing
        // and brow vectors live in world space, so shells and eye rows
        // foreshorten with the floor (or wall) instead of facing the canvas.
        const face = this.heading + this.headYaw;
        const o0 = toWorld(S, 0, 0, 0);
        const sub = (w) => ({ x: w.x - o0.x, y: w.y - o0.y, z: w.z - o0.z });
        J.fwd = sub(toWorld(S, ha, hb, 0));                 // body facing
        J.faceW = sub(toWorld(S, Math.cos(face), Math.sin(face), 0));
        J.browW = sub(toWorld(S, -Math.sin(face), Math.cos(face), 0));
        J.perpW = sub(toWorld(S, -hb, ha, 0));
        J.upW = { x: S.n[0], y: S.n[1], z: S.n[2] };

        const add = (w, v, k2) => ({ x: w.x + v.x * k2, y: w.y + v.y * k2, z: w.z + v.z * k2 });
        J.heads = [];
        const off = p.heads === 2 ? bodyR * 0.55 : 0;
        const headR = bodyR * p.headSize * 2.2;
        for (let i = 0; i < p.heads; i++) {
            const sgn = p.heads === 2 ? (i === 0 ? 1 : -1) : 0;
            const pos = add(this.headM, J.perpW, sgn * off);
            // Eyes: a row across the brow, on the upper-front of the head,
            // level with the gripped plane. Stalky morphs lift them off.
            const stalk = p.stalky > 0.6 ? (p.stalky - 0.6) * headR * 2.2 : 0;
            const eyes = [];
            for (let e = 0; e < p.eyeCount; e++) {
                const lat = p.eyeCount === 1 ? 0
                    : (e / (p.eyeCount - 1) - 0.5) * 2 * p.eyeSpread * headR;
                let ep = add(pos, J.faceW, headR * 0.55);
                ep = add(ep, J.browW, lat);
                ep = add(ep, J.upW, headR * 0.35 + stalk);
                eyes.push({ pos: ep, r: headR * p.eyeSize * (1 + 0.4 * (e % 2 === 0 ? 0 : -0.3)) });
            }
            J.heads.push({ pos, eyes, stalk, headR });
        }

        const gBody = this.groundAt(this.a, this.b);
        const grounded = this.h - gBody < p.stance * s * 1.3;
        J.legs = [];
        for (const leg of this.legs) {
            const host = this.hostLocal(leg.host, S);
            const reach = reachBase * (leg.hind ? 1 + 0.6 * p.hindBias : 1);
            const seg = reach * p.legSeg;
            const ang = host.dir + leg.sweep;
            const hipR = bodyR * (0.85 + 0.55 * Math.max(0, p.sprawl * (1 + this.mods.sprawl)));
            const hip = {
                a: host.a + Math.cos(ang) * hipR,
                b: host.b + Math.sin(ang) * hipR,
                h: host.h + bob,
            };
            let foot, toeDir;
            if (this.lie >= 0.7) {
                const fa3 = host.a + Math.cos(ang) * reach * 1.25;
                const fb3 = host.b + Math.sin(ang) * reach * 1.25;
                foot = { a: fa3, b: fb3, h: this.groundAt(fa3, fb3) + 0.01 };
                toeDir = ang;
            } else if (!grounded) {
                const back = leg.hind ? 0.9 : 0;   // hind legs trail in flight
                foot = {
                    a: host.a + Math.cos(ang) * reach * 0.5 - Math.cos(host.dir) * reach * back,
                    b: host.b + Math.sin(ang) * reach * 0.5 - Math.sin(host.dir) * reach * back,
                    h: bodyH - seg * (leg.hind ? 0.3 : 0.7),
                };
                toeDir = ang;
            } else if (leg.swing) {
                const sw = leg.swing;
                const e = sw.t < 0.5 ? 2 * sw.t * sw.t : 1 - Math.pow(-2 * sw.t + 2, 2) / 2;
                const fa3 = sw.fa + (sw.ta2 - sw.fa) * e;
                const fb3 = sw.fb + (sw.tb2 - sw.fb) * e;
                foot = {
                    a: fa3, b: fb3,
                    h: this.groundAt(fa3, fb3) + Math.sin(Math.PI * sw.t) * 0.16 * s,
                };
                toeDir = Math.atan2(sw.tb2 - sw.fb, sw.ta2 - sw.fa);
            } else if (leg.planted) {
                foot = { a: leg.planted.a, b: leg.planted.b, h: this.groundAt(leg.planted.a, leg.planted.b) };
                toeDir = host.dir;
            } else {
                const fa3 = host.a + Math.cos(ang) * reach * 0.7;
                const fb3 = host.b + Math.sin(ang) * reach * 0.7;
                foot = { a: fa3, b: fb3, h: this.groundAt(fa3, fb3) };
                toeDir = ang;
            }
            const sol = ik2local(hip, foot, seg, seg, p.arch * (1 + this.mods.arch));
            const joints = [hip, sol.knee];
            if (p.segs === 3) {
                joints.splice(1, 0, {
                    a: (hip.a + sol.knee.a) / 2,
                    b: (hip.b + sol.knee.b) / 2,
                    h: (hip.h + sol.knee.h) / 2 + seg * 0.22 * p.arch,
                });
            }
            joints.push(sol.foot);
            // The toe: a short flat segment ON the surface, conforming to
            // the terrain and pointing where the foot is headed.
            const ta3 = sol.foot.a + Math.cos(toeDir) * 0.1 * s;
            const tb3 = sol.foot.b + Math.sin(toeDir) * 0.1 * s;
            const toe = { a: ta3, b: tb3, h: this.groundAt(ta3, tb3) };
            J.legs.push({
                joints: joints.map(j => toWorld(S, j.a, j.b, j.h)),
                toe: toWorld(S, toe.a, toe.b, toe.h),
                grounded: !!leg.planted,
            });
        }

        J.shadow = [];
        for (let i = 0; i < 10; i++) {
            const a2 = (i / 10) * Math.PI * 2;
            const sa3 = this.a + Math.cos(a2) * 0.5 * s * p.elong * 0.8;
            const sb3 = this.b + Math.sin(a2) * 0.38 * s;
            J.shadow.push(toWorld(S, sa3, sb3, this.groundAt(sa3, sb3) + 0.01));
        }
        return J;
    }
}

// ------------------------------------------------------------------- ink

class Inker {
    constructor(rng, boilHz, sketch) {
        this.rng = rng;
        this.boilHz = boilHz;
        this.sketch = sketch;
        this.jit = new Map();
        this.last = -1e9;
    }

    boil(t) {
        if (t - this.last < 1 / this.boilHz) return;
        this.last = t;
        this.jit.clear();
    }

    jitter(key, n) {
        let a = this.jit.get(key);
        if (!a) {
            a = Array.from({ length: n }, () => (this.rng() * 2 - 1));
            a[0] *= 0.25; a[n - 1] *= 0.25;
            this.jit.set(key, a);
        }
        return a;
    }

    bone(ctx, a, b, key, amp) {
        const n = 7;
        const dx = b.x - a.x, dy = b.y - a.y;
        const len = Math.hypot(dx, dy) || 1;
        const nx = -dy / len, ny = dx / len;
        const j = this.jitter(key, n);
        const A = amp * this.sketch * Math.min(6, 1.2 + len * 0.05);
        const base = ctx.globalAlpha;
        for (let pass = 0; pass < 2; pass++) {
            ctx.beginPath();
            ctx.globalAlpha = base * (pass === 0 ? 0.95 : 0.30);
            const k = pass === 0 ? 1 : -0.7;
            for (let i = 0; i < n; i++) {
                const t = i / (n - 1);
                ctx[i ? 'lineTo' : 'moveTo'](
                    a.x + dx * t + nx * j[i] * A * k,
                    a.y + dy * t + ny * j[i] * A * k,
                );
            }
            ctx.stroke();
        }
        ctx.globalAlpha = base;
    }

    loop(ctx, pts, key, amp) {
        const j = this.jitter(key, pts.length);
        const A = amp * this.sketch;
        ctx.beginPath();
        pts.forEach((p, i) => {
            ctx[i ? 'lineTo' : 'moveTo'](p.x + j[i] * A, p.y + j[i] * A * 0.6);
        });
        ctx.closePath();
        ctx.stroke();
    }
}

// ------------------------------------------------------------ ecosystem

/* A shared terrain layer every habitat gets a hint of: rolling contour
   lines, boulders solid enough to hide behind, a tree or two, mold
   creeping on the back wall, motes drifting in the air. Counts ride the
   habitat genome - a sketch of what full generative terrain could be. */
function buildTerrain(rng, hab) {
    const f = hab.feat;
    const n = (k) => Math.round(k * hab.density);
    return {
        hills: Array.from({ length: 1 + n(2) }, () => ({
            z: ROOM_D * (0.2 + rng() * 0.7),
            amp: 0.1 + rng() * 0.3,
            waves: 1 + Math.floor(rng() * 3),
            ph: rng() * Math.PI * 2,
        })),
        boulders: Array.from({ length: 1 + n(f.rocks * 0.7) }, () => ({
            x: ROOM_W * (0.1 + rng() * 0.8),
            z: ROOM_D * (0.15 + rng() * 0.75),
            r: 0.45 + rng() * 0.65,
            squash: 0.55 + rng() * 0.3,
        })),
        trees: Array.from({ length: n(f.trees) }, () => ({
            x: ROOM_W * (0.1 + rng() * 0.8),
            z: ROOM_D * (0.3 + rng() * 0.65),
            h: Math.min(ROOM_H * 0.85, 1.6 + rng() * 1.6),
            r: 0.4 + rng() * 0.5,
        })),
        stalks: Array.from({ length: n(f.stalks) }, () => ({
            x: rng() * ROOM_W,
            z: ROOM_D * (0.15 + rng() * 0.8),
            h: Math.min(ROOM_H * 0.7, 0.8 + rng() * 2.4),
            lean: (rng() - 0.5) * 0.6,
        })),
        arcs: Array.from({ length: n(f.arcs) }, () => ({
            h: ROOM_H - 0.3 - rng() * 0.7,
            sag: 0.25 + rng() * 0.55,
            z: ROOM_D * (0.35 + rng() * 0.6),
        })),
        pebbles: Array.from({ length: n(f.pebbles) }, () => ({
            x: rng() * ROOM_W,
            z: ROOM_D * (0.05 + rng() * 0.9),
            r: 0.05 + rng() * 0.12,
        })),
        glass: f.glass,
        seams: n(f.seams),
        ring: f.ring > 0.4 ? { r: 1.0 + rng() * 1.2 } : null,
        rail: f.rail > 0.4 ? { h: 0.35 + rng() * 0.2 } : null,
        gridX: n(f.grid), gridZ: Math.round(n(f.grid) * 0.6),
        mounds: Array.from({ length: 1 + n(f.mounds) }, () => ({
            x: ROOM_W * (0.05 + rng() * 0.9),
            z: ROOM_D * (0.1 + rng() * 0.8),
            r: 0.5 + rng() * 0.9,
            h: 0.12 + rng() * 0.3,
        })),
        tufts: Array.from({ length: 2 + n(f.tufts) }, () => ({
            x: ROOM_W * rng(), z: ROOM_D * rng(),
            s: 0.08 + rng() * 0.14, lean: (rng() - 0.5) * 0.6,
        })),
        mold: Array.from({ length: n(f.mold) }, () => ({
            x: ROOM_W * rng(),
            y: ROOM_H * (0.05 + rng() * 0.6),
            spots: Array.from({ length: 4 + Math.floor(rng() * 9) }, () => ({
                dx: (rng() - 0.5) * 1.4, dy: (rng() - 0.5) * 0.9,
                r: 0.04 + rng() * 0.1,
            })),
        })),
        motes: Array.from({ length: 1 + n(1.5) }, () => ({
            x: ROOM_W * rng(), y: 0.8 + rng() * Math.max(0.5, ROOM_H - 1.6),
            z: ROOM_D * (0.2 + rng() * 0.7),
            r: 0.25 + rng() * 0.5, w: 0.3 + rng() * 0.7, ph: rng() * Math.PI * 2,
        })),
        ceilDots: Array.from({ length: n(40) + 20 }, () => ({
            x: rng() * ROOM_W, z: rng() * ROOM_D,
            r: 0.5 + rng() * 1.2, al: 0.02 + rng() * 0.05,
        })),
        shell: f.shell,
        // Machinery: angled girders and braced pylons.
        beams: Array.from({ length: n(f.beams) }, () => ({
            x0: ROOM_W * rng(), x1: ROOM_W * rng(),
            z: ROOM_D * (0.45 + rng() * 0.5),
            h0: 0.0, h1: Math.min(ROOM_H * 0.9, 1.2 + rng() * 2.4),
        })),
        pylons: Array.from({ length: n(f.pylons) }, () => ({
            x: ROOM_W * (0.08 + rng() * 0.84),
            z: ROOM_D * (0.4 + rng() * 0.55),
            h: Math.min(ROOM_H * 0.95, 1.4 + rng() * 2.2),
            brace: rng() < 0.7,
        })),
        // Natural backdrop: big angled rock faces instead of a flat wall.
        faces: Array.from({ length: n(f.faces) }, () => ({
            cx: ROOM_W * (rng() * 1.3 - 0.15),
            z: ROOM_D * (0.8 + rng() * 0.5),
            w: ROOM_W * (0.25 + rng() * 0.6),
            h: 1.0 + rng() * (ROOM_H * 0.9),
            tilt: (rng() - 0.5) * 2.2,
            jag: Array.from({ length: 5 }, () => (rng() - 0.5) * 0.5),
        })),
        treeline: f.treeline > 0.3 ? {
            z: ROOM_D + 1.5 + rng() * 1.5,
            pts: Array.from({ length: 12 }, () => 0.6 + rng() * 1.4),
            trunks: Array.from({ length: 3 + Math.floor(rng() * 4) }, () => ({
                x: rng() * 1.4 - 0.2, h: 0.5 + rng() * 0.9,
            })),
        } : null,
        // The world continues past the bounds: faded scatter off-stage.
        outTufts: Array.from({ length: n((f.tufts + f.stalks) * 0.6) }, () => ({
            x: ROOM_W * (rng() * 1.6 - 0.3),
            z: ROOM_D * (1.0 + rng() * 0.45),
            s: 0.1 + rng() * 0.18, lean: (rng() - 0.5) * 0.6,
        })),
        outRocks: Array.from({ length: n((f.rocks + f.mounds) * 0.5) }, () => ({
            x: ROOM_W * (rng() * 1.5 - 0.25),
            z: ROOM_D * (1.05 + rng() * 0.4),
            r: 0.3 + rng() * 0.7,
        })),
    };
}

/* The floor's true geometry: boulders are domes, mounds are gaussian
   humps, hills are low ridges. The creature's body height and every
   planted foot sample this, so it steps up onto rocks and self-corrects
   over dirt - the spring physics and IK do the rest. */
function groundHeight(terrain, a, b) {
    let g = 0;
    for (const bo of terrain.boulders) {
        const d = Math.hypot(a - bo.x, b - bo.z);
        const R = bo.r * 1.15;
        if (d < R) {
            g = Math.max(g, bo.r * bo.squash * 0.85 * Math.cos((d / R) * Math.PI / 2));
        }
    }
    for (const mo of terrain.mounds) {
        const d2 = (a - mo.x) ** 2 + (b - mo.z) ** 2;
        g = Math.max(g, mo.h * Math.exp(-d2 / (2 * mo.r * mo.r * 0.35)));
    }
    for (const hl of terrain.hills) {
        const ridge = Math.max(0, Math.sin((a / ROOM_W) * Math.PI * hl.waves + hl.ph));
        const band = Math.exp(-((b - hl.z) ** 2) / 0.5);
        g = Math.max(g, hl.amp * ridge * band);
    }
    return g;
}

/* Stable per-key decay: a decayed room has crumbled away some of its
   lines, and which ones never changes for a given seed. */
function survives(key, hab, salt) {
    return ((fnv1a(key + salt) % 1000) / 1000) >= hab.decay;
}

/* Emit the room as depth-sorted items. Long spanning lines split into
   thirds so the creature can pass in front of the near part of an edge
   and behind the far part. */
function envItems(hab, salt, ink, P, terrain, colors, t) {
    const items = [];
    const seg3 = (x0, y0, z0, x1, y1, z1, key, alpha, amp) => {
        for (let k = 0; k < 3; k++) {
            const t0 = k / 3, t1 = (k + 1) / 3;
            const subKey = key + ':' + k;
            if (!survives(subKey, hab, salt)) continue;
            const za = z0 + (z1 - z0) * t0, zb = z0 + (z1 - z0) * t1;
            items.push({
                z: Math.min(za, zb),
                draw: (ctx) => {
                    ctx.globalAlpha = alpha * hab.cohesion;
                    ink.bone(ctx,
                        P(x0 + (x1 - x0) * t0, y0 + (y1 - y0) * t0, za),
                        P(x0 + (x1 - x0) * t1, y0 + (y1 - y0) * t1, zb),
                        subKey, amp);
                },
            });
        }
    };

    // The containment shell is a TANK feature, not a given: its lines
    // fade in only as the habitat leans glass-tank. Open-world arenas
    // keep a whisper of the floor edge and nothing else.
    const sh = Math.min(1, terrain.shell);
    const floorA = 0.12 + 0.33 * sh;
    seg3(0, 0, 0, ROOM_W, 0, 0, 'fl-near', floorA, 1.0);
    seg3(0, 0, ROOM_D, ROOM_W, 0, ROOM_D, 'fl-far', floorA * 0.7, 0.8);
    seg3(0, 0, 0, 0, 0, ROOM_D, 'fl-left', floorA * 0.7, 0.8);
    seg3(ROOM_W, 0, 0, ROOM_W, 0, ROOM_D, 'fl-right', floorA * 0.7, 0.8);
    if (sh > 0.15) {
        seg3(0, ROOM_H, ROOM_D, ROOM_W, ROOM_H, ROOM_D, 'ce-far', 0.22 * sh, 0.8);
        seg3(0, ROOM_H, 0, ROOM_W, ROOM_H, 0, 'ce-near', 0.16 * sh, 0.7);
        seg3(0, ROOM_H, 0, 0, ROOM_H, ROOM_D, 'ce-left', 0.16 * sh, 0.7);
        seg3(ROOM_W, ROOM_H, 0, ROOM_W, ROOM_H, ROOM_D, 'ce-right', 0.16 * sh, 0.7);
        seg3(0, 0, ROOM_D, 0, ROOM_H, ROOM_D, 'post-bl', 0.22 * sh, 0.8);
        seg3(ROOM_W, 0, ROOM_D, ROOM_W, ROOM_H, ROOM_D, 'post-br', 0.22 * sh, 0.8);
        seg3(0, 0, 0, 0, ROOM_H, 0, 'post-fl', 0.10 * sh, 0.7);
        seg3(ROOM_W, 0, 0, ROOM_W, ROOM_H, 0, 'post-fr', 0.10 * sh, 0.7);
    }    // ---- blended archetype furniture ----
    for (let i = 1; i <= terrain.seams; i++) {
        const z = (ROOM_D * i) / (terrain.seams + 1);
        seg3(0, 0, z, ROOM_W, 0, z, 'fl-z' + i, 0.12, 0.7);
    }
    if (terrain.ring && survives('ring', hab, salt)) {
        items.push({
            z: ROOM_D / 2 - terrain.ring.r,
            draw: (ctx) => {
                ctx.globalAlpha = 0.12 * hab.cohesion;
                const ring = [];
                for (let i = 0; i < 14; i++) {
                    const a = (i / 14) * Math.PI * 2;
                    ring.push(P(ROOM_W / 2 + Math.cos(a) * terrain.ring.r, 0,
                        ROOM_D / 2 + Math.sin(a) * terrain.ring.r));
                }
                ink.loop(ctx, ring, 'fl-ring', 0.7);
            },
        });
    }
    if (terrain.rail) {
        seg3(-0.5, ROOM_H * terrain.rail.h, ROOM_D, ROOM_W + 0.5, ROOM_H * terrain.rail.h, ROOM_D, 'rail', 0.15, 0.7);
    }
    for (let i = 1; i < terrain.gridX; i++) {
        const x = (ROOM_W * i) / terrain.gridX;
        seg3(x, 0, 0, x, 0, ROOM_D, 'gx' + i, 0.10, 0.6);
    }
    for (let i = 1; i < terrain.gridZ; i++) {
        const z = (ROOM_D * i) / terrain.gridZ;
        seg3(0, 0, z, ROOM_W, 0, z, 'gz' + i, 0.10, 0.6);
    }
    terrain.stalks.forEach((st, i) => {
        if (!survives('stalk' + i, hab, salt)) return;
        items.push({
            z: st.z,
            draw: (ctx) => {
                ctx.globalAlpha = 0.28 * hab.cohesion;
                const top = P(st.x + st.lean, st.h, st.z);
                ink.bone(ctx, P(st.x, 0, st.z), top, 'stalk' + i, 1.1);
                ctx.globalAlpha = 0.18 * hab.cohesion;
                ink.bone(ctx, top, P(st.x + st.lean + 0.3, st.h - 0.35, st.z), 'frond-a' + i, 0.9);
                ink.bone(ctx, top, P(st.x + st.lean - 0.3, st.h - 0.3, st.z), 'frond-b' + i, 0.9);
            },
        });
    });
    terrain.arcs.forEach((arc, i) => {
        if (!survives('arc' + i, hab, salt)) return;
        items.push({
            z: arc.z,
            draw: (ctx) => {
                ctx.globalAlpha = 0.20 * hab.cohesion;
                let prev = null;
                for (let k = 0; k <= 8; k++) {
                    const x = (k / 8) * ROOM_W;
                    const dip = -Math.sin((k / 8) * Math.PI) * arc.sag;
                    const pt = P(x, arc.h + dip, arc.z);
                    if (prev) ink.bone(ctx, prev, pt, 'arc' + i + '-' + k, 0.8);
                    prev = pt;
                }
            },
        });
    });
    terrain.pebbles.forEach((pb, i) => {
        if (!survives('pebble' + i, hab, salt)) return;
        items.push({
            z: pb.z,
            draw: (ctx) => {
                ctx.globalAlpha = 0.25 * hab.cohesion;
                const pts = [];
                for (let k = 0; k < 6; k++) {
                    const a = (k / 6) * Math.PI * 2;
                    pts.push(P(pb.x + Math.cos(a) * pb.r,
                        Math.max(0, Math.sin(a) * pb.r * 0.6), pb.z));
                }
                ink.loop(ctx, pts, 'pebble' + i, 0.5);
            },
        });
    });
    if (terrain.glass > 0.25) {
        // Glass tank: translucent panes and a diagonal glint on the back
        // wall - drawn deepest, so everything lives in front of it.
        items.push({
            z: ROOM_D + 0.02,
            draw: (ctx) => {
                ctx.globalAlpha = 0.05 * terrain.glass;
                ctx.fillStyle = colors.grain;
                ctx.beginPath();
                const q = [P(0, ROOM_H, ROOM_D), P(ROOM_W, ROOM_H, ROOM_D),
                    P(ROOM_W, 0, ROOM_D), P(0, 0, ROOM_D)];
                q.forEach((p2, k) => ctx[k ? 'lineTo' : 'moveTo'](p2.x, p2.y));
                ctx.closePath();
                ctx.fill();
                ctx.globalAlpha = 0.18 * terrain.glass * hab.cohesion;
                ink.bone(ctx, P(ROOM_W * 0.18, ROOM_H * 0.9, ROOM_D),
                    P(ROOM_W * 0.42, ROOM_H * 0.15, ROOM_D), 'glint-a', 0.5);
                ink.bone(ctx, P(ROOM_W * 0.28, ROOM_H * 0.95, ROOM_D),
                    P(ROOM_W * 0.46, ROOM_H * 0.35, ROOM_D), 'glint-b', 0.5);
            },
        });
    }

    terrain.hills.forEach((hl, i) => {
        if (!survives('hill' + i, hab, salt)) return;
        items.push({
            z: hl.z,
            draw: (ctx) => {
                ctx.globalAlpha = 0.14 * hab.cohesion;
                let prev = null;
                for (let k = 0; k <= 10; k++) {
                    const x = (k / 10) * ROOM_W;
                    const y = Math.max(0, Math.sin((k / 10) * Math.PI * hl.waves + hl.ph) * hl.amp);
                    const pt = P(x, y, hl.z);
                    if (prev) ink.bone(ctx, prev, pt, `hill${i}-${k}`, 0.8);
                    prev = pt;
                }
            },
        });
    });
    terrain.boulders.forEach((bo, i) => {
        if (!survives('boulder' + i, hab, salt)) return;
        items.push({
            z: bo.z,
            draw: (ctx) => {
                // Filled with the paper color: the creature can disappear
                // entirely behind a boulder.
                const pts = [];
                for (let k = 0; k < 9; k++) {
                    const a = (k / 9) * Math.PI * 2;
                    const rr = bo.r * (1 + 0.18 * Math.sin(a * 3 + i));
                    pts.push(P(bo.x + Math.cos(a) * rr,
                        Math.max(0, Math.sin(a) * rr * bo.squash), bo.z));
                }
                ctx.globalAlpha = 1;
                ctx.fillStyle = colors.paper;
                ctx.beginPath();
                pts.forEach((p2, k) => ctx[k ? 'lineTo' : 'moveTo'](p2.x, p2.y));
                ctx.closePath();
                ctx.fill();
                ctx.globalAlpha = 0.35 * hab.cohesion;
                ink.loop(ctx, pts, 'boulder' + i, 1.0);
            },
        });
    });
    terrain.trees.forEach((tr, i) => {
        if (!survives('tree' + i, hab, salt)) return;
        items.push({
            z: tr.z,
            draw: (ctx) => {
                ctx.globalAlpha = 0.30 * hab.cohesion;
                const top = P(tr.x, tr.h, tr.z);
                ink.bone(ctx, P(tr.x, 0, tr.z), top, 'trunk' + i, 1.0);
                // Canopy: paper-filled so it occludes, faint rim.
                const pts = [];
                for (let k = 0; k < 9; k++) {
                    const a = (k / 9) * Math.PI * 2;
                    const rr = tr.r * (1 + 0.22 * Math.sin(a * 4 + i * 2));
                    pts.push(P(tr.x + Math.cos(a) * rr, tr.h + Math.sin(a) * rr * 0.75, tr.z));
                }
                ctx.globalAlpha = 1;
                ctx.fillStyle = colors.paper;
                ctx.beginPath();
                pts.forEach((p2, k) => ctx[k ? 'lineTo' : 'moveTo'](p2.x, p2.y));
                ctx.closePath();
                ctx.fill();
                ctx.globalAlpha = 0.28 * hab.cohesion;
                ink.loop(ctx, pts, 'canopy' + i, 1.1);
            },
        });
    });
    // Mold on the back wall: behind everything in the room.
    items.push({
        z: ROOM_D + 0.01,
        draw: (ctx) => {
            ctx.fillStyle = colors.grain;
            for (let i = 0; i < terrain.mold.length; i++) {
                const m = terrain.mold[i];
                for (const sp of m.spots) {
                    const pt = P(m.x + sp.dx, m.y + sp.dy, ROOM_D);
                    ctx.globalAlpha = 0.10;
                    ctx.beginPath();
                    ctx.arc(pt.x, pt.y, sp.r * pt.d, 0, Math.PI * 2);
                    ctx.fill();
                }
            }
            ctx.globalAlpha = 1;
        },
    });
    // Dirt mounds: low stacked arc strokes with a dusting of grain.
    terrain.mounds.forEach((mo, i) => {
        if (!survives('mound' + i, hab, salt)) return;
        items.push({
            z: mo.z,
            draw: (ctx) => {
                ctx.globalAlpha = 0.20 * hab.cohesion;
                for (let ring = 0; ring < 2; ring++) {
                    const rr = mo.r * (1 - ring * 0.4);
                    const hh = mo.h * (1 - ring * 0.35);
                    let prev = null;
                    for (let k = 0; k <= 6; k++) {
                        const x = mo.x - rr + (k / 6) * rr * 2;
                        const y = Math.sin((k / 6) * Math.PI) * hh;
                        const pt = P(x, y, mo.z);
                        if (prev) ink.bone(ctx, prev, pt, `mound${i}-${ring}-${k}`, 0.7);
                        prev = pt;
                    }
                }
                ctx.fillStyle = colors.grain;
                for (let k = 0; k < 5; k++) {
                    const pt = P(mo.x + Math.sin(k * 2.4 + i) * mo.r * 0.6, 0.03,
                        mo.z + Math.cos(k * 1.7 + i) * 0.2);
                    ctx.globalAlpha = 0.10;
                    ctx.beginPath();
                    ctx.arc(pt.x, pt.y, Math.max(0.6, pt.d * 0.008), 0, Math.PI * 2);
                    ctx.fill();
                }
            },
        });
    });
    // Grass tufts: two or three leaning blades, everywhere.
    terrain.tufts.forEach((tf, i) => {
        if (!survives('tuft' + i, hab, salt)) return;
        items.push({
            z: tf.z,
            draw: (ctx) => {
                ctx.globalAlpha = 0.22 * hab.cohesion;
                const base = P(tf.x, 0, tf.z);
                for (let k = -1; k <= 1; k++) {
                    const tip = P(tf.x + tf.lean * tf.s * 3 + k * tf.s * 0.8,
                        tf.s * (2.2 - Math.abs(k) * 0.6), tf.z);
                    ink.bone(ctx, base, tip, `tuft${i}-${k}`, 0.5);
                }
            },
        });
    });
    // Machinery: girders rake across the back, pylons stand braced.
    terrain.beams.forEach((bm, i) => {
        if (!survives('beam' + i, hab, salt)) return;
        items.push({
            z: bm.z,
            draw: (ctx) => {
                ctx.globalAlpha = 0.26 * hab.cohesion;
                ink.bone(ctx, P(bm.x0, bm.h0, bm.z), P(bm.x1, bm.h1, bm.z), 'beam' + i, 0.6);
            },
        });
    });
    terrain.pylons.forEach((py, i) => {
        if (!survives('pylon' + i, hab, salt)) return;
        items.push({
            z: py.z,
            draw: (ctx) => {
                ctx.globalAlpha = 0.30 * hab.cohesion;
                ink.bone(ctx, P(py.x, 0, py.z), P(py.x, py.h, py.z), 'pylon' + i, 0.6);
                if (py.brace) {
                    ctx.globalAlpha = 0.18 * hab.cohesion;
                    ink.bone(ctx, P(py.x - 0.5, 0, py.z), P(py.x, py.h * 0.55, py.z), 'pybr-a' + i, 0.5);
                    ink.bone(ctx, P(py.x + 0.5, 0, py.z), P(py.x, py.h * 0.55, py.z), 'pybr-b' + i, 0.5);
                }
            },
        });
    });
    // Natural backdrop: angled rock faces where a wall might have been -
    // jagged, tilted, sometimes spanning the full width.
    terrain.faces.forEach((fc, i) => {
        if (!survives('face' + i, hab, salt)) return;
        items.push({
            z: fc.z,
            draw: (ctx) => {
                const pts = [];
                pts.push(P(fc.cx - fc.w / 2, 0, fc.z));
                for (let k = 0; k < fc.jag.length; k++) {
                    const fr = (k + 0.5) / fc.jag.length;
                    pts.push(P(fc.cx - fc.w / 2 + fc.w * fr + fc.tilt * (fc.h * fr),
                        fc.h * Math.sin(fr * Math.PI) * (1 + fc.jag[k]), fc.z));
                }
                pts.push(P(fc.cx + fc.w / 2 + fc.tilt * 0.3, 0, fc.z));
                ctx.globalAlpha = 0.06;
                ctx.fillStyle = colors.grain;
                ctx.beginPath();
                pts.forEach((p2, k) => ctx[k ? 'lineTo' : 'moveTo'](p2.x, p2.y));
                ctx.closePath();
                ctx.fill();
                ctx.globalAlpha = 0.25 * hab.cohesion;
                ink.loop(ctx, pts, 'face' + i, 1.3);
            },
        });
    });
    // A treeline past the bounds: the world does not end at the pen.
    if (terrain.treeline) {
        const tl = terrain.treeline;
        items.push({
            z: tl.z,
            draw: (ctx) => {
                ctx.globalAlpha = 0.16 * hab.cohesion;
                let prev = null;
                for (let k = 0; k < tl.pts.length; k++) {
                    const x = ROOM_W * (k / (tl.pts.length - 1)) * 1.4 - ROOM_W * 0.2;
                    const pt = P(x, tl.pts[k], tl.z);
                    if (prev) ink.bone(ctx, prev, pt, 'tline' + k, 1.0);
                    prev = pt;
                }
                for (let k = 0; k < tl.trunks.length; k++) {
                    const tr = tl.trunks[k];
                    ink.bone(ctx, P(ROOM_W * tr.x, 0, tl.z), P(ROOM_W * tr.x, tr.h, tl.z), 'ttr' + k, 0.7);
                }
            },
        });
    }
    // Off-stage scatter, faded: tufts and rocks beyond the pen.
    terrain.outTufts.forEach((tf, i) => {
        items.push({
            z: tf.z,
            draw: (ctx) => {
                ctx.globalAlpha = 0.11 * hab.cohesion;
                const base = P(tf.x, 0, tf.z);
                for (let k = -1; k <= 1; k++) {
                    ink.bone(ctx, base, P(tf.x + tf.lean * tf.s * 3 + k * tf.s * 0.8,
                        tf.s * (2.2 - Math.abs(k) * 0.6), tf.z), `otuft${i}-${k}`, 0.5);
                }
            },
        });
    });
    terrain.outRocks.forEach((r, i) => {
        items.push({
            z: r.z,
            draw: (ctx) => {
                ctx.globalAlpha = 0.13 * hab.cohesion;
                const pts = [];
                for (let k = 0; k < 7; k++) {
                    const a = (k / 7) * Math.PI * 2;
                    pts.push(P(r.x + Math.cos(a) * r.r, Math.max(0, Math.sin(a) * r.r * 0.6), r.z));
                }
                ink.loop(ctx, pts, 'orock' + i, 0.8);
            },
        });
    });

    // Motes: tiny fauna drifting slow ellipses in the air.
    terrain.motes.forEach((mo, i) => {
        items.push({
            z: mo.z,
            draw: (ctx) => {
                const pt = P(mo.x + Math.cos(t * mo.w + mo.ph) * mo.r,
                    mo.y + Math.sin(t * mo.w * 1.7 + mo.ph) * mo.r * 0.4, mo.z);
                ctx.globalAlpha = 0.3;
                ctx.fillStyle = colors.grain;
                ctx.beginPath();
                ctx.arc(pt.x, pt.y, Math.max(0.8, pt.d * 0.012), 0, Math.PI * 2);
                ctx.fill();
                ctx.globalAlpha = 1;
            },
        });
    });
    return items;
}

// ------------------------------------------------------------------ arena

class Arena {
    constructor(canvas, seedStr) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        // Habitat first: it sets the room's scale, and the creature is
        // born into whatever space this seed grows.
        const envRng = mulberry32(fnv1a(seedStr + ':env'));
        this.hab = sampleHabitat(envRng);
        setRoom(this.hab.roomW, this.hab.roomD, this.hab.roomH);
        this.terrain = buildTerrain(envRng, this.hab);
        // Enclosure: how much of a box this habitat truly is. Tunnels
        // count - their rock arcs are a ceiling of sorts.
        this.enclosure = Math.min(1,
            this.hab.feat.shell + this.hab.weights.tunnel * 0.8);
        this.creature = new Creature(seedStr);
        this.creature.terrainRef = this.terrain;
        this.creature.enclosure = this.enclosure;
        this.inker = new Inker(
            mulberry32(fnv1a(seedStr + ':ink')),
            this.creature.p.boilHz,
            this.creature.p.sketch,
        );
        this.envSalt = seedStr;
        this.envInker = new Inker(
            mulberry32(fnv1a(seedStr + ':env-ink')), 0.12,
            this.creature.p.sketch * 0.6 * this.hab.noise,
        );
        this.flair = envRng() * Math.PI * 2;

        // Sparse seeded grain: stipple on the floor and the back wall -
        // roughness scales with the habitat's noise gene.
        const dots = Math.round(70 * this.hab.noise);
        this.floorDots = Array.from({ length: dots }, () => ({
            x: envRng() * ROOM_W, z: envRng() * ROOM_D,
            r: 0.6 + envRng() * 1.4, al: 0.02 + envRng() * 0.06,
        }));
        this.wallDots = Array.from({ length: Math.round(dots * 0.6) }, () => ({
            x: envRng() * ROOM_W, y: envRng() * ROOM_H,
            r: 0.6 + envRng() * 1.2, al: 0.02 + envRng() * 0.05,
        }));

        this.t = 0;
        this.lastTs = null;
        this.colors = { line: '#888', accent: '#4a8', shade: '#888' };
        this.colorTick = 0;
        this.raf = null;
        this.onFrame = null;
        this.ro = new ResizeObserver(() => this.resize());
        this.ro.observe(canvas);
        this.resize();
    }

    resize() {
        const dpr = window.devicePixelRatio || 1;
        const r = this.canvas.getBoundingClientRect();
        if (!r.width) return;
        this.canvas.width = Math.round(r.width * dpr);
        this.canvas.height = Math.round(r.height * dpr);
        this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        this.w = r.width;
        this.h = r.height;
    }

    /* Ink follows the live accent hue AND theme. The creature reads at
       text contrast (dark ink on light, bright on dark); the room leans
       off the accent by its own seeded hue and stays a register quieter.
       Washes are the gradient/grain tints for walls and floor. */
    refreshColors() {
        const cs = getComputedStyle(document.documentElement);
        const hue = parseFloat(cs.getPropertyValue('--accent-hue')) || 161;
        const dark = document.documentElement.getAttribute('data-theme') === 'dark';
        const envHue = (hue + this.hab.hueOff + 360) % 360;
        const sat = 20 + Math.abs(this.hab.hueOff) * 0.4;
        this.colors = {
            line: `hsl(${hue} 30% ${dark ? 88 : 14}%)`,
            shade: `hsl(${envHue} ${sat}% ${dark ? 62 : 42}%)`,
            accent: cs.getPropertyValue('--accent').trim() || `hsl(${hue} 87% 40%)`,
            washTop: `hsl(${envHue} ${sat}% ${dark ? 70 : 35}% / 0)`,
            washBot: `hsl(${envHue} ${sat}% ${dark ? 70 : 35}% / ${dark ? 0.10 : 0.08})`,
            grain: `hsl(${envHue} ${sat}% ${dark ? 75 : 30}%)`,
            glow: `hsl(${hue} 70% ${dark ? 60 : 45}%)`,
            paper: cs.getPropertyValue('--background').trim() || (dark ? '#101312' : '#f4f6f3'),
        };
    }

    start() {
        if (this.raf) return;
        const tick = (ts) => {
            if (!this.canvas.isConnected) { this.stop(); return; }
            if (this.lastTs == null) this.lastTs = ts;
            const dt = Math.min(0.05, (ts - this.lastTs) / 1000);
            this.lastTs = ts;
            this.t += dt;
            if (--this.colorTick <= 0) {
                this.refreshColors();
                this.onFrame?.();
                this.colorTick = 30;
            }
            this.creature.step(dt, this.t);
            this.draw(dt);
            this.raf = requestAnimationFrame(tick);
        };
        this.raf = requestAnimationFrame(tick);
    }

    stop() {
        if (this.raf) cancelAnimationFrame(this.raf);
        this.raf = null;
        this.lastTs = null;
        this.ro.disconnect();
    }

    /* Soft washes, grain, and a glow under the creature: enough surface
       treatment to give the box weight, sparse enough to stay a sketch. */
    drawBackdrop(ctx, P, J) {
        const cl = this.colors;

        // Enclosed habitats get a back-wall wash; open ones get a low
        // horizon band rising from the far floor edge instead.
        const enc = this.enclosure;
        if (enc > 0.15) {
            const wt = P(0, ROOM_H, ROOM_D), wb = P(0, 0, ROOM_D);
            const wallGrad = ctx.createLinearGradient(0, wt.y, 0, wb.y);
            wallGrad.addColorStop(0, cl.washTop);
            wallGrad.addColorStop(1, cl.washBot);
            ctx.fillStyle = wallGrad;
            ctx.globalAlpha = Math.min(1, enc);
            this.quad(ctx, P(0, ROOM_H, ROOM_D), P(ROOM_W, ROOM_H, ROOM_D),
                P(ROOM_W, 0, ROOM_D), P(0, 0, ROOM_D));
            ctx.globalAlpha = 1;
        } else {
            const hb2 = P(0, 1.2, ROOM_D * 1.4), fb2 = P(0, 0, ROOM_D);
            const sky = ctx.createLinearGradient(0, hb2.y, 0, fb2.y);
            sky.addColorStop(0, cl.washTop);
            sky.addColorStop(1, cl.washBot);
            ctx.fillStyle = sky;
            this.quad(ctx, P(-ROOM_W * 0.3, 1.2, ROOM_D * 1.4),
                P(ROOM_W * 1.3, 1.2, ROOM_D * 1.4),
                P(ROOM_W * 1.3, 0, ROOM_D), P(-ROOM_W * 0.3, 0, ROOM_D));
        }

        // Floor: the wash deepens toward the viewer - the tank's shadowed
        // near bed.
        const fn = P(0, 0, 0), ff = P(0, 0, ROOM_D);
        const floorGrad = ctx.createLinearGradient(0, ff.y, 0, fn.y);
        floorGrad.addColorStop(0, cl.washTop);
        floorGrad.addColorStop(1, cl.washBot);
        ctx.fillStyle = floorGrad;
        this.quad(ctx, P(0, 0, ROOM_D), P(ROOM_W, 0, ROOM_D),
            P(ROOM_W, 0, 0), P(0, 0, 0));

        // Grain: seeded stipple, the terrain's roughness.
        ctx.fillStyle = cl.grain;
        for (const d of this.floorDots) {
            const pt = P(d.x, 0, d.z);
            ctx.globalAlpha = d.al;
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, d.r * pt.d / 90, 0, Math.PI * 2);
            ctx.fill();
        }
        if (enc > 0.15) {
            for (const d of this.wallDots) {
                const pt = P(d.x, d.y, ROOM_D);
                ctx.globalAlpha = d.al * enc;
                ctx.beginPath();
                ctx.arc(pt.x, pt.y, d.r * pt.d / 110, 0, Math.PI * 2);
                ctx.fill();
            }
            // Ceiling grain: sparse stipple so the plane reads when the
            // creature walks it.
            for (const d of this.terrain.ceilDots) {
                const pt = P(d.x, ROOM_H, d.z);
                ctx.globalAlpha = d.al * enc;
                ctx.beginPath();
                ctx.arc(pt.x, pt.y, d.r * pt.d / 100, 0, Math.PI * 2);
                ctx.fill();
            }
        }
        ctx.globalAlpha = 1;

        // A soft pool of accent light under the creature.
        const sc = J.shadow[0] ? this.project(J.body) : null;
        if (sc) {
            const under = J.shadow.reduce((acc, s) => {
                const p = this.project(s);
                return { x: acc.x + p.x / J.shadow.length, y: acc.y + p.y / J.shadow.length };
            }, { x: 0, y: 0 });
            const r = sc.d * 1.1 * this.creature.p.scale * 2.2;
            const glow = ctx.createRadialGradient(under.x, under.y, 0, under.x, under.y, r);
            glow.addColorStop(0, cl.glow);
            glow.addColorStop(1, 'transparent');
            ctx.globalAlpha = 0.10;
            ctx.fillStyle = glow;
            ctx.beginPath();
            ctx.ellipse(under.x, under.y, r, r * 0.45, 0, 0, Math.PI * 2);
            ctx.fill();
            ctx.globalAlpha = 1;
        }
    }

    quad(ctx, p0, p1, p2, p3) {
        ctx.beginPath();
        ctx.moveTo(p0.x, p0.y);
        ctx.lineTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.lineTo(p3.x, p3.y);
        ctx.closePath();
        ctx.fill();
    }

    /* Tank camera: standing outside and above, looking down in. The
       seat scales with the room, so every arena fills the window. */
    project(p) {
        const camY = ROOM_H * 1.875, camD = ROOM_D;
        const f = (this.w * 0.97 * camD) / ROOM_W;
        const d = f / (camD + p.z);
        // Pin the near floor edge to the window's bottom: rooms of any
        // proportion fill the frame.
        const y0 = this.h * 0.96 - camY * (f / camD);
        return {
            x: this.w / 2 + (p.x - ROOM_W / 2) * d,
            y: y0 + (camY - p.y) * d,
            d, z: p.z,
        };
    }

    /* A body shell spanned by two screen-space basis vectors (the
       projections of world-plane axes), so it foreshortens with the
       surface the creature grips instead of facing the canvas. */
    shell(ctx, inker, center, vMaj, vMin, key, n = 16) {
        const p = this.creature.p;
        const pe = 2 + 6 * p.square;
        const pts = [];
        for (let i = 0; i < n; i++) {
            const th = (i / n) * Math.PI * 2;
            const ct = Math.cos(th), st = Math.sin(th);
            const r = 1 / Math.pow(
                Math.pow(Math.abs(ct), pe) + Math.pow(Math.abs(st), pe), 1 / pe);
            const teeth = Math.tanh(5 * Math.sin(th * 7 + this.flair)) * 0.07 * p.mech;
            const waver = Math.sin(th * 3 + this.flair * 2) * 0.08 * (1 - p.mech);
            const m = r * (1 + teeth + waver);
            pts.push({
                x: center.x + ct * m * vMaj.x + st * m * vMin.x,
                y: center.y + ct * m * vMaj.y + st * m * vMin.y,
            });
        }
        inker.loop(ctx, pts, key, 1.1);
    }

    gear(ctx, inker, center, radius, key) {
        const pts = [];
        const n = 18;
        for (let i = 0; i < n; i++) {
            const th = (i / n) * Math.PI * 2;
            const r = radius * (1 + 0.18 * Math.tanh(6 * Math.sin(th * 6 + this.t * 0.7)));
            pts.push({ x: center.x + Math.cos(th) * r, y: center.y + Math.sin(th) * r });
        }
        inker.loop(ctx, pts, key, 0.8);
    }

    drawCreature(ctx, J, dt) {
        const c = this.creature;
        const p = c.p;
        const inker = this.inker;

        const k = p.stiffness, dmp = p.damping;
        const chase = (name, target) => {
            let s = c.ink.get(name);
            if (!s) { s = { ...target, vx: 0, vy: 0, vz: 0 }; c.ink.set(name, s); }
            s.vx += (k * (target.x - s.x) - dmp * s.vx) * dt;
            s.vy += (k * (target.y - s.y) - dmp * s.vy) * dt;
            s.vz += (k * (target.z - s.z) - dmp * s.vz) * dt;
            s.x += s.vx * dt; s.y += s.vy * dt; s.z += s.vz * dt;
            return s;
        };

        const body = chase('body', J.body);
        const heads = J.heads.map((hd, i) => chase('head' + i, hd.pos));
        const legs = J.legs.map((leg, i) => ({
            joints: leg.joints.map((j, k2) => chase(`leg${i}-${k2}`, j)),
            toe: leg.toe,
            grounded: leg.grounded,
        }));

        ctx.strokeStyle = this.colors.shade;
        ctx.globalAlpha = 0.15;
        ctx.lineWidth = 1.1;
        inker.loop(ctx, J.shadow.map(s => this.project(s)), 'shadow', 1.0);
        ctx.globalAlpha = 1;

        const pBody = this.project(body);
        const lw = (base) => Math.max(1.25, base * pBody.d / 110);

        ctx.strokeStyle = this.colors.line;
        const order = legs.map((leg, i) => ({ ...leg, i, z: leg.joints[1].z }))
            .sort((a, b2) => b2.z - a.z);
        for (const leg of order) {
            const far = leg.z > body.z;
            ctx.globalAlpha = far ? 0.45 : 0.95;
            ctx.lineWidth = lw(far ? 1.3 : 1.8);
            const pts = leg.joints.map(j => this.project(j));
            const blades = p.mech > 0.4 && p.limbStyle > 0;
            for (let s2 = 0; s2 < pts.length - 1; s2++) {
                if (!blades) {
                    inker.bone(ctx, pts[s2], pts[s2 + 1], `leg${leg.i}-s${s2}`, 1.2);
                    continue;
                }
                // Shaped segments: thin closed polygons along the bone -
                // triangles, diamonds, or crossed scissor blades.
                const a3 = pts[s2], b3 = pts[s2 + 1];
                const dx3 = b3.x - a3.x, dy3 = b3.y - a3.y;
                const ln = Math.hypot(dx3, dy3) || 1;
                const nx3 = -dy3 / ln, ny3 = dx3 / ln;
                const wB = lw(2.6) * p.bladeW;
                const key = `leg${leg.i}-s${s2}`;
                if (p.limbStyle === 1) {
                    inker.loop(ctx, [
                        { x: a3.x + nx3 * wB, y: a3.y + ny3 * wB },
                        { x: a3.x - nx3 * wB, y: a3.y - ny3 * wB },
                        { x: b3.x, y: b3.y },
                    ], key, 0.8);
                } else if (p.limbStyle === 2) {
                    inker.loop(ctx, [
                        { x: a3.x, y: a3.y },
                        { x: (a3.x + b3.x) / 2 + nx3 * wB, y: (a3.y + b3.y) / 2 + ny3 * wB },
                        { x: b3.x, y: b3.y },
                        { x: (a3.x + b3.x) / 2 - nx3 * wB, y: (a3.y + b3.y) / 2 - ny3 * wB },
                    ], key, 0.8);
                } else {
                    inker.loop(ctx, [
                        { x: a3.x + nx3 * wB, y: a3.y + ny3 * wB },
                        { x: b3.x - nx3 * wB * 0.4, y: b3.y - ny3 * wB * 0.4 },
                        { x: b3.x, y: b3.y },
                    ], key + 'x', 0.7);
                    inker.loop(ctx, [
                        { x: a3.x - nx3 * wB, y: a3.y - ny3 * wB },
                        { x: b3.x + nx3 * wB * 0.4, y: b3.y + ny3 * wB * 0.4 },
                        { x: b3.x, y: b3.y },
                    ], key + 'y', 0.7);
                }
            }
            // The toe conforms to the terrain.
            if (leg.grounded) {
                ctx.lineWidth = lw(1.2);
                inker.bone(ctx, pts[pts.length - 1], this.project(leg.toe), `toe${leg.i}`, 0.7);
            }
            if (p.mech > 0.55) {
                ctx.fillStyle = this.colors.line;
                for (let s2 = 1; s2 < pts.length - 1; s2++) {
                    ctx.beginPath();
                    ctx.arc(pts[s2].x, pts[s2].y, lw(2.2), 0, Math.PI * 2);
                    ctx.fill();
                }
            }
        }
        ctx.globalAlpha = 1;

        // Screen-space basis of a world direction at a point: project the
        // point and a point one unit along the direction, take the delta.
        const basis = (w, dir, len) => {
            const a2 = this.project(w);
            const b2 = this.project({ x: w.x + dir.x * len, y: w.y + dir.y * len, z: w.z + dir.z * len });
            return { x: b2.x - a2.x, y: b2.y - a2.y };
        };
        const cross = (n, d) => ({
            x: n[1] * d.z - n[2] * d.y,
            y: n[2] * d.x - n[0] * d.z,
            z: n[0] * d.y - n[1] * d.x,
        });
        const S = SURFACES[c.surface];
        const rW = 0.32 * p.scale * p.chunk;   // body radius, world units

        // Spine shells, tail-first so nearer segments overdraw - each
        // spanned by its own in-plane direction so it lies on the surface.
        const chainP = J.chain.map(s => this.project(s));
        ctx.lineWidth = lw(1.7);
        for (let i = J.chain.length - 1; i >= 0; i--) {
            const aheadW = i === 0 ? body : J.chain[i - 1];
            const seg = J.chain[i];
            let dir = { x: aheadW.x - seg.x, y: aheadW.y - seg.y, z: aheadW.z - seg.z };
            const dl = Math.hypot(dir.x, dir.y, dir.z) || 1;
            dir = { x: dir.x / dl, y: dir.y / dl, z: dir.z / dl };
            const side = cross(S.n, dir);
            const shrink = 1 - p.taper * ((i + 1) / (J.chain.length + 1));
            const last = i === J.chain.length - 1;
            const size = rW * shrink * (last ? 0.9 + 0.7 * p.tail : 1);
            this.shell(ctx, inker, chainP[i],
                basis(seg, dir, size * (1 + (p.elong - 1) * 0.3)),
                basis(seg, side, size * 0.8), 'sh-c' + i, 12);
            const aheadP = i === 0 ? pBody : chainP[i - 1];
            inker.bone(ctx, chainP[i], aheadP, 'sp-c' + i, 0.9);
        }

        // Thorax, spanned by the body's facing and side vectors in-plane.
        ctx.lineWidth = lw(1.8);
        this.shell(ctx, inker, pBody,
            basis(body, J.fwd, rW * p.elong),
            basis(body, J.perpW, rW * 0.8), 'sh-body', 16);

        if (p.mech > 0.5) {
            ctx.globalAlpha = 0.5;
            ctx.lineWidth = lw(1.2);
            this.gear(ctx, inker, pBody, pBody.d * rW * 0.45, 'gear');
            ctx.globalAlpha = 1;
        }

        // Heads: shells, feelers, and eye rows all built on the head's
        // facing/brow vectors - level with the gripped plane, swiveling
        // with the gaze and dipping with the peck.
        ctx.lineWidth = lw(1.6);
        for (let i = 0; i < J.heads.length; i++) {
            const head = J.heads[i];
            const hw = heads[i];
            const ph = this.project(hw);
            const hr = head.headR;
            this.shell(ctx, inker, ph,
                basis(hw, J.faceW, hr),
                basis(hw, J.browW, hr * 0.85), 'sh-head' + i, 10);
            inker.bone(ctx, pBody, ph, 'sp-2-' + i, 0.8);

            ctx.lineWidth = lw(1.1);
            for (const sgn of [-1, 1]) {
                const tipW = {
                    x: hw.x + J.faceW.x * hr * 1.6 + J.browW.x * sgn * hr * 0.7,
                    y: hw.y + J.faceW.y * hr * 1.6 + J.browW.y * sgn * hr * 0.7
                        + Math.sin(this.t * 6 + sgn + i) * hr * 0.15,
                    z: hw.z + J.faceW.z * hr * 1.6 + J.browW.z * sgn * hr * 0.7,
                };
                inker.bone(ctx, ph, this.project(tipW), `feel${i}${sgn}`, 0.8);
            }

            // Eyes - stalked morphs get a little stem first; mechanical
            // morphs get lens rings, organic morphs get filled dots.
            ctx.lineWidth = lw(1.0);
            for (let e = 0; e < head.eyes.length; e++) {
                const eye = head.eyes[e];
                // Eyes ride the chased head: offset by the ink spring's lag.
                const pe2 = this.project({
                    x: eye.pos.x + (hw.x - head.pos.x),
                    y: eye.pos.y + (hw.y - head.pos.y),
                    z: eye.pos.z + (hw.z - head.pos.z),
                });
                if (head.stalk > 0) {
                    inker.bone(ctx, ph, pe2, `stalk${i}-${e}`, 0.6);
                }
                const rPx = Math.max(1.1, eye.r * pe2.d);
                if (p.mech > 0.55) {
                    ctx.strokeStyle = this.colors.accent;
                    ctx.beginPath();
                    ctx.arc(pe2.x, pe2.y, rPx, 0, Math.PI * 2);
                    ctx.stroke();
                    ctx.fillStyle = this.colors.accent;
                    ctx.beginPath();
                    ctx.arc(pe2.x, pe2.y, Math.max(0.6, rPx * 0.35), 0, Math.PI * 2);
                    ctx.fill();
                    ctx.strokeStyle = this.colors.line;
                } else {
                    ctx.fillStyle = this.colors.accent;
                    ctx.beginPath();
                    ctx.arc(pe2.x, pe2.y, rPx, 0, Math.PI * 2);
                    ctx.fill();
                }
            }
            ctx.lineWidth = lw(1.6);
        }
    }

    draw(dt) {
        const { ctx, w, h } = this;
        if (!w) return;
        const inker = this.inker;
        inker.boil(this.t);
        this.envInker.boil(this.t);

        ctx.clearRect(0, 0, w, h);
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        const P = (x, y, z) => this.project({ x, y, z });
        const J = this.creature.solve(this.t);
        this.drawBackdrop(ctx, P, J);

        // Painter's algorithm: the room's pieces and the creature sort by
        // depth, so stalks and rocks pass in front of or behind the
        // creature - and near wall edges overdraw it when it climbs.
        const items = envItems(this.hab, this.envSalt, this.envInker, P, this.terrain, this.colors, this.t);
        items.push({
            z: J.body.z,
            creature: true,
            draw: (ctx2) => {
                ctx2.globalAlpha = 1;
                this.drawCreature(ctx2, J, dt);
            },
        });
        items.sort((a, b) => b.z - a.z);
        ctx.strokeStyle = this.colors.shade;
        ctx.lineWidth = 1.2;
        for (const item of items) {
            if (!item.creature) {
                ctx.strokeStyle = this.colors.shade;
                ctx.lineWidth = 1.2;
            }
            item.draw(ctx);
        }
        ctx.globalAlpha = 1;
    }
}

// -------------------------------------------------------------- wiring

let arenaInstance = null;

export function renderArena() {
    return `<div class="arena-card">
        <div class="portal-frame arena-frame">
            <div class="portal-window">
                <canvas class="arena-canvas"></canvas>
                <div class="biz-card-actions arena-actions" hidden>
                    <button class="biz-btn" id="arena-reroll" type="button">Reroll</button>
                </div>
            </div>
        </div>
    </div>`;
}

export function wireArena(container, seedStr) {
    const canvas = container.querySelector('.arena-canvas');
    if (!canvas) return;
    if (arenaInstance) arenaInstance.stop();
    arenaInstance = new Arena(canvas, seedStr || 'praxis');
    window.__arena = arenaInstance; // dev introspection
    arenaInstance.start();

    // Debug-mode developer tool: reroll the creature's seed, overriding
    // the hash-bound default. Visibility tracks the Settings checkbox.
    const actions = container.querySelector('.arena-actions');
    const reroll = container.querySelector('#arena-reroll');
    if (!actions || !reroll) return;
    const sync = () => { actions.hidden = !state.settings.debugLogging; };
    sync();
    arenaInstance.onFrame = sync;
    let n = 0;
    reroll.addEventListener('click', () => {
        arenaInstance.stop();
        arenaInstance = new Arena(canvas, `reroll:${Date.now()}:${++n}`);
        window.__arena = arenaInstance;
        arenaInstance.onFrame = sync;
        arenaInstance.start();
    });
}
