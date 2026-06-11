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
    const m = {
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
        playful: u(0.01, 0.07),       // P(run circles) - a rare mood
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
        // Rodent/fish behavior dials (overridden per kind below).
        rear: 0, fidget: 0, tailStyle: 'none', tailLen: 0, swim: false,
        carapace: false, feelerLen: 1,
    };

    // Creature KIND is a discrete archetype roll, like the arena's faces:
    // the genome stays continuous underneath, the kind re-shapes it.
    const kr = rng();
    if (kr < 0.3) {
        m.kind = 'bug';
        m.components = ['legs'];
    } else if (kr < 0.42) {
        // The roach: a low scuttler under one dorsal shell.
        m.kind = 'roach';
        m.carapace = true;
        m.legPairs = 3; m.segs = 2; m.spine = 1; m.heads = 1;
        m.elong = u(1.7, 2.4); m.square = u(0.2, 0.5);
        m.chunk = u(0.9, 1.25);
        m.stance = u(0.18, 0.28);           // hugs the ground
        m.sprawl = Math.max(0.6, m.sprawl); // legs out from under the rim
        m.arch = u(0.9, 1.4);
        m.legSpan = u(1.1, 1.5);
        m.undulate *= 0.2; m.tail = 0.2;
        m.waveK = u(0.6, 1.2);              // rippling tripod scuttle
        m.speedy = Math.max(0.7, m.speedy); // bursts when it moves
        m.speed = u(2.6, 3.8);
        m.fidget = u(0.2, 0.4);
        m.restlessness = u(0.5, 1.6);
        m.sleepiness = u(0.02, 0.07);       // roaches barely sleep
        m.kick = Math.max(0.5, m.kick);
        m.climby = Math.max(0.5, m.climby);
        m.eyeCount = 2; m.stalky = 0;
        m.feelerLen = u(2.0, 3.0);          // the long antennae
        m.tailStyle = 'none';
        m.eyeSize = u(0.09, 0.14);
        m.components = ['legs', 'carapace'];
    } else if (kr < 0.53) {
        m.kind = 'squirrel';
        m.legPairs = 2; m.segs = 2; m.spine = 1; m.heads = 1;
        m.elong = u(1.15, 1.55); m.square = m.square * 0.4;
        m.sprawl *= 0.35; m.undulate *= 0.3;
        m.hindBias = Math.max(0.65, m.hindBias);
        m.waveK = Math.PI;                  // bounding gait
        m.stance = u(0.4, 0.52);
        m.restlessness = u(0.4, 1.2);       // busy little thing
        m.rear = u(0.3, 0.48);              // sits constantly
        m.fidget = u(0.2, 0.35);
        m.bounder = true;                   // gallop: paired feet
        m.pecky = u(0.08, 0.2);
        m.tailStyle = 'floof';
        m.tailLen = u(0.9, 1.4);
        m.mech = m.mech * 0.5;
        m.eyeCount = 2; m.stalky = 0;
        m.eyeSize = u(0.07, 0.11);          // small rodent eyes, side-set
        m.eyeSpread = u(0.55, 0.8);
        m.climby = Math.max(0.6, m.climby); // squirrels scale anything
        m.components = ['legs', 'paws', 'floofTail'];
    } else if (kr < 0.64) {
        m.kind = 'rat';
        m.legPairs = 2; m.segs = 2; m.spine = 2; m.heads = 1;
        m.elong = u(1.5, 2.1); m.scale *= 0.85;
        m.sprawl *= 0.5; m.undulate *= 0.4;
        m.stance = u(0.26, 0.36);           // low slink
        m.speedy = Math.max(0.6, m.speedy);
        m.restlessness = u(0.4, 1.4);
        m.rear = u(0.18, 0.3);
        m.fidget = u(0.25, 0.4);
        m.bounder = true;
        m.pecky = u(0.1, 0.22);
        m.tailStyle = 'whip';
        m.tailLen = u(1.2, 1.9);
        m.mech = m.mech * 0.5;
        m.eyeCount = 2; m.stalky = 0;
        m.eyeSize = u(0.07, 0.11);
        m.eyeSpread = u(0.55, 0.8);
        m.components = ['legs', 'paws', 'whipTail'];
    } else if (kr < 0.74) {
        m.kind = 'fish';
        m.swim = true;
        m.legPairs = 0; m.segs = 2; m.heads = 1;
        m.spine = 2 + Math.floor(t() * 3.0);
        m.elong = u(1.4, 2.0);
        m.undulate = Math.max(0.7, m.undulate);
        m.tailStyle = 'fin';
        m.tailLen = u(0.5, 0.9);
        m.fidget = 0; m.rear = 0;
        m.jumpiness = u(0.04, 0.12);        // the rare breach
        m.sleepiness = u(0.03, 0.1);
        m.fluid = Math.max(0.6, m.fluid);   // water is smooth
        m.stalky = 0;
        m.components = ['swimmer', 'caudalFin', 'pectorals'];
    } else if (kr < 0.83) {
        // The butterfly: slow, erratic, mostly wind; perches often.
        m.kind = 'butterfly';
        m.swim = true; m.flyStyle = 'butterfly';
        m.legPairs = 0; m.spine = 1; m.heads = 1;
        m.scale *= 0.62; m.elong = u(0.9, 1.3);
        m.chunk = u(0.5, 0.8);
        m.undulate *= 0.2; m.tail = 0.3;
        m.speed = u(1.0, 1.9);
        m.speedy = Math.min(0.35, m.speedy);  // drifts, never darts
        m.bumpy = Math.max(0.6, m.bumpy);     // the flutter
        m.fluid = Math.max(0.5, m.fluid);
        m.sleepiness = u(0.1, 0.26);          // perches constantly
        m.playful = u(0.03, 0.1);             // occasional lazy loops
        m.jumpiness = 0.02; m.fidget = 0;
        m.restlessness = u(0.5, 1.6);
        m.swimLo = 0.4; m.swimHi = Math.min(2.8, ROOM_H * 0.8);
        m.eyeCount = 2; m.stalky = 0;
        m.eyeSize = u(0.18, 0.3);           // big compound insect eyes
        m.feelerLen = u(1.4, 2.2);
        m.tailStyle = 'none';
        // Endurance: how long the wings last. The high tail flies forever
        // - a butterfly that is, in truth, a fish of the air.
        m.endurance = Math.min(1, u(0.15, 1.15));
        m.perchTime = u(2.0, 6.0);            // long lazy rests
        m.components = ['swimmer', 'wingsButterfly'];
    } else if (kr < 0.9) {
        // The wasp: fast hover, hard darts, a mean little pendulum.
        m.kind = 'wasp';
        m.swim = true; m.flyStyle = 'wasp';
        m.legPairs = 0; m.spine = 2; m.heads = 1;
        m.scale *= 0.68; m.elong = u(1.2, 1.6);
        m.taper = u(0.1, 0.25); m.tail = u(0.5, 0.9);
        m.undulate *= 0.15;
        m.speed = u(2.6, 4.0);
        m.speedy = Math.max(0.8, m.speedy);
        m.bumpy = Math.min(0.4, m.bumpy);
        m.sleepiness = u(0.01, 0.05);
        m.fidget = u(0.3, 0.5);               // the darting
        m.restlessness = u(0.3, 1.0);
        m.swimLo = 0.3; m.swimHi = ROOM_H * 0.7;
        m.eyeCount = 2; m.stalky = 0;
        m.eyeSize = u(0.18, 0.28);
        m.feelerLen = u(0.8, 1.2);
        m.tailStyle = 'none';
        m.endurance = Math.min(1, u(0.3, 1.1));
        m.perchTime = u(0.4, 1.4);            // touch-and-go
        m.components = ['swimmer', 'wingsWasp'];
    } else {
        // The Spore move: no preset at all - a spine and whatever parts
        // the roll attaches. Chimeras the named kinds never produce.
        m.kind = 'hybrid';
        const comps = [];
        if (rng() < 0.25) {
            comps.push('swimmer', 'caudalFin');
            m.swim = true;
            m.legPairs = rng() < 0.5 ? 0 : 1;
            m.undulate = Math.max(0.6, m.undulate);
            m.fluid = Math.max(0.5, m.fluid);
            if (rng() < 0.6) comps.push('pectorals');
        } else {
            comps.push('legs');
            m.legPairs = 1 + Math.floor(rng() * 3.0);
        }
        if (rng() < 0.35) {
            comps.push('carapace');
            m.carapace = true;
            m.stance = Math.min(m.stance, u(0.2, 0.34));
            m.elong = Math.max(1.5, m.elong);
        }
        if (!comps.includes('caudalFin')) {
            const tr = rng();
            if (tr < 0.25) { comps.push('floofTail'); m.tailStyle = 'floof'; m.tailLen = u(0.8, 1.4); }
            else if (tr < 0.5) { comps.push('whipTail'); m.tailStyle = 'whip'; m.tailLen = u(1.0, 1.8); }
        }
        if (!m.swim && m.legPairs >= 2 && rng() < 0.4) {
            comps.push('paws');
            m.rear = u(0.1, 0.3); m.fidget = u(0.15, 0.35);
        }
        m.feelerLen = 0.8 + rng() * 1.8;
        m.spine = 1 + Math.floor(rng() * 4);
        m.components = comps;
    }
    // Derived flags stay consistent with the parts actually attached.
    m.swim = m.components.includes('swimmer');
    m.carapace = m.components.includes('carapace');
    if (!m.components.includes('floofTail') && !m.components.includes('whipTail')
        && !m.components.includes('caudalFin')) m.tailStyle = 'none';
    return m;
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
    constructor(seedStr, epochSalt = '') {
        // Identity is hash-bound and eternal: the genome, the voter wiring,
        // the leg temperaments. The RUNTIME is epoch-salted: where it
        // spawns and what it decides differ each window - but identically
        // for every viewer on Earth, since the salt is UTC-derived.
        this.rng = mulberry32(fnv1a(seedStr));
        this.live = mulberry32(fnv1a(seedStr + epochSalt + ':live'));
        const spawn = mulberry32(fnv1a(seedStr + epochSalt + ':spawn'));
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

        this.rearAmt = 0;             // 0 standing .. 1 up on hind legs
        this.nibble = 0;              // paws-to-mouth eating envelope
        this.stamina = 1;             // wing fuel; flyers land to refill
        this.perchFold = 0;           // 0 flying .. 1 wings folded
        this.terrainRef = null;       // set by the Arena; floor geometry
        this.enclosure = 1;           // backdrop-scale enclosure
        this.faces = null;            // per-face materials; null = all solid

        this.surface = 'floor';
        this.a = ROOM_W * (0.25 + 0.5 * spawn());
        this.b = ROOM_D * (0.25 + 0.5 * spawn());
        this.va = 0; this.vb = 0;
        this.p.stance *= 1 - 0.35 * this.p.sprawl;   // lizards run low
        this.h = this.p.stance * this.p.scale;
        this.vh = 0;
        this.heading = spawn() * Math.PI * 2;
        this.spin = 0;
        this.phase = spawn() * Math.PI * 2;
        this.lie = 0;
        this.stagger = 0;
        this.state = 'idle';
        this.timer = this.expo(this.p.restlessness);
        this.ta = this.a; this.tb = this.b;
        this.playCool = 15;           // circles cannot chain back-to-back

        // Locomotor effectiveness derives from the body: legless or
        // overloaded morphs are slower, but they still try - the wriggle.
        const P = this.p.legPairs;
        this.effSpeed = this.p.swim
            ? this.p.speed * 0.9
            : this.p.speed
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
                    front: i === 0 && P > 1,   // the front pair: rodent paws
                    host,                  // -1 = driver body, 0.. = chain seg
                    side,
                    sweep: sweep * side,
                    hind: i === P - 1 && P > 1,
                    thresh: 0.6 + this.rng() * 0.5,   // independent tempo
                    // Bounders gallop: each pair lands together, pairs
                    // alternate. Everything else ripples or alternates.
                    gait: this.p.bounder
                        ? (i % 2) * Math.PI
                        : (this.legs.length % 2) * Math.PI
                            + i * this.p.waveK,
                    planted: null,         // surface coords {a,b}
                    swing: null,           // {fa,fb,ta2,tb2,t}
                });
            }
        }

        this.ink = new Map();
    }

    expo(mean) { return -Math.log(1 - this.live() + 1e-9) * mean; }

    has(component) { return this.p.components.includes(component); }

    /* Place every follower exactly where the rig wants it - called once
       the terrain is attached (constructor defaults know no ground), and
       the cure for heads that spawn on the floor and reel themselves in. */
    seat() {
        const S = SURFACES[this.surface];
        const p = this.p;
        const s = p.scale;
        const g = this.groundAt(this.a, this.b);
        this.h = g + p.stance * s;
        this.vh = 0;
        const bodyR = 0.32 * s * p.chunk;
        const ha = Math.cos(this.heading), hb = Math.sin(this.heading);
        const nose = toWorld(S,
            this.a + ha * bodyR * p.neck * p.elong,
            this.b + hb * bodyR * p.neck * p.elong,
            this.h + 0.12 * s);
        Object.assign(this.headM, { x: nose.x, y: nose.y, z: nose.z, vx: 0, vy: 0, vz: 0 });
        const spacing = bodyR * 1.3;
        this.chain.forEach((seg, i) => {
            const w = toWorld(S,
                this.a - ha * spacing * (i + 1),
                this.b - hb * spacing * (i + 1),
                Math.max(g + 0.05, this.h * (1 - 0.08 * (i + 1))));
            Object.assign(seg, { x: w.x, y: w.y, z: w.z, vx: 0, vy: 0, vz: 0 });
        });
        this.ink.clear();   // drawn twins re-init on the seated pose
    }

    solidFace(name) {
        if (name === 'floor') return true;
        return this.faces ? this.faces[name] !== 'none' : true;
    }

    /* Where this creature may roam: every open edge runs forever. */
    bounds() {
        const S = SURFACES[this.surface];
        if (this.surface === 'floor' && this.terrainRef?.infinite) {
            return {
                x0: this.solidFace('wallL') ? 0 : -1e9,
                x1: this.solidFace('wallR') ? S.A : 1e9,
                z0: this.solidFace('front') ? 0.15 : -1e9,
                z1: this.solidFace('wallB') ? S.B : 1e9,
            };
        }
        if (this.surface === 'floor' && this.terrainRef?.roam) {
            const rm = this.terrainRef.roam;
            return {
                x0: this.solidFace('wallL') ? 0 : rm.x0,
                x1: this.solidFace('wallR') ? S.A : rm.x1,
                z0: rm.z0,
                z1: this.solidFace('wallB') ? S.B : rm.z1,
            };
        }
        return { x0: 0, x1: S.A, z0: 0, z1: S.B };
    }

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
        // Tired wings land before anything else - except the enduring,
        // who never need to.
        if (p.flyStyle && (p.endurance ?? 1) < 1 && this.stamina < 0.25) {
            this.state = 'land';
            this.ta = this.a + (this.live() * 2 - 1) * 2;
            this.tb = this.b + (this.live() * 2 - 1) * 2;
            return;
        }
        if (this.has('paws') && r < p.rear) {
            // Up on the hind legs - sometimes higher still - looking
            // around, sometimes nibbling something held in the paws.
            this.state = 'rear';
            this.timer = 1.2 + this.expo(2.5);
            this.rearHigh = this.live() < 0.4;   // second stage: stretch up
            this.nibbleOn = this.live() < 0.5;
        } else if (p.fidget > 0 && r < p.rear + p.fidget) {
            this.state = 'fidget';
            this.timer = 0.4 + this.expo(0.6);
        } else if (r < p.rear + p.fidget + p.sleepiness) {
            this.state = 'sleep';
            this.timer = 3.0 + this.expo(4.0);
        } else if (r < p.rear + p.fidget + p.sleepiness + p.playful
            && this.playCool <= 0) {
            // Play: a rare mood - short laps, abandoned as easily as begun.
            this.state = 'play';
            this.timer = 2.5 + this.expo(2.5);
            this.playCool = 30 + this.expo(40);
            const S = SURFACES[this.surface];
            this.playR = 0.8 + this.live() * 1.8;
            const B2 = this.bounds();
            this.playCa = Math.min(Math.min(B2.x1, S.A) - EDGE - this.playR,
                Math.max(Math.max(B2.x0, -1e8) + EDGE + this.playR, this.a));
            this.playCb = Math.min(Math.min(B2.z1, S.B * 1e6) - EDGE - this.playR,
                Math.max(B2.z0 + EDGE + this.playR, this.b));
            this.playPhase = Math.atan2(this.b - this.playCb, this.a - this.playCa);
            this.playDir = this.live() < 0.5 ? 1 : -1;
        } else if (r < p.rear + p.fidget + p.sleepiness + p.playful + p.jumpiness) {
            // The leap: a quick dart AHEAD on a low arc - crouch, then
            // mostly-forward launch. Straight-up ceiling jumps survive
            // only as a rare move for climbers under a real ceiling.
            this.state = 'crouch';
            this.timer = 0.12 + 0.1 * this.live();
            const hard = this.surface === 'floor' && !p.swim
                && p.climby > 0.5 && this.solidFace('ceiling')
                && this.live() < 0.15;
            this.jumpVh = hard
                ? 4.6 * (0.75 + 0.5 * p.floaty)
                : (0.9 + this.live() * 0.9)
                    * (0.8 + 0.5 * p.floaty) * (1 + 0.35 * p.hindBias);
            this.jumpFwd = hard
                ? 0.5
                : (2.3 + 2.4 * p.hindBias) * (0.7 + 0.6 * this.live());
            // Rodents bound: more arc, and the head-down landing shows.
            if (!hard && this.has('paws')) {
                this.jumpVh *= 1.8;
                this.jumpFwd *= 1.15;
            }
        } else {
            this.state = 'run';
            const B = this.bounds();
            const over = this.live() < p.clumsiness;
            const pick = (lo, hi, cur, wallTarget) => {
                if (hi - lo > 100) {
                    // Unbounded axis: wander relative to where we stand.
                    return Math.min(hi, Math.max(lo, cur + (this.live() * 2 - 1) * 9));
                }
                return wallTarget != null ? wallTarget
                    : lo + (hi - lo) * (0.06 + 0.88 * this.live());
            };
            this.ta = pick(B.x0, B.x1, this.a,
                over && this.live() < 0.5
                    ? (this.live() < 0.5 ? B.x0 - 0.6 : B.x1 + 0.6) : null);
            this.tb = pick(B.z0, B.z1, this.b,
                over && this.live() >= 0.5 ? B.z1 + 0.6 : null);
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

        if (route && fast && !p.swim && this.solidFace(route[0]) && this.live() < p.climby) {
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
        this.playCool -= dt;
        // The peck is a WHOLE-BODY act: the body crouches over the front
        // legs, the rear teeters up, and the head only closes the last
        // span - no telescoping necks.
        const peckDip = this.peck > 0 ? Math.sin(Math.PI * this.peck) : 0;
        // Coming down from a leap, rodents look to the ground and reach
        // with the front paws - the landing is led by the head and hands.
        const wantDip = (this.state === 'jump' && this.vh < -0.2 && !p.swim) ? 1 : 0;
        this.airDip = (this.airDip || 0) + (wantDip - (this.airDip || 0)) * Math.min(1, 7 * dt);

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
            case 'rear': {
                this.va *= Math.exp(-10 * dt); this.vb *= Math.exp(-10 * dt);
                const high = this.rearHigh ? 1.0 : 0.6;
                this.rearAmt = Math.min(high, this.rearAmt + 2.5 * dt);
                if (this.nibbleOn) this.nibble = Math.min(1, this.nibble + 3 * dt);
                if (this.timer <= 0) this.rest();
                break;
            }
            case 'fidget': {
                // Quick darty jitter: tiny lunges, snapped heading.
                dva = ch.swayA * 1.6; dvb = ch.swayB * 1.6;
                this.spin += (this.live() - 0.5) * 30 * dt;
                this.phase += 6 * dt;
                if (this.timer <= 0) this.rest();
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
                // The mood passes as suddenly as it arrived.
                if (this.live() < 0.35 * dt) { this.rest(); break; }
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
            case 'land': {
                // Glide down toward the landing spot.
                const da2 = this.ta - this.a, db2 = this.tb - this.b;
                const d2 = Math.hypot(da2, db2) || 1;
                const gl = Math.min(1, d2) * this.effSpeed * 0.4;
                dva = (da2 / d2) * gl; dvb = (db2 / d2) * gl;
                if (this.h - this.groundAt(this.a, this.b) < 0.18) {
                    this.state = 'perch';
                    this.timer = (this.p.perchTime || 2)
                        * (0.7 + 0.6 * this.live());
                }
                break;
            }
            case 'perch':
                this.va *= Math.exp(-8 * dt); this.vb *= Math.exp(-8 * dt);
                if (this.timer <= 0 && this.stamina > 0.85) {
                    this.vh += 1.6 + this.live();        // the takeoff hop
                    this.swimH = null;
                    this.rest();
                }
                break;
        }

        // Wing fuel: drains aloft (slower for the enduring), refills on
        // the ground. Exhaustion is not a choice: tired wings land NOW,
        // whatever the creature was doing. perchFold eases the wings shut.
        if (p.flyStyle) {
            if ((p.endurance ?? 1) < 1 && this.stamina < 0.2
                && !['land', 'perch', 'sleep', 'rise'].includes(this.state)) {
                this.state = 'land';
                this.ta = this.a + (this.live() * 2 - 1) * 2;
                this.tb = this.b + (this.live() * 2 - 1) * 2;
            }
            const gAlt = this.h - this.groundAt(this.a, this.b);
            const aloft = gAlt > 0.3;
            const drain = 0.05 * (1 - (p.endurance ?? 1));
            this.stamina = Math.min(1, Math.max(0,
                this.stamina + (aloft ? -drain : 0.15) * dt));
            const wantFold = this.state === 'perch' || this.state === 'sleep'
                || (this.state === 'rise' && gAlt < 0.3) ? 1 : 0;
            this.perchFold += (wantFold - this.perchFold) * Math.min(1, 5 * dt);
        }

        if (this.state !== 'rear') {
            this.rearAmt = Math.max(0, this.rearAmt - 2.2 * dt);
            this.nibble = Math.max(0, this.nibble - 3 * dt);
        }

        // The environment moves bodies: wind pushes (light things more),
        // a heavy gust can even wrench the frozen axis around.
        const wind = this.env ? this.env.wind : 0;
        if (wind && this.surface === 'floor') {
            const sail = p.swim
                ? (p.flyStyle === 'butterfly' ? 1.8 : p.flyStyle === 'wasp' ? 0.5 : 0.3)
                : 0.35 * (1.3 - p.scale);
            this.va += wind * sail * dt;
            if (Math.abs(wind) > 1.0) {
                this.spin += wind * 0.22 * (1.3 - p.scale) * dt;
            }
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

        const edgeFaces = this.surface === 'floor'
            ? { a0: 'wallL', a1: 'wallR', b0: 'front', b1: 'wallB' }
            : null;
        const edgeSolid = (e) => !edgeFaces || this.solidFace(edgeFaces[e]);
        if (this.surface !== 'floor'
            || (edgeSolid('a0') && edgeSolid('a1') && edgeSolid('b0') && edgeSolid('b1'))) {
            if (this.a < EDGE) { this.a = EDGE; this.hitEdge('a0'); }
            else if (this.a > S.A - EDGE) { this.a = S.A - EDGE; this.hitEdge('a1'); }
            if (this.b < EDGE) { this.b = EDGE; this.hitEdge('b0'); }
            else if (this.b > SURFACES[this.surface].B - EDGE) {
                this.b = SURFACES[this.surface].B - EDGE; this.hitEdge('b1');
            }
        } else {
            // Mixed faces: solid edges collide, open edges soft-steer.
            if (edgeSolid('a0') && this.a < EDGE) { this.a = EDGE; this.hitEdge('a0'); }
            if (edgeSolid('a1') && this.a > S.A - EDGE) { this.a = S.A - EDGE; this.hitEdge('a1'); }
            if (edgeSolid('b0') && this.b < EDGE) { this.b = EDGE; this.hitEdge('b0'); }
            if (edgeSolid('b1') && this.b > S.B - EDGE) { this.b = S.B - EDGE; this.hitEdge('b1'); }
            // Open world: no walls to hit. The land rises at the roam
            // rim and the creature is gently steered home - it wanders
            // back over the berm instead of bouncing off unseen glass.
            const B = this.bounds();
            const M = 1.0;
            const push = 4.5;
            if (this.a < B.x0 + M) this.va += (B.x0 + M - this.a) * push * dt;
            if (this.a > B.x1 - M) this.va -= (this.a - (B.x1 - M)) * push * dt;
            if (this.b < B.z0 + M) this.vb += (B.z0 + M - this.b) * push * dt;
            if (this.b > B.z1 - M) this.vb -= (this.b - (B.z1 - M)) * push * dt;
            // Failsafe rail just past the rim.
            this.a = Math.min(B.x1 + 0.6, Math.max(B.x0 - 0.6, this.a));
            this.b = Math.min(B.z1 + 0.6, Math.max(B.z0 - 0.6, this.b));
            // A runner aimed past the rim picks a new target inside.
            if (this.state === 'run'
                && (this.ta < B.x0 || this.ta > B.x1 || this.tb < B.z0 || this.tb > B.z1)
                && (this.a < B.x0 + M || this.a > B.x1 - M
                    || this.b < B.z0 + M || this.b > B.z1 - M)) {
                this.ta = B.x0 + (B.x1 - B.x0) * (0.15 + 0.7 * this.live());
                this.tb = B.z0 + (B.z1 - B.z0) * (0.15 + 0.7 * this.live());
            }
        }

        // Solid trunks: trees push the body out and around, never through.
        if (this.surface === 'floor' && this.terrainRef) {
            for (const tr of treesNear(this.terrainRef, this.a, this.b)) {
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
        if (p.swim) {
            // Swimmers and flyers: no gravity, no ground spring - a smooth
            // glide toward a wandering altitude, breathing with the buoy
            // voter. Perching flyers drop to the ground to rest.
            const lo = p.swimLo != null ? p.swimLo : 0.5;
            const hi = p.swimHi != null ? p.swimHi : Math.max(1.1, ROOM_H * 0.75);
            if (this.swimH == null || this.live() < 0.25 * dt) {
                this.swimH = lo + this.live() * Math.max(0.3, hi - lo);
            }
            const gFloor = this.groundAt(this.a, this.b);
            const perched = p.flyStyle
                && (this.state === 'sleep' || this.state === 'rise'
                    || this.state === 'perch' || this.state === 'land');
            let targetH2 = Math.max(gFloor + 0.4,
                Math.min(ROOM_H * 0.85, this.swimH))
                + 0.25 * this.ch.buoy;
            if (perched) targetH2 = gFloor + 0.1;
            if (p.flyStyle === 'butterfly' && !perched) {
                // The flutter: every wingbeat lifts, gravity answers.
                this.vh += Math.sin(t * 9 + this.phase) * 2.2 * dt;
            }
            this.vh += ((targetH2 - this.h) * 2.2 - this.vh * 2.5) * dt
                + (this.state === 'jump' ? 0 : 0);
            if (this.state === 'jump') this.vh += 2.0 * dt; // the breach
            this.h += this.vh * dt;
            const hMin2 = perched ? gFloor + 0.06 : gFloor + 0.25;
            if (this.h < hMin2) { this.h = hMin2; this.vh = Math.abs(this.vh) * 0.4; }
            if (this.h > ROOM_H * 0.9) { this.h = ROOM_H * 0.9; this.vh = -Math.abs(this.vh) * 0.4; }
            if (this.state === 'jump' && this.timer <= 0) this.rest();
        } else {
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
        // Low rear = SITTING on the haunches (chest up, rear sunk);
        // rearHigh = the full upright stretch to look around.
        const rearLift = 1 + this.rearAmt * (this.rearHigh ? 0.85 : 0.22);
        const peckCrouch = 1 - 0.38 * peckDip * (1 - this.lie);
        const standLive = standH * posture * rearLift * peckCrouch * (1 + this.mods.stance);
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

        if (this.surface === 'floor' && !p.swim && this.solidFace('ceiling')
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
        }

        // The axis is FROZEN unless the creature's own locomotion turns
        // it: below a real stride speed there is no torque at all - no
        // effortless rotating in place. (The environment may still shove
        // it: see the wind coupling above.)
        const sp = Math.hypot(this.va, this.vb);
        const spThresh = 0.22 * this.effSpeed;
        if (sp > spThresh) {
            const want = Math.atan2(this.vb, this.va);
            let diff = want - this.heading;
            while (diff > Math.PI) diff -= 2 * Math.PI;
            while (diff < -Math.PI) diff += 2 * Math.PI;
            const authority = Math.min(1, (sp - spThresh) / (this.effSpeed * 0.5));
            this.spin += (diff * (6 + 16 * p.speedy) * authority
                - this.spin * (5 + 4 * p.fluid)) * dt;
        } else {
            this.spin -= this.spin * 8 * dt;
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
                Math.max(0.04, prevL.h * (1 - 0.08 * i)
                    * (1 - this.rearAmt * 0.55 * ((i + 1) / this.chain.length))
                    + peckDip * 0.14 * s * ((i + 1) / this.chain.length)));
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
            && !p.flyStyle && this.live() < p.pecky) {
            this.peck = 1;
            this.peckCool = 1.5 + this.expo(3.0);
        }
        // Nibbling: the head bobs down toward the held paws.
        const nib = this.nibble * (0.4 + 0.2 * Math.sin(t * 7));

        // Head follower rides the nose, swiveled by the yaw, dipped by
        // the peck.
        const face = this.heading + this.headYaw;
        const fa2 = Math.cos(face), fb2 = Math.sin(face);
        const nose = toWorld(S,
            this.a + fa2 * bodyR * p.neck * p.elong * (1 + 0.12 * peckDip),
            this.b + fb2 * bodyR * p.neck * p.elong * (1 + 0.12 * peckDip),
            Math.max(0.04, this.h + (0.12 + this.rearAmt * 0.45) * s * (1 - this.lie)
                - peckDip * this.h * 0.45 - nib * this.h * 0.4
                - (this.has('paws') ? (this.airDip || 0) * this.h * 0.3 : 0)));
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
            if (leg.front && this.rearAmt > 0.2) {
                leg.planted = null; leg.swing = null;
                continue;
            }
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

        const bob = Math.sin(this.phase * 2) * 0.03 * s
            * (p.bounder ? 2.4 : 1) * Math.min(1, speed + 0.1);
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
                let ep = add(pos, J.faceW, headR * 0.38);
                ep = add(ep, J.browW, lat * 1.15);
                ep = add(ep, J.upW, headR * 0.12 + stalk);
                eyes.push({ pos: ep, r: headR * p.eyeSize * (1 + 0.4 * (e % 2 === 0 ? 0 : -0.3)) });
            }
            J.heads.push({ pos, eyes, stalk, headR });
        }

        const gBody = this.groundAt(this.a, this.b);
        const grounded = this.h - gBody < p.stance * s * 1.3;
        J.legs = [];
        const pending = [];
        for (const leg of this.legs) {
            if (leg.front && this.rearAmt > 0.2) {
                leg.planted = null; leg.swing = null;
                continue;
            }
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
            if (leg.front && this.rearAmt > 0.2) {
                // Paws held to the chest - and busy: they groom and fuss
                // in quick alternation, more so while nibbling.
                const ha2 = Math.cos(this.heading), hb2 = Math.sin(this.heading);
                const lat = Math.sin(leg.sweep) * bodyR * 0.45;
                const fuss = Math.sin(t * 11 + (leg.side > 0 ? 0 : Math.PI))
                    * (0.05 + 0.08 * this.nibble);
                foot = {
                    a: this.a + ha2 * bodyR * (0.9 + fuss * 0.5) - hb2 * lat,
                    b: this.b + hb2 * bodyR * (0.9 + fuss * 0.5) + ha2 * lat,
                    h: bodyH * (0.62 + fuss
                        + 0.12 * Math.sin(this.phase * 2) * this.nibble),
                };
                toeDir = this.heading;
            } else if (leg.hind && this.rearAmt > 0.4) {
                // Sitting: the hind feet fold forward under the haunches.
                const ha2 = Math.cos(this.heading), hb2 = Math.sin(this.heading);
                const lat = Math.sin(leg.sweep) * bodyR * 0.8;
                const fa3 = this.a + ha2 * reach * 0.15 - hb2 * lat;
                const fb3 = this.b + hb2 * reach * 0.15 + ha2 * lat;
                foot = { a: fa3, b: fb3, h: this.groundAt(fa3, fb3) };
                toeDir = this.heading;
            } else if (this.lie >= 0.7) {
                const fa3 = host.a + Math.cos(ang) * reach * 1.25;
                const fb3 = host.b + Math.sin(ang) * reach * 1.25;
                foot = { a: fa3, b: fb3, h: this.groundAt(fa3, fb3) + 0.01 };
                toeDir = ang;
            } else if (!grounded) {
                const back = leg.hind ? 0.9 : 0;   // hind legs trail in flight
                // Falling: the front feet stretch down and ahead to take
                // the landing first.
                const reachF = leg.front ? (this.airDip || 0) : 0;
                foot = {
                    a: host.a + Math.cos(ang) * reach * 0.5
                        - Math.cos(host.dir) * reach * back
                        + Math.cos(host.dir) * reach * 0.5 * reachF,
                    b: host.b + Math.sin(ang) * reach * 0.5
                        - Math.sin(host.dir) * reach * back
                        + Math.sin(host.dir) * reach * 0.5 * reachF,
                    h: bodyH - seg * (leg.hind ? 0.3 : 0.7 + 0.5 * reachF),
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
            pending.push({ leg, hip, foot, toeDir, seg });
        }

        // Self-collision: the body is its own boundary. Feet never cross
        // the midline to the other side, and same-side feet keep a gap -
        // legs stop passing through each other before the IK ever runs.
        if (this.lie < 0.7) {
            const ha2 = Math.cos(this.heading), hb2 = Math.sin(this.heading);
            const minLat = bodyR * 0.35;
            for (const pd of pending) {
                const side = Math.sign(pd.leg.sweep) || 1;
                const ra2 = pd.foot.a - this.a, rb2 = pd.foot.b - this.b;
                const lat = -hb2 * ra2 + ha2 * rb2;
                if (side * lat < minLat) {
                    const fix = (minLat - side * lat) * side;
                    pd.foot.a += -hb2 * fix;
                    pd.foot.b += ha2 * fix;
                }
            }
            const sep = 0.22 * s;
            for (let i = 0; i < pending.length; i++) {
                for (let k2 = i + 1; k2 < pending.length; k2++) {
                    const A2 = pending[i], B2 = pending[k2];
                    if (Math.sign(A2.leg.sweep) !== Math.sign(B2.leg.sweep)) continue;
                    let dxf = B2.foot.a - A2.foot.a, dbf = B2.foot.b - A2.foot.b;
                    const df = Math.hypot(dxf, dbf);
                    if (df < sep && df > 1e-4) {
                        const push = (sep - df) / 2 / df;
                        A2.foot.a -= dxf * push; A2.foot.b -= dbf * push;
                        B2.foot.a += dxf * push; B2.foot.b += dbf * push;
                    }
                }
            }
        }

        for (const pd of pending) {
            const { leg, hip, foot, toeDir, seg } = pd;
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
/* Sliding-window world: beyond the origin room, terrain generates in
   seeded chunks on demand - heights, rocks, mounds, trees, grass, grain.
   Deterministic per (seed, chunk), cached, evicted when the cache grows;
   re-entering a region regenerates it identically. */
const CH = 8;

function chunkData(terrain, ci, cj) {
    if (!terrain._chunks) terrain._chunks = new Map();
    const key = ci + ',' + cj;
    let c = terrain._chunks.get(key);
    if (c) return c;
    if (terrain._chunks.size > 600) terrain._chunks.clear();
    const rng = mulberry32(fnv1a(terrain.chunkSalt + ':ck:' + key));
    const f = terrain.featRef;
    const ox = ci * CH, oz = cj * CH;
    const inRoom = (x, z) => x >= 0 && x <= ROOM_W && z >= 0 && z <= ROOM_D;
    const scatter = (n2, mk) => {
        const out = [];
        for (let i = 0; i < n2; i++) {
            const x = ox + rng() * CH, z = oz + rng() * CH;
            const item = mk(x, z);
            if (!inRoom(x, z)) out.push(item);   // origin room keeps its own
        }
        return out;
    };
    c = {
        gauss: scatter(rng() < 0.55 ? 1 : 2, (x, z) => ({
            x, z, r: 1.2 + rng() * 2.6, amp: (rng() - 0.35) * 1.1,
        })),
        boulders: scatter(rng() < 0.12 + f.rocks * 0.05 ? 1 : 0, (x, z) => ({
            x, z, r: 0.45 + rng() * 0.6, squash: 0.55 + rng() * 0.3,
        })),
        mounds: scatter(rng() < 0.3 + f.mounds * 0.06 ? 1 : 0, (x, z) => ({
            x, z, r: 0.5 + rng() * 0.9, h: 0.12 + rng() * 0.3,
        })),
        trees: scatter(rng() < 0.08 + f.trees * 0.07 ? 1 : 0, (x, z) => ({
            x, z, h: 1.6 + rng() * 1.6, r: 0.4 + rng() * 0.5,
        })),
        tufts: scatter(Math.round(rng() * (1 + f.tufts * 0.15)), (x, z) => ({
            x, z, s: 0.08 + rng() * 0.14, lean: (rng() - 0.5) * 0.6,
        })),
        grain: scatter(3 + Math.floor(rng() * 4), (x, z) => ({
            x, z, r: 0.6 + rng() * 1.4, al: 0.02 + rng() * 0.05,
        })),
    };
    terrain._chunks.set(key, c);
    return c;
}

function chunksAround(terrain, a, b, radius = 1) {
    const ci0 = Math.floor(a / CH), cj0 = Math.floor(b / CH);
    const out = [];
    for (let i = -radius; i <= radius; i++) {
        for (let j = -radius; j <= radius; j++) {
            out.push(chunkData(terrain, ci0 + i, cj0 + j));
        }
    }
    return out;
}

function treesNear(terrain, a, b) {
    const out = terrain.trees.slice();
    if (terrain.infinite) {
        for (const c of chunksAround(terrain, a, b)) out.push(...c.trees);
    }
    return out;
}

function buildTerrain(rng, hab, salt = 'praxis') {
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
        featRef: f,
        chunkSalt: salt,
        _fillRoam: true,
        // Geometry is built face by face, DISCRETELY: each wall and the
        // ceiling rolls its own material from the archetype weights. One
        // scene has a glass right wall and nothing on the left; the cube
        // only exists when every roll lands solid.
        facesMat: (() => {
            const w = hab.weights;
            const wallTable = [
                ['none', 0.3 + w.void * 1.6 + w.thicket * 0.7],
                ['box', w.gym * 1.3 + w.tank * 0.4],
                ['glass', w.tank * 1.8],
                ['rock', w.tunnel * 1.8 + 0.12],
                ['hedge', w.thicket * 1.5],
                ['machine', w.machine * 1.6],
            ];
            const ceilTable = [
                ['none', 0.5 + w.void * 1.6 + w.thicket * 1.3 + w.tank * 0.5],
                ['box', w.gym * 0.8 + w.tank * 0.4],
                ['rock', w.tunnel * 2.2],
                ['machine', w.machine * 1.6],
            ];
            const roll2 = (table) => {
                const tot = table.reduce((a2, e) => a2 + e[1], 0);
                let r = rng() * tot;
                for (const [m2, wt] of table) { r -= wt; if (r <= 0) return m2; }
                return 'none';
            };
            return {
                wallB: roll2(wallTable),
                wallL: roll2(wallTable),
                wallR: roll2(wallTable),
                ceiling: roll2(ceilTable),
                front: w.tank > 0.35 ? 'glass' : 'none',
            };
        })(),
        // Asymmetric macro relief: big signed gaussians - a high hill far
        // right, a sunken valley near left, wherever the rolls land.
        macro: Array.from({ length: 3 + Math.floor(rng() * 3) }, () => ({
            x: ROOM_W * (rng() * 1.5 - 0.25),
            z: ROOM_D * (rng() * 1.2),
            r: 1.5 + rng() * 3.5,
            amp: (rng() - 0.35) * 1.3 * (1 - Math.min(1, f.shell) * 0.7),
        })),
        tilt: {
            gx: (rng() - 0.5) * 0.08 * (1 - Math.min(1, f.shell)),
            gz: (rng() - 0.5) * 0.08 * (1 - Math.min(1, f.shell)),
        },
        // The floor is a heightmap, not a plane: a seeded low-frequency
        // rolling field that physics and rendering both sample. Tanks
        // stay flat; open country rolls.
        roll: {
            amp: (1 - Math.min(1, f.shell)) * (0.12 + rng() * 0.4),
            k1: 0.25 + rng() * 0.45, k2: 0.25 + rng() * 0.45,
            k3: 0.15 + rng() * 0.35, k4: 0.15 + rng() * 0.35,
            p1: rng() * Math.PI * 2, p2: rng() * Math.PI * 2, p3: rng() * Math.PI * 2,
        },
        // How far the world extends: each open edge runs off-stage.
        roam: null, // filled below from facesMat
        // Open worlds: ground swells into a berm at the old boundary and
        // rolls off in ridgelines - no edge, just rising land.
        rim: (1 - Math.min(1, f.shell)) * (0.2 + rng() * 0.45),
        ridges: Math.min(1, f.shell) < 0.3
            ? Array.from({ length: 2 + n(1.6) }, () => ({
                z: ROOM_D * (0.95 + rng() * 2.6),
                amp: 0.25 + rng() * 0.55,
                waves: 1 + Math.floor(rng() * 3),
                ph: rng() * Math.PI * 2,
                jag: Array.from({ length: 11 }, () => (rng() - 0.5) * 0.4),
            }))
            : [],
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
            z: ROOM_D * (1.6 + rng() * 1.6),
            pts: Array.from({ length: 12 }, () => 0.6 + rng() * 1.4),
            trunks: Array.from({ length: 3 + Math.floor(rng() * 4) }, () => ({
                x: rng() * 1.4 - 0.2, h: 0.5 + rng() * 0.9,
            })),
        } : null,
        // The world continues past the bounds: faded scatter off-stage.
        outTufts: Array.from({ length: n((f.tufts + f.stalks) * 0.9) }, () => ({
            x: ROOM_W * (rng() * 3.0 - 1.0),
            z: ROOM_D * (1.0 + rng() * 1.8),
            s: 0.1 + rng() * 0.18, lean: (rng() - 0.5) * 0.6,
        })),
        outRocks: Array.from({ length: n((f.rocks + f.mounds) * 0.8) }, () => ({
            x: ROOM_W * (rng() * 3.0 - 1.0),
            z: ROOM_D * (1.05 + rng() * 1.9),
            r: 0.3 + rng() * 0.7,
        })),
    };
}

function finishTerrain(terrain) {
    const fm = terrain.facesMat;
    // Any missing wall opens that direction to the infinite world.
    terrain.infinite = fm.wallL === 'none' || fm.wallR === 'none' || fm.wallB === 'none';
    if (terrain.infinite) terrain.rim = 0;   // no berm: nothing to fence
    terrain.roam = {
        x0: fm.wallL === 'none' ? -ROOM_W * 0.25 : 0,
        x1: fm.wallR === 'none' ? ROOM_W * 1.25 : ROOM_W,
        z0: 0.15,
        z1: fm.wallB === 'none' ? ROOM_D * 1.3 : ROOM_D,
    };
    // The RENDERED ground: where no face stands, the plane runs far
    // beyond anything the camera frames. Only the box ends at the box.
    terrain.vast = {
        x0: fm.wallL === 'none' ? -ROOM_W * 2.5 : 0,
        x1: fm.wallR === 'none' ? ROOM_W * 3.5 : ROOM_W,
        // No front pane: the ground continues toward (and past) the
        // camera, so its near edge falls outside the viewport entirely.
        z0: fm.front === 'none' ? -ROOM_D * 0.45 : 0.15,
        z1: fm.wallB === 'none' ? ROOM_D * 5.5 : ROOM_D,
    };
    return terrain;
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
    if (terrain.macro) {
        for (const m of terrain.macro) {
            const d2 = (a - m.x) ** 2 + (b - m.z) ** 2;
            g += m.amp * Math.exp(-d2 / (2 * m.r * m.r));
        }
    }
    if (terrain.tilt) {
        g += terrain.tilt.gx * (a - ROOM_W / 2) + terrain.tilt.gz * (b - ROOM_D / 2);
    }
    if (terrain.infinite) {
        for (const c of chunksAround(terrain, a, b)) {
            for (const g2 of c.gauss) {
                const d2 = (a - g2.x) ** 2 + (b - g2.z) ** 2;
                g += g2.amp * Math.exp(-d2 / (2 * g2.r * g2.r));
            }
            for (const bo of c.boulders) {
                const d = Math.hypot(a - bo.x, b - bo.z);
                const R = bo.r * 1.15;
                if (d < R) g = Math.max(g, bo.r * bo.squash * 0.85 * Math.cos((d / R) * Math.PI / 2));
            }
            for (const mo of c.mounds) {
                const d2 = (a - mo.x) ** 2 + (b - mo.z) ** 2;
                g = Math.max(g, mo.h * Math.exp(-d2 / (2 * mo.r * mo.r * 0.35)));
            }
        }
    }
    const roll = terrain.roll;
    if (roll && roll.amp > 0.02) {
        g += roll.amp * (
            Math.sin(a * roll.k1 + roll.p1) * Math.sin(b * roll.k2 + roll.p2) * 0.6
            + Math.sin(a * roll.k3 + b * roll.k4 + roll.p3) * 0.4
            + 1.0
        ) * 0.5;
    }
    if (terrain.rim > 0.05) {
        const rm = terrain.roam || { x0: 0, x1: ROOM_W, z0: 0, z1: ROOM_D };
        const dEdge = Math.min(a - rm.x0, b - rm.z0, rm.x1 - a, rm.z1 - b);
        if (dEdge < 1.6) {
            const r2 = Math.min(1.3, 1 - dEdge / 1.6);
            g += terrain.rim * r2 * r2;
        }
    }
    return Math.max(-0.7, g);
}

/* Stable per-key decay: a decayed room has crumbled away some of its
   lines, and which ones never changes for a given seed. */
function survives(key, hab, salt) {
    return ((fnv1a(key + salt) % 1000) / 1000) >= hab.decay;
}

/* Emit the room as depth-sorted items. Long spanning lines split into
   thirds so the creature can pass in front of the near part of an edge
   and behind the far part. */
function envItems(hab, salt, ink, P, terrain, colors, t, cam) {
    const items = [];
    const gy = (x, z) => groundHeight(terrain, x, z);
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
    // ---- geometry, face by face ----
    const fm = terrain.facesMat;
    const solid = (m) => m && m !== 'none';
    const rollAmp = (terrain.roll ? terrain.roll.amp : 0)
        + (terrain.macro ? terrain.macro.reduce((a2, m) => a2 + Math.abs(m.amp), 0) / 4 : 0);

    // The ground: flowing contour lines over the heightmap whenever the
    // land actually rolls; straight floor edges only along solid faces.
    if (rollAmp > 0.05) {
        const vt = terrain.infinite
            ? { x0: cam.x - 32, x1: cam.x + 32,
                z0: cam.z - ROOM_D * 0.4, z1: cam.z + 38 }
            : terrain.vast;
        const LINES = 9, PTS = 19;
        for (let li = 0; li < LINES; li++) {
            // Quadratic spacing: dense near the viewer, reaching far out.
            const fr = (li + 0.5) / LINES;
            const z = vt.z0 + (vt.z1 - vt.z0) * fr * fr;
            const fade = 1 - 0.75 * fr;
            items.push({
                z,
                draw: (ctx) => {
                    ctx.globalAlpha = 0.11 * hab.cohesion * fade;
                    let prev = null;
                    for (let k = 0; k < PTS; k++) {
                        const x = vt.x0 + (vt.x1 - vt.x0) * (k / (PTS - 1));
                        const pt = P(x, gy(x, z) + 0.01, z);
                        if (prev) ink.bone(ctx, prev, pt, `flow${li}-${k}`, 0.6);
                        prev = pt;
                    }
                },
            });
        }
    }
    if (solid(fm.front)) seg3(0, 0, 0, ROOM_W, 0, 0, 'fl-near', 0.4, 1.0);
    if (solid(fm.wallB)) seg3(0, 0, ROOM_D, ROOM_W, 0, ROOM_D, 'fl-far', 0.25, 0.8);
    if (solid(fm.wallL)) seg3(0, 0, 0, 0, 0, ROOM_D, 'fl-left', 0.25, 0.8);
    if (solid(fm.wallR)) seg3(ROOM_W, 0, 0, ROOM_W, 0, ROOM_D, 'fl-right', 0.25, 0.8);

    // Each wall face is its own construction in its own plane.
    const facePlanes = {
        wallB: { pt: (u, v) => P(u * ROOM_W, v, ROOM_D), zOf: ROOM_D, span: ROOM_W },
        wallL: { pt: (u, v) => P(0, v, u * ROOM_D), zOf: ROOM_D * 0.55, span: ROOM_D },
        wallR: { pt: (u, v) => P(ROOM_W, v, u * ROOM_D), zOf: ROOM_D * 0.55, span: ROOM_D },
    };
    for (const fname of ['wallB', 'wallL', 'wallR']) {
        const mat = fm[fname];
        if (!solid(mat)) continue;
        const FP = facePlanes[fname];
        items.push({
            z: FP.zOf,
            draw: (ctx) => {
                if (mat === 'box' || mat === 'glass') {
                    ctx.globalAlpha = 0.2 * hab.cohesion;
                    ink.bone(ctx, FP.pt(0, 0), FP.pt(0, ROOM_H), fname + '-p0', 0.8);
                    ink.bone(ctx, FP.pt(1, 0), FP.pt(1, ROOM_H), fname + '-p1', 0.8);
                    ink.bone(ctx, FP.pt(0, ROOM_H), FP.pt(1, ROOM_H), fname + '-top', 0.8);
                    if (mat === 'glass') {
                        ctx.globalAlpha = 0.05;
                        ctx.fillStyle = colors.grain;
                        const q = [FP.pt(0, ROOM_H), FP.pt(1, ROOM_H), FP.pt(1, 0), FP.pt(0, 0)];
                        ctx.beginPath();
                        q.forEach((p2, k) => ctx[k ? 'lineTo' : 'moveTo'](p2.x, p2.y));
                        ctx.closePath();
                        ctx.fill();
                        ctx.globalAlpha = 0.16 * hab.cohesion;
                        ink.bone(ctx, FP.pt(0.2, ROOM_H * 0.9), FP.pt(0.45, ROOM_H * 0.15), fname + '-gl', 0.5);
                    }
                } else if (mat === 'rock') {
                    const pts = [FP.pt(-0.04, 0)];
                    for (let k = 0; k < 6; k++) {
                        const fr = (k + 0.5) / 6;
                        pts.push(FP.pt(fr, ROOM_H * (0.35 + 0.55 * Math.abs(Math.sin(fr * 7 + FP.span)))));
                    }
                    pts.push(FP.pt(1.04, 0));
                    ctx.globalAlpha = 0.06;
                    ctx.fillStyle = colors.grain;
                    ctx.beginPath();
                    pts.forEach((p2, k) => ctx[k ? 'lineTo' : 'moveTo'](p2.x, p2.y));
                    ctx.closePath();
                    ctx.fill();
                    ctx.globalAlpha = 0.26 * hab.cohesion;
                    ink.loop(ctx, pts, fname + '-rk', 1.3);
                } else if (mat === 'hedge') {
                    ctx.globalAlpha = 0.2 * hab.cohesion;
                    for (let k = 0; k < 8; k++) {
                        const u = (k + 0.5) / 8;
                        const hgt = ROOM_H * (0.3 + 0.35 * Math.abs(Math.sin(u * 9 + FP.span * 2)));
                        ink.bone(ctx, FP.pt(u, 0), FP.pt(u + 0.02, hgt), fname + '-hg' + k, 1.0);
                    }
                } else if (mat === 'machine') {
                    ctx.globalAlpha = 0.22 * hab.cohesion;
                    ink.bone(ctx, FP.pt(0, 0), FP.pt(0.6, ROOM_H * 0.85), fname + '-bm0', 0.6);
                    ink.bone(ctx, FP.pt(1, 0), FP.pt(0.4, ROOM_H * 0.8), fname + '-bm1', 0.6);
                    ink.bone(ctx, FP.pt(0.1, ROOM_H * 0.55), FP.pt(0.9, ROOM_H * 0.55), fname + '-bm2', 0.6);
                }
            },
        });
    }
    // The ceiling face, when it exists at all.
    if (solid(fm.ceiling)) {
        items.push({
            z: ROOM_D * 0.98,
            draw: (ctx) => {
                if (fm.ceiling === 'box') {
                    ctx.globalAlpha = 0.16 * hab.cohesion;
                    ink.bone(ctx, P(0, ROOM_H, 0), P(ROOM_W, ROOM_H, 0), 'ce-near', 0.7);
                    ink.bone(ctx, P(0, ROOM_H, ROOM_D), P(ROOM_W, ROOM_H, ROOM_D), 'ce-far', 0.7);
                    ink.bone(ctx, P(0, ROOM_H, 0), P(0, ROOM_H, ROOM_D), 'ce-l', 0.7);
                    ink.bone(ctx, P(ROOM_W, ROOM_H, 0), P(ROOM_W, ROOM_H, ROOM_D), 'ce-r', 0.7);
                } else if (fm.ceiling === 'rock') {
                    ctx.globalAlpha = 0.2 * hab.cohesion;
                    for (let i = 0; i < 3; i++) {
                        let prev = null;
                        for (let k = 0; k <= 7; k++) {
                            const x = (k / 7) * ROOM_W;
                            const dip = -Math.sin((k / 7) * Math.PI) * (0.3 + i * 0.18);
                            const pt = P(x, ROOM_H - 0.2 - i * 0.22 + dip, ROOM_D * (0.55 + i * 0.15));
                            if (prev) ink.bone(ctx, prev, pt, `cerk${i}-${k}`, 0.8);
                            prev = pt;
                        }
                    }
                } else if (fm.ceiling === 'machine') {
                    ctx.globalAlpha = 0.18 * hab.cohesion;
                    for (let i = 1; i < 4; i++) {
                        const x = (ROOM_W * i) / 4;
                        ink.bone(ctx, P(x, ROOM_H, 0), P(x, ROOM_H, ROOM_D), 'cebm' + i, 0.6);
                    }
                    ink.bone(ctx, P(0, ROOM_H, ROOM_D / 2), P(ROOM_W, ROOM_H, ROOM_D / 2), 'cebmx', 0.6);
                }
            },
        });
    }
    // ---- blended archetype furniture ----
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
            z: st.z, x: st.x,
            draw: (ctx) => {
                ctx.globalAlpha = 0.28 * hab.cohesion;
                const g0 = gy(st.x, st.z);
                const top = P(st.x + st.lean, g0 + st.h, st.z);
                ink.bone(ctx, P(st.x, g0, st.z), top, 'stalk' + i, 1.1);
                ctx.globalAlpha = 0.18 * hab.cohesion;
                ink.bone(ctx, top, P(st.x + st.lean + 0.3, g0 + st.h - 0.35, st.z), 'frond-a' + i, 0.9);
                ink.bone(ctx, top, P(st.x + st.lean - 0.3, g0 + st.h - 0.3, st.z), 'frond-b' + i, 0.9);
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
            z: pb.z, x: pb.x,
            draw: (ctx) => {
                ctx.globalAlpha = 0.25 * hab.cohesion;
                const pts = [];
                const g0 = gy(pb.x, pb.z);
                for (let k = 0; k < 6; k++) {
                    const a = (k / 6) * Math.PI * 2;
                    pts.push(P(pb.x + Math.cos(a) * pb.r,
                        g0 + Math.max(0, Math.sin(a) * pb.r * 0.6), pb.z));
                }
                ink.loop(ctx, pts, 'pebble' + i, 0.5);
            },
        });
    });
    // Machinery: girders rake across the back, pylons stand braced.
    terrain.beams.forEach((bm, i) => {
        if (!survives('beam' + i, hab, salt)) return;
        items.push({
            z: bm.z, x: (bm.x0 + bm.x1) / 2,
            draw: (ctx) => {
                ctx.globalAlpha = 0.26 * hab.cohesion;
                ink.bone(ctx, P(bm.x0, gy(bm.x0, bm.z) + bm.h0, bm.z),
                    P(bm.x1, gy(bm.x1, bm.z) + bm.h1, bm.z), 'beam' + i, 0.6);
            },
        });
    });
    terrain.pylons.forEach((py, i) => {
        if (!survives('pylon' + i, hab, salt)) return;
        items.push({
            z: py.z, x: py.x,
            draw: (ctx) => {
                ctx.globalAlpha = 0.30 * hab.cohesion;
                const g0 = gy(py.x, py.z);
                ink.bone(ctx, P(py.x, g0, py.z), P(py.x, g0 + py.h, py.z), 'pylon' + i, 0.6);
                if (py.brace) {
                    ctx.globalAlpha = 0.18 * hab.cohesion;
                    ink.bone(ctx, P(py.x - 0.5, g0, py.z), P(py.x, g0 + py.h * 0.55, py.z), 'pybr-a' + i, 0.5);
                    ink.bone(ctx, P(py.x + 0.5, g0, py.z), P(py.x, g0 + py.h * 0.55, py.z), 'pybr-b' + i, 0.5);
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
    // Rolling ridgelines where a far edge would have been.
    terrain.ridges.forEach((rd, i) => {
        items.push({
            z: rd.z,
            draw: (ctx) => {
                ctx.globalAlpha = 0.18 * hab.cohesion;
                let prev = null;
                for (let k = 0; k < rd.jag.length; k++) {
                    const fr = k / (rd.jag.length - 1);
                    const x = ROOM_W * (fr * 4.0 - 1.5);
                    const y = Math.max(0.02,
                        rd.amp * (Math.sin(fr * Math.PI * rd.waves + rd.ph) * 0.5 + 0.5)
                        * (1 + rd.jag[k]));
                    const pt = P(x, y, rd.z);
                    if (prev) ink.bone(ctx, prev, pt, `ridge${i}-${k}`, 1.0);
                    prev = pt;
                }
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
                    const x = ROOM_W * ((k / (tl.pts.length - 1)) * 4.5 - 1.75);
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
            z: tf.z, x: tf.x,
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
            z: r.z, x: r.x,
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

    // The sliding window: chunk furniture around the camera, drawn
    // from the same seeded store the physics reads.
    if (terrain.infinite) {
        const ci0 = Math.floor((cam.x - 26) / CH), ci1 = Math.ceil((cam.x + 26) / CH);
        const cj0 = Math.floor(Math.max(cam.z - 2, -1e8) / CH), cj1 = Math.ceil((cam.z + 36) / CH);
        for (let ci = ci0; ci <= ci1; ci++) {
            for (let cj = cj0; cj <= cj1; cj++) {
                const c = chunkData(terrain, ci, cj);
                const ck = ci + ',' + cj;
                c.tufts.forEach((tf, i) => {
                    items.push({
                        z: tf.z, x: tf.x,
                        draw: (ctx) => {
                            ctx.globalAlpha = 0.2 * hab.cohesion;
                            const g0 = gy(tf.x, tf.z);
                            const base = P(tf.x, g0, tf.z);
                            for (let k = -1; k <= 1; k++) {
                                ink.bone(ctx, base, P(tf.x + tf.lean * tf.s * 3 + k * tf.s * 0.8,
                                    g0 + tf.s * (2.2 - Math.abs(k) * 0.6), tf.z), `ct${ck}-${i}-${k}`, 0.5);
                            }
                        },
                    });
                });
                c.boulders.forEach((bo, i) => {
                    items.push({
                        z: bo.z, x: bo.x,
                        draw: (ctx) => {
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
                            ctx.globalAlpha = 0.32 * hab.cohesion;
                            ink.loop(ctx, pts, `cb${ck}-${i}`, 1.0);
                        },
                    });
                });
                c.mounds.forEach((mo, i) => {
                    items.push({
                        z: mo.z, x: mo.x,
                        draw: (ctx) => {
                            ctx.globalAlpha = 0.2 * hab.cohesion;
                            for (let ring = 0; ring < 2; ring++) {
                                const rr = mo.r * (1 - ring * 0.4);
                                const hh = mo.h * (1 - ring * 0.35);
                                let prev = null;
                                for (let k = 0; k <= 6; k++) {
                                    const x = mo.x - rr + (k / 6) * rr * 2;
                                    const pt = P(x, Math.sin((k / 6) * Math.PI) * hh + gy(x, mo.z) * 0, mo.z);
                                    if (prev) ink.bone(ctx, prev, pt, `cm${ck}-${i}-${ring}-${k}`, 0.7);
                                    prev = pt;
                                }
                            }
                        },
                    });
                });
                c.trees.forEach((tr, i) => {
                    items.push({
                        z: tr.z, x: tr.x,
                        draw: (ctx) => {
                            ctx.globalAlpha = 0.3 * hab.cohesion;
                            const g0 = gy(tr.x, tr.z);
                            const top = P(tr.x, g0 + tr.h, tr.z);
                            ink.bone(ctx, P(tr.x, g0, tr.z), top, `ctr${ck}-${i}`, 1.0);
                            const pts = [];
                            for (let k = 0; k < 9; k++) {
                                const a = (k / 9) * Math.PI * 2;
                                const rr = tr.r * (1 + 0.22 * Math.sin(a * 4 + i * 2));
                                pts.push(P(tr.x + Math.cos(a) * rr, g0 + tr.h + Math.sin(a) * rr * 0.75, tr.z));
                            }
                            ctx.globalAlpha = 1;
                            ctx.fillStyle = colors.paper;
                            ctx.beginPath();
                            pts.forEach((p2, k) => ctx[k ? 'lineTo' : 'moveTo'](p2.x, p2.y));
                            ctx.closePath();
                            ctx.fill();
                            ctx.globalAlpha = 0.28 * hab.cohesion;
                            ink.loop(ctx, pts, `cc${ck}-${i}`, 1.1);
                        },
                    });
                });
                c.grain.forEach((d, i) => {
                    items.push({
                        z: d.z, x: d.x,
                        draw: (ctx) => {
                            const pt = P(d.x, gy(d.x, d.z) + 0.01, d.z);
                            ctx.globalAlpha = d.al;
                            ctx.fillStyle = colors.grain;
                            ctx.beginPath();
                            ctx.arc(pt.x, pt.y, d.r * pt.d / 90, 0, Math.PI * 2);
                            ctx.fill();
                        },
                    });
                });
            }
        }
    }

    return items;
}

/* Animated ambience rebuilt every frame (cheap, a handful of items). */
function envDynamicItems(terrain, P, colors, t) {
    const items = [];
    // Motes: tiny fauna drifting slow ellipses in the air.
    terrain.motes.forEach((mo, i) => {
        items.push({
            z: mo.z, x: mo.x,
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

// -------------------------------------------------------------- weather

/* The sky's genome: one discrete material roll (weighted by the same
   archetype mixture as the room) and continuous dials - density from
   drizzle to downpour, signed wind, gust depth, and a slow swell that
   makes the whole system ebb and flow like a tide. A clear-sky roll
   yields pure stillness. Tank-leaning scenes weather UNDERWATER: bubbles
   on a drifting current. */
function sampleWeather(rng, hab, facesMat) {
    const t = genome(rng, 4, 6);
    const w = hab.weights;
    const indoor = facesMat.ceiling !== 'none';
    const clear = rng() < 0.22;
    const table = [
        ['rain', indoor ? 0 : 0.5 + w.thicket + w.gym * 0.4],
        ['snow', indoor ? 0 : 0.35 + w.void * 1.2],
        ['ash', 0.15 + w.tunnel * 1.3 + w.machine * 0.5],
        ['dust', 0.25 + w.machine * 1.1 + w.tunnel * 0.4],
        ['leaves', indoor ? 0 : w.thicket * 1.6],
        ['bubbles', w.tank * 1.9],
    ];
    const tot = table.reduce((a, e) => a + e[1], 0);
    let r = rng() * tot;
    let material = 'dust';
    for (const [m, wt] of table) { r -= wt; if (r <= 0) { material = m; break; } }
    return {
        material,
        clear,
        density: clear ? 0 : 0.15 + t() * 0.85,
        wind: (t() - 0.5) * 2.4,
        gust: t(),
        swell: 0.04 + t() * 0.25,
        flutter: t(),
        // Fog rolls independently of precipitation - a still, clear day
        // can still be socked in. 0 = none, 1 = heavy banks.
        fog: rng() < 0.35 ? 0.25 + t() * 0.75 : 0,
    };
}

const WEATHER_COUNT = { rain: 300, snow: 220, ash: 150, dust: 200, leaves: 60, bubbles: 90 };
const WEATHER_FALL = { rain: -7.5, snow: -0.7, ash: -0.3, dust: -0.05, leaves: -0.7, bubbles: 0.7 };

// ----------------------------------------------------------- components

/* Spore-style parts: every creature is a shared spine; kinds are preset
   bundles of COMPONENTS and hybrids are free assemblies. Each component
   may shape traits at birth (done in sampleMorphology), grant behaviors
   (gated via creature.has()), and draw itself here. `drawUnder` runs
   before the heads (shells that cover the body), `draw` after (tails,
   fins, anything trailing). All hooks receive the same bag. */
function tailFrame(d) {
    const { c, J, body } = d;
    const S2 = SURFACES[c.surface];
    const last = J.chain[J.chain.length - 1];
    const ahead = J.chain.length > 1 ? J.chain[J.chain.length - 2] : body;
    const L = fromWorld(S2, last);
    const A2 = fromWorld(S2, ahead);
    let da = L.a - A2.a, db = L.b - A2.b;
    const dl = Math.hypot(da, db) || 1;
    return { S2, last, L, da: da / dl, db: db / dl, len: d.p.tailLen * d.p.scale };
}

const COMPONENTS = {
    legs: {},        // the leg machinery lives in the spine solve itself
    swimmer: {},     // locomotion override lives in step(); marker part
    paws: {},        // grants rear/nibble/fidget; poses live in the solve

    carapace: {
        drawUnder(d) {
            const { arena, ctx, p, J, body, inker, rW, lw, basis } = d;
            const tailW = J.chain[J.chain.length - 1];
            const mid = {
                x: (body.x * 1.1 + tailW.x * 0.9) / 2,
                y: Math.max(body.y, tailW.y) + 0.06 * p.scale,
                z: (body.z * 1.1 + tailW.z * 0.9) / 2,
            };
            const span = Math.hypot(body.x - tailW.x, body.y - tailW.y, body.z - tailW.z);
            const major = span * 0.75 + rW * 1.5;
            const pMid = arena.project(mid);
            ctx.lineWidth = lw(1.9);
            arena.shell(ctx, inker, pMid,
                basis(mid, J.fwd, major),
                basis(mid, J.perpW, rW * 1.25), 'carapace', 18);
            ctx.globalAlpha = 0.55;
            ctx.lineWidth = lw(1.2);
            inker.bone(ctx,
                arena.project({ x: mid.x - J.fwd.x * major * 0.85, y: mid.y - J.fwd.y * major * 0.85, z: mid.z - J.fwd.z * major * 0.85 }),
                arena.project({ x: mid.x + J.fwd.x * major * 0.7, y: mid.y + J.fwd.y * major * 0.7, z: mid.z + J.fwd.z * major * 0.7 }),
                'seam', 0.7);
            ctx.globalAlpha = 1;
        },
    },

    floofTail: {
        draw(d) {
            const { arena, ctx, c, p, inker, lw } = d;
            const { S2, L, da, db, len } = tailFrame(d);
            ctx.strokeStyle = arena.colors.line;
            // The classic S: back, then HIGH over the body; taller still
            // when sitting up.
            const curl = 0.62 * (1 + 0.35 * (c.rearAmt || 0));
            for (const off of [0, 0.16, -0.16]) {
                let prev = null;
                for (let k = 0; k <= 7; k++) {
                    const th = (k / 7) * 2.6;
                    const r = len * (curl + off * Math.sin((k / 7) * Math.PI));
                    const pt = arena.project(toWorld(S2,
                        L.a + da * Math.sin(th) * r * 0.75,
                        L.b + db * Math.sin(th) * r * 0.75
                            + Math.cos(arena.t * 2) * 0.03,
                        Math.max(0.05, L.h + (1 - Math.cos(th)) * r * 1.2)));
                    ctx.lineWidth = lw(1.5 - Math.abs(off) * 2);
                    if (prev) inker.bone(ctx, prev, pt, `tail${off}-${k}`, 1.1);
                    prev = pt;
                }
            }
        },
    },

    whipTail: {
        draw(d) {
            const { arena, ctx, c, inker, lw } = d;
            const { S2, L, da, db, len } = tailFrame(d);
            ctx.strokeStyle = arena.colors.line;
            let prev = null;
            ctx.lineWidth = lw(1.1);
            for (let k = 0; k <= 6; k++) {
                const tt = k / 6;
                const sway = Math.sin(c.phase * 0.7 + tt * 3) * 0.12 * len;
                // The rat tail DRAGS: on the ground from halfway out.
                const pt = arena.project(toWorld(S2,
                    L.a + da * len * tt - db * sway,
                    L.b + db * len * tt + da * sway,
                    Math.max(0.02, L.h * Math.max(0, 1 - tt * 1.8))));
                if (prev) inker.bone(ctx, prev, pt, `whip${k}`, 0.7);
                prev = pt;
            }
        },
    },

    caudalFin: {
        draw(d) {
            const { arena, ctx, c, inker, lw } = d;
            const { S2, last, L, da, db, len } = tailFrame(d);
            ctx.strokeStyle = arena.colors.line;
            const sweep = Math.sin(c.phase * 1.4) * 0.25 * len;
            const tip = toWorld(S2, L.a + da * len - db * sweep,
                L.b + db * len + da * sweep, L.h);
            const up2 = toWorld(S2, L.a + da * len * 0.8 - db * sweep,
                L.b + db * len * 0.8 + da * sweep, L.h + len * 0.5);
            const dn = toWorld(S2, L.a + da * len * 0.8 - db * sweep,
                L.b + db * len * 0.8 + da * sweep, Math.max(0.05, L.h - len * 0.4));
            ctx.lineWidth = lw(1.4);
            inker.loop(ctx, [arena.project(last), arena.project(up2),
                arena.project(tip), arena.project(dn)], 'caudal', 1.0);
        },
    },

    wingsButterfly: {
        draw(d) {
            const { arena, ctx, c, p, J, body, pBody, inker, lw } = d;
            ctx.strokeStyle = arena.colors.line;
            // Wingbeat: fold foreshortens the span; perching folds them up.
            const rest = 1 - Math.min(1, Math.max(c.lie * 1.2, c.perchFold || 0));
            const beat = Math.sin(c.phase * 3.5);
            const fold = (0.25 + 0.75 * Math.abs(Math.cos(c.phase * 3.5))) * rest + 0.12;
            const liftY = beat * 0.45 * p.scale * rest;
            const span = p.scale * 1.5;
            for (const sgn of [-1, 1]) {
                for (const [lobe, back, sc] of [['u', 0.12, 1.0], ['l', -0.3, 0.62]]) {
                    const tip = {
                        x: body.x + J.perpW.x * sgn * span * fold * sc + J.fwd.x * back * p.scale,
                        y: body.y + liftY * sc + 0.15 * p.scale,
                        z: body.z + J.perpW.z * sgn * span * fold * sc + J.fwd.z * back * p.scale,
                    };
                    const fore = {
                        x: body.x + J.fwd.x * (back + 0.25) * p.scale,
                        y: body.y + 0.1 * p.scale,
                        z: body.z + J.fwd.z * (back + 0.25) * p.scale,
                    };
                    const aft = {
                        x: body.x + J.fwd.x * (back - 0.2) * p.scale,
                        y: body.y + 0.05 * p.scale,
                        z: body.z + J.fwd.z * (back - 0.2) * p.scale,
                    };
                    ctx.lineWidth = lw(1.4);
                    inker.loop(ctx, [arena.project(fore),
                        arena.project({ x: tip.x + J.fwd.x * 0.12 * p.scale, y: tip.y, z: tip.z + J.fwd.z * 0.12 * p.scale }),
                        arena.project(tip), arena.project(aft)],
                        `bw${sgn}${lobe}`, 1.0);
                }
            }
        },
    },

    wingsWasp: {
        draw(d) {
            const { arena, ctx, c, p, J, body, inker, lw } = d;
            ctx.strokeStyle = arena.colors.line;
            // A blur of wing: two ghost positions per side, beating fast.
            const span = p.scale * 0.8;
            for (const sgn of [-1, 1]) {
                const restW = 1 - 0.9 * (c.perchFold || 0);
                for (const ghost of [0, 1]) {
                    const beat = Math.sin(arena.t * 46 + ghost * 1.8 + sgn) * restW;
                    const liftY = 0.25 * p.scale + beat * 0.18 * p.scale;
                    const tip = {
                        x: body.x + J.perpW.x * sgn * span - J.fwd.x * 0.2 * p.scale,
                        y: body.y + liftY,
                        z: body.z + J.perpW.z * sgn * span - J.fwd.z * 0.2 * p.scale,
                    };
                    ctx.globalAlpha = ghost ? 0.22 : 0.45;
                    ctx.lineWidth = lw(1.1);
                    inker.loop(ctx, [d.pBody, arena.project(tip),
                        arena.project({ x: tip.x - J.fwd.x * 0.25 * p.scale, y: tip.y - 0.05 * p.scale, z: tip.z - J.fwd.z * 0.25 * p.scale })],
                        `ww${sgn}${ghost}`, 0.7);
                }
            }
            ctx.globalAlpha = 1;
            // Stripes across the abdomen, and the sting.
            const tailW = J.chain[J.chain.length - 1];
            const dir = { x: body.x - tailW.x, y: body.y - tailW.y, z: body.z - tailW.z };
            const dl = Math.hypot(dir.x, dir.y, dir.z) || 1;
            ctx.lineWidth = lw(1.2);
            for (const fr of [0.3, 0.6]) {
                const cx2 = { x: tailW.x + dir.x * fr, y: tailW.y + dir.y * fr, z: tailW.z + dir.z * fr };
                const r2 = 0.16 * p.scale;
                inker.bone(ctx,
                    arena.project({ x: cx2.x + J.perpW.x * r2, y: cx2.y, z: cx2.z + J.perpW.z * r2 }),
                    arena.project({ x: cx2.x - J.perpW.x * r2, y: cx2.y, z: cx2.z - J.perpW.z * r2 }),
                    `stripe${fr}`, 0.5);
            }
            inker.bone(ctx, arena.project(tailW),
                arena.project({ x: tailW.x - dir.x / dl * 0.18 * p.scale, y: tailW.y - dir.y / dl * 0.18 * p.scale, z: tailW.z - dir.z / dl * 0.18 * p.scale }),
                'sting', 0.5);
        },
    },

    pectorals: {
        draw(d) {
            const { arena, ctx, c, p, J, body, pBody, inker, lw } = d;
            ctx.strokeStyle = arena.colors.line;
            const flap = Math.sin(c.phase * 2.2) * 0.25;
            for (const sgn of [-1, 1]) {
                const finTip = {
                    x: body.x + J.perpW.x * sgn * p.scale * (0.5 + flap),
                    y: body.y + J.perpW.y * sgn * p.scale * (0.5 + flap) - p.scale * 0.18,
                    z: body.z + J.perpW.z * sgn * p.scale * (0.5 + flap),
                };
                const finBk = {
                    x: body.x - J.fwd.x * p.scale * 0.35,
                    y: body.y - J.fwd.y * p.scale * 0.35,
                    z: body.z - J.fwd.z * p.scale * 0.35,
                };
                ctx.lineWidth = lw(1.1);
                inker.loop(ctx, [pBody, arena.project(finTip), arena.project(finBk)],
                    'pec' + sgn, 0.8);
            }
        },
    },
};

// ------------------------------------------------------------------ arena

class Arena {
    constructor(canvas, seedStr) {
        this.canvas = canvas;
        this.seedStr = seedStr;
        this.ctx = canvas.getContext('2d');
        // Observation is 1-to-1: every page load is a fresh sighting of
        // the same animal in the same REGION, but never the same spot or
        // the same moment. Regional traits stay hash-bound; geometry,
        // spawn, and the decision stream re-roll per observation. Weather
        // keeps an hourly regional cadence.
        const obsSalt = ':o' + Date.now();
        const weatherEpoch = ':w' + Math.floor(Date.now() / 3600e3);

        // Habitat first: it sets the room's scale, and the creature is
        // born into whatever space this seed grows.
        const envRng = mulberry32(fnv1a(seedStr + ':env'));
        this.hab = sampleHabitat(envRng);
        setRoom(this.hab.roomW, this.hab.roomD, this.hab.roomH);
        const geoRng = mulberry32(fnv1a(seedStr + ':geo' + obsSalt));
        this.terrain = finishTerrain(buildTerrain(geoRng, this.hab, seedStr + obsSalt));
        // Enclosure is now literal: the fraction of faces that exist.
        const fmat = this.terrain.facesMat;
        this.enclosure = ['wallB', 'wallL', 'wallR', 'ceiling']
            .filter(f2 => fmat[f2] !== 'none').length / 4;
        this.creature = new Creature(seedStr, obsSalt);
        this.creature.terrainRef = this.terrain;
        this.creature.enclosure = this.enclosure;
        this.creature.faces = fmat;
        this.creature.seat();
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
        const dotRng = mulberry32(fnv1a(seedStr + ':dots' + obsSalt));
        const rm = this.terrain.roam, vt = this.terrain.vast;
        // Two-thirds of the grain lands where the creature lives, the
        // rest scatters across the vast plane.
        this.floorDots = Array.from({ length: Math.round(dots * 1.4) }, (_, i) => {
            const R = i % 3 === 2 ? vt : rm;
            return {
                x: R.x0 + dotRng() * (R.x1 - R.x0),
                z: R.z0 + dotRng() * (R.z1 - R.z0),
                r: 0.6 + dotRng() * 1.4, al: 0.02 + dotRng() * 0.06,
            };
        });
        this.wallDots = Array.from({ length: Math.round(dots * 0.6) }, () => ({
            x: dotRng() * ROOM_W, y: dotRng() * ROOM_H,
            r: 0.6 + dotRng() * 1.2, al: 0.02 + dotRng() * 0.05,
        }));

        this.t = 0;
        this.lastTs = null;
        // The following camera: world-anchored center, gentle pans, slow
        // distance zoom. Boxed scenes never move it.
        this.camX = ROOM_W / 2;
        this.camZ = 0;
        this.zoom = 1;
        this.panX = false;
        this.panZ = false;
        this.camYoff = 0;
        this.lostT = 0;       // seconds the creature has been off-canvas
        // Distant scenery freezes into an offscreen layer; only the band
        // around the creature renders live. See draw().
        this.scene = document.createElement('canvas');
        this.sceneCtx = this.scene.getContext('2d');
        this.sceneStamp = { x: NaN, z: NaN, zoom: NaN, boil: NaN, band: NaN, colors: '', frame: -9 };
        this.itemsCache = null;
        this.itemsEpoch = '';
        this.frameNo = 0;

        // Weather: its own seed stream so habitats never reroll.
        const wRng = mulberry32(fnv1a(seedStr + ':weather' + weatherEpoch));
        this.weather = sampleWeather(wRng, this.hab, this.terrain.facesMat);
        const wth = this.weather;
        const count = Math.round((WEATHER_COUNT[wth.material] || 0) * wth.density);
        const yMax = wth.material === 'bubbles'
            ? ROOM_H * 0.85 : Math.max(6, ROOM_H * 1.5);
        this.partYMax = yMax;
        // Fog banks: big soft blobs that drift with the wind and billow,
        // masking some of the scene while other patches stay clear.
        const fogN = wth.fog > 0 ? 6 + Math.round(wth.fog * 16) : 0;
        // Fog lives in WORLD space: banks sit where they sit while the
        // camera travels; leaving one behind, a new one wraps in ahead.
        this.fogBlobs = Array.from({ length: fogN }, () => ({
            x: ROOM_W / 2 + (wRng() - 0.5) * 64,
            z: 2 + wRng() * 26,
            y: 0.2 + wRng() * 1.8,
            rx: 4 + wRng() * 8,
            ry: 1.0 + wRng() * 2.0,
            ph: wRng() * Math.PI * 2,
            sp: 0.5 + wRng(),
        }));
        if (wth.fog > 0.55) {
            // Heavy cover: wide sheets hugging the ground.
            for (let i = 0; i < 3; i++) {
                this.fogBlobs.push({
                    x: ROOM_W / 2 + (wRng() - 0.5) * 60,
                    z: 3 + wRng() * 20,
                    y: 0.15 + wRng() * 0.4,
                    rx: 12 + wRng() * 10,
                    ry: 1.2 + wRng() * 1.2,
                    ph: wRng() * Math.PI * 2,
                    sp: 0.3 + wRng() * 0.5,
                });
            }
        }
        // Particles live in WORLD space and wrap around the camera
        // window: real parallax (near flakes sweep, far ones crawl)
        // instead of a field glued to the lens.
        this.particles = Array.from({ length: count }, () => ({
            x: ROOM_W / 2 + (wRng() - 0.5) * 48,
            y: wRng() * yMax,
            z: wRng() * 30,
            ph: wRng() * Math.PI * 2,
            sp: 0.7 + wRng() * 0.6,
        }));
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
        this.scene.width = this.canvas.width;
        this.scene.height = this.canvas.height;
        this.sceneCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
        this.sceneStamp.boil = NaN;   // stale after resize
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
        // Mutate, never replace: cached scenery closures hold this object.
        Object.assign(this.colors, {
            line: `hsl(${hue} 30% ${dark ? 88 : 14}%)`,
            shade: `hsl(${envHue} ${sat}% ${dark ? 62 : 42}%)`,
            accent: cs.getPropertyValue('--accent').trim() || `hsl(${hue} 87% 40%)`,
            washTop: `hsl(${envHue} ${sat}% ${dark ? 70 : 35}% / 0)`,
            washBot: `hsl(${envHue} ${sat}% ${dark ? 70 : 35}% / ${dark ? 0.10 : 0.08})`,
            grain: `hsl(${envHue} ${sat}% ${dark ? 75 : 30}%)`,
            glow: `hsl(${hue} 70% ${dark ? 60 : 45}%)`,
            paper: cs.getPropertyValue('--background').trim() || (dark ? '#101312' : '#f4f6f3'),
            fog: `hsl(${envHue} ${Math.min(20, sat)}% ${dark ? 34 : 86}%)`,
        });
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
            this.creature.env = { wind: this.windSignal() };
            this.creature.step(dt, this.t);
            this.updateCamera(dt);
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
    drawBackdrop(ctx, P) {
        const cl = this.colors;

        // A back-wall wash only when that face exists; otherwise a low
        // horizon band rising from the far floor edge.
        const enc = this.enclosure;
        const fmat = this.terrain.facesMat;
        if (fmat.wallB !== 'none') {
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
            // A full-width atmospheric band at the horizon - screen
            // space, so it has no edges of its own.
            const vt = this.terrain.vast;
            const yTop = P(0, 1.4, vt.z1).y;
            const yBot = P(0, 0, vt.z1 * 0.55).y;
            const sky = ctx.createLinearGradient(0, yTop, 0, yBot);
            sky.addColorStop(0, cl.washTop);
            sky.addColorStop(1, cl.washBot);
            ctx.fillStyle = sky;
            ctx.fillRect(0, yTop, this.w, Math.max(1, yBot - yTop));
        }

        // Ground wash over the whole rendered plane, deepening toward
        // the viewer - the near bed.
        const rm = this.terrain.infinite
            ? { x0: this.camX - 36, x1: this.camX + 36,
                z0: this.camZ - ROOM_D * 0.45, z1: this.camZ + 40 }
            : this.terrain.vast;
        const fn = P(rm.x0, 0, rm.z0), ff = P(rm.x0, 0, rm.z1);
        const floorGrad = ctx.createLinearGradient(0, ff.y, 0, fn.y);
        floorGrad.addColorStop(0, cl.washTop);
        floorGrad.addColorStop(1, cl.washBot);
        ctx.fillStyle = floorGrad;
        this.quad(ctx, P(rm.x0, 0, rm.z1), P(rm.x1, 0, rm.z1),
            P(rm.x1, 0, rm.z0), P(rm.x0, 0, rm.z0));

        // Grain: seeded stipple riding the actual heightmap. Infinite
        // worlds carry their grain in the chunks instead.
        ctx.fillStyle = cl.grain;
        for (const d of this.terrain.infinite ? [] : this.floorDots) {
            const pt = P(d.x, groundHeight(this.terrain, d.x, d.z) + 0.01, d.z);
            ctx.globalAlpha = d.al;
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, d.r * pt.d / 90, 0, Math.PI * 2);
            ctx.fill();
        }
        if (fmat.wallB !== 'none') {
            for (const d of this.wallDots) {
                const pt = P(d.x, d.y, ROOM_D);
                ctx.globalAlpha = d.al;
                ctx.beginPath();
                ctx.arc(pt.x, pt.y, d.r * pt.d / 110, 0, Math.PI * 2);
                ctx.fill();
            }
        }
        // Ceiling grain only when there IS a ceiling.
        if (fmat.ceiling !== 'none') {
            for (const d of this.terrain.ceilDots) {
                const pt = P(d.x, ROOM_H, d.z);
                ctx.globalAlpha = d.al;
                ctx.beginPath();
                ctx.arc(pt.x, pt.y, d.r * pt.d / 100, 0, Math.PI * 2);
                ctx.fill();
            }
        }
        ctx.globalAlpha = 1;
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
       seat scales with the room and FOLLOWS the creature through the
       infinite world - pan and zoom live in here. */
    project(p) {
        const camY = ROOM_H * 1.875, camD = ROOM_D;
        const fBase = (this.w * 0.97 * camD) / ROOM_W;
        const f = fBase * this.zoom;
        const zc = Math.max(-camD + 0.6, p.z - this.camZ);
        const d = f / (camD + zc);
        // The floor pin must NOT depend on zoom (a zoomed pin slides the
        // whole scene upward and the camera "looks down" forever); zoomed
        // framing is handled by camYoff re-centering on the creature.
        const y0 = this.h * 0.96 - camY * (fBase / camD) + (this.camYoff || 0);
        return {
            x: this.w / 2 + (p.x - this.camX) * d,
            y: y0 + (camY - p.y) * d,
            d, z: p.z,
        };
    }

    /* Touch the edge of the screen and the camera glides after you,
       revealing new terrain; run deep and it slowly zooms to keep you. */
    updateCamera(dt) {
        if (!this.terrain.infinite) return;
        const c = this.creature;
        const S = SURFACES[c.surface];
        const w0 = toWorld(S, c.a, c.b, c.h);
        // Velocity EMA: the camera aims a beat AHEAD of a walker, so the
        // long marchers stay framed instead of forever outrunning the pan.
        if (this.lastW0) {
            const ivx = (w0.x - this.lastW0.x) / Math.max(1e-3, dt);
            const ivz = (w0.z - this.lastW0.z) / Math.max(1e-3, dt);
            const k2 = Math.min(1, 1.2 * dt);
            this.vemaX = (this.vemaX || 0) + (ivx - (this.vemaX || 0)) * k2;
            this.vemaZ = (this.vemaZ || 0) + (ivz - (this.vemaZ || 0)) * k2;
        }
        this.lastW0 = { x: w0.x, z: w0.z };
        const lead = 1.1;
        const aimX = w0.x + (this.vemaX || 0) * lead;
        const aimZ = w0.z + (this.vemaZ || 0) * lead;
        const pb = this.project(w0);
        // The camera sleeps until the creature touches the outside of the
        // screen, then drifts after it - slowly, so as not to disturb it -
        // and settles once the creature is centered again.
        if (!this.panX && (pb.x < this.w * 0.03 || pb.x > this.w * 0.97)) {
            this.panX = true;
        }
        if (this.panX) {
            this.camX += (aimX - this.camX) * Math.min(1, 0.22 * dt);
            if (Math.abs(aimX - this.camX) < 0.5) this.panX = false;
        }
        const zRel = aimZ - this.camZ;
        const zHome = ROOM_D * 0.55;
        if (!this.panZ && (zRel > ROOM_D * 2.2 || zRel < 0.25)) this.panZ = true;
        if (this.panZ) {
            this.camZ += (zRel - zHome) * Math.min(1, 0.18 * dt);
            if (Math.abs(zRel - zHome) < 0.6) this.panZ = false;
        }
        // While the pan catches up, an even slower zoom keeps the runner
        // in view.
        const tz = Math.min(2.0, Math.max(1, (ROOM_D + zRel) / (ROOM_D + zHome)));
        this.zoom += (tz - this.zoom) * Math.min(1, 0.15 * dt);

        // Zoomed framing re-centers vertically on the creature; at rest
        // (zoom ~1) the offset eases home so the floor pin rules again.
        const pb2 = this.project(w0);
        const zBlend = Math.min(1, Math.max(0, (this.zoom - 1) / 0.6));
        const wantY = zBlend * Math.max(-this.h * 0.45,
            Math.min(this.h * 0.45, this.h * 0.52 - (pb2.y - this.camYoff)));
        this.camYoff += (wantY - this.camYoff) * Math.min(1, 0.3 * dt);

        // Watchdog: if the creature has been fully off-canvas for a couple
        // of seconds - whatever the cause - recover focus directly.
        const off = pb2.x < -40 || pb2.x > this.w + 40
            || pb2.y < -40 || pb2.y > this.h + 40;
        this.lostT = off ? this.lostT + dt : 0;
        if (this.lostT > 2) {
            this.panX = true; this.panZ = true;
            this.camX += (w0.x - this.camX) * Math.min(1, 0.7 * dt);
            this.camZ += (zRel - zHome) * Math.min(1, 0.7 * dt);
            this.camYoff += (0 - this.camYoff) * Math.min(1, 0.7 * dt);
        }
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
        // The component bag: every part hook gets the same view of the
        // creature - Spore-style shared spine, parts drawn on top.
        const bag = {
            arena: this, ctx, c, p, J, inker,
            body, pBody, heads, chainP, rW, lw, basis,
        };
        if (p.carapace) {
            for (const name of p.components) {
                COMPONENTS[name]?.drawUnder?.(bag);
            }
        } else {
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
        }

        if (!p.carapace && p.mech > 0.5) {
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
                const fLen = hr * 1.6 * p.feelerLen;
                const fSpread = hr * 0.7 * (0.6 + 0.4 * p.feelerLen);
                const tipW = {
                    x: hw.x + J.faceW.x * fLen + J.browW.x * sgn * fSpread,
                    y: hw.y + J.faceW.y * fLen + J.browW.y * sgn * fSpread
                        + Math.sin(this.t * 6 + sgn + i) * hr * 0.2 * p.feelerLen,
                    z: hw.z + J.faceW.z * fLen + J.browW.z * sgn * fSpread,
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
                // The common black eye: solid ink, no glow, no accent.
                const rPx = Math.max(1.0, eye.r * pe2.d);
                if (p.mech > 0.55) {
                    ctx.strokeStyle = this.colors.line;
                    ctx.beginPath();
                    ctx.arc(pe2.x, pe2.y, rPx, 0, Math.PI * 2);
                    ctx.stroke();
                    ctx.fillStyle = this.colors.line;
                    ctx.beginPath();
                    ctx.arc(pe2.x, pe2.y, Math.max(0.5, rPx * 0.4), 0, Math.PI * 2);
                    ctx.fill();
                } else {
                    ctx.fillStyle = this.colors.line;
                    ctx.beginPath();
                    ctx.arc(pe2.x, pe2.y, rPx, 0, Math.PI * 2);
                    ctx.fill();
                }
            }
            ctx.lineWidth = lw(1.6);
        }

        // ---- attached parts: tails, fins, whatever the spine carries ----
        for (const name of p.components) {
            COMPONENTS[name]?.draw?.(bag);
        }
    }

    /* Should this item be skipped as outside the viewport? Cheap world-x
       test against the projected half-width at the item's depth. */
    culled(item, fEff) {
        if (item.x === undefined) return false;
        const d = fEff / (ROOM_D + Math.max(0.6, item.z - this.camZ));
        return Math.abs((item.x - this.camX) * d) > this.w * 0.62;
    }

    /* Render backdrop + everything beyond the near band into the frozen
       scene layer. Runs only when the camera/boil/theme actually change. */
    renderScene(items, band, fEff) {
        const ctx = this.sceneCtx;
        ctx.clearRect(0, 0, this.w, this.h);
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        const P = (x, y, z) => this.project({ x, y, z });
        this.drawBackdrop(ctx, P);
        const zCull = this.camZ - ROOM_D * 0.7;
        for (const item of items) {
            if (item.z <= band || item.z < zCull) continue;
            if (this.culled(item, fEff)) continue;
            ctx.strokeStyle = this.colors.shade;
            ctx.lineWidth = 1.2;
            item.draw(ctx);
        }
        ctx.globalAlpha = 1;
    }

    draw(dt) {
        const { ctx, w, h } = this;
        if (!w) return;
        this.frameNo++;
        const inker = this.inker;
        inker.boil(this.t);
        this.envInker.boil(this.t);

        ctx.clearRect(0, 0, w, h);
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        const P = (x, y, z) => this.project({ x, y, z });
        const J = this.creature.solve(this.t);
        const fEff = ((this.w * 0.97 * ROOM_D) / ROOM_W) * this.zoom;

        // Item epoch: rebuild the (camera-independent) item closures only
        // when the window slides 4+ units - this also PRE-SPAWNS chunks a
        // full view ahead of the camera, and unloads distant ones.
        const epochX = Math.round(this.camX / 4) * 4;
        const epochZ = Math.round(this.camZ / 4) * 4;
        const epochKey = epochX + '|' + epochZ;
        if (!this.itemsCache || this.itemsEpoch !== epochKey) {
            this.itemsEpoch = epochKey;
            this.itemsCache = envItems(this.hab, this.envSalt, this.envInker, P,
                this.terrain, this.colors, this.t, { x: this.camX, z: this.camZ });
            this.itemsCache.sort((a, b) => b.z - a.z);
            const m = this.terrain._chunks;
            if (m && m.size > 280) {
                for (const k of m.keys()) {
                    const [ci, cj] = k.split(',');
                    if (Math.hypot(ci * CH - this.camX, cj * CH - this.camZ) > 90) m.delete(k);
                }
            }
            this.sceneStamp.boil = NaN;   // scene layer must repaint
        }
        const items = this.itemsCache;

        // The near band renders live so the creature sorts among it; the
        // rest is frozen in the scene layer. Quantized so the boundary
        // only moves when the creature crosses a 2-unit stripe.
        const band = Math.ceil((J.body.z + 1.5) / 2) * 2;
        const s = this.sceneStamp;
        const colKey = this.colors.line + this.colors.shade;
        const moved = Math.abs(this.camX - s.x) > 0.05
            || Math.abs(this.camZ - s.z) > 0.05
            || Math.abs(this.zoom - s.zoom) > 0.004;
        const hard = s.band !== band || s.boil !== this.envInker.last || s.colors !== colKey;
        if ((hard || moved) && (hard || this.frameNo - s.frame >= 2)) {
            this.renderScene(items, band, fEff);
            s.x = this.camX; s.z = this.camZ; s.zoom = this.zoom;
            s.boil = this.envInker.last; s.band = band; s.colors = colKey;
            s.frame = this.frameNo;
        }
        // Blit the frozen layer, shift-compensated for the pan since the
        // last repaint (parallax error is sub-pixel at these speeds).
        const dMid = fEff / (ROOM_D + 8);
        const dx = (s.x - this.camX) * dMid;
        ctx.drawImage(this.scene, dx, 0, this.scene.width / (window.devicePixelRatio || 1),
            this.scene.height / (window.devicePixelRatio || 1));

        // Live pass: near items, the creature, animated ambience.
        const zCull = this.camZ - ROOM_D * 0.7;
        const live = [];
        for (const item of items) {
            if (item.z > band || item.z < zCull) continue;
            if (this.culled(item, fEff)) continue;
            live.push(item);
        }
        live.push(...envDynamicItems(this.terrain, P, this.colors, this.t));
        if (this.fogBlobs && this.fogBlobs.length) {
            const wind = this.windSignal();
            const fogA = 0.1 + 0.32 * this.weather.fog;
            for (const fb of this.fogBlobs) {
                fb.x += (wind * 0.55 + Math.sin(this.t * 0.13 + fb.ph) * 0.12) * dt * fb.sp * 8;
                // World-anchored: wrap into the window around the camera,
                // so departed banks regenerate ahead instead of tagging along.
                while (fb.x - this.camX > 34) fb.x -= 68;
                while (fb.x - this.camX < -34) fb.x += 68;
                while (fb.z - this.camZ > 28) fb.z -= 28;
                while (fb.z - this.camZ < 0) fb.z += 28;
                const wz = fb.z;
                live.push({
                    z: wz,
                    draw: (ctx2) => {
                        const cx2 = fb.x;
                        // Billow: each bank swells and thins on its own slow clock.
                        const billow = 0.65 + 0.35 * Math.sin(this.t * 0.21 * fb.sp + fb.ph);
                        const pc = this.project({ x: cx2, y: fb.y, z: wz });
                        if (pc.x < -300 || pc.x > this.w + 300) return;
                        const rx = fb.rx * pc.d * billow;
                        const ry = fb.ry * pc.d * 0.45 * billow;
                        const g2 = ctx2.createRadialGradient(pc.x, pc.y, 0, pc.x, pc.y, Math.max(1, rx));
                        g2.addColorStop(0, this.colors.fog);
                        g2.addColorStop(1, 'transparent');
                        ctx2.globalAlpha = fogA * billow;
                        ctx2.fillStyle = g2;
                        ctx2.beginPath();
                        ctx2.ellipse(pc.x, pc.y, rx, ry, 0, 0, Math.PI * 2);
                        ctx2.fill();
                        ctx2.globalAlpha = 1;
                    },
                });
            }
        }
        live.push({
            z: J.body.z,
            creature: true,
            draw: (ctx2) => {
                ctx2.globalAlpha = 1;
                this.drawCreature(ctx2, J, dt);
            },
        });
        live.sort((a, b) => b.z - a.z);
        for (const item of live) {
            if (!item.creature) {
                ctx.strokeStyle = this.colors.shade;
                ctx.lineWidth = 1.2;
            }
            item.draw(ctx);
        }
        ctx.globalAlpha = 1;

        this.drawWeather(ctx, dt);
    }

    /* The particle system: a camera-wrapped volume of simple strokes.
       Wind is one shared signal - base strength, gusts and reductions,
       and the slow swell - so the whole sky moves as a body. */
    /* One wind for the whole world: weather particles, swaying bodies,
       gusts that shove a frozen axis. Zero when the sky rolled clear. */
    windSignal() {
        const wth = this.weather;
        if (!wth || wth.clear) return 0;
        const t = this.t;
        const sw = Math.sin(t * wth.swell * 6.283);
        return wth.wind * (1 - wth.gust * 0.45
            + wth.gust * (0.55 * sw + 0.35 * Math.sin(t * wth.swell * 17 + 1.7)))
            * (this.terrain.facesMat.ceiling !== 'none' ? 0.25 : 1);
    }

    drawWeather(ctx, dt) {
        const wth = this.weather;
        if (!wth || !this.particles.length) return;
        const t = this.t;
        const windNow = this.windSignal();
        const fall = WEATHER_FALL[wth.material] * (0.6 + 0.7 * wth.density);
        const mat = wth.material;
        const bubbles = mat === 'bubbles';
        const yMax = this.partYMax;
        ctx.strokeStyle = mat === 'ash' || mat === 'dust' || mat === 'leaves'
            ? this.colors.grain : this.colors.line;
        ctx.fillStyle = ctx.strokeStyle;
        ctx.lineWidth = 1;

        for (const pt of this.particles) {
            const flut = wth.flutter * Math.sin(t * (2 + pt.sp) + pt.ph);
            pt.y += fall * pt.sp * dt;
            pt.x += (windNow * (mat === 'dust' ? 2.2 : 1) + flut * 0.6) * dt;
            pt.z += Math.cos(t * 0.7 + pt.ph) * 0.2 * dt;
            // World-anchored, camera-wrapped: the region of weather is
            // wherever the view is, but each particle holds its world
            // position - parallax comes free from the projection.
            if (pt.y < 0) pt.y += yMax;
            if (pt.y > yMax) pt.y -= yMax;
            while (pt.x - this.camX > 24) pt.x -= 48;
            while (pt.x - this.camX < -24) pt.x += 48;
            while (pt.z - this.camZ > 30) pt.z -= 30;
            while (pt.z - this.camZ < 0) pt.z += 30;

            const wx = pt.x;
            const wz = pt.z;
            const pp = this.project({ x: wx, y: pt.y, z: wz });
            if (pp.x < -10 || pp.x > this.w + 10 || pp.y < -10 || pp.y > this.h + 10) continue;
            const depth = Math.min(1, pp.d / 30);
            ctx.globalAlpha = (mat === 'dust' ? 0.12 : 0.3) * (0.35 + 0.65 * depth) * (0.5 + 0.5 * wth.density);

            if (mat === 'rain') {
                const p2 = this.project({
                    x: wx + windNow * 0.05, y: pt.y - fall * pt.sp * 0.045, z: wz,
                });
                ctx.beginPath();
                ctx.moveTo(pp.x, pp.y);
                ctx.lineTo(p2.x, p2.y);
                ctx.stroke();
            } else if (mat === 'leaves') {
                const a2 = t * 3 * pt.sp + pt.ph;
                const r2 = Math.max(1.4, pp.d * 0.02);
                ctx.beginPath();
                ctx.moveTo(pp.x - Math.cos(a2) * r2, pp.y - Math.sin(a2) * r2 * 0.5);
                ctx.lineTo(pp.x, pp.y + r2 * 0.4);
                ctx.lineTo(pp.x + Math.cos(a2) * r2, pp.y - Math.sin(a2) * r2 * 0.5);
                ctx.stroke();
            } else if (mat === 'bubbles') {
                ctx.beginPath();
                ctx.arc(pp.x, pp.y, Math.max(0.8, pp.d * (0.008 + 0.006 * pt.sp)), 0, Math.PI * 2);
                ctx.stroke();
            } else {
                // snow, ash, dust: dots of falling matter.
                ctx.beginPath();
                ctx.arc(pp.x, pp.y, Math.max(0.6, pp.d * (mat === 'snow' ? 0.012 : 0.007)), 0, Math.PI * 2);
                ctx.fill();
            }
        }
        ctx.globalAlpha = 1;
    }
}

// -------------------------------------------------------------- wiring

let arenaInstance = null;
let arenaSeedBase = null;     // the hash-bound seed the tab handed us
let arenaSeedActive = null;   // what is actually running (reroll survives)

export function renderArena() {
    return `<div class="arena-card">
        <div class="portal-frame arena-frame">
            <div class="portal-window">
                <canvas class="arena-canvas"></canvas>
                <div class="biz-card-actions arena-actions" hidden>
                    <button class="biz-btn" id="arena-draw" type="button">Draw</button>
                </div>
            </div>
        </div>
    </div>`;
}

export function wireArena(container, seedStr) {
    const canvas = container.querySelector('.arena-canvas');
    if (!canvas) return;
    const base = seedStr || 'praxis';
    // Idempotent against the global tab refresh: if the living arena is
    // already wired to this very canvas for this model, leave it alone.
    if (arenaInstance && arenaInstance.canvas === canvas
        && arenaSeedBase === base) return;
    // Re-rendered DOM, same model: resume whatever seed was ACTIVE - a
    // reroll must survive the tab's re-render, not revert to the hash.
    const seed = arenaSeedBase === base && arenaSeedActive
        ? arenaSeedActive : base;
    arenaSeedBase = base;
    arenaSeedActive = seed;
    if (arenaInstance) arenaInstance.stop();
    arenaInstance = new Arena(canvas, seed);
    window.__arena = arenaInstance; // dev introspection
    arenaInstance.start();

    // Debug-mode developer tool: DRAW a fresh sample from the genome
    // pool, overriding the hash-bound default. Full random - it's a
    // debugging tool, determinism is not wanted here. Only the first
    // card (the hash-bound one) is stable across debug on/off.
    const actions = container.querySelector('.arena-actions');
    const draw = container.querySelector('#arena-draw');
    if (!actions || !draw) return;
    const sync = () => { actions.hidden = !state.settings.debugLogging; };
    sync();
    arenaInstance.onFrame = sync;
    draw.addEventListener('click', () => {
        arenaInstance.stop();
        arenaSeedActive = `draw:${Math.random().toString(36).slice(2)}`;
        arenaInstance = new Arena(canvas, arenaSeedActive);
        window.__arena = arenaInstance;
        arenaInstance.onFrame = sync;
        arenaInstance.start();
    });
}
