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
        hindBias: t(),                // rear legs longer, jumps harder (frog)
        undulate: t(),                // lateral spine wave (snake, swimmer)
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
        // Ink.
        sketch: u(0.5, 1.3),
        boilHz: u(5, 9),
        stiffness: u(45, 110),
        damping: u(7, 13),
    };
}

/* The room's own genome: rooms lean colorful or plain, noisy or calm,
   cohesive or decayed - coupled the same way the creature is. */
function sampleHabitat(rng) {
    const t = genome(rng, 4, 6);
    return {
        style: Math.floor(rng() * 4),
        hueOff: (t() - 0.5) * 90,     // color lean off the accent hue
        noise: 0.4 + t() * 1.3,       // jitter of the room's ink
        cohesion: 0.55 + t() * 0.45,  // overall presence of the lines
        decay: t() * 0.45,            // chance any given line has crumbled
        density: 0.5 + t() * 1.2,     // how much furniture
    };
}

// ------------------------------------------------------------ 3D helpers

const ROOM_W = 14, ROOM_D = 7, ROOM_H = 4;
const EDGE = 0.4;

const SURFACES = {
    floor: { o: [0, 0, 0], ea: [1, 0, 0], eb: [0, 0, 1], n: [0, 1, 0], A: ROOM_W, B: ROOM_D },
    ceiling: { o: [0, ROOM_H, 0], ea: [1, 0, 0], eb: [0, 0, 1], n: [0, -1, 0], A: ROOM_W, B: ROOM_D },
    wallB: { o: [0, 0, ROOM_D], ea: [1, 0, 0], eb: [0, 1, 0], n: [0, 0, -1], A: ROOM_W, B: ROOM_H },
    wallL: { o: [0, 0, 0], ea: [0, 0, 1], eb: [0, 1, 0], n: [1, 0, 0], A: ROOM_D, B: ROOM_H },
    wallR: { o: [ROOM_W, 0, 0], ea: [0, 0, 1], eb: [0, 1, 0], n: [-1, 0, 0], A: ROOM_D, B: ROOM_H },
};

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

        this.surface = 'floor';
        this.a = ROOM_W * (0.25 + 0.5 * this.rng());
        this.b = ROOM_D * (0.25 + 0.5 * this.rng());
        this.va = 0; this.vb = 0;
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
        return this.ch;
    }

    act() {
        const r = this.live();
        const p = this.p;
        if (r < p.sleepiness) {
            this.state = 'sleep';
            this.timer = 3.0 + this.expo(4.0);
        } else if (r < p.sleepiness + p.jumpiness) {
            this.state = 'jump';
            const hard = this.surface === 'floor' && this.live() < 0.35;
            this.vh += (hard ? 4.6 : 2.2 + this.live() * 1.6)
                * (0.75 + 0.5 * p.floaty) * (1 + 0.5 * p.hindBias);
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

        if (route && fast && this.live() < p.climby) {
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
            case 'jump':
                if (this.vh <= 0 && this.h <= p.stance * p.scale * 1.15) this.rest();
                break;
            case 'bump':
                if (this.timer <= 0) this.rest();
                break;
        }

        const agility = (2.2 + 7 * p.speedy) * (this.state === 'bump' ? 0.4 : 1);
        const drag = 1.2 + 3.5 * (1 - p.fluid);
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

        const standH = p.stance * p.scale;
        const lieH = 0.08 * p.scale;
        const targetH = (standH * (1 - this.lie) + lieH * this.lie)
            * (1 + 0.12 * this.ch.buoy * p.floaty);
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
        if (this.h < lieH * 0.6) { this.h = lieH * 0.6; this.vh = Math.abs(this.vh) * 0.3; }

        if (this.surface === 'floor' && this.vh > 0 && this.h >= ROOM_H - standH * 1.4) {
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
                * p.undulate * bodyR * 1.2 * speedF;
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

        this.phase += (2.0 + 3.0 * sp) * p.stride * (1 + 0.3 * ch.tempo) * dt;
    }

    stepLegs(dt, sp, S) {
        const p = this.p;
        const s = p.scale;
        const grounded = this.h < p.stance * s * 1.3 && this.lie < 0.7;
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

        const grounded = this.h < p.stance * s * 1.3;
        J.legs = [];
        for (const leg of this.legs) {
            const host = this.hostLocal(leg.host, S);
            const reach = reachBase * (leg.hind ? 1 + 0.6 * p.hindBias : 1);
            const seg = reach * p.legSeg;
            const ang = host.dir + leg.sweep;
            const hip = {
                a: host.a + Math.cos(ang) * bodyR * 0.85,
                b: host.b + Math.sin(ang) * bodyR * 0.85,
                h: host.h + bob,
            };
            let foot, toeDir;
            if (this.lie >= 0.7) {
                foot = { a: host.a + Math.cos(ang) * reach * 1.25, b: host.b + Math.sin(ang) * reach * 1.25, h: 0.01 };
                toeDir = ang;
            } else if (!grounded) {
                foot = {
                    a: host.a + Math.cos(ang) * reach * 0.5,
                    b: host.b + Math.sin(ang) * reach * 0.5,
                    h: bodyH - seg * 0.7,
                };
                toeDir = ang;
            } else if (leg.swing) {
                const sw = leg.swing;
                const e = sw.t < 0.5 ? 2 * sw.t * sw.t : 1 - Math.pow(-2 * sw.t + 2, 2) / 2;
                foot = {
                    a: sw.fa + (sw.ta2 - sw.fa) * e,
                    b: sw.fb + (sw.tb2 - sw.fb) * e,
                    h: Math.sin(Math.PI * sw.t) * 0.16 * s,
                };
                toeDir = Math.atan2(sw.tb2 - sw.fb, sw.ta2 - sw.fa);
            } else if (leg.planted) {
                foot = { a: leg.planted.a, b: leg.planted.b, h: 0 };
                toeDir = host.dir;
            } else {
                foot = { a: host.a + Math.cos(ang) * reach * 0.7, b: host.b + Math.sin(ang) * reach * 0.7, h: 0 };
                toeDir = ang;
            }
            const sol = ik2local(hip, foot, seg, seg, p.arch);
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
            const toe = {
                a: sol.foot.a + Math.cos(toeDir) * 0.1 * s,
                b: sol.foot.b + Math.sin(toeDir) * 0.1 * s,
                h: 0,
            };
            J.legs.push({
                joints: joints.map(j => toWorld(S, j.a, j.b, j.h)),
                toe: toWorld(S, toe.a, toe.b, toe.h),
                grounded: !!leg.planted,
            });
        }

        J.shadow = [];
        for (let i = 0; i < 10; i++) {
            const a2 = (i / 10) * Math.PI * 2;
            J.shadow.push(toWorld(S,
                this.a + Math.cos(a2) * 0.5 * s * p.elong * 0.8,
                this.b + Math.sin(a2) * 0.38 * s, 0.01));
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

/* Habitat furniture, sampled per seed under the habitat genome (density
   scales counts). Every element carries a depth z so the painter can
   interleave it with the creature. */
function buildFurniture(rng, hab) {
    const count = (n) => Math.max(1, Math.round(n * hab.density));
    const style = hab.style;
    if (style === 0) {
        return {
            kind: 'gym',
            seams: count(3),
            ringR: 1.0 + rng() * 1.2,
            railH: 0.35 + rng() * 0.2,
        };
    }
    if (style === 1) {
        return {
            kind: 'grove',
            stalks: Array.from({ length: count(10) }, () => ({
                x: rng() * ROOM_W,
                z: ROOM_D * (0.15 + rng() * 0.8),
                h: 0.8 + rng() * 2.4,
                lean: (rng() - 0.5) * 0.6,
            })),
        };
    }
    if (style === 2) {
        return {
            kind: 'cavern',
            arcs: Array.from({ length: count(4) }, () => ({
                h: ROOM_H - 0.4 - rng() * 0.8,
                sag: 0.3 + rng() * 0.6,
                z: ROOM_D * (0.5 + rng() * 0.45),
            })),
            rocks: Array.from({ length: count(8) }, () => ({
                x: rng() * ROOM_W,
                z: ROOM_D * (0.1 + rng() * 0.85),
                r: 0.15 + rng() * 0.4,
            })),
        };
    }
    return {
        kind: 'void',
        gridX: count(7),
        gridZ: count(4),
    };
}

/* Stable per-key decay: a decayed room has crumbled away some of its
   lines, and which ones never changes for a given seed. */
function survives(key, hab, salt) {
    return ((fnv1a(key + salt) % 1000) / 1000) >= hab.decay;
}

/* Emit the room as depth-sorted items. Long spanning lines split into
   thirds so the creature can pass in front of the near part of an edge
   and behind the far part. */
function envItems(env, hab, salt, ink, P) {
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

    // Room shell.
    seg3(0, 0, 0, ROOM_W, 0, 0, 'fl-near', 0.45, 1.0);
    seg3(0, 0, ROOM_D, ROOM_W, 0, ROOM_D, 'fl-far', 0.3, 0.8);
    seg3(0, 0, 0, 0, 0, ROOM_D, 'fl-left', 0.3, 0.8);
    seg3(ROOM_W, 0, 0, ROOM_W, 0, ROOM_D, 'fl-right', 0.3, 0.8);
    seg3(0, ROOM_H, ROOM_D, ROOM_W, ROOM_H, ROOM_D, 'ce-far', 0.22, 0.8);
    seg3(0, ROOM_H, 0, ROOM_W, ROOM_H, 0, 'ce-near', 0.16, 0.7);
    seg3(0, ROOM_H, 0, 0, ROOM_H, ROOM_D, 'ce-left', 0.16, 0.7);
    seg3(ROOM_W, ROOM_H, 0, ROOM_W, ROOM_H, ROOM_D, 'ce-right', 0.16, 0.7);
    seg3(0, 0, ROOM_D, 0, ROOM_H, ROOM_D, 'post-bl', 0.22, 0.8);
    seg3(ROOM_W, 0, ROOM_D, ROOM_W, ROOM_H, ROOM_D, 'post-br', 0.22, 0.8);
    seg3(0, 0, 0, 0, ROOM_H, 0, 'post-fl', 0.10, 0.7);
    seg3(ROOM_W, 0, 0, ROOM_W, ROOM_H, 0, 'post-fr', 0.10, 0.7);

    if (env.kind === 'gym') {
        for (let i = 1; i <= env.seams; i++) {
            const z = (ROOM_D * i) / (env.seams + 1);
            seg3(0, 0, z, ROOM_W, 0, z, 'fl-z' + i, 0.12, 0.7);
        }
        if (survives('ring', hab, salt)) {
            items.push({
                z: ROOM_D / 2 - env.ringR,
                draw: (ctx) => {
                    ctx.globalAlpha = 0.12 * hab.cohesion;
                    const ring = [];
                    for (let i = 0; i < 14; i++) {
                        const a = (i / 14) * Math.PI * 2;
                        ring.push(P(ROOM_W / 2 + Math.cos(a) * env.ringR, 0,
                            ROOM_D / 2 + Math.sin(a) * env.ringR));
                    }
                    ink.loop(ctx, ring, 'fl-ring', 0.7);
                },
            });
        }
        seg3(-0.5, ROOM_H * env.railH, ROOM_D, ROOM_W + 0.5, ROOM_H * env.railH, ROOM_D, 'rail', 0.15, 0.7);
    } else if (env.kind === 'grove') {
        env.stalks.forEach((st, i) => {
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
    } else if (env.kind === 'cavern') {
        env.arcs.forEach((arc, i) => {
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
        env.rocks.forEach((r, i) => {
            if (!survives('rock' + i, hab, salt)) return;
            items.push({
                z: r.z,
                draw: (ctx) => {
                    ctx.globalAlpha = 0.28 * hab.cohesion;
                    const pts = [];
                    for (let k = 0; k < 7; k++) {
                        const a = (k / 7) * Math.PI * 2;
                        pts.push(P(r.x + Math.cos(a) * r.r, Math.max(0, Math.sin(a) * r.r * 0.7), r.z));
                    }
                    ink.loop(ctx, pts, 'rock' + i, 0.9);
                },
            });
        });
    } else if (env.kind === 'void') {
        for (let i = 1; i < env.gridX; i++) {
            const x = (ROOM_W * i) / env.gridX;
            seg3(x, 0, 0, x, 0, ROOM_D, 'gx' + i, 0.10, 0.6);
        }
        for (let i = 1; i < env.gridZ; i++) {
            const z = (ROOM_D * i) / env.gridZ;
            seg3(0, 0, z, ROOM_W, 0, z, 'gz' + i, 0.10, 0.6);
        }
    }
    return items;
}

// ------------------------------------------------------------------ arena

class Arena {
    constructor(canvas, seedStr) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.creature = new Creature(seedStr);
        this.inker = new Inker(
            mulberry32(fnv1a(seedStr + ':ink')),
            this.creature.p.boilHz,
            this.creature.p.sketch,
        );
        const envRng = mulberry32(fnv1a(seedStr + ':env'));
        this.hab = sampleHabitat(envRng);
        this.env = buildFurniture(envRng, this.hab);
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

        // Back wall: a vertical wash, heavier toward the floor line.
        const wt = P(0, ROOM_H, ROOM_D), wb = P(0, 0, ROOM_D);
        const wallGrad = ctx.createLinearGradient(0, wt.y, 0, wb.y);
        wallGrad.addColorStop(0, cl.washTop);
        wallGrad.addColorStop(1, cl.washBot);
        ctx.fillStyle = wallGrad;
        this.quad(ctx, P(0, ROOM_H, ROOM_D), P(ROOM_W, ROOM_H, ROOM_D),
            P(ROOM_W, 0, ROOM_D), P(0, 0, ROOM_D));

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
        for (const d of this.wallDots) {
            const pt = P(d.x, d.y, ROOM_D);
            ctx.globalAlpha = d.al;
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, d.r * pt.d / 110, 0, Math.PI * 2);
            ctx.fill();
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

    /* Tank camera: standing outside and above, looking down in. */
    project(p) {
        const camY = 7.5, camD = 7.0;
        const f = this.w * 0.50;
        const d = f / (camD + p.z);
        return {
            x: this.w / 2 + (p.x - ROOM_W / 2) * d,
            y: this.h * 0.04 + (camY - p.y) * d,
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
            for (let s2 = 0; s2 < pts.length - 1; s2++) {
                inker.bone(ctx, pts[s2], pts[s2 + 1], `leg${leg.i}-s${s2}`, 1.2);
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
        const items = envItems(this.env, this.hab, this.envSalt, this.envInker, P);
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
