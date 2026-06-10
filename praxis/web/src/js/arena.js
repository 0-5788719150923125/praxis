/**
 * The spider, raised - a seeded robot bug living in a large 3D room,
 * watched from the bleachers (the Arena card on the Identity tab). Pure
 * mechanics, no backend: all randomness derives from the model hash, so
 * each model is one creature.
 *
 * World space is 3D: x along the room, z into depth, y up. The creature
 * lives on SURFACES (floor, walls, ceiling) in surface-local coordinates
 * and is flattened only at draw time through a perspective camera.
 *
 * Morphology is evolutionary, the business-card move in 3D: a seeded
 * latent genome passes through a small random tanh network, so traits
 * (leg count, elbows, heads, body squareness, mech-vs-organic...) couple
 * non-linearly instead of being independent dials.
 *
 * Three layers:
 *   1. Behavior - a probabilistic FSM (idle/run/sleep/jump/bump/climb)
 *      plus a voter pool: seeded slow oscillators pooled into continuous
 *      control channels that inform the rigged parts.
 *   2. Skeleton - the true mechanical rig in surface-local space: body,
 *      abdomen, 1-2 heads, and 4-8 legs of 2-3 segments via analytic IK.
 *   3. Ink - every rendered joint is a spring-damper twin chasing its
 *      true joint, and every bone is a jagged polyline whose jitter
 *      re-rolls a few times a second - the living-newspaper boil.
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

// -------------------------------------------------------------- morphology

/* A latent genome through a random two-layer tanh net: every trait is a
   non-linear projection of the same hidden state, so proportions and
   temperament co-vary - lanky things tend whole families of ways. */
function sampleMorphology(rng) {
    const z = Array.from({ length: 6 }, () => rng() * 2 - 1);
    const hidden = Array.from({ length: 10 }, () => {
        let s = (rng() * 2 - 1) * 0.5;
        for (const zi of z) s += (rng() * 2 - 1) * zi;
        return Math.tanh(s * 1.2);
    });
    const t = () => {
        let s = (rng() * 2 - 1) * 0.4;
        for (const hi of hidden) s += (rng() * 2 - 1) * hi;
        return Math.tanh(s) * 0.5 + 0.5;             // [0,1], non-uniform
    };
    const u = (lo, hi) => lo + (hi - lo) * t();

    return {
        // The master spectrum: organic blob .. tooled machine.
        mech: t(),
        // Body plan.
        scale: u(0.36, 0.6),          // small creatures, large room
        elong: u(1.0, 2.3),           // body length / width
        square: t(),                  // superellipse: oval .. slab
        chunk: u(0.7, 1.2),           // body minor radius
        tail: t(),                    // abdomen: tucked .. trailing
        legPairs: 2 + Math.floor(t() * 2.999),  // 4 / 6 / 8 legs
        segs: t() > 0.62 ? 3 : 2,     // elbows per leg
        heads: t() > 0.72 ? 2 : 1,
        legSpan: u(0.95, 1.5),
        legSeg: u(0.6, 0.84),         // segment/reach: stubby .. lanky
        arch: u(0.55, 1.5),           // knee arch: crouched .. cathedral
        sweepBase: u(0.3, 0.65),
        sweepRange: u(1.4, 2.3),
        neck: u(1.1, 2.0),
        headSize: u(0.4, 0.72),
        torsoLift: u(0.0, 0.2),
        stance: u(0.3, 0.5),
        // Temperament.
        speed: u(1.8, 3.6),           // units/sec - they RUN
        restlessness: u(0.5, 2.2),    // mean idle seconds (short: energy)
        jumpiness: u(0.12, 0.42),
        sleepiness: u(0.04, 0.16),
        clumsiness: u(0.05, 0.3),     // P(run target inside a wall)
        kick: u(0.25, 0.85),          // P(wall-kick on a fast bump)
        climby: u(0.25, 0.85),        // P(scaling the wall instead)
        stride: u(0.8, 1.5),
        // Motion spectrums the voter pool plays.
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

// ------------------------------------------------------------ 3D helpers

const ROOM_W = 14, ROOM_D = 7, ROOM_H = 4;   // world units; y is up
const EDGE = 0.4;                            // soft margin at every bound

/* Surfaces the bug can grip: local axes (ea, eb), outward normal n, and
   bounds. Local coords: a along ea, b along eb, h along n. */
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

/* 2-bone IK in the surface-local frame: root->knee->foot, knee arched
   along +h (away from the gripped surface - spider knees point out). */
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
        this.rng = mulberry32(fnv1a(seedStr));            // morphology
        this.live = mulberry32(fnv1a(seedStr + ':live')); // runtime
        this.p = sampleMorphology(this.rng);

        // Voter pool: seeded slow oscillators pooled through per-channel
        // weight vectors into continuous control signals - the incremental
        // sample-voter loop the generative process will one day replace.
        const N = 12;
        this.voters = Array.from({ length: N }, () => ({
            w1: 0.3 + this.rng() * 2.2, p1: this.rng() * Math.PI * 2,
            w2: 0.05 + this.rng() * 0.6, p2: this.rng() * Math.PI * 2,
        }));
        this.weights = {};
        for (const c of ['energy', 'buoy', 'tempo', 'swayA', 'swayB']) {
            this.weights[c] = Array.from(
                { length: N }, () => ((this.rng() * 2 - 1) * 2.2) / Math.sqrt(N),
            );
        }
        this.ch = { energy: 0, buoy: 0, tempo: 0, swayA: 0, swayB: 0 };

        // Surface-local state.
        this.surface = 'floor';
        this.a = ROOM_W * (0.25 + 0.5 * this.rng());
        this.b = ROOM_D * (0.25 + 0.5 * this.rng());
        this.va = 0; this.vb = 0;
        this.h = this.p.stance * this.p.scale;   // height off the surface
        this.vh = 0;
        this.heading = this.rng() * Math.PI * 2; // angle in the (ea,eb) plane
        this.spin = 0;
        this.phase = this.rng() * Math.PI * 2;
        this.lie = 0;                            // 0 standing .. 1 asleep flat
        this.stagger = 0;
        this.state = 'idle';
        this.timer = this.expo(this.p.restlessness);
        this.ta = this.a; this.tb = this.b;

        // Leg fan around the body rim, mirrored pairs.
        const pairs = this.p.legPairs;
        this.legAngles = [];
        for (let i = 0; i < pairs; i++) {
            const sweep = this.p.sweepBase
                + (pairs > 1 ? (i / (pairs - 1)) : 0.5) * this.p.sweepRange;
            this.legAngles.push(sweep, -sweep);
        }
        this.legPhase = this.legAngles.map((a, i) => (i % 2) * Math.PI + i * 0.25);

        // Head and abdomen are physical followers in WORLD space: they
        // chase anchors with momentum, so turns and surface transitions
        // whip and settle elastically instead of pivoting rigidly.
        const w0 = toWorld(SURFACES.floor, this.a, this.b, this.h);
        this.headM = { ...w0, vx: 0, vy: 0, vz: 0 };
        this.abdM = { ...w0, vx: 0, vy: 0, vz: 0 };

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

    /* ---------------- behavior ---------------- */

    act() {
        const r = this.live();
        const p = this.p;
        if (r < p.sleepiness) {
            this.state = 'sleep';
            this.timer = 3.0 + this.expo(4.0);
        } else if (r < p.sleepiness + p.jumpiness) {
            this.state = 'jump';
            // On the ceiling a jump is a push-off toward the floor; on the
            // floor a hard jump can reach the ceiling and grip it.
            const hard = this.surface === 'floor' && this.live() < 0.35;
            this.vh += (hard ? 4.6 : 2.2 + this.live() * 1.6)
                * (0.75 + 0.5 * p.floaty);
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

    /* Move to another surface, keeping coordinates and motion continuous. */
    transfer(name, a, b, va, vb) {
        this.surface = name;
        const S = SURFACES[name];
        this.a = Math.min(S.A - EDGE, Math.max(EDGE, a));
        this.b = Math.min(S.B - EDGE, Math.max(EDGE, b));
        this.va = va; this.vb = vb;
        this.h = this.p.stance * this.p.scale;
        this.vh = 0;
        this.heading = Math.atan2(vb, va) || this.heading;
        if (this.state === 'run') {
            // Re-aim somewhere on the new surface, biased onward.
            this.ta = S.A * (0.1 + 0.8 * this.live());
            this.tb = S.B * (0.1 + 0.8 * this.live());
        }
    }

    /* The bug hit edge (`which` is 'a0','a1','b0','b1') of its surface. */
    hitEdge(which) {
        const p = this.p;
        const sp = Math.hypot(this.va, this.vb);
        const S = SURFACES[this.surface];
        const fast = sp > 0.4 * p.speed;

        // Climbing routes between surfaces, where they exist.
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
            // Wall-kick: rebound hard and keep running the other way.
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
            // A plain crash.
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

        // Desired surface velocity from the FSM; reality is physics.
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
                const sp = p.speed * (1 + 0.45 * ch.energy);
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

        // Edges: climbing routes, wall-kicks, or crashes.
        if (this.a < EDGE) { this.a = EDGE; this.hitEdge('a0'); }
        else if (this.a > S.A - EDGE) { this.a = S.A - EDGE; this.hitEdge('a1'); }
        if (this.b < EDGE) { this.b = EDGE; this.hitEdge('b0'); }
        else if (this.b > SURFACES[this.surface].B - EDGE) {
            this.b = SURFACES[this.surface].B - EDGE; this.hitEdge('b1');
        }

        // Height physics along the surface normal. Bugs grip walls (no
        // gravity there); floor and ceiling have real flight.
        const standH = p.stance * p.scale;
        const lieH = 0.08 * p.scale;
        const targetH = (standH * (1 - this.lie) + lieH * this.lie)
            * (1 + 0.12 * this.ch.buoy * p.floaty);
        const airborne = this.h > targetH + 0.12;
        const onWall = this.surface.startsWith('wall');
        if (airborne && !onWall) {
            this.vh -= (9.0 - 5.5 * p.floaty) * dt;
        } else {
            const k = 60 - 38 * p.floaty;
            const damp = 6 + 6 * p.fluid;
            this.vh += (k * (targetH - this.h) - damp * this.vh) * dt;
        }
        this.h += this.vh * dt;
        if (this.h < lieH * 0.6) { this.h = lieH * 0.6; this.vh = Math.abs(this.vh) * 0.3; }

        // Cross-room flight: a hard floor jump grips the ceiling; pushing
        // off the ceiling falls all the way to the floor.
        if (this.surface === 'floor' && this.vh > 0 && this.h >= ROOM_H - standH * 1.4) {
            this.surface = 'ceiling';
            this.h = ROOM_H - this.h;
            this.vh = -this.vh * 0.25;
            this.rest();
        } else if (this.surface === 'ceiling' && this.h >= ROOM_H - standH * 1.2) {
            this.surface = 'floor';
            this.h = ROOM_H - this.h;
            this.vh = -Math.abs(this.vh) * 0.2;
        }

        // Heading earns its authority from momentum: no twirling in place.
        const sp = Math.hypot(this.va, this.vb);
        if (sp > 0.08) {
            const want = Math.atan2(this.vb, this.va);
            let diff = want - this.heading;
            while (diff > Math.PI) diff -= 2 * Math.PI;
            while (diff < -Math.PI) diff += 2 * Math.PI;
            const authority = Math.min(1, sp / (p.speed * 0.5));
            this.spin += (diff * (6 + 16 * p.speedy) * authority
                - this.spin * (5 + 4 * p.fluid)) * dt;
        } else {
            this.spin -= this.spin * 6 * dt;
        }
        this.heading += this.spin * dt;

        // Followers chase their world anchors; the lag is the elasticity.
        const s = p.scale;
        const bodyR = 0.32 * s * p.chunk;
        const ha = Math.cos(this.heading), hb = Math.sin(this.heading);
        const lift = (1 - this.lie);
        const kF = 30 + 60 * p.speedy, dF = 6 + 5 * p.fluid;
        const nose = toWorld(S, this.a + ha * bodyR * p.neck * p.elong,
            this.b + hb * bodyR * p.neck * p.elong,
            this.h + (0.10 + p.torsoLift) * s * lift);
        const tail = toWorld(S, this.a - ha * bodyR * (1.0 + 0.9 * p.tail) * p.elong,
            this.b - hb * bodyR * (1.0 + 0.9 * p.tail) * p.elong,
            this.h + (0.08 + 0.12 * p.tail + p.torsoLift * 0.5) * s * lift);
        this.follow(this.headM, nose, kF, dF, dt);
        this.follow(this.abdM, tail, kF * 0.6, dF, dt);

        this.phase += (2.0 + 3.0 * sp) * p.stride * (1 + 0.3 * ch.tempo) * dt;
    }

    follow(m, w, k, damp, dt) {
        m.vx += (k * (w.x - m.x) - damp * m.vx) * dt;
        m.vy += (k * (w.y - m.y) - damp * m.vy) * dt;
        m.vz += (k * (w.z - m.z) - damp * m.vz) * dt;
        m.x += m.vx * dt; m.y += m.vy * dt; m.z += m.vz * dt;
    }

    /* ---------------- skeleton solve ----------------
       Solves the rig in surface-local space, then transforms every joint
       to world units. Returns { body, heads[], abdomen, legs[], up }. */

    solve(t) {
        const p = this.p;
        const S = SURFACES[this.surface];
        const s = p.scale;
        const bodyR = 0.32 * s * p.chunk;
        const reach = 0.95 * p.legSpan * s;
        const seg = reach * p.legSeg;
        const sp = Math.hypot(this.va, this.vb);
        const speed = sp / p.speed;

        const bob = Math.sin(this.phase * 2) * 0.03 * s * Math.min(1, speed + 0.1);
        const wob = this.stagger > 0 ? Math.sin(t * 22) * 0.10 * this.stagger : 0;
        const bodyH = this.h + bob + wob;

        const ha = Math.cos(this.heading), hb = Math.sin(this.heading);
        const J = {};
        J.up = { x: S.n[0], y: S.n[1], z: S.n[2] };
        J.body = toWorld(S, this.a, this.b, bodyH);
        J.abdomen = { x: this.abdM.x, y: this.abdM.y, z: this.abdM.z };

        // Heads ride the nose follower; two-headed morphs split laterally.
        const px = -hb, pb2 = ha;     // surface-plane perpendicular
        J.heads = [];
        const off = p.heads === 2 ? bodyR * 0.55 : 0;
        for (let i = 0; i < p.heads; i++) {
            const sgn = p.heads === 2 ? (i === 0 ? 1 : -1) : 0;
            const d = toWorld(S, sgn * px * off, sgn * pb2 * off, 0);
            const o0 = toWorld(S, 0, 0, 0);
            J.heads.push({
                x: this.headM.x + (d.x - o0.x),
                y: this.headM.y + (d.y - o0.y),
                z: this.headM.z + (d.z - o0.z),
            });
        }

        const grounded = this.h < p.stance * s * 1.3;
        const strideLen = 0.45 * s * Math.min(1, speed + 0.05);
        const mva = sp > 0.05 ? this.va / sp : ha;
        const mvb = sp > 0.05 ? this.vb / sp : hb;

        J.legs = [];
        for (let i = 0; i < this.legAngles.length; i++) {
            const ang = this.heading + this.legAngles[i];
            const oa = Math.cos(ang), ob = Math.sin(ang);
            const hip = {
                a: this.a + oa * bodyR * 0.9 * (1 + (p.elong - 1) * 0.4 * Math.abs(Math.cos(this.legAngles[i]))),
                b: this.b + ob * bodyR * 0.9,
                h: bodyH,
            };
            const ph = this.phase + this.legPhase[i];
            let foot;
            if (this.lie >= 0.7) {
                foot = { a: this.a + oa * reach * 1.25, b: this.b + ob * reach * 1.25, h: 0.01 };
            } else if (!grounded) {
                foot = {
                    a: this.a + oa * reach * 0.5,
                    b: this.b + ob * reach * 0.5,
                    h: bodyH - seg * 0.7,
                };
            } else {
                const swing = Math.sin(ph) * strideLen;
                const liftF = Math.max(0, Math.cos(ph)) * 0.22 * s * Math.min(1, speed + 0.03);
                foot = {
                    a: this.a + oa * reach * 0.7 + mva * swing,
                    b: this.b + ob * reach * 0.7 + mvb * swing,
                    h: liftF,
                };
            }
            const sol = ik2local(hip, foot, seg, seg, p.arch);
            const joints = [hip, sol.knee];
            if (p.segs === 3) {
                // Third segment: split the thigh with a second elbow pushed
                // a little further out - a stylized extra articulation.
                const mid = {
                    a: (hip.a + sol.knee.a) / 2,
                    b: (hip.b + sol.knee.b) / 2,
                    h: (hip.h + sol.knee.h) / 2 + seg * 0.22 * p.arch,
                };
                joints.splice(1, 0, mid);
            }
            joints.push(sol.foot);
            J.legs.push(joints.map(j => toWorld(S, j.a, j.b, j.h)));
        }

        // Shadow ring on the gripped surface, under the body.
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

/* Environment styles, sampled per seed like the business-card fields.
   Furniture rolls once at construction; drawing replays through the slow
   environment inker, so the room itself barely breathes. */

const ENV_STYLES = [
    (rng) => ({
        kind: 'gym',
        seams: 2 + Math.floor(rng() * 4),
        ringR: 1.0 + rng() * 1.2,
        railH: 0.35 + rng() * 0.2,
    }),
    (rng) => ({
        kind: 'grove',
        stalks: Array.from({ length: 8 + Math.floor(rng() * 9) }, () => ({
            x: rng() * ROOM_W,
            z: ROOM_D * (0.7 + rng() * 0.3),
            h: 0.8 + rng() * 2.4,
            lean: (rng() - 0.5) * 0.6,
        })),
    }),
    (rng) => ({
        kind: 'cavern',
        arcs: Array.from({ length: 3 + Math.floor(rng() * 3) }, (_, i) => ({
            h: ROOM_H - 0.4 - rng() * 0.8,
            sag: 0.3 + rng() * 0.6,
        })),
        rocks: Array.from({ length: 6 + Math.floor(rng() * 7) }, () => ({
            x: rng() * ROOM_W,
            z: rng() < 0.5 ? 0.3 + rng() * 0.5 : ROOM_D - 0.3 - rng() * 0.5,
            r: 0.15 + rng() * 0.4,
        })),
    }),
    (rng) => ({
        kind: 'void',
        gridX: 5 + Math.floor(rng() * 6),
        gridZ: 3 + Math.floor(rng() * 4),
    }),
];

function drawEnvironment(ctx, ink, P, env) {
    // The room shell: floor and ceiling perimeters, corner posts.
    const c000 = P(0, 0, 0), c100 = P(ROOM_W, 0, 0);
    const c001 = P(0, 0, ROOM_D), c101 = P(ROOM_W, 0, ROOM_D);
    const c010 = P(0, ROOM_H, 0), c110 = P(ROOM_W, ROOM_H, 0);
    const c011 = P(0, ROOM_H, ROOM_D), c111 = P(ROOM_W, ROOM_H, ROOM_D);
    ctx.globalAlpha = 0.45;
    ink.bone(ctx, c000, c100, 'fl-near', 1.0);
    ctx.globalAlpha = 0.3;
    ink.bone(ctx, c001, c101, 'fl-far', 0.8);
    ink.bone(ctx, c000, c001, 'fl-left', 0.8);
    ink.bone(ctx, c100, c101, 'fl-right', 0.8);
    ctx.globalAlpha = 0.22;
    ink.bone(ctx, c011, c111, 'ce-far', 0.8);
    ink.bone(ctx, c010, c110, 'ce-near', 0.7);
    ink.bone(ctx, c010, c011, 'ce-left', 0.7);
    ink.bone(ctx, c110, c111, 'ce-right', 0.7);
    ink.bone(ctx, c001, c011, 'post-bl', 0.8);
    ink.bone(ctx, c101, c111, 'post-br', 0.8);
    ctx.globalAlpha = 0.10;
    ink.bone(ctx, c000, c010, 'post-fl', 0.7);
    ink.bone(ctx, c100, c110, 'post-fr', 0.7);

    if (env.kind === 'gym') {
        ctx.globalAlpha = 0.12;
        for (let i = 1; i <= env.seams; i++) {
            const z = (ROOM_D * i) / (env.seams + 1);
            ink.bone(ctx, P(0, 0, z), P(ROOM_W, 0, z), 'fl-z' + i, 0.7);
        }
        const ring = [];
        for (let i = 0; i < 14; i++) {
            const a = (i / 14) * Math.PI * 2;
            ring.push(P(ROOM_W / 2 + Math.cos(a) * env.ringR, 0, ROOM_D / 2 + Math.sin(a) * env.ringR));
        }
        ink.loop(ctx, ring, 'fl-ring', 0.7);
        ctx.globalAlpha = 0.15;
        ink.bone(ctx, P(-0.5, ROOM_H * env.railH, ROOM_D),
            P(ROOM_W + 0.5, ROOM_H * env.railH, ROOM_D), 'rail', 0.7);
    } else if (env.kind === 'grove') {
        env.stalks.forEach((st, i) => {
            ctx.globalAlpha = 0.28;
            const top = P(st.x + st.lean, st.h, st.z);
            ink.bone(ctx, P(st.x, 0, st.z), top, 'stalk' + i, 1.1);
            ctx.globalAlpha = 0.18;
            ink.bone(ctx, top, P(st.x + st.lean + 0.3, st.h - 0.35, st.z), 'frond-a' + i, 0.9);
            ink.bone(ctx, top, P(st.x + st.lean - 0.3, st.h - 0.3, st.z), 'frond-b' + i, 0.9);
        });
    } else if (env.kind === 'cavern') {
        env.arcs.forEach((arc, i) => {
            ctx.globalAlpha = 0.20;
            let prev = null;
            for (let k = 0; k <= 8; k++) {
                const x = (k / 8) * ROOM_W;
                const dip = -Math.sin((k / 8) * Math.PI) * arc.sag;
                const pt = P(x, arc.h + dip, ROOM_D * 0.9);
                if (prev) ink.bone(ctx, prev, pt, 'arc' + i + '-' + k, 0.8);
                prev = pt;
            }
        });
        env.rocks.forEach((r, i) => {
            ctx.globalAlpha = 0.28;
            const pts = [];
            for (let k = 0; k < 7; k++) {
                const a = (k / 7) * Math.PI * 2;
                pts.push(P(r.x + Math.cos(a) * r.r, Math.max(0, Math.sin(a) * r.r * 0.7), r.z));
            }
            ink.loop(ctx, pts, 'rock' + i, 0.9);
        });
    } else if (env.kind === 'void') {
        ctx.globalAlpha = 0.10;
        for (let i = 1; i < env.gridX; i++) {
            const x = (ROOM_W * i) / env.gridX;
            ink.bone(ctx, P(x, 0, 0), P(x, 0, ROOM_D), 'gx' + i, 0.6);
        }
        for (let i = 1; i < env.gridZ; i++) {
            const z = (ROOM_D * i) / env.gridZ;
            ink.bone(ctx, P(0, 0, z), P(ROOM_W, 0, z), 'gz' + i, 0.6);
        }
    }
    ctx.globalAlpha = 1;
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
        this.envInker = new Inker(
            mulberry32(fnv1a(seedStr + ':env-ink')), 0.12, this.creature.p.sketch * 0.8,
        );
        const envRng = mulberry32(fnv1a(seedStr + ':env'));
        this.env = ENV_STYLES[Math.floor(envRng() * ENV_STYLES.length)](envRng);
        // A static phase offset so two-headed shells etc. differ per seed.
        this.flair = envRng() * Math.PI * 2;

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

    /* Strokes follow the live accent hue AND the theme: ink is the accent
       hue desaturated toward the theme's text lightness, eyes the full
       accent. Re-read every ~30 frames so LOGS / dark-toggle re-ink live. */
    refreshColors() {
        const cs = getComputedStyle(document.documentElement);
        const hue = parseFloat(cs.getPropertyValue('--accent-hue')) || 161;
        const dark = document.documentElement.getAttribute('data-theme') !== 'light';
        this.colors = {
            line: `hsl(${hue} 38% ${dark ? 80 : 24}%)`,
            shade: `hsl(${hue} 30% ${dark ? 65 : 40}%)`,
            accent: cs.getPropertyValue('--accent').trim() || `hsl(${hue} 87% 40%)`,
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

    /* Bleachers camera: above the near sideline of a large room, looking
       across. Perspective divide gives depth. */
    project(p) {
        const camY = 2.2, camD = 8.0;
        const f = this.w * 0.52;
        const d = f / (camD + p.z);
        return {
            x: this.w / 2 + (p.x - ROOM_W / 2) * d,
            y: this.h * 0.42 + (camY - p.y) * d,
            d, z: p.z,
        };
    }

    /* A body shell: superellipse (oval..slab) elongated along the body
       axis, its rim modulated by mech teeth and organic waver - the
       business-card discrete..fractal blend, in 3D. */
    shell(ctx, inker, center, axis, major, minor, key, n = 16) {
        const p = this.creature.p;
        const pe = 2 + 6 * p.square;
        const ca = Math.cos(axis), sa = Math.sin(axis);
        const pts = [];
        for (let i = 0; i < n; i++) {
            const th = (i / n) * Math.PI * 2;
            const ct = Math.cos(th), st = Math.sin(th);
            // Superellipse radius.
            const r = 1 / Math.pow(
                Math.pow(Math.abs(ct), pe) + Math.pow(Math.abs(st), pe), 1 / pe);
            // Rim modulation: square teeth when mechanical, slow waver when
            // organic - coupled, never both at full strength.
            const teeth = Math.tanh(5 * Math.sin(th * 7 + this.flair)) * 0.07 * p.mech;
            const waver = Math.sin(th * 3 + this.flair * 2) * 0.08 * (1 - p.mech);
            const m = 1 + teeth + waver;
            const ex = ct * r * major * m, ey = st * r * minor * m;
            pts.push({ x: center.x + ex * ca - ey * sa, y: center.y + ex * sa + ey * ca });
        }
        inker.loop(ctx, pts, key, 1.1);
    }

    /* A gear ring: the mechanical heart, drawn only on machine-leaning
       morphs. Teeth emerge from radius modulation, never drawn discretely. */
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

    draw(dt) {
        const { ctx, w, h } = this;
        if (!w) return;
        const c = this.creature;
        const p = c.p;
        const inker = this.inker;
        inker.boil(this.t);
        this.envInker.boil(this.t);

        ctx.clearRect(0, 0, w, h);
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        const P = (x, y, z) => this.project({ x, y, z });

        // ---- the room ----
        ctx.strokeStyle = this.colors.shade;
        ctx.lineWidth = 1.2;
        drawEnvironment(ctx, this.envInker, P, this.env);

        // ---- the creature ----
        const J = c.solve(this.t);

        // Ink springs chase true joints in 3D, then project.
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
        const abdomen = chase('abd', J.abdomen);
        const heads = J.heads.map((hd, i) => chase('head' + i, hd));
        const legs = J.legs.map((joints, i) =>
            joints.map((j, k2) => chase(`leg${i}-${k2}`, j)));

        // Shadow on the gripped surface.
        ctx.strokeStyle = this.colors.shade;
        ctx.globalAlpha = 0.15;
        ctx.lineWidth = 1.1;
        inker.loop(ctx, J.shadow.map(s => this.project(s)), 'shadow', 1.0);
        ctx.globalAlpha = 1;

        const pBody = this.project(body);
        const lw = (base) => Math.max(0.8, base * pBody.d / 160);

        // Far legs first, fainter and thinner: depth in the ink.
        ctx.strokeStyle = this.colors.line;
        const order = legs.map((joints, i) => ({ joints, i, z: joints[1].z }))
            .sort((a, b2) => b2.z - a.z);
        for (const leg of order) {
            const far = leg.z > body.z;
            ctx.globalAlpha = far ? 0.45 : 0.95;
            ctx.lineWidth = lw(far ? 1.3 : 1.8);
            const pts = leg.joints.map(j => this.project(j));
            for (let s2 = 0; s2 < pts.length - 1; s2++) {
                inker.bone(ctx, pts[s2], pts[s2 + 1], `leg${leg.i}-s${s2}`, 1.2);
            }
            // Axle dots at the joints on machine-leaning morphs.
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

        // Body shells: an elongated superellipse thorax, a trailing
        // abdomen, and 1-2 heads - all on the mech..organic rim blend.
        const pAbd = this.project(abdomen);
        const pHead0 = this.project(heads[0]);
        const axis = Math.atan2(pHead0.y - pAbd.y, pHead0.x - pAbd.x);
        const u = pBody.d * 0.32 * p.scale * p.chunk;
        ctx.lineWidth = lw(1.8);
        const tailR = u * (0.9 + 0.8 * p.tail);
        this.shell(ctx, inker, pAbd, axis, tailR, tailR * 0.78, 'sh-abd', 14);
        this.shell(ctx, inker, pBody, axis, u * p.elong, u * 0.8, 'sh-body', 16);
        inker.bone(ctx, pAbd, pBody, 'sp-1', 0.9);

        if (p.mech > 0.5) {
            ctx.globalAlpha = 0.5;
            ctx.lineWidth = lw(1.2);
            this.gear(ctx, inker, pBody, u * 0.45, 'gear');
            ctx.globalAlpha = 1;
        }

        ctx.lineWidth = lw(1.6);
        for (let i = 0; i < heads.length; i++) {
            const ph = this.project(heads[i]);
            const hr = u * p.headSize;
            this.shell(ctx, inker, ph, axis, hr, hr * 0.85, 'sh-head' + i, 10);
            inker.bone(ctx, pBody, ph, 'sp-2-' + i, 0.8);

            // Feelers and accent eyes per head.
            const fwd = { x: ph.x - pBody.x, y: ph.y - pBody.y };
            const fl = Math.hypot(fwd.x, fwd.y) || 1;
            ctx.lineWidth = lw(1.1);
            for (const sgn of [-1, 1]) {
                const tip = {
                    x: ph.x + (fwd.x / fl) * hr * 1.6 - (fwd.y / fl) * sgn * hr * 0.7,
                    y: ph.y + (fwd.y / fl) * hr * 1.6 + (fwd.x / fl) * sgn * hr * 0.7
                        + Math.sin(this.t * 6 + sgn + i) * hr * 0.15,
                };
                inker.bone(ctx, ph, tip, `feel${i}${sgn}`, 0.8);
            }
            ctx.lineWidth = lw(1.6);
            ctx.fillStyle = this.colors.accent;
            for (const sgn of [-1, 1]) {
                ctx.beginPath();
                ctx.arc(
                    ph.x + (fwd.x / fl) * hr * 0.45 - (fwd.y / fl) * sgn * hr * 0.32,
                    ph.y + (fwd.y / fl) * hr * 0.45 + (fwd.x / fl) * sgn * hr * 0.32,
                    Math.max(1.1, hr * 0.16), 0, Math.PI * 2,
                );
                ctx.fill();
            }
        }
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
