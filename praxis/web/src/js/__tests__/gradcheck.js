// gradcheck.js - finite-difference check that the autograd is exact.
// Perturbs a sampling of parameters by +-eps and compares the numeric gradient
// to the analytic one from backward(). Run: `node gradcheck.js`

import { Nanoformer, crossEntropy, backward } from "../nanoformer.js";

function lossOf(m, ids, targets) {
  let x = m.inputs(ids);
  for (const b of m.blocks) x = b.forward(x);
  return crossEntropy(m.blocks[m.blocks.length - 1].logits(x), targets).data[0];
}

const VOCAB = 16;
const m = new Nanoformer({ vocab: VOCAB, d: 16, hidden: 32, layers: 2, maxT: 32, seed: 3 });
const ids = [1, 5, 2, 9, 0, 7];
const targets = [5, 2, 9, 0, 7, 3];

// Analytic grads.
const params = [m.tok, m.pos, ...m.blocks.flatMap((b) => b.params())];
for (const p of params) p.grad.fill(0);
let x = m.inputs(ids);
for (const b of m.blocks) x = b.forward(x);
backward(crossEntropy(m.blocks[m.blocks.length - 1].logits(x), targets));

const eps = 1e-6;
let worst = 0;
const rng = (() => { let s = 11; return () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff; })();
const targetsList = [
  ["tok", m.tok], ["pos", m.pos],
  ["Wq0", m.blocks[0].Wq], ["Wv0", m.blocks[0].Wv], ["W1_0", m.blocks[0].W1],
  ["n1_0", m.blocks[0].n1], ["M0", m.blocks[0].M], ["M1", m.blocks[1].M],
  ["Wo1", m.blocks[1].Wo],
];

for (const [name, p] of targetsList) {
  for (let trial = 0; trial < 4; trial++) {
    const i = Math.floor(rng() * p.data.length);
    const orig = p.data[i];
    p.data[i] = orig + eps;
    const lp = lossOf(m, ids, targets);
    p.data[i] = orig - eps;
    const lm = lossOf(m, ids, targets);
    p.data[i] = orig;
    const numeric = (lp - lm) / (2 * eps);
    const analytic = p.grad[i];
    const denom = Math.max(1e-8, Math.abs(numeric) + Math.abs(analytic));
    const rel = Math.abs(numeric - analytic) / denom;
    worst = Math.max(worst, rel);
    const flag = rel > 1e-4 ? "  <-- FAIL" : "";
    console.log(
      `${name}[${i}]  analytic ${analytic.toExponential(3)}  numeric ${numeric.toExponential(3)}  rel ${rel.toExponential(2)}${flag}`
    );
  }
}

console.log(`\nworst relative error: ${worst.toExponential(2)}  ${worst < 1e-4 ? "PASS" : "FAIL"}`);
